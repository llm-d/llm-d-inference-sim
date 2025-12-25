/*
Copyright 2025 The llm-d-inference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package llmdinferencesim

import (
	"context"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

// worker runs simulators requests
type worker struct {
	ctx    context.Context
	logger logr.Logger
	// worker's id
	id int
	// a channel for requests
	reqChan chan *openaiserverapi.CompletionReqCtx
	// a channel to indicate that the worker finished processing a request
	finishedChan chan *requestCompleted
	// the request processor
	processor requestProcessor
}

func (w *worker) waitForRequests() {
	for {
		select {
		case <-w.ctx.Done():
			w.logger.V(logging.TRACE).Info("worker done", "id", w.id)
			return
		case req := <-w.reqChan:
			w.processor.processRequest(req, nil)
			w.finishedChan <- &requestCompleted{worker: w, model: req.CompletionReq.GetModel()}
		}

	}
}

type requestProcessor interface {
	processRequest(reqCtx *openaiserverapi.CompletionReqCtx, wg *sync.WaitGroup)
}

func (s *VllmSimulator) processRequest(reqCtx *openaiserverapi.CompletionReqCtx, _ *sync.WaitGroup) {
	startTime := time.Now()
	wg := sync.WaitGroup{}
	wg.Add(1)

	go s.processRequestAsync(reqCtx, &wg)

	wg.Wait()
	// calculate inference time and finish e2e latency calculation only when sure that request processing was finished for streaming requests too
	common.WriteToChannel(s.context.metrics.e2eReqLatencyChan, time.Since(reqCtx.StartProcessing).Seconds(), s.context.logger, "metrics.e2eReqLatencyChan")
	common.WriteToChannel(s.context.metrics.reqInferenceTimeChan, time.Since(startTime).Seconds(), s.context.logger, "metrics.reqInferenceTimeChan")
}

func (s *VllmSimulator) processRequestAsync(reqCtx *openaiserverapi.CompletionReqCtx, wg *sync.WaitGroup) {
	req := reqCtx.CompletionReq
	respCtx, badRequestErr, serverErr := s.context.handleRequest(reqCtx)
	switch {
	case badRequestErr != "":
		s.setBadRequestError(reqCtx.HTTPReqCtx, badRequestErr)
	case serverErr != nil:
		s.sendError(reqCtx.HTTPReqCtx, *serverErr, false)
	default:
		if req.IsStream() {
			s.sendStreamingResponse(respCtx, reqCtx.HTTPReqCtx, wg)
		} else {
			s.sendResponse(reqCtx, respCtx)
			wg.Done()
		}

		common.WriteToChannel(s.context.metrics.requestSuccessChan,
			requestSuccessEvent{
				promptTokens:     respCtx.usageData.PromptTokens,
				generationTokens: respCtx.usageData.CompletionTokens,
				// currently only responses with a single choice are supported
				genTokensPerChoice: []int{respCtx.usageData.CompletionTokens},
				maxTokens:          reqCtx.CompletionReq.GetMaxCompletionTokens(),
				finishReason:       *respCtx.finishReason},
			s.context.logger, "metrics.requestSuccessChan")
	}
	s.context.logger.V(logging.DEBUG).Info("Finished processing request", "id", req.GetRequestID())
	reqCtx.Wg.Done()
}

// getFreeWorker returns a free worker or nil if none are available (non-blocking)
func (s *VllmSimulator) getFreeWorker() *worker {
	select {
	case w := <-s.freeWorkers:
		return w
	default:
		return nil
	}
}

func (s *simContext) handleRequest(reqCtx *openaiserverapi.CompletionReqCtx) (*responseContext, string, *openaiserverapi.Error) {
	req := reqCtx.CompletionReq
	model := req.GetModel()
	displayModel := s.getDisplayedModelName(model)

	// increment running requests count
	common.WriteToChannel(s.metrics.runReqChan, 1, s.logger, "metrics.runReqChan")

	if s.isLora(model) {
		// update loraInfo metric to reflect that
		// the request has changed its status from waiting to running
		common.WriteToChannel(s.metrics.lorasChan, loraUsage{model, runningUsageState}, s.logger,
			"metrics.lorasChan")
	}

	if s.config.EnableKVCache && !reqCtx.IsChatCompletion {
		// kv cache is currently supported for /completion API only
		if err := s.kvcacheHelper.OnRequestStart(req); err != nil {
			severError := openaiserverapi.NewError(err.Error(), fasthttp.StatusInternalServerError, nil)
			return nil, "", &severError
		}
	}

	var responseTokens []string
	var finishReason string
	var err error
	var toolCalls []openaiserverapi.ToolCall
	var completionTokens int
	if reqCtx.IsChatCompletion &&
		!common.IsToolChoiceNone(req.GetToolChoice()) &&
		req.GetTools() != nil {
		toolCalls, completionTokens, err =
			common.CreateToolCalls(req.GetTools(), req.GetToolChoice(), s.config, s.random)
		finishReason = common.ToolsFinishReason
	}
	if toolCalls == nil && err == nil {
		// Either no tool calls were defined, or we randomly chose not to create tool calls,
		// so we generate a response text.
		responseTokens, finishReason, err = s.dataset.GetTokens(req, s.config.Mode)
		completionTokens += len(responseTokens)
	}
	if err != nil {
		prefix := ""
		if reqCtx.IsChatCompletion {
			prefix = "failed to create chat response"
		} else {
			prefix = "failed to create text response"
		}
		s.logger.Error(err, prefix)
		return nil, prefix + err.Error(), nil
	}

	numOfInputTokens := getNumberOfPromptTokens(req)
	usageData := openaiserverapi.Usage{
		PromptTokens:     numOfInputTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      numOfInputTokens + completionTokens,
	}

	// Extract logprob data from request (unified approach)
	var logprobs *int
	if toolCalls == nil {
		logprobs = reqCtx.CompletionReq.GetLogprobs()
	}

	respCtx := &responseContext{
		responseTokens:      responseTokens,
		toolCalls:           toolCalls,
		displayModel:        displayModel,
		finishReason:        &finishReason,
		usageData:           &usageData,
		logprobs:            logprobs,
		requestID:           reqCtx.CompletionReq.GetRequestID(),
		doRemotePrefill:     reqCtx.CompletionReq.IsDoRemotePrefill(),
		doRemoteDecode:      reqCtx.CompletionReq.IsDoRemoteDecode(),
		isChatCompletion:    reqCtx.IsChatCompletion,
		nCachedPromptTokens: reqCtx.CompletionReq.GetNumberOfCachedPromptTokens(),
	}

	if req.IsStream() {
		respCtx.sendUsageData = req.IncludeUsage()
	} else {
		respCtx.sendUsageData = true
		if req.IsDoRemoteDecode() {
			// in case this is prefill pod processing, return special finish reason
			respCtx.finishReason = strPtr(common.RemoteDecodeFinishReason)
		}
	}
	return respCtx, "", nil
}
