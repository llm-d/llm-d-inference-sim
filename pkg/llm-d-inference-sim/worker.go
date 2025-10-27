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

// Package vllmsim implements the vLLM simulator.
package llmdinferencesim

import (
	"context"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
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
			w.logger.V(4).Info("worker done", "id", w.id)
			return
		case req := <-w.reqChan:
			w.processor.processRequest(req)
			w.finishedChan <- &requestCompleted{worker: w, model: req.CompletionReq.GetModel()}
		}
	}
}

type requestProcessor interface {
	processRequest(reqCtx *openaiserverapi.CompletionReqCtx)
}

func (s *VllmSimulator) processRequest(reqCtx *openaiserverapi.CompletionReqCtx) {
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
			s.sendCompletionError(reqCtx.HTTPReqCtx,
				openaiserverapi.NewCompletionError(err.Error(), fasthttp.StatusInternalServerError, nil),
				false)
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
			common.CreateToolCalls(req.GetTools(), req.GetToolChoice(), s.config)
		finishReason = dataset.ToolsFinishReason
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
		reqCtx.HTTPReqCtx.Error(prefix+err.Error(), fasthttp.StatusBadRequest)
	} else {
		usageData := openaiserverapi.Usage{
			PromptTokens:     s.getNumberOfPromptTokens(req),
			CompletionTokens: completionTokens,
			TotalTokens:      s.getNumberOfPromptTokens(req) + completionTokens,
		}
		if req.IsStream() {
			var usageDataToSend *openaiserverapi.Usage
			if req.IncludeUsage() {
				usageDataToSend = &usageData
			}
			s.sendStreamingResponse(
				&streamingContext{
					ctx:                 reqCtx.HTTPReqCtx,
					isChatCompletion:    reqCtx.IsChatCompletion,
					model:               displayModel,
					doRemotePrefill:     req.IsDoRemotePrefill(),
					nPromptTokens:       usageData.PromptTokens,
					nCachedPromptTokens: reqCtx.CompletionReq.GetNumberOfCachedPromptTokens(),
				},
				responseTokens, toolCalls, finishReason, usageDataToSend,
			)
		} else {
			if req.IsDoRemoteDecode() {
				// in case this is prefill pod processing, return special finish reason
				finishReason = dataset.RemoteDecodeFinishReason
			}
			s.sendResponse(reqCtx, responseTokens, toolCalls, displayModel, finishReason, &usageData)
		}

		common.WriteToChannel(s.metrics.requestSuccessChan,
			requestSuccessEvent{
				promptTokens:     usageData.PromptTokens,
				generationTokens: usageData.CompletionTokens,
				maxTokens:        reqCtx.CompletionReq.GetMaxCompletionTokens(),
				finishReason:     finishReason},
			s.logger, "metrics.requestSuccessChan")
	}

	s.logger.V(4).Info("Finished processing request", "id", req.GetRequestID())

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
