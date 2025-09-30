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

	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

type worker struct {
	ctx      context.Context
	id       int
	wc       chan *openaiserverapi.CompletionReqCtx
	finished chan *worker
	s        *VllmSimulator
}

func (w *worker) waitForRequests() {
	for {
		select {
		case <-w.ctx.Done():
			w.s.logger.Info("worker done")
			return
		case req := <-w.wc:
			// w.s.incrementLora(req.CompletionReq.GetModel())
			w.s.processRequest(req)
			w.s.decrementLora(req.CompletionReq.GetModel())
			w.finished <- w
		}
	}
}

func (s *VllmSimulator) processRequest(reqCtx *openaiserverapi.CompletionReqCtx) {
	req := reqCtx.CompletionReq
	model := req.GetModel()
	displayModel := s.getDisplayedModelName(model)

	// decrement waiting and increment running requests count
	s.metrics.waitingReqChan <- -1
	s.metrics.runReqChan <- 1

	if s.isLora(model) {
		// update loraInfo metric to reflect that
		// the request has changed its status from waiting to running
		s.metrics.lorasChan <- loraUsage{model, runningUsageState}
	}

	if s.config.EnableKVCache && !reqCtx.IsChatCompletion {
		// kv cache is currently supported for /completion API only
		if err := s.kvcacheHelper.OnRequestStart(req); err != nil {
			s.sendCompletionError(reqCtx.HTTPReqCtx, openaiserverapi.NewCompletionError(err.Error(), fasthttp.StatusInternalServerError, nil), false)
		}
	}

	s.logger.Info("handling", "req", reqCtx.CompletionReq.GetRequestID())

	var responseTokens []string
	var finishReason string
	var err error
	var toolCalls []openaiserverapi.ToolCall
	var completionTokens int
	if reqCtx.IsChatCompletion &&
		req.GetToolChoice() != openaiserverapi.ToolChoiceNone &&
		req.GetTools() != nil {
		toolCalls, completionTokens, err =
			openaiserverapi.CreateToolCalls(req.GetTools(), req.GetToolChoice(), s.config)
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
		s.logger.Info("handling OK", "req", reqCtx.CompletionReq.GetRequestID())

		usageData := openaiserverapi.Usage{
			PromptTokens:     req.GetNumberOfPromptTokens(),
			CompletionTokens: completionTokens,
			TotalTokens:      req.GetNumberOfPromptTokens() + completionTokens,
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
	}
	s.logger.Info("response sent", "id", req.GetRequestID())

	reqCtx.Wg.Done()
}

// getFreeWorker returns a free worker or nil if none are available (non-blocking)
func (s *VllmSimulator) getFreeWorker() *worker {
	select {
	case w := <-s.workers:
		s.logger.Info("GetFreeWorker", "worker", w.id)
		return w
	default:
		s.logger.Info("GetFreeWorker", "worker", "no free workers")
		return nil
	}
}
