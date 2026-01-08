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
	"encoding/json"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

// Implementation of request for /completions requests
type textCompletionRequest struct {
	openaiserverapi.TextCompletionRequest
}

// reads and parses data from the body of the given request
func (t *textCompletionRequest) unmarshal(data []byte) error {
	return json.Unmarshal(data, t)
}

func (t *textCompletionRequest) validate(config *common.Configuration, toolsValidator *common.ToolsValidator) (string, int) {
	return validateRequest(t, config)
}

func (t *textCompletionRequest) buildRequestContext(simCtx *simContext, ctx *fasthttp.RequestCtx, wg *sync.WaitGroup) requestContext {
	reqCtx := &textCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, ctx, wg),
		req:                t,
	}
	// wire textCompletionReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (t *textCompletionRequest) asString() string {
	return "text completion request (req id " + t.RequestID + ")"
}

func (t *textCompletionRequest) createResponseContext(displayModel string, responseTokens []string, finishReason *string,
	usageData *openaiserverapi.Usage, sendUsageData bool, logprobs *int, toolCalls []openaiserverapi.ToolCall) responseContext {
	base := newBaseResponseContext(displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, t.GetRequestID(), t.IsDoRemotePrefill(), t.IsDoRemoteDecode(), t.GetNumberOfCachedPromptTokens())
	return &textCompletionResponseCtx{
		baseResponseContext: base,
	}
}

var _ request = (*textCompletionRequest)(nil)

// Implementation of requestContext for /completions requests
type textCompletionReqCtx struct {
	baseRequestContext
	req *textCompletionRequest
}

func (t *textCompletionReqCtx) request() request {
	return t.req
}

func (t *textCompletionReqCtx) kvCacheGetCacheHitInfo() (*kvcache.CacheHitInfo, *openaiserverapi.Error) {
	if t.sim.config.EnableKVCache {
		cacheHitInfo, err := t.sim.kvcacheHelper.GetCacheHitInfo(t.request())
		if err != nil {
			serverError := openaiserverapi.NewError(err.Error(), fasthttp.StatusInternalServerError, nil)
			return nil, &serverError
		}
		return &cacheHitInfo, nil
	}
	return nil, nil
}

func (t *textCompletionReqCtx) kvCacheOnRequestStart() *openaiserverapi.Error {
	if t.sim.config.EnableKVCache {
		if err := t.sim.kvcacheHelper.OnRequestStart(t.request()); err != nil {
			serverError := openaiserverapi.NewError(err.Error(), fasthttp.StatusInternalServerError, nil)
			return &serverError
		}
	}
	return nil
}

func (t *textCompletionReqCtx) kvCacheOnRequestEnd() {
	if t.sim.config.EnableKVCache {
		if err := t.sim.kvcacheHelper.OnRequestEnd(t.request().GetRequestID()); err != nil {
			t.sim.logger.Error(err, "kv cache failed to process request end")
		}
	}
}

func (t *textCompletionReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

var _ requestContext = (*textCompletionReqCtx)(nil)

// Implementation of responseContext for /completions requests
type textCompletionResponseCtx struct {
	baseResponseContext
}

// createResponse creates the response for completion requests
func (respCtx *textCompletionResponseCtx) createResponse() openaiserverapi.CompletionResponse {
	baseResp := openaiserverapi.CreateBaseCompletionResponse(
		time.Now().Unix(), respCtx.displayModelName, respCtx.usage, respCtx.id, respCtx.remoteDecode)
	baseChoice := openaiserverapi.CreateBaseResponseChoice(0, respCtx.finishReasonPtr)
	respText := strings.Join(respCtx.respTokens, "")

	choice := openaiserverapi.CreateTextRespChoice(baseChoice, respText)

	// Generate logprobs if requested for text completion
	if respCtx.logprobs != nil && *respCtx.logprobs > 0 {
		if logprobsData := common.GenerateTextLogprobs(respCtx.respTokens, *respCtx.logprobs); logprobsData != nil &&
			len(logprobsData.Tokens) > 0 {
			choice.Logprobs = logprobsData
		} else {
			// Set to nil if generation failed or tokens is empty
			choice.Logprobs = nil
		}
	} else {
		// Explicitly ensure logprobs is nil when not requested
		choice.Logprobs = nil
	}

	baseResp.Object = textCompletionObject
	return openaiserverapi.CreateTextCompletionResponse(baseResp, []openaiserverapi.TextRespChoice{choice})
}

// createUsageChunk creates and returns a CompletionRespChunk with usage data, a single chunk of streamed completion API response,
// supports both modes
func (respCtx *textCompletionResponseCtx) createUsageChunk() openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModelName, respCtx.usageData(), respCtx.id, false)
	baseChunk.Object = textCompletionObject
	return openaiserverapi.CreateTextCompletionResponse(baseChunk, []openaiserverapi.TextRespChoice{})
}

// createTextCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion API response,
// for text completion.
func (respCtx *textCompletionResponseCtx) createCompletionChunk(token string, tool *openaiserverapi.ToolCall,
	role string, finishReason *string) openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModelName, nil, respCtx.id, false)
	baseChunk.Object = textCompletionObject

	choice := openaiserverapi.CreateTextRespChoice(openaiserverapi.CreateBaseResponseChoice(0, finishReason), token)

	// Generate logprobs if requested and token is not empty
	if respCtx.logprobs != nil && token != "" && *respCtx.logprobs > 0 {
		// Use token position based on current time
		tokenPosition := int(respCtx.creationTime) % 1000 // Simple position simulation
		logprobs := common.GenerateSingleTokenTextLogprobs(token, tokenPosition, *respCtx.logprobs)
		if logprobs != nil {
			choice.Logprobs = logprobs
		}
	}

	return openaiserverapi.CreateTextCompletionResponse(baseChunk, []openaiserverapi.TextRespChoice{choice})
}

// in text completion there is no special first chunk
func (respCtx *textCompletionResponseCtx) createFirstCompletionChunk() openaiserverapi.CompletionRespChunk {
	return nil
}

func (respCtx *textCompletionResponseCtx) toolCalls() []openaiserverapi.ToolCall {
	return nil
}

var _ responseContext = (*textCompletionResponseCtx)(nil)
