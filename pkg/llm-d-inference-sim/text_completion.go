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
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
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

func (t *textCompletionRequest) validate(toolsValidator *toolsValidator) (string, int) {
	return validateRequest(t)
}

func (t *textCompletionRequest) buildRequestContext(simCtx *simContext, channel chan *responseInfo) requestContext {
	reqCtx := &textCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel),
		req:                t,
	}
	// wire textCompletionReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (t *textCompletionRequest) asString() string {
	return "text completion request (req id " + t.RequestID + ")"
}

func (t *textCompletionRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *openaiserverapi.Tokenized, finishReason *string, usageData *openaiserverapi.Usage, sendUsageData bool,
	logprobs *int, toolCalls []openaiserverapi.ToolCall) responseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
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

func (t *textCompletionReqCtx) kvCacheOnRequestStart() (hitRate float64, oaiServerError *openaiserverapi.Error) {
	if t.sim.config.EnableKVCache {
		var err error
		hitRate, err = t.sim.kvcacheHelper.OnRequestStart(t.request())
		if err != nil {
			serverError := openaiserverapi.NewError(err.Error(), fasthttp.StatusInternalServerError, nil)
			return 0, &serverError
		}
		return hitRate, nil
	}
	return 0, nil
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

func (respCtx *textCompletionResponseCtx) createResponse(tokens *openaiserverapi.Tokenized) openaiserverapi.CompletionResponse {
	baseResp := openaiserverapi.CreateBaseCompletionResponse(
		time.Now().Unix(), respCtx.displayModelName, respCtx.usage, respCtx.id, respCtx.remoteDecode)
	baseChoice := openaiserverapi.CreateBaseResponseChoice(0, respCtx.finishReasonPtr)
	respText := strings.Join(tokens.Strings, "")

	choice := openaiserverapi.CreateTextRespChoice(baseChoice, respText)

	// Generate logprobs if requested for text completion
	if respCtx.logprobs != nil && *respCtx.logprobs > 0 {
		if logprobsData := common.GenerateTextLogprobs(tokens.Strings, *respCtx.logprobs); logprobsData != nil &&
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
func (respCtx *textCompletionResponseCtx) createUsageChunk() openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModelName, respCtx.usageData(), respCtx.id, false)
	baseChunk.Object = textCompletionObject
	return openaiserverapi.CreateTextCompletionResponse(baseChunk, []openaiserverapi.TextRespChoice{})
}

// createTextCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion API response,
// for text completion.
func (respCtx *textCompletionResponseCtx) createCompletionChunk(tokens []string, tool *openaiserverapi.ToolCall,
	role string, finishReason *string) openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModelName, nil, respCtx.id, false)
	baseChunk.Object = textCompletionObject

	tokensStr := strings.Join(tokens, "")
	choice := openaiserverapi.CreateTextRespChoice(openaiserverapi.CreateBaseResponseChoice(0, finishReason), tokensStr)

	// Generate logprobs if requested and tokens is not empty
	if respCtx.logprobs != nil && len(tokens) > 0 && *respCtx.logprobs > 0 {
		// Use token position based on current time
		tokenPosition := int(respCtx.creationTime) % 1000 // Simple position simulation
		logprobs := common.GenerateSingleTokenTextLogprobs(tokensStr, tokenPosition, *respCtx.logprobs)
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
