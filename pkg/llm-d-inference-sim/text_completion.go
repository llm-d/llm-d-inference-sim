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

func (t *textCompletionRequest) buildRequestContext(simCtx *simContext, channel chan *responseInfo, respBuilder responseBuilder) requestContext {
	reqCtx := &textCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel, respBuilder),
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

func (t *textCompletionReqCtx) tokenizedPromptForEcho() (*openaiserverapi.Tokenized, error) {
	return t.req.TokenizedPrompt(), nil
}

var _ requestContext = (*textCompletionReqCtx)(nil)

// Implementation of responseContext for /completions requests
type textCompletionResponseCtx struct {
	baseResponseContext
}

func (respCtx *textCompletionResponseCtx) toolCalls() []openaiserverapi.ToolCall {
	return nil
}

var _ responseContext = (*textCompletionResponseCtx)(nil)
