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

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

// Implementation of request for /completions requests
type TextCompletionsRequest struct {
	openaiserverapi.TextCompletionsRequest
}

// reads and parses data from the body of the given request
func (t *TextCompletionsRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, t)
}

func (t *TextCompletionsRequest) validate(toolsValidator *toolsValidator) (string, int) {
	return validateRequest(t)
}

func (t *TextCompletionsRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo]) requestContext {
	reqCtx := &textCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel),
		req:                t,
	}
	// wire textCompletionReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (t *TextCompletionsRequest) AsString() string {
	return "text completion request (req id " + t.RequestID + ")"
}

func (t *TextCompletionsRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *openaiserverapi.Tokenized, finishReason *string, usageData *openaiserverapi.Usage, sendUsageData bool,
	logprobs *int, toolCalls []openaiserverapi.ToolCall) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, t.GetRequestID(), t.IsDoRemotePrefill(), t.IsDoRemoteDecode(), t.GetNumberOfCachedPromptTokens())
	return &textCompletionsResponseCtx{
		baseResponseContext: base,
	}
}

var _ Request = (*TextCompletionsRequest)(nil)

// Implementation of requestContext for /completions requests
type textCompletionReqCtx struct {
	baseRequestContext
	req *TextCompletionsRequest
}

func (t *textCompletionReqCtx) request() Request {
	return t.req
}

func (t *textCompletionReqCtx) encode() ([]uint32, []string, *openaiserverapi.RenderMMFeatures, error) {
	tokens, strTokens, err := t.sim.Tokenizer.RenderText(t.req.Prompt)
	return tokens, strTokens, nil, err
}

func (t *textCompletionReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

func (t *textCompletionReqCtx) tokenizedPromptForEcho() (*openaiserverapi.Tokenized, error) {
	return t.req.TokenizedPrompt(), nil
}

var _ requestContext = (*textCompletionReqCtx)(nil)

// Implementation of responseContext for /completions requests
type textCompletionsResponseCtx struct {
	baseResponseContext
}

func (respCtx *textCompletionsResponseCtx) ToolCalls() []openaiserverapi.ToolCall {
	return nil
}

var _ ResponseContext = (*textCompletionsResponseCtx)(nil)
