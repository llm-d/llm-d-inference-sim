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
	"fmt"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

// TextCompletionsParsedRequest is the wire form of /completions requests:
// it is unmarshaled from the HTTP body and immediately split into one or more
// TextCompletionsRequest values inside HandleRequest. Workers never see this
// type, so buildRequestContext / createResponseContext are unreachable on it.
type TextCompletionsParsedRequest struct {
	openaiserverapi.TextCompletionsParsedRequest
}

func (t *TextCompletionsParsedRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, t)
}

func (t *TextCompletionsParsedRequest) validate(_ *toolsValidator) (string, int) {
	if t.Prompt.IsArray() && len(t.Prompt.Array()) == 0 {
		return "prompt array must contain at least one prompt", fasthttp.StatusBadRequest
	}
	return validateRequest(t)
}

func (t *TextCompletionsParsedRequest) AsString() string {
	return textCompletionsAsString(t.RequestID)
}

// split converts the parsed wire form into one or more processing-form
// TextCompletionsRequest values. When the client sent a single string the
// result has length 1 and the original RequestID is reused; when the client
// sent an array (even of length 1) each element gets a "<requestID>-<i>"
// suffix.
func (t *TextCompletionsParsedRequest) split() []Request {
	if !t.Prompt.IsArray() {
		return []Request{t.toSingle(t.Prompt.String(), t.RequestID)}
	}
	arr := t.Prompt.Array()
	out := make([]Request, len(arr))
	for i, prompt := range arr {
		out[i] = t.toSingle(prompt, fmt.Sprintf("%s-%d", t.RequestID, i))
	}
	return out
}

// toSingle builds a processing-form TextCompletionsRequest sharing the parsed
// request's envelope (model, lora, KV params, stream options, …) with the
// given prompt and requestID. The envelope copy is delegated to
// openaiserverapi.TextCompletionsParsedRequest.AsSingle so this package
// doesn't need to reach into the unexported base.
func (t *TextCompletionsParsedRequest) toSingle(prompt, requestID string) *TextCompletionsRequest {
	sub := &TextCompletionsRequest{TextCompletionsRequest: t.AsSingle(prompt)}
	sub.SetRequestID(requestID)
	return sub
}

// buildRequestContext / createResponseContext are required by the Request
// interface but are unreachable: HandleRequest always calls split first and
// only the resulting TextCompletionsRequest sub-requests reach a worker.
func (t *TextCompletionsParsedRequest) buildRequestContext(_ *SimContext, _ common.Channel[*ResponseInfo],
	_ int, _ func()) requestContext {
	panic("TextCompletionsParsedRequest.buildRequestContext: split must be called first")
}

func (t *TextCompletionsParsedRequest) createResponseContext(_ requestContext, _ string,
	_ *openaiserverapi.Tokenized, _ *string, _ *openaiserverapi.Usage, _ bool,
	_ *int, _ []openaiserverapi.ToolCall, _ bool) ResponseContext {
	panic("TextCompletionsParsedRequest.createResponseContext: split must be called first")
}

var _ Request = (*TextCompletionsParsedRequest)(nil)

// TextCompletionsRequest is the processing form: a /completions request that
// always carries a single prompt. Produced by TextCompletionsParsedRequest.split.
type TextCompletionsRequest struct {
	openaiserverapi.TextCompletionsRequest
}

func (t *TextCompletionsRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, t)
}

func (t *TextCompletionsRequest) validate(_ *toolsValidator) (string, int) {
	return validateRequest(t)
}

func (t *TextCompletionsRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo],
	choiceIdx int, doneFn func()) requestContext {
	reqCtx := &textCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel, choiceIdx, doneFn),
		req:                t,
	}
	// wire textCompletionReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx
	return reqCtx
}

// split is a no-op: TextCompletionsRequest is already the single-prompt form.
func (t *TextCompletionsRequest) split() []Request {
	return []Request{t}
}

func (t *TextCompletionsRequest) AsString() string {
	return textCompletionsAsString(t.RequestID)
}

func (t *TextCompletionsRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *openaiserverapi.Tokenized, finishReason *string, usageData *openaiserverapi.Usage, sendUsageData bool,
	logprobs *int, toolCalls []openaiserverapi.ToolCall, _ bool) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, t.GetRequestID(), t.IsDoRemotePrefill(), t.IsDoRemoteDecode(), t.GetNumberOfCachedPromptTokens())
	return &textCompletionsResponseCtx{
		baseResponseContext: base,
	}
}

var _ Request = (*TextCompletionsRequest)(nil)

func textCompletionsAsString(requestID string) string {
	return "text completion request (req id " + requestID + ")"
}

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
