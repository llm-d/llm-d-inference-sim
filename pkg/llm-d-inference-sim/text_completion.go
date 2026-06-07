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
	"strconv"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
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

// ValidateBody checks that a /v1/completions/render body has the required text
// shape — at minimum, a non-empty prompt with non-empty entries. Catches
// chat-shaped bodies (which JSON-unmarshal cleanly into
// TextCompletionsParsedRequest with empty Prompt).
func (t *TextCompletionsParsedRequest) ValidateBody() *openaiserverapi.Error {
	if len(t.Prompt) == 0 {
		serverErr := openaiserverapi.NewError("prompt array must contain at least one prompt",
			fasthttp.StatusBadRequest, nil)
		return &serverErr
	}
	for _, p := range t.Prompt {
		if p.IsTokens() {
			if len(p.Tokens) == 0 {
				serverErr := openaiserverapi.NewError("prompt must not contain an empty token-id array",
					fasthttp.StatusBadRequest, nil)
				return &serverErr
			}
		} else if p.Text == "" {
			serverErr := openaiserverapi.NewError("prompt must not contain an empty string",
				fasthttp.StatusBadRequest, nil)
			return &serverErr
		}
	}
	return nil
}

// Render tokenizes each prompt for /v1/completions/render (passing
// pre-tokenized prompts through verbatim) and returns one token slice per
// prompt. Features is always nil for text completions.
func (t *TextCompletionsParsedRequest) Render(tk tokenizer.Tokenizer) ([][]uint32, *openaiserverapi.RenderMMFeatures, error) {
	result := make([][]uint32, len(t.Prompt))
	for i, p := range t.Prompt {
		tokens := p.Tokens
		if !p.IsTokens() {
			var err error
			tokens, _, err = tk.RenderText(p.Text)
			if err != nil {
				return nil, nil, err
			}
		}
		result[i] = tokens
	}
	return result, nil, nil
}

func (t *TextCompletionsParsedRequest) validate(_ *toolsValidator) *openaiserverapi.Error {
	if err := t.ValidateBody(); err != nil {
		return err
	}
	return validateRequest(t)
}

func (t *TextCompletionsParsedRequest) AsString() string {
	return textCompletionsAsString(t.RequestID)
}

// split converts the parsed wire form into one or more processing-form
// TextCompletionsRequest values. Each sub-request gets the parent envelope and
// a "<requestID>-<i>" id stamped by openaiserverapi.AsSingle. When n > 1 each
// prompt produces n sub-requests, so the total is len(Prompt) * n.
func (t *TextCompletionsParsedRequest) split() []Request {
	n := t.GetN()
	out := make([]Request, 0, len(t.Prompt)*n)
	for i := range t.Prompt {
		for range n {
			out = append(out, &TextCompletionsRequest{TextCompletionsRequest: t.AsSingle(i)})
		}
	}
	return out
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

func (t *TextCompletionsRequest) validate(_ *toolsValidator) *openaiserverapi.Error {
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

// split returns n copies of this request, one per completion choice. Each
// sub-request shares the same underlying TextCompletionsRequest. When n==1
// (the default) this degenerates to the original single-element slice.
func (t *TextCompletionsRequest) split() []Request {
	n := t.GetN()
	out := make([]Request, n)
	for i := range n {
		cp := *t
		out[i] = &cp
	}
	return out
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
	if t.req.Prompt.IsTokens() {
		return t.req.Prompt.Tokens, nil, nil, nil
	}

	tokens, strTokens, err := t.sim.Tokenizer.RenderText(t.req.Prompt.Text)
	return tokens, strTokens, nil, err
}

func (t *textCompletionReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

func (t *textCompletionReqCtx) tokenizedPromptForEcho() (*openaiserverapi.Tokenized, error) {
	if t.req.Prompt.IsTokens() {
		// prompt arrived as token ids; render each id as its decimal form with
		// a trailing comma on all but the last so joining the Strings with ""
		// yields "id0,id1,...,idN" — what echo mode replays back to the client
		ids := t.req.Prompt.Tokens
		strs := make([]string, len(ids))
		for i, id := range ids {
			strs[i] = strconv.FormatUint(uint64(id), 10)
			if i < len(ids)-1 {
				strs[i] += ","
			}
		}
		return &openaiserverapi.Tokenized{Tokens: ids, Strings: strs}, nil
	}
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
