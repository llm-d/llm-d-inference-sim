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

	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

// Implementation of request for /chat/completions requests
type ChatCompletionsRequest struct {
	api.ChatCompletionsRequest
	willSendImage bool
}

func (c *ChatCompletionsRequest) SetSendImage(v bool) { c.willSendImage = v }
func (c *ChatCompletionsRequest) SendImage() bool     { return c.willSendImage }

// reads and parses data from the body of the given request
func (c *ChatCompletionsRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, c)
}

// ValidateBody checks that a /v1/chat/completions/render body has the required
// chat shape — at minimum, a non-empty messages array. Catches text-shaped
// bodies (which JSON-unmarshal cleanly into ChatCompletionsRequest with empty
// Messages because Go ignores unknown fields by default).
func (c *ChatCompletionsRequest) ValidateBody() *api.Error {
	if len(c.Messages) == 0 {
		serverErr := api.NewError("messages must not be empty", fasthttp.StatusBadRequest, nil)
		return &serverErr
	}
	return nil
}

// Render tokenizes the chat messages for /v1/chat/completions/render and
// returns the tokens (wrapped as a single-element slice for shape parity with
// /v1/completions/render) and any mm_features.
func (c *ChatCompletionsRequest) Render(tk tokenizer.Tokenizer) ([][]uint32, *api.RenderMMFeatures, error) {
	tokens, _, features, err := tk.RenderMessages(c.Messages)
	if err != nil {
		return nil, nil, err
	}
	return [][]uint32{tokens}, features, nil
}

func (c *ChatCompletionsRequest) validate(toolsValidator *toolsValidator) *api.Error {
	for _, tool := range c.Tools {
		toolJson, err := json.Marshal(tool.Function)
		if err != nil {
			serverErr := api.NewError("Failed to marshal request tools: "+err.Error(),
				fasthttp.StatusBadRequest, nil)
			return &serverErr
		}
		err = toolsValidator.validateTool(toolJson)
		if err != nil {
			serverErr := api.NewError("Tool validation failed: "+err.Error(),
				fasthttp.StatusBadRequest, nil)
			return &serverErr
		}
	}

	return validateRequest(c)
}

func (c *ChatCompletionsRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo],
	choiceIdx int, doneFn func()) requestContext {
	reqCtx := &chatCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel, choiceIdx, doneFn),
		req:                c,
		toolIDPrefix:       common.ChatCmplToolIDPrefix,
	}
	// wire chatCompletionReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (c *ChatCompletionsRequest) AsString() string {
	return "chat completion request (req id " + c.RequestID + ")"
}

func (c *ChatCompletionsRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *api.Tokenized, finishReason *string, usageData *api.Usage,
	sendUsageData bool, logprobs *int, toolCalls []api.ToolCall, mmEncoderOnlyMode bool) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, c.GetRequestID(), c.IsDoRemotePrefill(), c.IsDoRemoteDecode(), c.GetNumberOfCachedPromptTokens())

	var ecParams map[string]api.ECTransferParams
	if mmEncoderOnlyMode {
		if features := c.MMFeatures(); features != nil {
			ecParams = buildECTransferParams(features.MMHashes)
		}
	}

	return &chatCompletionsResponseCtx{
		baseResponseContext: base,
		toolsCalls:          toolCalls,
		ecTransferParams:    ecParams,
	}
}

func (c *chatCompletionReqCtx) tokenizedPromptForEcho() (*api.Tokenized, error) {
	lastMsg := ""
	if len(c.req.Messages) > 0 {
		// in echo mode return the last message without role
		lastMsg = c.req.Messages[len(c.req.Messages)-1].PlainText(false)
	}
	tokens, strTokens, err := c.sim.Tokenizer.RenderText(lastMsg)
	if err != nil {
		return nil, err
	}
	return &api.Tokenized{Tokens: tokens, Strings: strTokens}, nil
}

// split returns n copies of this request, one per completion choice. Each
// sub-request shares the same underlying ChatCompletionsRequest so tokenization
// and prompt data are computed once and reused. When n==1 (the default) this
// degenerates to the original single-element slice.
func (c *ChatCompletionsRequest) split() []Request {
	n := c.GetN()
	out := make([]Request, n)
	for i := range n {
		cp := *c
		out[i] = &cp
	}
	return out
}

var _ Request = (*ChatCompletionsRequest)(nil)

// Implementation of requestContext for /chat/completions requests
type chatCompletionReqCtx struct {
	baseRequestContext
	req          *ChatCompletionsRequest
	toolIDPrefix string
}

func (c *chatCompletionReqCtx) request() Request {
	return c.req
}

func (c *chatCompletionReqCtx) encode() ([]uint32, []string, *api.RenderMMFeatures, error) {
	return c.sim.Tokenizer.RenderMessages(c.req.Messages)
}

func (c *chatCompletionReqCtx) createToolCalls() ([]api.ToolCall, int, string, error) {
	req := c.request()
	if !isToolChoiceNone(req.GetToolChoice()) &&
		req.GetTools() != nil {
		toolCalls, completionTokens, err :=
			createToolCalls(req.GetTools(), req.GetToolChoice(), c.sim.Config(), c.sim.Random, c.sim.Tokenizer, c.toolIDPrefix)
		finishReason := common.ToolsFinishReason
		return toolCalls, completionTokens, finishReason, err
	}
	return nil, 0, "", nil
}

var _ requestContext = (*chatCompletionReqCtx)(nil)

// Implementation of responseContext for /chat/completions requests
type chatCompletionsResponseCtx struct {
	baseResponseContext
	// tool calls to be sent in the response
	toolsCalls []api.ToolCall
	// ecTransferParams holds simulated encoder-cache transfer params per mm hash
	ecTransferParams map[string]api.ECTransferParams
}

func (respCtx *chatCompletionsResponseCtx) ToolCalls() []api.ToolCall {
	return respCtx.toolsCalls
}

func (respCtx *chatCompletionsResponseCtx) ECTransferParams() map[string]api.ECTransferParams {
	return respCtx.ecTransferParams
}

var _ ResponseContext = (*chatCompletionsResponseCtx)(nil)
