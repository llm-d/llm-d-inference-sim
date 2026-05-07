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

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

// Implementation of request for /chat/completions requests
type ChatCompletionsRequest struct {
	openaiserverapi.ChatCompletionsRequest
}

// reads and parses data from the body of the given request
func (c *ChatCompletionsRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, c)
}

func (c *ChatCompletionsRequest) validate(toolsValidator *toolsValidator) (string, int) {
	for _, tool := range c.Tools {
		toolJson, err := json.Marshal(tool.Function)
		if err != nil {
			return "Failed to marshal request tools: " + err.Error(), fasthttp.StatusBadRequest
		}
		err = toolsValidator.validateTool(toolJson)
		if err != nil {
			return "Tool validation failed: " + err.Error(), fasthttp.StatusBadRequest
		}
	}

	return validateRequest(c)
}

func (c *ChatCompletionsRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo]) requestContext {
	reqCtx := &chatCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel),
		req:                c,
	}
	// wire chatCompletionReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (c *ChatCompletionsRequest) AsString() string {
	return "chat completion request (req id " + c.RequestID + ")"
}

func (c *ChatCompletionsRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *openaiserverapi.Tokenized, finishReason *string, usageData *openaiserverapi.Usage,
	sendUsageData bool, logprobs *int, toolCalls []openaiserverapi.ToolCall) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, c.GetRequestID(), c.IsDoRemotePrefill(), c.IsDoRemoteDecode(), c.GetNumberOfCachedPromptTokens())
	return &chatCompletionsResponseCtx{
		baseResponseContext: base,
		toolsCalls:          toolCalls,
	}
}

func (c *chatCompletionReqCtx) tokenizedPromptForEcho() (*openaiserverapi.Tokenized, error) {
	lastMsg := ""
	if len(c.req.Messages) > 0 {
		// in echo mode return the last message without role
		lastMsg = c.req.Messages[len(c.req.Messages)-1].PlainText(false)
	}
	tokens, strTokens, err := c.sim.Tokenizer.RenderText(lastMsg)
	if err != nil {
		return nil, err
	}
	return &openaiserverapi.Tokenized{Tokens: tokens, Strings: strTokens}, nil
}

var _ Request = (*ChatCompletionsRequest)(nil)

// Implementation of requestContext for /chat/completions requests
type chatCompletionReqCtx struct {
	baseRequestContext
	req *ChatCompletionsRequest
}

func (c *chatCompletionReqCtx) request() Request {
	return c.req
}

func (c *chatCompletionReqCtx) encode() ([]uint32, []string, *openaiserverapi.RenderMMFeatures, error) {
	return c.sim.Tokenizer.RenderMessages(c.req.Messages)
}

func (c *chatCompletionReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	req := c.request()
	if !isToolChoiceNone(req.GetToolChoice()) &&
		req.GetTools() != nil {
		toolCalls, completionTokens, err :=
			createToolCalls(req.GetTools(), req.GetToolChoice(), c.sim.Config, c.sim.Random, c.sim.Tokenizer)
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
	toolsCalls []openaiserverapi.ToolCall
}

func (respCtx *chatCompletionsResponseCtx) ToolCalls() []openaiserverapi.ToolCall {
	return respCtx.toolsCalls
}

var _ ResponseContext = (*chatCompletionsResponseCtx)(nil)
