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

	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

// Implementation of request for /chat/completions requests
type chatCompletionRequest struct {
	openaiserverapi.ChatCompletionRequest
}

// reads and parses data from the body of the given request
func (c *chatCompletionRequest) unmarshal(data []byte) error {
	return json.Unmarshal(data, c)
}

func (c *chatCompletionRequest) validate(toolsValidator *toolsValidator) (string, int) {
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

func (c *chatCompletionRequest) buildRequestContext(simCtx *simContext, respSender responseSender,
	wg *sync.WaitGroup) requestContext {
	reqCtx := &chatCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, respSender, wg),
		req:                c,
	}
	// wire chatCompletionReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (c *chatCompletionRequest) asString() string {
	return "chat completion request (req id " + c.RequestID + ")"
}

func (c *chatCompletionRequest) createResponseContext(displayModel string, responseTokens *openaiserverapi.Tokenized,
	finishReason *string, usageData *openaiserverapi.Usage, sendUsageData bool, logprobs *int,
	toolCalls []openaiserverapi.ToolCall) responseContext {
	base := newBaseResponseContext(displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, c.GetRequestID(), c.IsDoRemotePrefill(), c.IsDoRemoteDecode(), c.GetNumberOfCachedPromptTokens())
	return &chatCompletionResponseCtx{
		baseResponseContext: base,
		toolsCalls:          toolCalls,
	}
}

var _ request = (*chatCompletionRequest)(nil)

// Implementation of requestContext for /chat/completions requests
type chatCompletionReqCtx struct {
	baseRequestContext
	req *chatCompletionRequest
}

func (c *chatCompletionReqCtx) request() request {
	return c.req
}

func (c *chatCompletionReqCtx) kvCacheOnRequestStart() (hitRate float64, oaiServerError *openaiserverapi.Error) {
	// kv cache is currently supported for /completion API only
	return 0, nil
}

func (c *chatCompletionReqCtx) kvCacheOnRequestEnd() {
}

func (c *chatCompletionReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	req := c.request()
	if !isToolChoiceNone(req.GetToolChoice()) &&
		req.GetTools() != nil {
		toolCalls, completionTokens, err :=
			createToolCalls(req.GetTools(), req.GetToolChoice(), c.sim.config, c.sim.random, c.sim.tokenizer)
		finishReason := common.ToolsFinishReason
		return toolCalls, completionTokens, finishReason, err
	}
	return nil, 0, "", nil
}

var _ requestContext = (*chatCompletionReqCtx)(nil)

// Implementation of responseContext for /chat/completions requests
type chatCompletionResponseCtx struct {
	baseResponseContext
	// tool calls to be sent in the response
	toolsCalls []openaiserverapi.ToolCall
}

// createResponse creates the response for chat completion requests
func (respCtx *chatCompletionResponseCtx) createResponse() openaiserverapi.CompletionResponse {
	baseResp := openaiserverapi.CreateBaseCompletionResponse(
		time.Now().Unix(), respCtx.displayModelName, respCtx.usage, respCtx.id, respCtx.remoteDecode)
	baseChoice := openaiserverapi.CreateBaseResponseChoice(0, respCtx.finishReasonPtr)
	baseResp.Object = chatCompletionObject

	message := openaiserverapi.Message{Role: openaiserverapi.RoleAssistant}
	if respCtx.toolsCalls != nil {
		message.ToolCalls = respCtx.toolsCalls
	} else {
		respText := strings.Join(respCtx.respTokens.Strings, "")
		message.Content = openaiserverapi.Content{Raw: respText}
	}

	choice := openaiserverapi.CreateChatRespChoice(baseChoice, message)

	// Generate logprobs if requested
	if respCtx.logprobs != nil && respCtx.toolsCalls == nil {
		if logprobsData := common.GenerateChatLogprobs(respCtx.respTokens.Strings, *respCtx.logprobs); logprobsData != nil &&
			len(logprobsData.Content) > 0 {
			choice.Logprobs = logprobsData
		} else {
			// Set to nil if generation failed or content is empty
			choice.Logprobs = nil
		}
	} else {
		// Explicitly ensure logprobs is nil when not requested
		choice.Logprobs = nil
	}

	return openaiserverapi.CreateChatCompletionResponse(baseResp, []openaiserverapi.ChatRespChoice{choice})
}

// createUsageChunk creates and returns a CompletionRespChunk with usage data, a single chunk of streamed completion API response,
// supports both modes
func (respCtx *chatCompletionResponseCtx) createUsageChunk() openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModelName, respCtx.usageData(), respCtx.id, false)
	baseChunk.Object = chatCompletionChunkObject
	return openaiserverapi.CreateChatCompletionResponse(baseChunk, []openaiserverapi.ChatRespChoice{})
}

// createChatCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion
// API response, for chat completion. It sets either role, or token, or tool call info in the message.
func (respCtx *chatCompletionResponseCtx) createCompletionChunk(token string, tool *openaiserverapi.ToolCall,
	role string, finishReason *string) openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModelName, nil, respCtx.id, false)
	baseChunk.Object = chatCompletionChunkObject
	chunk := openaiserverapi.CreateChatCompletionRespChunk(baseChunk,
		[]openaiserverapi.ChatRespChunkChoice{
			openaiserverapi.CreateChatRespChunkChoice(
				openaiserverapi.CreateBaseResponseChoice(0, finishReason), openaiserverapi.Message{})})

	if len(role) > 0 {
		chunk.Choices[0].Delta.Role = role
	}
	if tool != nil {
		chunk.Choices[0].Delta.ToolCalls = []openaiserverapi.ToolCall{*tool}
	} else if len(token) > 0 {
		chunk.Choices[0].Delta.Content.Raw = token

		// Generate logprobs if requested and token is not empty
		if respCtx.logprobs != nil {
			// Use token position based on current time
			tokenPosition := int(respCtx.creationTime) % 1000 // Simple position simulation
			logprobs := common.GenerateSingleTokenChatLogprobs(token, tokenPosition, *respCtx.logprobs)
			if logprobs != nil {
				chunk.Choices[0].Logprobs = &openaiserverapi.ChatLogprobs{
					Content: []openaiserverapi.LogprobsContent{*logprobs},
				}
			}
		}
	}

	return &chunk
}

// in chat completion first chunk contains the role
func (respCtx *chatCompletionResponseCtx) createFirstCompletionChunk() openaiserverapi.CompletionRespChunk {
	return respCtx.createCompletionChunk("", nil, openaiserverapi.RoleAssistant, nil)
}

func (respCtx *chatCompletionResponseCtx) toolCalls() []openaiserverapi.ToolCall {
	return respCtx.toolsCalls
}

var _ responseContext = (*chatCompletionResponseCtx)(nil)
