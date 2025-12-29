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
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

type responseContext interface {
	createCompletionResponse() openaiserverapi.CompletionResponse
	createUsageChunk() openaiserverapi.CompletionRespChunk
	createCompletionChunk(token string, tool *openaiserverapi.ToolCall,
		role string, finishReason *string) openaiserverapi.CompletionRespChunk
	createFirstCompletionChunk() openaiserverapi.CompletionRespChunk
	requestID() string
	usageData() *openaiserverapi.Usage
	displayModel() string
	doRemotePrefill() bool
	doRemoteDecode() bool
	numberCachedPromptTokens() int
	responseTokens() []string
	finishReason() *string
	sendUsageData() bool
	toolCalls() []openaiserverapi.ToolCall
	setCreationTime(int64)
}

type baseResponseContext struct {
	// the ID of the request
	id string
	// creation time of the response
	creationTime int64
	// indicates whether do_remote_prefill field is true in the request
	remotePrefill bool
	// indicates whether do_remote_decode field is true in the request
	remoteDecode bool
	// the number of prompt tokens that are in the local KV Cache
	nCachedPromptTokens int
	// tokenized content to be sent in the response
	respTokens []string
	// display model name returned to the client and used in metrics. It is either the first alias
	// from --served-model-name (for a base-model request) or the LoRA adapter name (for a LoRA request)
	displayModelName string
	// a pointer to a string that represents finish reason, can be nil or stop or length, ...
	finishReasonPtr *string
	// usage (tokens statistics) for this response
	usage *openaiserverapi.Usage
	// indicates whether to send usage data in this response
	sendUsage bool
	// number of logprob options to include or nil if no logprobs needed
	logprobs *int
}

type chatCompletionResponseCtx struct {
	baseResponseContext
	// tool calls to be sent in the response
	toolsCalls []openaiserverapi.ToolCall
}

type textCompletionResponseCtx struct {
	baseResponseContext
}

// createCompletionResponse creates the response for completion requests
func (respCtx *chatCompletionResponseCtx) createCompletionResponse() openaiserverapi.CompletionResponse {
	baseResp := openaiserverapi.CreateBaseCompletionResponse(
		time.Now().Unix(), respCtx.displayModelName, respCtx.usage, respCtx.id, respCtx.remoteDecode)
	baseChoice := openaiserverapi.CreateBaseResponseChoice(0, respCtx.finishReasonPtr)
	respText := strings.Join(respCtx.respTokens, "")
	baseResp.Object = chatCompletionObject

	message := openaiserverapi.Message{Role: openaiserverapi.RoleAssistant}
	if respCtx.toolsCalls != nil {
		message.ToolCalls = respCtx.toolsCalls
	} else {
		message.Content = openaiserverapi.Content{Raw: respText}
	}

	choice := openaiserverapi.CreateChatRespChoice(baseChoice, message)

	// Generate logprobs if requested
	if respCtx.logprobs != nil && respCtx.toolsCalls == nil {
		if logprobsData := common.GenerateChatLogprobs(respCtx.respTokens, *respCtx.logprobs); logprobsData != nil &&
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

// createCompletionResponse creates the response for completion requests
func (respCtx *textCompletionResponseCtx) createCompletionResponse() openaiserverapi.CompletionResponse {
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
// supports both modes (text and chat)
func (respCtx *chatCompletionResponseCtx) createUsageChunk() openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModelName, respCtx.usageData(), respCtx.id, false)
	baseChunk.Object = chatCompletionChunkObject
	return openaiserverapi.CreateChatCompletionResponse(baseChunk, []openaiserverapi.ChatRespChoice{})
}

// createUsageChunk creates and returns a CompletionRespChunk with usage data, a single chunk of streamed completion API response,
// supports both modes (text and chat)
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

// in text completion there is no special first chunk
func (respCtx *textCompletionResponseCtx) createFirstCompletionChunk() openaiserverapi.CompletionRespChunk {
	return nil
}

func (respCtx *baseResponseContext) usageData() *openaiserverapi.Usage {
	return respCtx.usage
}
func (respCtx *baseResponseContext) displayModel() string {
	return respCtx.displayModelName
}
func (respCtx *baseResponseContext) requestID() string {
	return respCtx.id
}
func (respCtx *baseResponseContext) doRemotePrefill() bool {
	return respCtx.remotePrefill
}
func (respCtx *baseResponseContext) doRemoteDecode() bool {
	return respCtx.remoteDecode
}
func (respCtx *baseResponseContext) numberCachedPromptTokens() int {
	return respCtx.nCachedPromptTokens
}
func (respCtx *baseResponseContext) responseTokens() []string {
	return respCtx.respTokens
}
func (respCtx *baseResponseContext) finishReason() *string {
	return respCtx.finishReasonPtr
}
func (respCtx *baseResponseContext) sendUsageData() bool {
	return respCtx.sendUsage
}
func (respCtx *baseResponseContext) setCreationTime(creationTime int64) {
	respCtx.creationTime = creationTime
}

func (respCtx *chatCompletionResponseCtx) toolCalls() []openaiserverapi.ToolCall {
	return respCtx.toolsCalls
}
func (respCtx *textCompletionResponseCtx) toolCalls() []openaiserverapi.ToolCall {
	return nil
}
