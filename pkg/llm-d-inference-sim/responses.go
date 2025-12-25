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

type responseContext struct {
	// the ID of the request
	requestID string
	// is this chat completion or text completion
	isChatCompletion bool
	// creation time of the response
	creationTime int64
	// indicates whether do_remote_prefill field is true in the request
	doRemotePrefill bool
	// indicates whether do_remote_decode field is true in the request
	doRemoteDecode bool
	// the number of prompt tokens that are in the local KV Cache
	nCachedPromptTokens int
	// tokenized content to be sent in the response
	responseTokens []string
	// tool calls to be sent in the response
	toolCalls []openaiserverapi.ToolCall
	// display model name returned to the client and used in metrics. It is either the first alias
	// from --served-model-name (for a base-model request) or the LoRA adapter name (for a LoRA request)
	displayModel string
	// a pointer to a string that represents finish reason, can be nil or stop or length, ...
	finishReason *string
	// usage (tokens statistics) for this response
	usageData *openaiserverapi.Usage
	// indicates whether to send usage data in this response
	sendUsageData bool
	// number of logprob options to include or nil if no logprobs needed
	logprobs *int
}

// createCompletionResponse creates the response for completion requests, supports both completion request types (text and chat)
// as defined by isChatCompletion
func createCompletionResponse(respCtx *responseContext) openaiserverapi.CompletionResponse {
	baseResp := openaiserverapi.CreateBaseCompletionResponse(
		time.Now().Unix(), respCtx.displayModel, respCtx.usageData, respCtx.requestID)

	if respCtx.doRemoteDecode {
		baseResp.KVParams = &openaiserverapi.KVTransferParams{}
		// add special fields related to the prefill pod special behavior
		baseResp.KVParams.DoRemoteDecode = false
		baseResp.KVParams.DoRemotePrefill = true
		// currently remote prefill information is hard-coded
		baseResp.KVParams.RemoteBlockIds = []string{"DUMMY_ID"}
		baseResp.KVParams.RemoteEngineId = "DUMMY_ID"
		baseResp.KVParams.RemoteHost = "DUMMY"
		baseResp.KVParams.RemotePort = 1234
		baseResp.KVParams.TPSize = 1
	}

	baseChoice := openaiserverapi.CreateBaseResponseChoice(0, respCtx.finishReason)

	respText := strings.Join(respCtx.responseTokens, "")
	if respCtx.isChatCompletion {
		baseResp.Object = chatCompletionObject

		message := openaiserverapi.Message{Role: openaiserverapi.RoleAssistant}
		if respCtx.toolCalls != nil {
			message.ToolCalls = respCtx.toolCalls
		} else {
			message.Content = openaiserverapi.Content{Raw: respText}
		}

		choice := openaiserverapi.CreateChatRespChoice(baseChoice, message)

		// Generate logprobs if requested
		if respCtx.logprobs != nil && respCtx.toolCalls == nil {
			if logprobsData := common.GenerateChatLogprobs(respCtx.responseTokens, *respCtx.logprobs); logprobsData != nil &&
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

	choice := openaiserverapi.CreateTextRespChoice(baseChoice, respText)

	// Generate logprobs if requested for text completion
	if respCtx.logprobs != nil && *respCtx.logprobs > 0 {
		if logprobsData := common.GenerateTextLogprobs(respCtx.responseTokens, *respCtx.logprobs); logprobsData != nil &&
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
func createUsageChunk(respCtx *responseContext, usageData *openaiserverapi.Usage) openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModel, usageData, respCtx.requestID)

	if respCtx.isChatCompletion {
		baseChunk.Object = chatCompletionChunkObject
		return openaiserverapi.CreateChatCompletionResponse(baseChunk, []openaiserverapi.ChatRespChoice{})
	}
	baseChunk.Object = textCompletionObject

	return openaiserverapi.CreateTextCompletionResponse(baseChunk, []openaiserverapi.TextRespChoice{})
}

// createTextCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion API response,
// for text completion.
func createTextCompletionChunk(respCtx *responseContext, token string, finishReason *string) openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModel, nil, respCtx.requestID)
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
func createChatCompletionChunk(respCtx *responseContext, token string, tool *openaiserverapi.ToolCall,
	role string, finishReason *string) openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.creationTime, respCtx.displayModel, nil, respCtx.requestID)
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
