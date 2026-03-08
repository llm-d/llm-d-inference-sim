/*
Copyright 2026 The llm-d-inference-sim Authors.

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

package communication

import (
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

type response interface{}

type responseBuilder interface {
	createResponse(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized) response
	createUsageChunk(respCtx vllmsim.ResponseContext) response
	createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized, tool *openaiserverapi.ToolCall,
		role string, finishReason *string) response
	createFirstChunk(respCtx vllmsim.ResponseContext) response
	createLastChunk(respCtx vllmsim.ResponseContext) response
}

type textComplHTTPRespBuilder struct{}

func (respBuilder *textComplHTTPRespBuilder) createResponse(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized) response {
	baseResp := openaiserverapi.CreateBaseCompletionResponse(
		time.Now().Unix(), respCtx.DisplayModel(), respCtx.UsageData(), respCtx.RequestID(), respCtx.DoRemoteDecode())
	baseChoice := openaiserverapi.CreateBaseResponseChoice(0, respCtx.FinishReason())
	respText := strings.Join(tokens.Strings, "")

	choice := openaiserverapi.CreateTextRespChoice(baseChoice, respText)

	// Generate logprobs if requested for text completion
	if respCtx.Logprobs() != nil && *respCtx.Logprobs() > 0 {
		if logprobsData := common.GenerateTextLogprobs(tokens.Strings, *respCtx.Logprobs()); logprobsData != nil &&
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

	baseResp.Object = openaiserverapi.TextCompletionObject
	return openaiserverapi.CreateTextCompletionResponse(baseResp, []openaiserverapi.TextRespChoice{choice})
}

func (respBuilder *textComplHTTPRespBuilder) createUsageChunk(respCtx vllmsim.ResponseContext) response {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), respCtx.UsageData(), respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.TextCompletionObject
	return openaiserverapi.CreateTextCompletionResponse(baseChunk, []openaiserverapi.TextRespChoice{})
}

// createChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion API response,
// for text completion.
func (respBuilder *textComplHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized,
	tool *openaiserverapi.ToolCall, role string, finishReason *string) response {

	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), nil, respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.TextCompletionObject

	var tokensStr string
	if tokens != nil {
		tokensStr = strings.Join(tokens.Strings, "")
	}
	choice := openaiserverapi.CreateTextRespChoice(openaiserverapi.CreateBaseResponseChoice(0, finishReason), tokensStr)

	// Generate logprobs if requested and tokens is not empty
	if respCtx.Logprobs() != nil && tokens != nil && len(tokens.Strings) > 0 && *respCtx.Logprobs() > 0 {
		// Use token position based on current time
		tokenPosition := int(respCtx.CreationTime()) % 1000 // Simple position simulation
		logprobs := common.GenerateSingleTokenTextLogprobs(tokensStr, tokenPosition, *respCtx.Logprobs())
		if logprobs != nil {
			choice.Logprobs = logprobs
		}
	}

	return openaiserverapi.CreateTextCompletionResponse(baseChunk, []openaiserverapi.TextRespChoice{choice})
}

func (respBuilder *textComplHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext) response {
	return nil
}

func (respBuilder *textComplHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext) response {
	return respBuilder.createChunk(respCtx, nil, nil, "", respCtx.FinishReason())
}

var _ responseBuilder = (*textComplHTTPRespBuilder)(nil)

type chatComplHTTPRespBuilder struct{}

func (respBuilder *chatComplHTTPRespBuilder) createResponse(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized) response {
	baseResp := openaiserverapi.CreateBaseCompletionResponse(
		time.Now().Unix(), respCtx.DisplayModel(), respCtx.UsageData(), respCtx.RequestID(), respCtx.DoRemoteDecode())
	baseChoice := openaiserverapi.CreateBaseResponseChoice(0, respCtx.FinishReason())
	baseResp.Object = openaiserverapi.ChatCompletionObject

	message := openaiserverapi.Message{Role: openaiserverapi.RoleAssistant}
	if respCtx.ToolCalls() != nil {
		message.ToolCalls = respCtx.ToolCalls()
	} else {
		respText := strings.Join(tokens.Strings, "")
		message.Content = openaiserverapi.Content{Raw: respText}
	}

	choice := openaiserverapi.CreateChatRespChoice(baseChoice, message)

	// Generate logprobs if requested
	if respCtx.Logprobs() != nil && respCtx.ToolCalls() == nil {
		if logprobsData := common.GenerateChatLogprobs(tokens.Strings, *respCtx.Logprobs()); logprobsData != nil &&
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

func (respBuilder *chatComplHTTPRespBuilder) createUsageChunk(respCtx vllmsim.ResponseContext) response {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), respCtx.UsageData(), respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.ChatCompletionChunkObject
	return openaiserverapi.CreateChatCompletionResponse(baseChunk, []openaiserverapi.ChatRespChoice{})
}

// createChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion
// API response, for chat completion. It sets either role, or token, or tool call info in the message.
func (respBuilder *chatComplHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized,
	tool *openaiserverapi.ToolCall, role string, finishReason *string) response {
	baseChunk := openaiserverapi.CreateBaseCompletionResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), nil, respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.ChatCompletionChunkObject
	chunk := openaiserverapi.CreateChatCompletionRespChunk(baseChunk,
		[]openaiserverapi.ChatRespChunkChoice{
			openaiserverapi.CreateChatRespChunkChoice(
				openaiserverapi.CreateBaseResponseChoice(0, finishReason), openaiserverapi.Message{})})

	if len(role) > 0 {
		chunk.Choices[0].Delta.Role = role
	}
	if tool != nil {
		chunk.Choices[0].Delta.ToolCalls = []openaiserverapi.ToolCall{*tool}
	} else if tokens != nil && len(tokens.Strings) > 0 {
		tokensStr := strings.Join(tokens.Strings, "")
		chunk.Choices[0].Delta.Content.Raw = tokensStr

		// Generate logprobs if requested and token is not empty
		if respCtx.Logprobs() != nil {
			// Use token position based on current time
			tokenPosition := int(respCtx.CreationTime()) % 1000 // Simple position simulation
			logprobs := common.GenerateSingleTokenChatLogprobs(tokensStr, tokenPosition, *respCtx.Logprobs())
			if logprobs != nil {
				chunk.Choices[0].Logprobs = &openaiserverapi.ChatLogprobs{
					Content: []openaiserverapi.LogprobsContent{*logprobs},
				}
			}
		}
	}

	return &chunk
}

func (respBuilder *chatComplHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext) response {
	return respBuilder.createChunk(respCtx, nil, nil, openaiserverapi.RoleAssistant, nil)
}

func (respBuilder *chatComplHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext) response {
	return respBuilder.createChunk(respCtx, nil, nil, "", respCtx.FinishReason())
}

var _ responseBuilder = (*chatComplHTTPRespBuilder)(nil)

type generationGRPCRespBuilder struct{}

func (respBuilder *generationGRPCRespBuilder) createResponse(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized) response {

	var completionTokens uint32
	var outputIds []uint32
	if tokens != nil {
		completionTokens = uint32(respCtx.UsageData().CompletionTokens)
		outputIds = tokens.Tokens
	}

	return &pb.GenerateResponse{
		Response: &pb.GenerateResponse_Complete{
			Complete: &pb.GenerateComplete{
				OutputIds:        outputIds,
				PromptTokens:     uint32(respCtx.UsageData().PromptTokens),
				CompletionTokens: completionTokens,
				CachedTokens:     uint32(respCtx.NumberCachedPromptTokens()),
				FinishReason:     *respCtx.FinishReason(),
			},
		},
	}
}

func (respBuilder *generationGRPCRespBuilder) createUsageChunk(respCtx vllmsim.ResponseContext) response {
	return nil
}

func (respBuilder *generationGRPCRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized,
	tool *openaiserverapi.ToolCall, role string, finishReason *string) response {
	return &pb.GenerateResponse{
		Response: &pb.GenerateResponse_Chunk{
			Chunk: &pb.GenerateStreamChunk{
				TokenIds:         tokens.Tokens,
				PromptTokens:     uint32(respCtx.UsageData().PromptTokens),
				CachedTokens:     uint32(respCtx.NumberCachedPromptTokens()),
				CompletionTokens: uint32(len(tokens.Tokens)),
			},
		},
	}
}

func (respBuilder *generationGRPCRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext) response {
	return nil
}

func (respBuilder *generationGRPCRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext) response {
	return respBuilder.createResponse(respCtx, nil)
}

var _ responseBuilder = (*generationGRPCRespBuilder)(nil)
