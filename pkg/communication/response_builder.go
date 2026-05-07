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
	"encoding/json"
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

// sseChunk knows how to format itself as SSE wire bytes.
type sseChunk interface {
	SSEBytes() ([]byte, error)
}

// jsonDataChunk formats its value as "data: <json>\n\n".
type jsonDataChunk struct{ data any }

func (j *jsonDataChunk) SSEBytes() ([]byte, error) {
	b, err := json.Marshal(j.data)
	if err != nil {
		return nil, err
	}
	return []byte("data: " + string(b) + "\n\n"), nil
}

// namedEventChunk formats its value as "event: <name>\ndata: <json>\n\n".
type namedEventChunk struct {
	names []string
	data  []any
}

func (e *namedEventChunk) SSEBytes() ([]byte, error) {
	var result strings.Builder
	for i, data := range e.data {
		b, err := json.Marshal(data)
		if err != nil {
			return nil, err
		}
		result.WriteString("event: ")
		result.WriteString(e.names[i])
		result.WriteString("\ndata: ")
		result.Write(b)
		result.WriteString("\n\n")
	}
	return []byte(result.String()), nil
}

// doneMarker emits the SSE stream terminator "data: [DONE]\n\n".
type doneMarker struct{}

func (*doneMarker) SSEBytes() ([]byte, error) { return []byte("data: [DONE]\n\n"), nil }

// responseBuilder is the HTTP streaming builder interface.
type responseBuilder interface {
	createResponse(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized) any
	createUsageChunk(respCtx vllmsim.ResponseContext) sseChunk
	createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized, tool *openaiserverapi.ToolCall,
		role string, finishReason *string) sseChunk
	createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk
	createFirstChunk(respCtx vllmsim.ResponseContext) sseChunk
	createLastChunk(respCtx vllmsim.ResponseContext, finishReason string) sseChunk
	createDoneChunk() sseChunk
}

type textComplHTTPRespBuilder struct{}

func (respBuilder *textComplHTTPRespBuilder) createResponse(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized) any {
	baseResp := openaiserverapi.CreateBaseCompletionsResponse(
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
	return openaiserverapi.CreateTextCompletionsResponse(baseResp, []openaiserverapi.TextRespChoice{choice})
}

func (respBuilder *textComplHTTPRespBuilder) createUsageChunk(respCtx vllmsim.ResponseContext) sseChunk {
	if !respCtx.SendUsageData() {
		return nil
	}
	baseChunk := openaiserverapi.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), respCtx.UsageData(), respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.TextCompletionObject
	return &jsonDataChunk{data: openaiserverapi.CreateTextCompletionsResponse(baseChunk, []openaiserverapi.TextRespChoice{})}
}

// createChunk creates and returns a CompletionsRespChunk, a single chunk of streamed completion API response,
// for text completion.
func (respBuilder *textComplHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized,
	tool *openaiserverapi.ToolCall, role string, finishReason *string) sseChunk {

	baseChunk := openaiserverapi.CreateBaseCompletionsResponse(
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

	return &jsonDataChunk{data: openaiserverapi.CreateTextCompletionsResponse(baseChunk, []openaiserverapi.TextRespChoice{choice})}
}

func (respBuilder *textComplHTTPRespBuilder) createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk {
	return nil
}

func (respBuilder *textComplHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext) sseChunk {
	return nil
}

func (respBuilder *textComplHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, finishReason string) sseChunk {
	if finishReason != common.StopFinishReason {
		return nil
	}
	return respBuilder.createChunk(respCtx, nil, nil, "", respCtx.FinishReason())
}

func (*textComplHTTPRespBuilder) createDoneChunk() sseChunk { return &doneMarker{} }

var _ responseBuilder = (*textComplHTTPRespBuilder)(nil)

type chatComplHTTPRespBuilder struct{}

func (respBuilder *chatComplHTTPRespBuilder) createResponse(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized) any {
	baseResp := openaiserverapi.CreateBaseCompletionsResponse(
		time.Now().Unix(), respCtx.DisplayModel(), respCtx.UsageData(), respCtx.RequestID(), respCtx.DoRemoteDecode())
	baseChoice := openaiserverapi.CreateBaseResponseChoice(0, respCtx.FinishReason())
	baseResp.Object = openaiserverapi.ChatCompletionObject

	message := openaiserverapi.Message{Role: openaiserverapi.RoleAssistant}
	if respCtx.ToolCalls() != nil {
		message.ToolCalls = respCtx.ToolCalls()
	} else {
		respText := strings.Join(tokens.Strings, "")
		message.Content = openaiserverapi.ChatComplContent{Raw: respText}
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

	return openaiserverapi.CreateChatCompletionsResponse(baseResp, []openaiserverapi.ChatRespChoice{choice})
}

func (respBuilder *chatComplHTTPRespBuilder) createUsageChunk(respCtx vllmsim.ResponseContext) sseChunk {
	if !respCtx.SendUsageData() {
		return nil
	}
	baseChunk := openaiserverapi.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), respCtx.UsageData(), respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.ChatCompletionChunkObject
	return &jsonDataChunk{data: openaiserverapi.CreateChatCompletionsResponse(baseChunk, []openaiserverapi.ChatRespChoice{})}
}

// createChunk creates and returns a CompletionsRespChunk, a single chunk of streamed completion
// API response, for chat completion. It sets either role, or token, or tool call info in the message.
func (respBuilder *chatComplHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized,
	tool *openaiserverapi.ToolCall, role string, finishReason *string) sseChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), nil, respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.ChatCompletionChunkObject
	chunk := openaiserverapi.CreateChatCompletionsRespChunk(baseChunk,
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

	return &jsonDataChunk{data: &chunk}
}

func (respBuilder *chatComplHTTPRespBuilder) createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk {
	return nil
}

func (respBuilder *chatComplHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext) sseChunk {
	return respBuilder.createChunk(respCtx, nil, nil, openaiserverapi.RoleAssistant, nil)
}

func (respBuilder *chatComplHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, finishReason string) sseChunk {
	if finishReason != common.StopFinishReason {
		return nil
	}
	return respBuilder.createChunk(respCtx, nil, nil, "", respCtx.FinishReason())
}

func (*chatComplHTTPRespBuilder) createDoneChunk() sseChunk { return &doneMarker{} }

var _ responseBuilder = (*chatComplHTTPRespBuilder)(nil)

type generationGRPCRespBuilder struct{}

func (respBuilder *generationGRPCRespBuilder) createResponse(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized) any {

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

func (respBuilder *generationGRPCRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized) any {
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

func (respBuilder *generationGRPCRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext) any {
	return respBuilder.createResponse(respCtx, nil)
}

type responsesHTTPRespBuilder struct {
	accumulated strings.Builder
}

func (respBuilder *responsesHTTPRespBuilder) createResponse(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized) any {
	text := strings.Join(tokens.Strings, "")
	usage := respCtx.UsageData()
	return openaiserverapi.CreateResponsesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		time.Now().Unix(),
		respCtx.Instructions(),
		[]openaiserverapi.OutputItem{
			openaiserverapi.MessageOutput{
				Type:   openaiserverapi.ResponsesOutputMessage,
				Role:   openaiserverapi.RoleAssistant,
				Status: openaiserverapi.ResponsesStatusCompleted,
				Content: []openaiserverapi.OutputContent{
					{Type: openaiserverapi.ResponsesOutputText, Text: text},
				},
			},
		},
		&openaiserverapi.ResponsesUsage{
			InputTokens:  usage.PromptTokens,
			OutputTokens: usage.CompletionTokens,
			TotalTokens:  usage.TotalTokens,
		},
	)
}

func (respBuilder *responsesHTTPRespBuilder) createUsageChunk(respCtx vllmsim.ResponseContext) sseChunk {
	usage := respCtx.UsageData()
	text := respBuilder.accumulated.String()
	resp := openaiserverapi.CreateResponsesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		respCtx.CreationTime(),
		respCtx.Instructions(),
		[]openaiserverapi.OutputItem{
			openaiserverapi.MessageOutput{
				Type:   openaiserverapi.ResponsesOutputMessage,
				ID:     openaiserverapi.ResponsesMessageIDPrefix + respCtx.RequestID(),
				Role:   openaiserverapi.RoleAssistant,
				Status: openaiserverapi.ResponsesStatusCompleted,
				Content: []openaiserverapi.OutputContent{
					{Type: openaiserverapi.ResponsesOutputText, Text: text},
				},
			},
		},
		&openaiserverapi.ResponsesUsage{
			InputTokens:  usage.PromptTokens,
			OutputTokens: usage.CompletionTokens,
			TotalTokens:  usage.TotalTokens,
		},
	)
	return &namedEventChunk{
		names: []string{openaiserverapi.ResponsesEventCompleted},
		data: []any{&openaiserverapi.ResponsesResponseEvent{
			Type:     openaiserverapi.ResponsesEventCompleted,
			Response: resp,
		}}}
}

func (respBuilder *responsesHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized, tool *openaiserverapi.ToolCall, role string, finishReason *string) sseChunk {
	if tokens == nil || len(tokens.Strings) == 0 {
		return nil
	}
	delta := strings.Join(tokens.Strings, "")
	respBuilder.accumulated.WriteString(delta)
	return &namedEventChunk{
		names: []string{openaiserverapi.ResponsesEventTextDelta},
		data: []any{&openaiserverapi.ResponsesItemEvent{
			Type:   openaiserverapi.ResponsesEventTextDelta,
			ItemID: openaiserverapi.ResponsesMessageIDPrefix + respCtx.RequestID(),
			Delta:  delta,
		}}}
}

func (respBuilder *responsesHTTPRespBuilder) createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk {
	resp := openaiserverapi.CreateResponsesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		respCtx.CreationTime(),
		respCtx.Instructions(),
		nil,
		nil,
	)
	resp.Status = openaiserverapi.ResponsesStatusInProgress
	created := openaiserverapi.ResponsesResponseEvent{Type: openaiserverapi.ResponsesEventCreated, Response: resp}
	inProgress := openaiserverapi.ResponsesResponseEvent{Type: openaiserverapi.ResponsesEventInProgress, Response: resp}

	return &namedEventChunk{
		names: []string{created.Type, inProgress.Type},
		data:  []any{created, inProgress},
	}
}

func (respBuilder *responsesHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext) sseChunk {
	itemID := openaiserverapi.ResponsesMessageIDPrefix + respCtx.RequestID()
	outputItemAdded := openaiserverapi.ResponsesItemEvent{
		Type: openaiserverapi.ResponsesEventOutputItemAdded,
		Item: openaiserverapi.MessageOutput{
			Type:    openaiserverapi.ResponsesOutputMessage,
			ID:      itemID,
			Role:    openaiserverapi.RoleAssistant,
			Status:  "in_progress",
			Content: []openaiserverapi.OutputContent{},
		},
	}
	part := openaiserverapi.OutputContent{Type: openaiserverapi.ResponsesOutputText, Text: ""}
	contentPartAdded := openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventContentPartAdded,
		ItemID: itemID,
		Part:   &part,
	}
	return &namedEventChunk{
		names: []string{outputItemAdded.Type, contentPartAdded.Type},
		data:  []any{outputItemAdded, contentPartAdded},
	}
}

func (respBuilder *responsesHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, _ string) sseChunk {
	itemID := openaiserverapi.ResponsesMessageIDPrefix + respCtx.RequestID()
	text := respBuilder.accumulated.String()

	textDone := openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventTextDone,
		ItemID: itemID,
		Text:   text,
	}
	part := openaiserverapi.OutputContent{Type: openaiserverapi.ResponsesOutputText, Text: text}
	contentPartDone := openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventContentPartDone,
		ItemID: itemID,
		Part:   &part,
	}
	outputItemDone := openaiserverapi.ResponsesItemEvent{
		Type: openaiserverapi.ResponsesEventOutputItemDone,
		Item: openaiserverapi.MessageOutput{
			Type:   openaiserverapi.ResponsesOutputMessage,
			ID:     itemID,
			Role:   openaiserverapi.RoleAssistant,
			Status: openaiserverapi.ResponsesStatusCompleted,
			Content: []openaiserverapi.OutputContent{
				{Type: openaiserverapi.ResponsesOutputText, Text: text},
			},
		},
	}

	return &namedEventChunk{
		names: []string{textDone.Type, contentPartDone.Type, outputItemDone.Type},
		data:  []any{textDone, contentPartDone, outputItemDone},
	}
}

func (*responsesHTTPRespBuilder) createDoneChunk() sseChunk { return nil }

var _ responseBuilder = (*responsesHTTPRespBuilder)(nil)
