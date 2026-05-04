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
type jsonDataChunk struct{ v any }

func (j *jsonDataChunk) SSEBytes() ([]byte, error) {
	b, err := json.Marshal(j.v)
	if err != nil {
		return nil, err
	}
	return []byte("data: " + string(b) + "\n\n"), nil
}

// namedEventChunk formats its value as "event: <name>\ndata: <json>\n\n".
type namedEventChunk struct {
	name string
	v    any
}

func (e *namedEventChunk) SSEBytes() ([]byte, error) {
	b, err := json.Marshal(e.v)
	if err != nil {
		return nil, err
	}
	return []byte("event: " + e.name + "\ndata: " + string(b) + "\n\n"), nil
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
	return &jsonDataChunk{v: openaiserverapi.CreateTextCompletionsResponse(baseChunk, []openaiserverapi.TextRespChoice{})}
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

	return &jsonDataChunk{v: openaiserverapi.CreateTextCompletionsResponse(baseChunk, []openaiserverapi.TextRespChoice{choice})}
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

	message := openaiserverapi.ChatComplMessage{Role: openaiserverapi.RoleAssistant}
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
	return &jsonDataChunk{v: openaiserverapi.CreateChatCompletionsResponse(baseChunk, []openaiserverapi.ChatRespChoice{})}
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
				openaiserverapi.CreateBaseResponseChoice(0, finishReason), openaiserverapi.ChatComplMessage{})})

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

	return &jsonDataChunk{v: &chunk}
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

func (respBuilder *generationGRPCRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized,
	tool *openaiserverapi.ToolCall, role string, finishReason *string) any {
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

func (respBuilder *generationGRPCRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, finishReason string) any {
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
				Type:   "message",
				Role:   openaiserverapi.RoleAssistant,
				Status: "completed",
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
				Type:   "message",
				ID:     "msg_" + respCtx.RequestID(),
				Role:   openaiserverapi.RoleAssistant,
				Status: "completed",
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
	return &namedEventChunk{name: openaiserverapi.ResponsesEventCompleted, v: &openaiserverapi.ResponsesResponseEvent{
		Type:     openaiserverapi.ResponsesEventCompleted,
		Response: resp,
	}}
}

func (respBuilder *responsesHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized, tool *openaiserverapi.ToolCall, role string, finishReason *string) sseChunk {
	if tokens == nil || len(tokens.Strings) == 0 {
		return nil
	}
	delta := strings.Join(tokens.Strings, "")
	respBuilder.accumulated.WriteString(delta)
	return &namedEventChunk{name: openaiserverapi.ResponsesEventTextDelta, v: &openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventTextDelta,
		ItemID: "msg_" + respCtx.RequestID(),
		Delta:  delta,
	}}
}

// responsesInitialEvents emits response.created + response.in_progress as one sendChunk call.
type responsesInitialEvents struct {
	resp *openaiserverapi.ResponsesResponse
}

func (e *responsesInitialEvents) SSEBytes() ([]byte, error) {
	created := openaiserverapi.ResponsesResponseEvent{Type: openaiserverapi.ResponsesEventCreated, Response: e.resp}
	inProgress := openaiserverapi.ResponsesResponseEvent{Type: openaiserverapi.ResponsesEventInProgress, Response: e.resp}
	b1, err := json.Marshal(created)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(inProgress)
	if err != nil {
		return nil, err
	}
	return []byte(
		"event: " + created.Type + "\ndata: " + string(b1) + "\n\n" +
			"event: " + inProgress.Type + "\ndata: " + string(b2) + "\n\n",
	), nil
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
	resp.Status = "in_progress"
	return &responsesInitialEvents{resp: resp}
}

// responsesFirstEvents is a composite that emits output_item.added + content_part.added in one sendChunk call.
type responsesFirstEvents struct {
	itemID string
}

func (e *responsesFirstEvents) SSEBytes() ([]byte, error) {
	outputItemAdded := openaiserverapi.ResponsesItemEvent{
		Type: openaiserverapi.ResponsesEventOutputItemAdded,
		Item: openaiserverapi.MessageOutput{
			Type:    "message",
			ID:      e.itemID,
			Role:    openaiserverapi.RoleAssistant,
			Status:  "in_progress",
			Content: []openaiserverapi.OutputContent{},
		},
	}
	part := openaiserverapi.OutputContent{Type: openaiserverapi.ResponsesOutputText, Text: ""}
	contentPartAdded := openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventContentPartAdded,
		ItemID: e.itemID,
		Part:   &part,
	}
	b1, err := json.Marshal(outputItemAdded)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(contentPartAdded)
	if err != nil {
		return nil, err
	}
	return []byte(
		"event: " + outputItemAdded.Type + "\ndata: " + string(b1) + "\n\n" +
			"event: " + contentPartAdded.Type + "\ndata: " + string(b2) + "\n\n",
	), nil
}

func (respBuilder *responsesHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext) sseChunk {
	return &responsesFirstEvents{itemID: "msg_" + respCtx.RequestID()}
}

// responsesLastEvents emits text.done + content_part.done + output_item.done as one sendChunk call.
type responsesLastEvents struct {
	itemID string
	text   string
}

func (e *responsesLastEvents) SSEBytes() ([]byte, error) {
	textDone := openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventTextDone,
		ItemID: e.itemID,
		Text:   e.text,
	}
	part := openaiserverapi.OutputContent{Type: openaiserverapi.ResponsesOutputText, Text: e.text}
	contentPartDone := openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventContentPartDone,
		ItemID: e.itemID,
		Part:   &part,
	}
	outputItemDone := openaiserverapi.ResponsesItemEvent{
		Type: openaiserverapi.ResponsesEventOutputItemDone,
		Item: openaiserverapi.MessageOutput{
			Type:   "message",
			ID:     e.itemID,
			Role:   openaiserverapi.RoleAssistant,
			Status: "completed",
			Content: []openaiserverapi.OutputContent{
				{Type: openaiserverapi.ResponsesOutputText, Text: e.text},
			},
		},
	}
	b1, err := json.Marshal(textDone)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(contentPartDone)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(outputItemDone)
	if err != nil {
		return nil, err
	}
	return []byte(
		"event: " + textDone.Type + "\ndata: " + string(b1) + "\n\n" +
			"event: " + contentPartDone.Type + "\ndata: " + string(b2) + "\n\n" +
			"event: " + outputItemDone.Type + "\ndata: " + string(b3) + "\n\n",
	), nil
}

func (respBuilder *responsesHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, _ string) sseChunk {
	return &responsesLastEvents{
		itemID: "msg_" + respCtx.RequestID(),
		text:   respBuilder.accumulated.String(),
	}
}

func (*responsesHTTPRespBuilder) createDoneChunk() sseChunk { return nil }

var _ responseBuilder = (*responsesHTTPRespBuilder)(nil)
