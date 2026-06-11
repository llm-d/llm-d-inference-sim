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
	return []byte(openaiserverapi.SSEDataPrefix + string(b) + "\n\n"), nil
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

func (*doneMarker) SSEBytes() ([]byte, error) {
	return []byte(openaiserverapi.SSEDataPrefix + openaiserverapi.SSEDoneMarker + "\n\n"), nil
}

// responseBuilder is the HTTP streaming builder interface.
type responseBuilder interface {
	createResponse(respCtxPerChoice []vllmsim.ResponseContext, tokens []openaiserverapi.Tokenized) any
	createUsageChunk(respCtxPerChoice []vllmsim.ResponseContext) sseChunk
	createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized, tool *openaiserverapi.ToolCall,
		role string, finishReason *string, choiceIdx int) sseChunk
	createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk
	createFirstChunk(respCtx vllmsim.ResponseContext, choiceIdx int) sseChunk
	createLastChunk(respCtx vllmsim.ResponseContext, finishReason string, choiceIdx int) sseChunk
	createDoneChunk() sseChunk
	createRenderResponse(tokens [][]uint32, features *openaiserverapi.RenderMMFeatures) any
	// sendFinishReasonWithTokens returns true if the builder wants the finish
	// reason included in the last tokens chunk rather than a separate empty chunk.
	sendFinishReasonWithTokens() bool
}

// aggregateUsage combines per-choice usages. Completion tokens are always
// summed across all choices. For prompt tokens, the aggregation depends on
// whether choices share the same prompt (n parameter — same request ID) or
// have different prompts (array-prompt text completions — different request
// IDs). When all choices share the same request ID, prompt tokens are counted
// once; otherwise they are summed.
// Callers must ensure every slot is populated — by the time we reach here, every
// non-error response has carried a non-nil RespCtx, so any nil slot is a bug and
// a nil deref here is the right signal.
func aggregateUsage(respCtxPerChoice []vllmsim.ResponseContext) *openaiserverapi.Usage {
	if len(respCtxPerChoice) == 1 {
		return respCtxPerChoice[0].UsageData()
	}
	agg := &openaiserverapi.Usage{}
	// Track which request IDs we've already counted prompt tokens for.
	// With the n parameter, multiple choices share the same request ID and
	// prompt tokens should be counted once per unique ID. With array-prompt
	// text completions each prompt has a distinct ID, so prompt tokens are
	// summed across prompts. This handles the combined case (array + n)
	// correctly: prompt tokens are counted once per prompt, not once per choice.
	seenIDs := make(map[string]bool)
	for _, rc := range respCtxPerChoice {
		u := rc.UsageData()
		agg.CompletionTokens += u.CompletionTokens
		if !seenIDs[rc.RequestID()] {
			seenIDs[rc.RequestID()] = true
			agg.PromptTokens += u.PromptTokens
			if u.PromptTokensDetail != nil && agg.PromptTokensDetail == nil {
				agg.PromptTokensDetail = u.PromptTokensDetail
			}
		}
	}
	agg.TotalTokens = agg.PromptTokens + agg.CompletionTokens
	return agg
}

type textComplHTTPRespBuilder struct{}

func (respBuilder *textComplHTTPRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []openaiserverapi.Tokenized) any {
	respCtx := respCtxPerChoice[0]
	baseResp := openaiserverapi.CreateBaseCompletionsResponse(
		time.Now().Unix(), respCtx.DisplayModel(), aggregateUsage(respCtxPerChoice), respCtx.RequestID(), respCtx.DoRemoteDecode())

	choices := make([]openaiserverapi.TextRespChoice, len(tokens))
	for i, t := range tokens {
		choiceCtx := respCtxPerChoice[i]
		baseChoice := openaiserverapi.CreateBaseResponseChoice(i, choiceCtx.FinishReason())
		respText := strings.Join(t.Strings, "")
		choice := openaiserverapi.CreateTextRespChoice(baseChoice, respText)

		// Generate logprobs if requested for text completion
		if choiceCtx.TopLogprobs() != nil && *choiceCtx.TopLogprobs() > 0 {
			if logprobsData := common.GenerateTextLogprobs(t.Strings, *choiceCtx.TopLogprobs()); logprobsData != nil &&
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
		choices[i] = choice
	}

	baseResp.Object = openaiserverapi.TextCompletionObject
	return openaiserverapi.CreateTextCompletionsResponse(baseResp, choices)
}

func (respBuilder *textComplHTTPRespBuilder) createUsageChunk(respCtxPerChoice []vllmsim.ResponseContext) sseChunk {
	respCtx := respCtxPerChoice[0]
	if !respCtx.SendUsageData() {
		return nil
	}
	baseChunk := openaiserverapi.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), aggregateUsage(respCtxPerChoice), respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.TextCompletionObject
	return &jsonDataChunk{data: openaiserverapi.CreateTextCompletionsResponse(baseChunk, []openaiserverapi.TextRespChoice{})}
}

// createChunk creates and returns a CompletionsRespChunk, a single chunk of streamed completion API response,
// for text completion.
func (respBuilder *textComplHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized,
	tool *openaiserverapi.ToolCall, role string, finishReason *string, choiceIdx int) sseChunk {

	baseChunk := openaiserverapi.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), nil, respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.TextCompletionObject

	var tokensStr string
	if tokens != nil {
		tokensStr = strings.Join(tokens.Strings, "")
	}
	choice := openaiserverapi.CreateTextRespChoice(openaiserverapi.CreateBaseResponseChoice(choiceIdx, finishReason), tokensStr)

	// Generate logprobs if requested and tokens is not empty
	if respCtx.TopLogprobs() != nil && tokens != nil && len(tokens.Strings) > 0 && *respCtx.TopLogprobs() > 0 {
		// Use token position based on current time
		tokenPosition := int(respCtx.CreationTime()) % 1000 // Simple position simulation
		logprobs := common.GenerateSingleTokenTextLogprobs(tokensStr, tokenPosition, *respCtx.TopLogprobs())
		if logprobs != nil {
			choice.Logprobs = logprobs
		}
	}

	return &jsonDataChunk{data: openaiserverapi.CreateTextCompletionsResponse(baseChunk, []openaiserverapi.TextRespChoice{choice})}
}

func (respBuilder *textComplHTTPRespBuilder) createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk {
	return nil
}

func (respBuilder *textComplHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext, choiceIdx int) sseChunk {
	return nil
}

func (respBuilder *textComplHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, finishReason string, choiceIdx int) sseChunk {
	if finishReason == common.CacheThresholdFinishReason {
		return nil
	}
	return respBuilder.createChunk(respCtx, nil, nil, "", respCtx.FinishReason(), choiceIdx)
}

func (*textComplHTTPRespBuilder) createDoneChunk() sseChunk              { return &doneMarker{} }
func (*textComplHTTPRespBuilder) sendFinishReasonWithTokens() bool       { return false }

// createRenderResponse builds the wire payload for /v1/completions/render: an
// array with one RenderResponse per prompt, mirroring vLLM which always
// returns an array even for a single string prompt. features is unused — the
// text endpoint never carries multimodal features.
func (*textComplHTTPRespBuilder) createRenderResponse(tokens [][]uint32,
	_ *openaiserverapi.RenderMMFeatures) any {
	responses := make([]openaiserverapi.RenderResponse, len(tokens))
	for i, t := range tokens {
		responses[i] = openaiserverapi.RenderResponse{TokenIDs: t}
	}
	return responses
}

var _ responseBuilder = (*textComplHTTPRespBuilder)(nil)

type chatComplHTTPRespBuilder struct{}

func (respBuilder *chatComplHTTPRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []openaiserverapi.Tokenized) any {
	respCtx := respCtxPerChoice[0]
	baseResp := openaiserverapi.CreateBaseCompletionsResponse(
		time.Now().Unix(), respCtx.DisplayModel(), aggregateUsage(respCtxPerChoice), respCtx.RequestID(), respCtx.DoRemoteDecode())
	baseResp.Object = openaiserverapi.ChatCompletionObject

	choices := make([]openaiserverapi.ChatRespChoice, len(tokens))
	for i, t := range tokens {
		choiceCtx := respCtxPerChoice[i]
		baseChoice := openaiserverapi.CreateBaseResponseChoice(i, choiceCtx.FinishReason())

		message := openaiserverapi.Message{Role: openaiserverapi.RoleAssistant}
		if choiceCtx.ToolCalls() != nil {
			message.ToolCalls = choiceCtx.ToolCalls()
		} else {
			respText := strings.Join(t.Strings, "")
			message.Content = openaiserverapi.ChatComplContent{Raw: respText}
		}

		choice := openaiserverapi.CreateChatRespChoice(baseChoice, message)

		// Generate logprobs if requested
		if choiceCtx.TopLogprobs() != nil && choiceCtx.ToolCalls() == nil {
			if logprobsData := common.GenerateChatLogprobs(t.Strings, *choiceCtx.TopLogprobs()); logprobsData != nil &&
				len(logprobsData.Content) > 0 {
				choice.Logprobs = logprobsData
			}
		}
		choices[i] = choice
	}

	resp := openaiserverapi.CreateChatCompletionsResponse(baseResp, choices)
	resp.ECTransferParams = respCtx.ECTransferParams()
	return resp
}

func (respBuilder *chatComplHTTPRespBuilder) createUsageChunk(respCtxPerChoice []vllmsim.ResponseContext) sseChunk {
	respCtx := respCtxPerChoice[0]
	if !respCtx.SendUsageData() {
		return nil
	}
	baseChunk := openaiserverapi.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), aggregateUsage(respCtxPerChoice), respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.ChatCompletionChunkObject
	return &jsonDataChunk{data: openaiserverapi.CreateChatCompletionsResponse(baseChunk, []openaiserverapi.ChatRespChoice{})}
}

// createChunk creates and returns a CompletionsRespChunk, a single chunk of streamed completion
// API response, for chat completion. It sets either role, or token, or tool call info in the message.
func (respBuilder *chatComplHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *openaiserverapi.Tokenized,
	tool *openaiserverapi.ToolCall, role string, finishReason *string, choiceIdx int) sseChunk {
	baseChunk := openaiserverapi.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), nil, respCtx.RequestID(), false)
	baseChunk.Object = openaiserverapi.ChatCompletionChunkObject
	chunk := openaiserverapi.CreateChatCompletionsRespChunk(baseChunk,
		[]openaiserverapi.ChatRespChunkChoice{
			openaiserverapi.CreateChatRespChunkChoice(
				openaiserverapi.CreateBaseResponseChoice(choiceIdx, finishReason), openaiserverapi.Message{})})

	if len(role) > 0 {
		chunk.Choices[0].Delta.Role = role
	}
	if tool != nil {
		chunk.Choices[0].Delta.ToolCalls = []openaiserverapi.ToolCall{*tool}
	} else if tokens != nil && len(tokens.Strings) > 0 {
		tokensStr := strings.Join(tokens.Strings, "")
		chunk.Choices[0].Delta.Content.Raw = tokensStr

		// Generate logprobs if requested and token is not empty
		if respCtx.TopLogprobs() != nil {
			// Use token position based on current time
			tokenPosition := int(respCtx.CreationTime()) % 1000 // Simple position simulation
			logprobs := common.GenerateSingleTokenChatLogprobs(tokensStr, tokenPosition, *respCtx.TopLogprobs())
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

func (respBuilder *chatComplHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext, choiceIdx int) sseChunk {
	return respBuilder.createChunk(respCtx, nil, nil, openaiserverapi.RoleAssistant, nil, choiceIdx)
}

func (respBuilder *chatComplHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, finishReason string, choiceIdx int) sseChunk {
	if finishReason == common.ToolsFinishReason || finishReason == common.CacheThresholdFinishReason {
		return nil
	}
	return respBuilder.createChunk(respCtx, nil, nil, "", respCtx.FinishReason(), choiceIdx)
}

func (*chatComplHTTPRespBuilder) createDoneChunk() sseChunk              { return &doneMarker{} }
func (*chatComplHTTPRespBuilder) sendFinishReasonWithTokens() bool       { return false }

// createRenderResponse builds the wire payload for /v1/chat/completions/render:
// a single RenderResponse object (not an array) carrying the tokens for the
// flattened prompt and any mm_features produced by the tokenizer.
func (*chatComplHTTPRespBuilder) createRenderResponse(tokens [][]uint32,
	features *openaiserverapi.RenderMMFeatures) any {
	return openaiserverapi.RenderResponse{TokenIDs: tokens[0], Features: features}
}

var _ responseBuilder = (*chatComplHTTPRespBuilder)(nil)

type generationGRPCRespBuilder struct{}

func (respBuilder *generationGRPCRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []openaiserverapi.Tokenized) any {
	respCtx := respCtxPerChoice[0]

	var completionTokens uint32
	var outputIds []uint32
	if len(tokens) > 0 {
		completionTokens = uint32(respCtx.UsageData().CompletionTokens)
		outputIds = tokens[0].Tokens
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
	return respBuilder.createResponse([]vllmsim.ResponseContext{respCtx}, nil)
}

type responsesHTTPRespBuilder struct {
	// contains the accumulated text for the response,
	// used to populate the final chunk and usage chunk
	accumulated strings.Builder
	// number of tokens in the accumulated text, used for logprobs generation
	accumulatedTokens int
	// logprobs for the accumulated text
	accumulatedLogprobs []openaiserverapi.ResponsesLogprob
}

func (respBuilder *responsesHTTPRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []openaiserverapi.Tokenized) any {
	respCtx := respCtxPerChoice[0]
	text := strings.Join(tokens[0].Strings, "")
	usage := respCtx.UsageData()

	outputContent := openaiserverapi.OutputContent{
		Type: openaiserverapi.ResponsesOutputText,
		Text: text,
	}

	if respCtx.TopLogprobs() != nil {
		logprobs := common.GenerateMessagesLogprobs(tokens[0].Strings, *respCtx.TopLogprobs())
		outputContent.Logprobs = &logprobs
	}

	return openaiserverapi.CreateResponsesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		time.Now().Unix(),
		respCtx.Instructions(),
		respCtx.TopLogprobs(),
		[]openaiserverapi.OutputItem{
			openaiserverapi.MessageOutput{
				Type:    openaiserverapi.ResponsesOutputMessage,
				Role:    openaiserverapi.RoleAssistant,
				Status:  openaiserverapi.ResponsesStatusCompleted,
				Content: []openaiserverapi.OutputContent{outputContent},
			},
		},
		&openaiserverapi.ResponsesUsage{
			InputTokens:  usage.PromptTokens,
			OutputTokens: usage.CompletionTokens,
			TotalTokens:  usage.TotalTokens,
		},
	)
}

func (respBuilder *responsesHTTPRespBuilder) createUsageChunk(respCtxPerChoice []vllmsim.ResponseContext) sseChunk {
	respCtx := respCtxPerChoice[0]
	usage := aggregateUsage(respCtxPerChoice)
	text := respBuilder.accumulated.String()
	completedContent := openaiserverapi.OutputContent{Type: openaiserverapi.ResponsesOutputText, Text: text}
	if respCtx.TopLogprobs() != nil && len(respBuilder.accumulatedLogprobs) > 0 {
		logprobs := make([]openaiserverapi.ResponsesLogprob, len(respBuilder.accumulatedLogprobs))
		copy(logprobs, respBuilder.accumulatedLogprobs)
		completedContent.Logprobs = &logprobs
	}
	resp := openaiserverapi.CreateResponsesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		respCtx.CreationTime(),
		respCtx.Instructions(),
		respCtx.TopLogprobs(),
		[]openaiserverapi.OutputItem{
			openaiserverapi.MessageOutput{
				Type:    openaiserverapi.ResponsesOutputMessage,
				ID:      openaiserverapi.ResponsesMessageIDPrefix + respCtx.RequestID(),
				Role:    openaiserverapi.RoleAssistant,
				Status:  openaiserverapi.ResponsesStatusCompleted,
				Content: []openaiserverapi.OutputContent{completedContent},
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
	tokens *openaiserverapi.Tokenized, tool *openaiserverapi.ToolCall, role string, finishReason *string, choiceIdx int) sseChunk {
	if tokens == nil || len(tokens.Strings) == 0 {
		return nil
	}
	delta := strings.Join(tokens.Strings, "")
	respBuilder.accumulated.WriteString(delta)

	itemID := openaiserverapi.ResponsesMessageIDPrefix + respCtx.RequestID()
	deltaEvent := &openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventTextDelta,
		ItemID: itemID,
		Delta:  delta,
	}

	if respCtx.TopLogprobs() != nil {
		var logprobs []openaiserverapi.ResponsesLogprob
		for _, tok := range tokens.Strings {
			lp := common.GenerateSingleTokenChatLogprobs(tok, respBuilder.accumulatedTokens, *respCtx.TopLogprobs())
			respBuilder.accumulatedTokens++
			if lp == nil {
				continue
			}
			topLogprobs := make([]openaiserverapi.TopLogprob, len(lp.TopLogprobs))
			for i, top := range lp.TopLogprobs {
				topLogprobs[i] = openaiserverapi.TopLogprob{Token: top.Token, Logprob: top.Logprob, Bytes: top.Bytes}
			}
			entry := openaiserverapi.ResponsesLogprob{
				Token:       lp.Token,
				Logprob:     lp.Logprob,
				Bytes:       lp.Bytes,
				TopLogprobs: topLogprobs,
			}
			logprobs = append(logprobs, entry)
			respBuilder.accumulatedLogprobs = append(respBuilder.accumulatedLogprobs, entry)
		}
		deltaEvent.Logprobs = &logprobs
	} else {
		respBuilder.accumulatedTokens += len(tokens.Strings)
	}

	return &namedEventChunk{names: []string{openaiserverapi.ResponsesEventTextDelta}, data: []any{deltaEvent}}
}

func (respBuilder *responsesHTTPRespBuilder) createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk {
	resp := openaiserverapi.CreateResponsesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		respCtx.CreationTime(),
		respCtx.Instructions(),
		respCtx.TopLogprobs(),
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

func (respBuilder *responsesHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext, choiceIdx int) sseChunk {
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
	if respCtx.TopLogprobs() != nil {
		emptyLogprobs := []openaiserverapi.ResponsesLogprob{}
		part.Logprobs = &emptyLogprobs
	}
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

func (respBuilder *responsesHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, _ string, choiceIdx int) sseChunk {
	itemID := openaiserverapi.ResponsesMessageIDPrefix + respCtx.RequestID()
	text := respBuilder.accumulated.String()

	textDone := openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventTextDone,
		ItemID: itemID,
		Text:   text,
	}
	if respCtx.TopLogprobs() != nil {
		emptyLogprobs := []openaiserverapi.ResponsesLogprob{}
		textDone.Logprobs = &emptyLogprobs
	}
	part := openaiserverapi.OutputContent{Type: openaiserverapi.ResponsesOutputText, Text: text}
	doneContent := openaiserverapi.OutputContent{Type: openaiserverapi.ResponsesOutputText, Text: text}
	if respCtx.TopLogprobs() != nil {
		// null signals that per-token logprobs were already streamed in delta events
		var nullLogprobs []openaiserverapi.ResponsesLogprob
		part.Logprobs = &nullLogprobs
		doneContent.Logprobs = &nullLogprobs
	}
	contentPartDone := openaiserverapi.ResponsesItemEvent{
		Type:   openaiserverapi.ResponsesEventContentPartDone,
		ItemID: itemID,
		Part:   &part,
	}
	outputItemDone := openaiserverapi.ResponsesItemEvent{
		Type: openaiserverapi.ResponsesEventOutputItemDone,
		Item: openaiserverapi.MessageOutput{
			Type:    openaiserverapi.ResponsesOutputMessage,
			ID:      itemID,
			Role:    openaiserverapi.RoleAssistant,
			Status:  openaiserverapi.ResponsesStatusCompleted,
			Content: []openaiserverapi.OutputContent{doneContent},
		},
	}

	return &namedEventChunk{
		names: []string{textDone.Type, contentPartDone.Type, outputItemDone.Type},
		data:  []any{textDone, contentPartDone, outputItemDone},
	}
}

func (*responsesHTTPRespBuilder) createDoneChunk() sseChunk              { return nil }
func (*responsesHTTPRespBuilder) sendFinishReasonWithTokens() bool       { return false }

func (*responsesHTTPRespBuilder) createRenderResponse(_ [][]uint32,
	_ *openaiserverapi.RenderMMFeatures) any {
	panic("responsesHTTPRespBuilder: /v1/responses has no /render endpoint")
}

var _ responseBuilder = (*responsesHTTPRespBuilder)(nil)

type generateHTTPRespBuilder struct{}

func (respBuilder *generateHTTPRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []openaiserverapi.Tokenized) any {
	respCtx := respCtxPerChoice[0]
	var tokenIDs []uint32
	if len(tokens) > 0 {
		tokenIDs = tokens[0].Tokens
	}
	choice := openaiserverapi.GenerateRespChoice{TokenIDs: tokenIDs}
	choice.Index = 0
	choice.FinishReason = respCtx.FinishReason()
	resp := &openaiserverapi.GenerateResponse{
		Choices:          []openaiserverapi.GenerateRespChoice{choice},
		GenRequestID:     respCtx.RequestID(),
		ECTransferParams: respCtx.ECTransferParams(),
	}
	if respCtx.DoRemoteDecode() {
		resp.KVParams = openaiserverapi.BuildPrefillKVTransferParams()
	}
	return resp
}

func (respBuilder *generateHTTPRespBuilder) createUsageChunk(respCtxPerChoice []vllmsim.ResponseContext) sseChunk {
	respCtx := respCtxPerChoice[0]
	if !respCtx.SendUsageData() {
		return nil
	}
	return &jsonDataChunk{data: &openaiserverapi.GenerateStreamResponse{
		RequestID: respCtx.RequestID(),
		Choices:   []openaiserverapi.GenerateRespChoice{},
		Usage:     aggregateUsage(respCtxPerChoice),
	}}
}

func (respBuilder *generateHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext,
	tokens *openaiserverapi.Tokenized, _ *openaiserverapi.ToolCall, _ string, finishReason *string, choiceIdx int) sseChunk {
	choice := openaiserverapi.GenerateRespChoice{}
	choice.Index = choiceIdx
	choice.FinishReason = finishReason
	if tokens != nil {
		choice.TokenIDs = tokens.Tokens
	}
	return &jsonDataChunk{data: &openaiserverapi.GenerateStreamResponse{
		RequestID: respCtx.RequestID(),
		Choices:   []openaiserverapi.GenerateRespChoice{choice},
	}}
}

func (respBuilder *generateHTTPRespBuilder) createInitialChunk(_ vllmsim.ResponseContext) sseChunk {
	return nil
}

func (respBuilder *generateHTTPRespBuilder) createFirstChunk(_ vllmsim.ResponseContext, _ int) sseChunk {
	return nil
}

func (respBuilder *generateHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, finishReason string, choiceIdx int) sseChunk {
	return nil
}

func (*generateHTTPRespBuilder) createDoneChunk() sseChunk              { return &doneMarker{} }
func (*generateHTTPRespBuilder) sendFinishReasonWithTokens() bool       { return true }

func (*generateHTTPRespBuilder) createRenderResponse(_ [][]uint32,
	_ *openaiserverapi.RenderMMFeatures) any {
	panic("generateHTTPRespBuilder: /inference/v1/generate has no /render endpoint")
}

var _ responseBuilder = (*generateHTTPRespBuilder)(nil)
