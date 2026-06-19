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

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
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
	return []byte(api.SSEDataPrefix + string(b) + "\n\n"), nil
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
	return []byte(api.SSEDataPrefix + api.SSEDoneMarker + "\n\n"), nil
}

// responseBuilder is the HTTP streaming builder interface.
type responseBuilder interface {
	createResponse(respCtxPerChoice []vllmsim.ResponseContext, tokens []api.Tokenized) any
	createUsageChunk(respCtxPerChoice []vllmsim.ResponseContext) sseChunk
	createChunk(respCtx vllmsim.ResponseContext, tokens *api.Tokenized, tool *api.ToolCall,
		role string, finishReason *string, choiceIdx int) sseChunk
	createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk
	createFirstChunk(respCtx vllmsim.ResponseContext, choiceIdx int) sseChunk
	createLastChunk(respCtx vllmsim.ResponseContext, finishReason string, choiceIdx int) sseChunk
	createDoneChunk() sseChunk
	createRenderResponse(tokens [][]uint32, features *api.RenderMMFeatures) any
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
func aggregateUsage(respCtxPerChoice []vllmsim.ResponseContext) *api.Usage {
	if len(respCtxPerChoice) == 1 {
		return respCtxPerChoice[0].UsageData()
	}
	agg := &api.Usage{}
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
	tokens []api.Tokenized) any {
	respCtx := respCtxPerChoice[0]
	baseResp := api.CreateBaseCompletionsResponse(
		time.Now().Unix(), respCtx.DisplayModel(), aggregateUsage(respCtxPerChoice), respCtx.RequestID(), respCtx.DoRemoteDecode())

	choices := make([]api.TextRespChoice, len(tokens))
	for i, t := range tokens {
		choiceCtx := respCtxPerChoice[i]
		baseChoice := api.CreateBaseResponseChoice(i, choiceCtx.FinishReason())
		respText := strings.Join(t.Strings, "")
		choice := api.CreateTextRespChoice(baseChoice, respText)

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

	baseResp.Object = api.TextCompletionObject
	return api.CreateTextCompletionsResponse(baseResp, choices)
}

func (respBuilder *textComplHTTPRespBuilder) createUsageChunk(respCtxPerChoice []vllmsim.ResponseContext) sseChunk {
	respCtx := respCtxPerChoice[0]
	if !respCtx.SendUsageData() {
		return nil
	}
	baseChunk := api.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), aggregateUsage(respCtxPerChoice), respCtx.RequestID(), false)
	baseChunk.Object = api.TextCompletionObject
	return &jsonDataChunk{data: api.CreateTextCompletionsResponse(baseChunk, []api.TextRespChoice{})}
}

// createChunk creates and returns a CompletionsRespChunk, a single chunk of streamed completion API response,
// for text completion.
func (respBuilder *textComplHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *api.Tokenized,
	tool *api.ToolCall, role string, finishReason *string, choiceIdx int) sseChunk {

	baseChunk := api.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), nil, respCtx.RequestID(), false)
	baseChunk.Object = api.TextCompletionObject

	var tokensStr string
	if tokens != nil {
		tokensStr = strings.Join(tokens.Strings, "")
	}
	choice := api.CreateTextRespChoice(api.CreateBaseResponseChoice(choiceIdx, finishReason), tokensStr)

	// Generate logprobs if requested and tokens is not empty
	if respCtx.TopLogprobs() != nil && tokens != nil && len(tokens.Strings) > 0 && *respCtx.TopLogprobs() > 0 {
		// Use token position based on current time
		tokenPosition := int(respCtx.CreationTime()) % 1000 // Simple position simulation
		logprobs := common.GenerateSingleTokenTextLogprobs(tokensStr, tokenPosition, *respCtx.TopLogprobs())
		if logprobs != nil {
			choice.Logprobs = logprobs
		}
	}

	return &jsonDataChunk{data: api.CreateTextCompletionsResponse(baseChunk, []api.TextRespChoice{choice})}
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

func (*textComplHTTPRespBuilder) createDoneChunk() sseChunk        { return &doneMarker{} }
func (*textComplHTTPRespBuilder) sendFinishReasonWithTokens() bool { return false }

// createRenderResponse builds the wire payload for /v1/completions/render: an
// array with one RenderResponse per prompt, mirroring vLLM which always
// returns an array even for a single string prompt. features is unused — the
// text endpoint never carries multimodal features.
func (*textComplHTTPRespBuilder) createRenderResponse(tokens [][]uint32,
	_ *api.RenderMMFeatures) any {
	responses := make([]api.RenderResponse, len(tokens))
	for i, t := range tokens {
		responses[i] = api.RenderResponse{TokenIDs: t}
	}
	return responses
}

var _ responseBuilder = (*textComplHTTPRespBuilder)(nil)

type chatComplHTTPRespBuilder struct{}

func (respBuilder *chatComplHTTPRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []api.Tokenized) any {
	respCtx := respCtxPerChoice[0]
	baseResp := api.CreateBaseCompletionsResponse(
		time.Now().Unix(), respCtx.DisplayModel(), aggregateUsage(respCtxPerChoice), respCtx.RequestID(), respCtx.DoRemoteDecode())
	baseResp.Object = api.ChatCompletionObject

	choices := make([]api.ChatRespChoice, len(tokens))
	for i, t := range tokens {
		choiceCtx := respCtxPerChoice[i]
		baseChoice := api.CreateBaseResponseChoice(i, choiceCtx.FinishReason())

		message := api.Message{Role: api.RoleAssistant}
		if choiceCtx.ToolCalls() != nil {
			message.ToolCalls = choiceCtx.ToolCalls()
		} else {
			respText := strings.Join(t.Strings, "")
			message.Content = api.ChatComplContent{Raw: respText}
		}

		choice := api.CreateChatRespChoice(baseChoice, message)

		// Generate logprobs if requested
		if choiceCtx.TopLogprobs() != nil && choiceCtx.ToolCalls() == nil {
			if logprobsData := common.GenerateChatLogprobs(t.Strings, *choiceCtx.TopLogprobs()); logprobsData != nil &&
				len(logprobsData.Content) > 0 {
				choice.Logprobs = logprobsData
			}
		}
		choices[i] = choice
	}

	resp := api.CreateChatCompletionsResponse(baseResp, choices)
	resp.ECTransferParams = respCtx.ECTransferParams()
	return resp
}

func (respBuilder *chatComplHTTPRespBuilder) createUsageChunk(respCtxPerChoice []vllmsim.ResponseContext) sseChunk {
	respCtx := respCtxPerChoice[0]
	if !respCtx.SendUsageData() {
		return nil
	}
	baseChunk := api.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), aggregateUsage(respCtxPerChoice), respCtx.RequestID(), false)
	baseChunk.Object = api.ChatCompletionChunkObject
	return &jsonDataChunk{data: api.CreateChatCompletionsResponse(baseChunk, []api.ChatRespChoice{})}
}

// createChunk creates and returns a CompletionsRespChunk, a single chunk of streamed completion
// API response, for chat completion. It sets either role, or token, or tool call info in the message.
func (respBuilder *chatComplHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *api.Tokenized,
	tool *api.ToolCall, role string, finishReason *string, choiceIdx int) sseChunk {
	baseChunk := api.CreateBaseCompletionsResponse(
		respCtx.CreationTime(), respCtx.DisplayModel(), nil, respCtx.RequestID(), false)
	baseChunk.Object = api.ChatCompletionChunkObject
	chunk := api.CreateChatCompletionsRespChunk(baseChunk,
		[]api.ChatRespChunkChoice{
			api.CreateChatRespChunkChoice(
				api.CreateBaseResponseChoice(choiceIdx, finishReason), api.Message{})})

	if len(role) > 0 {
		chunk.Choices[0].Delta.Role = role
	}
	if tool != nil {
		chunk.Choices[0].Delta.ToolCalls = []api.ToolCall{*tool}
	} else if tokens != nil && len(tokens.Strings) > 0 {
		tokensStr := strings.Join(tokens.Strings, "")
		chunk.Choices[0].Delta.Content.Raw = tokensStr

		// Generate logprobs if requested and token is not empty
		if respCtx.TopLogprobs() != nil {
			// Use token position based on current time
			tokenPosition := int(respCtx.CreationTime()) % 1000 // Simple position simulation
			logprobs := common.GenerateSingleTokenChatLogprobs(tokensStr, tokenPosition, *respCtx.TopLogprobs())
			if logprobs != nil {
				chunk.Choices[0].Logprobs = &api.ChatLogprobs{
					Content: []api.LogprobsContent{*logprobs},
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
	return respBuilder.createChunk(respCtx, nil, nil, api.RoleAssistant, nil, choiceIdx)
}

func (respBuilder *chatComplHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, finishReason string, choiceIdx int) sseChunk {
	if finishReason == common.ToolsFinishReason || finishReason == common.CacheThresholdFinishReason {
		return nil
	}
	return respBuilder.createChunk(respCtx, nil, nil, "", respCtx.FinishReason(), choiceIdx)
}

func (*chatComplHTTPRespBuilder) createDoneChunk() sseChunk        { return &doneMarker{} }
func (*chatComplHTTPRespBuilder) sendFinishReasonWithTokens() bool { return false }

// createRenderResponse builds the wire payload for /v1/chat/completions/render:
// a single RenderResponse object (not an array) carrying the tokens for the
// flattened prompt and any mm_features produced by the tokenizer.
func (*chatComplHTTPRespBuilder) createRenderResponse(tokens [][]uint32,
	features *api.RenderMMFeatures) any {
	return api.RenderResponse{TokenIDs: tokens[0], Features: features}
}

var _ responseBuilder = (*chatComplHTTPRespBuilder)(nil)

type generationGRPCRespBuilder struct{}

func (respBuilder *generationGRPCRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []api.Tokenized) any {
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

func (respBuilder *generationGRPCRespBuilder) createChunk(respCtx vllmsim.ResponseContext, tokens *api.Tokenized) any {
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
	accumulatedLogprobs []api.ResponsesLogprob
}

func (respBuilder *responsesHTTPRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []api.Tokenized) any {
	respCtx := respCtxPerChoice[0]
	text := strings.Join(tokens[0].Strings, "")
	usage := respCtx.UsageData()

	outputContent := api.OutputContent{
		Type: api.ResponsesOutputText,
		Text: text,
	}

	if respCtx.TopLogprobs() != nil {
		logprobs := common.GenerateMessagesLogprobs(tokens[0].Strings, *respCtx.TopLogprobs())
		outputContent.Logprobs = &logprobs
	}

	return api.CreateResponsesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		time.Now().Unix(),
		respCtx.Instructions(),
		[]api.OutputItem{
			api.MessageOutput{
				Type:    api.ResponsesOutputMessage,
				Role:    api.RoleAssistant,
				Status:  api.ResponsesStatusCompleted,
				Content: []api.OutputContent{outputContent},
			},
		},
		&api.ResponsesUsage{
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
	completedContent := api.OutputContent{Type: api.ResponsesOutputText, Text: text}
	if respCtx.TopLogprobs() != nil && len(respBuilder.accumulatedLogprobs) > 0 {
		logprobs := make([]api.ResponsesLogprob, len(respBuilder.accumulatedLogprobs))
		copy(logprobs, respBuilder.accumulatedLogprobs)
		completedContent.Logprobs = &logprobs
	}
	resp := api.CreateResponsesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		respCtx.CreationTime(),
		respCtx.Instructions(),
		[]api.OutputItem{
			api.MessageOutput{
				Type:    api.ResponsesOutputMessage,
				ID:      api.ResponsesMessageIDPrefix + respCtx.RequestID(),
				Role:    api.RoleAssistant,
				Status:  api.ResponsesStatusCompleted,
				Content: []api.OutputContent{completedContent},
			},
		},
		&api.ResponsesUsage{
			InputTokens:  usage.PromptTokens,
			OutputTokens: usage.CompletionTokens,
			TotalTokens:  usage.TotalTokens,
		},
	)
	return &namedEventChunk{
		names: []string{api.ResponsesEventCompleted},
		data: []any{&api.ResponsesResponseEvent{
			Type:     api.ResponsesEventCompleted,
			Response: resp,
		}}}
}

func (respBuilder *responsesHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext,
	tokens *api.Tokenized, tool *api.ToolCall, role string, finishReason *string, choiceIdx int) sseChunk {
	if tokens == nil || len(tokens.Strings) == 0 {
		return nil
	}
	delta := strings.Join(tokens.Strings, "")
	respBuilder.accumulated.WriteString(delta)

	itemID := api.ResponsesMessageIDPrefix + respCtx.RequestID()
	deltaEvent := &api.ResponsesItemEvent{
		Type:   api.ResponsesEventTextDelta,
		ItemID: itemID,
		Delta:  delta,
	}

	if respCtx.TopLogprobs() != nil {
		var logprobs []api.ResponsesLogprob
		for _, tok := range tokens.Strings {
			lp := common.GenerateSingleTokenChatLogprobs(tok, respBuilder.accumulatedTokens, *respCtx.TopLogprobs())
			respBuilder.accumulatedTokens++
			if lp == nil {
				continue
			}
			topLogprobs := make([]api.TopLogprob, len(lp.TopLogprobs))
			for i, top := range lp.TopLogprobs {
				topLogprobs[i] = api.TopLogprob{Token: top.Token, Logprob: top.Logprob, Bytes: top.Bytes}
			}
			entry := api.ResponsesLogprob{
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

	return &namedEventChunk{names: []string{api.ResponsesEventTextDelta}, data: []any{deltaEvent}}
}

func (respBuilder *responsesHTTPRespBuilder) createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk {
	resp := api.CreateResponsesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		respCtx.CreationTime(),
		respCtx.Instructions(),
		nil,
		nil,
	)
	resp.Status = api.ResponsesStatusInProgress
	created := api.ResponsesResponseEvent{Type: api.ResponsesEventCreated, Response: resp}
	inProgress := api.ResponsesResponseEvent{Type: api.ResponsesEventInProgress, Response: resp}

	return &namedEventChunk{
		names: []string{created.Type, inProgress.Type},
		data:  []any{created, inProgress},
	}
}

func (respBuilder *responsesHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext, choiceIdx int) sseChunk {
	itemID := api.ResponsesMessageIDPrefix + respCtx.RequestID()
	outputItemAdded := api.ResponsesItemEvent{
		Type: api.ResponsesEventOutputItemAdded,
		Item: api.MessageOutput{
			Type:    api.ResponsesOutputMessage,
			ID:      itemID,
			Role:    api.RoleAssistant,
			Status:  "in_progress",
			Content: []api.OutputContent{},
		},
	}
	part := api.OutputContent{Type: api.ResponsesOutputText, Text: ""}
	if respCtx.TopLogprobs() != nil {
		emptyLogprobs := []api.ResponsesLogprob{}
		part.Logprobs = &emptyLogprobs
	}
	contentPartAdded := api.ResponsesItemEvent{
		Type:   api.ResponsesEventContentPartAdded,
		ItemID: itemID,
		Part:   &part,
	}
	return &namedEventChunk{
		names: []string{outputItemAdded.Type, contentPartAdded.Type},
		data:  []any{outputItemAdded, contentPartAdded},
	}
}

func (respBuilder *responsesHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, _ string, choiceIdx int) sseChunk {
	itemID := api.ResponsesMessageIDPrefix + respCtx.RequestID()
	text := respBuilder.accumulated.String()

	textDone := api.ResponsesItemEvent{
		Type:   api.ResponsesEventTextDone,
		ItemID: itemID,
		Text:   text,
	}
	if respCtx.TopLogprobs() != nil {
		emptyLogprobs := []api.ResponsesLogprob{}
		textDone.Logprobs = &emptyLogprobs
	}
	part := api.OutputContent{Type: api.ResponsesOutputText, Text: text}
	doneContent := api.OutputContent{Type: api.ResponsesOutputText, Text: text}
	if respCtx.TopLogprobs() != nil {
		// null signals that per-token logprobs were already streamed in delta events
		var nullLogprobs []api.ResponsesLogprob
		part.Logprobs = &nullLogprobs
		doneContent.Logprobs = &nullLogprobs
	}
	contentPartDone := api.ResponsesItemEvent{
		Type:   api.ResponsesEventContentPartDone,
		ItemID: itemID,
		Part:   &part,
	}
	outputItemDone := api.ResponsesItemEvent{
		Type: api.ResponsesEventOutputItemDone,
		Item: api.MessageOutput{
			Type:    api.ResponsesOutputMessage,
			ID:      itemID,
			Role:    api.RoleAssistant,
			Status:  api.ResponsesStatusCompleted,
			Content: []api.OutputContent{doneContent},
		},
	}

	return &namedEventChunk{
		names: []string{textDone.Type, contentPartDone.Type, outputItemDone.Type},
		data:  []any{textDone, contentPartDone, outputItemDone},
	}
}

func (*responsesHTTPRespBuilder) createDoneChunk() sseChunk        { return nil }
func (*responsesHTTPRespBuilder) sendFinishReasonWithTokens() bool { return false }

func (*responsesHTTPRespBuilder) createRenderResponse(_ [][]uint32,
	_ *api.RenderMMFeatures) any {
	panic("responsesHTTPRespBuilder: /v1/responses has no /render endpoint")
}

var _ responseBuilder = (*responsesHTTPRespBuilder)(nil)

type generateHTTPRespBuilder struct{}

func (respBuilder *generateHTTPRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []api.Tokenized) any {
	respCtx := respCtxPerChoice[0]
	var tokenIDs []uint32
	if len(tokens) > 0 {
		tokenIDs = tokens[0].Tokens
	}
	choice := api.GenerateRespChoice{TokenIDs: tokenIDs}
	choice.Index = 0
	choice.FinishReason = respCtx.FinishReason()
	resp := &api.GenerateResponse{
		Choices:          []api.GenerateRespChoice{choice},
		GenRequestID:     respCtx.RequestID(),
		ECTransferParams: respCtx.ECTransferParams(),
	}
	if respCtx.DoRemoteDecode() {
		resp.KVParams = api.BuildPrefillKVTransferParams()
	}
	return resp
}

func (respBuilder *generateHTTPRespBuilder) createUsageChunk(respCtxPerChoice []vllmsim.ResponseContext) sseChunk {
	respCtx := respCtxPerChoice[0]
	if !respCtx.SendUsageData() {
		return nil
	}
	return &jsonDataChunk{data: &api.GenerateStreamResponse{
		RequestID: respCtx.RequestID(),
		Choices:   []api.GenerateRespChoice{},
		Usage:     aggregateUsage(respCtxPerChoice),
	}}
}

func (respBuilder *generateHTTPRespBuilder) createChunk(respCtx vllmsim.ResponseContext,
	tokens *api.Tokenized, _ *api.ToolCall, _ string, finishReason *string, choiceIdx int) sseChunk {
	choice := api.GenerateRespChoice{}
	choice.Index = choiceIdx
	choice.FinishReason = finishReason
	if tokens != nil {
		choice.TokenIDs = tokens.Tokens
	}
	return &jsonDataChunk{data: &api.GenerateStreamResponse{
		RequestID: respCtx.RequestID(),
		Choices:   []api.GenerateRespChoice{choice},
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

func (*generateHTTPRespBuilder) createDoneChunk() sseChunk        { return &doneMarker{} }
func (*generateHTTPRespBuilder) sendFinishReasonWithTokens() bool { return true }

func (*generateHTTPRespBuilder) createRenderResponse(_ [][]uint32,
	_ *api.RenderMMFeatures) any {
	panic("generateHTTPRespBuilder: /inference/v1/generate has no /render endpoint")
}

var _ responseBuilder = (*generateHTTPRespBuilder)(nil)

// messagesHTTPRespBuilder implements responseBuilder for /v1/messages (Anthropic Messages API).
type messagesHTTPRespBuilder struct {
	contentBlockIndex int  // tracks which content block index we are on for streaming
	inToolMode        bool // true when streaming tool call arguments
}

func (b *messagesHTTPRespBuilder) stopReason(finishReason string) string {
	switch finishReason {
	case common.ToolsFinishReason:
		return api.MessagesStopReasonToolUse
	case common.LengthFinishReason, common.CacheThresholdFinishReason:
		return api.MessagesStopReasonMaxTokens
	default:
		return api.MessagesStopReasonEndTurn
	}
}

func (b *messagesHTTPRespBuilder) createResponse(respCtxPerChoice []vllmsim.ResponseContext,
	tokens []api.Tokenized) any {
	respCtx := respCtxPerChoice[0]
	usage := respCtx.UsageData()
	finishReason := ""
	if respCtx.FinishReason() != nil {
		finishReason = *respCtx.FinishReason()
	}

	var content []api.MessagesContentBlock
	if toolCalls := respCtx.ToolCalls(); len(toolCalls) > 0 {
		for _, tc := range toolCalls {
			var input map[string]any
			if tc.Function.Arguments != "" {
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
			}
			if input == nil {
				input = map[string]any{}
			}
			name := ""
			if tc.Function.Name != nil {
				name = *tc.Function.Name
			}
			content = append(content, api.MessagesContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  name,
				Input: input,
			})
		}
	} else {
		text := strings.Join(tokens[0].Strings, "")
		content = []api.MessagesContentBlock{{Type: "text", Text: text}}
	}

	return api.CreateMessagesResponse(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		b.stopReason(finishReason),
		content,
		api.MessagesUsage{
			InputTokens:  usage.PromptTokens,
			OutputTokens: usage.CompletionTokens,
		},
	)
}

func (b *messagesHTTPRespBuilder) createUsageChunk(_ []vllmsim.ResponseContext) sseChunk {
	return nil // usage is delivered in the message_delta streaming event
}

func (b *messagesHTTPRespBuilder) createInitialChunk(respCtx vllmsim.ResponseContext) sseChunk {
	msg := api.CreateMessagesStreamStartMessage(
		respCtx.DisplayModel(),
		respCtx.RequestID(),
		respCtx.UsageData().PromptTokens,
	)
	return &namedEventChunk{
		names: []string{api.MessagesEventMessageStart},
		data:  []any{api.MessagesMessageStartEvent{Type: api.MessagesEventMessageStart, Message: msg}},
	}
}

func (b *messagesHTTPRespBuilder) createFirstChunk(respCtx vllmsim.ResponseContext, _ int) sseChunk {
	b.inToolMode = len(respCtx.ToolCalls()) > 0
	ping := api.MessagesPingEvent{Type: api.MessagesEventPing}

	if b.inToolMode {
		// content_block_start for tool blocks is emitted in createChunk when the
		// first argument token arrives (signalled by tool.Function.Name != nil).
		return &namedEventChunk{
			names: []string{api.MessagesEventPing},
			data:  []any{ping},
		}
	}

	blockStart := api.MessagesContentBlockStartEvent{
		Type:         api.MessagesEventContentBlockStart,
		Index:        0,
		ContentBlock: api.MessagesContentBlock{Type: "text", Text: ""},
	}
	return &namedEventChunk{
		names: []string{api.MessagesEventContentBlockStart, api.MessagesEventPing},
		data:  []any{blockStart, ping},
	}
}

func (b *messagesHTTPRespBuilder) createChunk(_ vllmsim.ResponseContext, tokens *api.Tokenized,
	tool *api.ToolCall, _ string, _ *string, _ int) sseChunk {

	if tool != nil {
		var names []string
		var data []any

		// tool.Function.Name != nil on the first argument token of a new tool call.
		if tool.Function.Name != nil {
			// Stop the previous block if this is not the first tool call.
			if b.contentBlockIndex > 0 {
				stop := api.MessagesContentBlockStopEvent{
					Type:  api.MessagesEventContentBlockStop,
					Index: b.contentBlockIndex - 1,
				}
				names = append(names, stop.Type)
				data = append(data, stop)
			}
			blockStart := api.MessagesContentBlockStartEvent{
				Type:  api.MessagesEventContentBlockStart,
				Index: b.contentBlockIndex,
				ContentBlock: api.MessagesContentBlock{
					Type:  "tool_use",
					ID:    tool.ID,
					Name:  *tool.Function.Name,
					Input: map[string]any{},
				},
			}
			names = append(names, blockStart.Type)
			data = append(data, blockStart)
			b.contentBlockIndex++
		}

		delta := api.MessagesContentBlockDeltaEvent{
			Type:  api.MessagesEventContentBlockDelta,
			Index: b.contentBlockIndex - 1,
			Delta: api.MessagesContentBlockDelta{
				Type:        "input_json_delta",
				PartialJSON: tool.Function.Arguments,
			},
		}
		names = append(names, delta.Type)
		data = append(data, delta)
		return &namedEventChunk{names: names, data: data}
	}

	if tokens == nil || len(tokens.Strings) == 0 {
		return nil
	}
	text := strings.Join(tokens.Strings, "")
	delta := api.MessagesContentBlockDeltaEvent{
		Type:  api.MessagesEventContentBlockDelta,
		Index: 0,
		Delta: api.MessagesContentBlockDelta{Type: "text_delta", Text: text},
	}
	return &namedEventChunk{
		names: []string{api.MessagesEventContentBlockDelta},
		data:  []any{delta},
	}
}

func (b *messagesHTTPRespBuilder) createLastChunk(respCtx vllmsim.ResponseContext, finishReason string, _ int) sseChunk {
	blockIdx := 0
	if b.inToolMode && b.contentBlockIndex > 0 {
		blockIdx = b.contentBlockIndex - 1
	}
	sr := b.stopReason(finishReason)

	blockStop := api.MessagesContentBlockStopEvent{
		Type:  api.MessagesEventContentBlockStop,
		Index: blockIdx,
	}
	msgDelta := api.MessagesMessageDeltaEvent{
		Type:  api.MessagesEventMessageDelta,
		Delta: api.MessagesMessageDeltaPayload{StopReason: &sr, StopSequence: nil},
		Usage: api.MessagesStreamUsage{OutputTokens: respCtx.UsageData().CompletionTokens},
	}
	msgStop := api.MessagesMessageStopEvent{Type: api.MessagesEventMessageStop}

	return &namedEventChunk{
		names: []string{blockStop.Type, msgDelta.Type, msgStop.Type},
		data:  []any{blockStop, msgDelta, msgStop},
	}
}

func (*messagesHTTPRespBuilder) createDoneChunk() sseChunk        { return nil }
func (*messagesHTTPRespBuilder) sendFinishReasonWithTokens() bool { return false }

func (*messagesHTTPRespBuilder) createRenderResponse(_ [][]uint32, _ *api.RenderMMFeatures) any {
	panic("messagesHTTPRespBuilder: /v1/messages has no /render endpoint")
}

var _ responseBuilder = (*messagesHTTPRespBuilder)(nil)
