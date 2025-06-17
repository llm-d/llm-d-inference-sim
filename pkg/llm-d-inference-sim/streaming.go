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
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/valyala/fasthttp"
)

type streamingContext struct {
	ctx              *fasthttp.RequestCtx
	isChatCompletion bool
	model            string
	creationTime     int64
}

// sendStreamingResponse creates and sends a streaming response for completion request of both types (text and chat) as defined by isChatCompletion
// response content is wrapped according SSE format
// First token is send after timeToFirstToken milliseconds, every other token is sent after interTokenLatency milliseconds
func (s *VllmSimulator) sendStreamingResponse(context *streamingContext, responseTxt string, toolCalls []toolCall, finishReason string, includeUsage bool, promptTokens int) {
	context.ctx.SetContentType("text/event-stream")
	context.ctx.SetStatusCode(fasthttp.StatusOK)

	context.ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		context.creationTime = time.Now().Unix()

		tokens := strings.Fields(responseTxt)

		if len(tokens) > 0 || len(toolCalls) > 0 {
			if context.isChatCompletion {
				// in chat completion first chunk contains the role
				chunk := s.createChatCompletionChunk(context, "", nil, roleAssistant, nil)
				if err := s.sendChunk(w, chunk, ""); err != nil {
					context.ctx.Error("Sending stream first chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
					return
				}
			}
			if len(toolCalls) > 0 {
				s.logger.Info("Going to send tools calls")
				for _, tc := range toolCalls {
					argsTokens := strings.Fields(tc.Function.Arguments)
					s.sendTokenChunks(context, w, argsTokens, &tc, finishReason)
				}
			} else {
				s.logger.Info("Going to send text", "resp body", responseTxt, "tokens num", len(tokens))
				s.sendTokenChunks(context, w, tokens, nil, finishReason)
			}
		}

		// send usage
		if includeUsage {
			completionTokens := len(tokens)
			if toolCalls != nil {
				completionTokens = countTokensForToolCalls(toolCalls)
			}
			chunk := s.createUsageChunk(context, promptTokens, completionTokens)
			if err := s.sendChunk(w, chunk, ""); err != nil {
				context.ctx.Error("Sending usage chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
				return
			}
		}

		// finish sse events stream
		if err := s.sendChunk(w, nil, "[DONE]"); err != nil {
			context.ctx.Error("Sending last stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
			return
		}
		s.responseSentCallback(context.model)
	})
}

// sendTokenChunks creates and sends response chunks
func (s *VllmSimulator) sendTokenChunks(context *streamingContext, w *bufio.Writer, tokens []string, tc *toolCall, finishReason string) {
	// time to first token delay
	time.Sleep(time.Duration(s.timeToFirstToken) * time.Millisecond)

	for i, token := range tokens {
		if i != 0 {
			time.Sleep(time.Duration(s.interTokenLatency) * time.Millisecond)
			token = " " + token
		}
		var toolChunkInsert *toolCall
		if tc != nil {
			toolChunkInsert = &toolCall{
				ID:    tc.ID,
				Type:  tc.Type,
				Index: tc.Index,
				Function: functionCall{
					Arguments: token,
				},
			}
			if i == 0 {
				toolChunkInsert.Function.Name = tc.Function.Name
			}
		}

		var chunk completionRespChunk
		var finishReasonToSend *string
		if i == len(tokens)-1 && (finishReason == lengthFinishReason || finishReason == toolsFinishReason) {
			finishReasonToSend = &finishReason
		}
		if context.isChatCompletion {
			chunk = s.createChatCompletionChunk(context, token, toolChunkInsert, "", finishReasonToSend)
		} else {
			chunk = s.createTextCompletionChunk(context, token, finishReasonToSend)
		}

		if err := s.sendChunk(w, chunk, ""); err != nil {
			context.ctx.Error("Sending stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
			return
		}
	}

	// send the last chunk if finish reason is stop
	var chunk completionRespChunk
	if finishReason == stopFinishReason {
		if context.isChatCompletion {
			chunk = s.createChatCompletionChunk(context, "", nil, "", &finishReason)
		} else {
			chunk = s.createTextCompletionChunk(context, "", &finishReason)
		}
		if err := s.sendChunk(w, chunk, ""); err != nil {
			context.ctx.Error("Sending last stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
			return
		}
	}
}

// createUsageChunk creates and returns a CompletionRespChunk with usage data, a single chunk of streamed completion API response,
// supports both modes (text and chat)
func (s *VllmSimulator) createUsageChunk(context *streamingContext, promptTokens int, completionTokens int) completionRespChunk {
	baseChunk := baseCompletionResponse{
		ID:      chatComplIDPrefix + uuid.NewString(),
		Created: context.creationTime,
		Model:   context.model,
		Usage: &usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}
	if context.isChatCompletion {
		baseChunk.Object = chatCompletionChunkObject
		return &chatCompletionResponse{
			baseCompletionResponse: baseChunk,
			Choices:                []chatRespChoice{},
		}
	}
	baseChunk.Object = textCompletionObject

	return &textCompletionResponse{
		baseCompletionResponse: baseChunk,
		Choices:                []textRespChoice{},
	}
}

// createTextCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion API response,
// for text completion
func (s *VllmSimulator) createTextCompletionChunk(context *streamingContext, token string, finishReason *string) completionRespChunk {
	return &textCompletionResponse{
		baseCompletionResponse: baseCompletionResponse{
			ID:      chatComplIDPrefix + uuid.NewString(),
			Created: context.creationTime,
			Model:   context.model,
			Object:  textCompletionObject,
		},
		Choices: []textRespChoice{
			{
				baseResponseChoice: baseResponseChoice{Index: 0, FinishReason: finishReason},
				Text:               token,
			},
		},
	}
}

// createChatCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion
// API response, for chat completion. It sets either role, or token, or tool call info in the message.
func (s *VllmSimulator) createChatCompletionChunk(context *streamingContext, token string, tool *toolCall,
	role string, finishReason *string) completionRespChunk {
	chunk := chatCompletionRespChunk{
		baseCompletionResponse: baseCompletionResponse{
			ID:      chatComplIDPrefix + uuid.NewString(),
			Created: context.creationTime,
			Model:   context.model,
			Object:  chatCompletionChunkObject,
		},
		Choices: []chatRespChunkChoice{
			{
				Delta:              message{},
				baseResponseChoice: baseResponseChoice{Index: 0, FinishReason: finishReason},
			},
		},
	}

	if len(role) > 0 {
		chunk.Choices[0].Delta.Role = role
	}
	if tool != nil {
		chunk.Choices[0].Delta.ToolCalls = []toolCall{*tool}
	} else if len(token) > 0 {
		chunk.Choices[0].Delta.Content.Raw = token
	}

	return &chunk
}

// sendChunk send a single token chunk in a streamed completion API response,
// receives either a completionRespChunk or a string with the data to send.
func (s *VllmSimulator) sendChunk(w *bufio.Writer, chunk completionRespChunk, dataString string) error {
	if dataString == "" {
		data, err := json.Marshal(chunk)
		if err != nil {
			return err
		}
		dataString = string(data)
	}

	_, err := fmt.Fprintf(w, "data: %s\n\n", dataString)
	if err != nil {
		return err
	}
	err = w.Flush()
	if err != nil {
		return err
	}

	return nil
}
