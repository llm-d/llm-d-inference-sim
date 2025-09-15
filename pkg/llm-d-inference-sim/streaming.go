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
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

type streamingContext struct {
	ctx                 *fasthttp.RequestCtx
	isChatCompletion    bool
	model               string
	creationTime        int64
	doRemotePrefill     bool
	nPromptTokens       int
	nCachedPromptTokens int
	requestID           string
}

// sendStreamingResponse creates and sends a streaming response for completion requests of both types (text and chat)
// as defined by isChatCompletion
// response content is wrapped according SSE format
// First token is send after timeToFirstToken milliseconds, every other token is sent after interTokenLatency milliseconds
func (s *VllmSimulator) sendStreamingResponse(context *streamingContext, responseTokens []string, toolCalls []openaiserverapi.ToolCall,
	finishReason string, usageData *openaiserverapi.Usage) {
	context.ctx.SetContentType("text/event-stream")
	context.ctx.SetStatusCode(fasthttp.StatusOK)

	// Add pod and namespace information to response headers for testing/debugging
	if s.pod != "" {
		context.ctx.Response.Header.Add(podHeader, s.pod)
	}
	if s.namespace != "" {
		context.ctx.Response.Header.Add(namespaceHeader, s.namespace)
	}

	context.ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		context.creationTime = time.Now().Unix()

		if len(responseTokens) > 0 || len(toolCalls) > 0 {
			if context.isChatCompletion {
				// in chat completion first chunk contains the role
				chunk := s.createChatCompletionChunk(context, "", nil, openaiserverapi.RoleAssistant, nil)
				if err := s.sendChunk(w, chunk, ""); err != nil {
					context.ctx.Error("Sending stream first chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
					return
				}
			}
			if len(toolCalls) > 0 {
				s.logger.Info("Going to send tools calls")
				for _, tc := range toolCalls {
					s.sendTokenChunks(context, w, tc.Function.TokenizedArguments, &tc, finishReason)
				}
			} else {
				s.logger.Info("Going to send text", "number of tokens", len(responseTokens))
				s.sendTokenChunks(context, w, responseTokens, nil, finishReason)
			}
		}

		// send usage
		if usageData != nil {
			chunk := s.createUsageChunk(context, usageData)
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
		s.responseSentCallback(context.model, context.isChatCompletion, context.requestID)
	})
}

// sendTokenChunks creates and sends response chunks
func (s *VllmSimulator) sendTokenChunks(context *streamingContext, w *bufio.Writer, genTokens []string,
	tc *openaiserverapi.ToolCall, finishReason string) {
	// time to first token delay
	ttft := s.getWaitTimeToFirstToken(context.nPromptTokens, context.nCachedPromptTokens, context.doRemotePrefill)
	time.Sleep(time.Duration(ttft) * time.Millisecond)

	for i, token := range genTokens {
		if i != 0 {
			time.Sleep(time.Duration(s.getInterTokenLatency()) * time.Millisecond)
		}
		var toolChunkInsert *openaiserverapi.ToolCall
		if tc != nil {
			toolChunkInsert = &openaiserverapi.ToolCall{
				ID:    tc.ID,
				Type:  tc.Type,
				Index: tc.Index,
				Function: openaiserverapi.FunctionCall{
					Arguments: token,
				},
			}
			if i == 0 {
				toolChunkInsert.Function.Name = tc.Function.Name
			}
		}

		var chunk openaiserverapi.CompletionRespChunk
		var finishReasonToSend *string
		if i == len(genTokens)-1 && (finishReason == dataset.LengthFinishReason || finishReason == dataset.ToolsFinishReason) {
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
	var chunk openaiserverapi.CompletionRespChunk
	if finishReason == dataset.StopFinishReason {
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
func (s *VllmSimulator) createUsageChunk(context *streamingContext, usageData *openaiserverapi.Usage) openaiserverapi.CompletionRespChunk {
	baseChunk := openaiserverapi.BaseCompletionResponse{
		ID:      chatComplIDPrefix + common.GenerateUUIDString(),
		Created: context.creationTime,
		Model:   context.model,
		Usage:   usageData,
	}
	if context.isChatCompletion {
		baseChunk.Object = chatCompletionChunkObject
		return &openaiserverapi.ChatCompletionResponse{
			BaseCompletionResponse: baseChunk,
			Choices:                []openaiserverapi.ChatRespChoice{},
		}
	}
	baseChunk.Object = textCompletionObject

	return &openaiserverapi.TextCompletionResponse{
		BaseCompletionResponse: baseChunk,
		Choices:                []openaiserverapi.TextRespChoice{},
	}
}

// createTextCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion API response,
// for text completion
func (s *VllmSimulator) createTextCompletionChunk(context *streamingContext, token string, finishReason *string) openaiserverapi.CompletionRespChunk {
	return &openaiserverapi.TextCompletionResponse{
		BaseCompletionResponse: openaiserverapi.BaseCompletionResponse{
			ID:      chatComplIDPrefix + common.GenerateUUIDString(),
			Created: context.creationTime,
			Model:   context.model,
			Object:  textCompletionObject,
		},
		Choices: []openaiserverapi.TextRespChoice{
			{
				BaseResponseChoice: openaiserverapi.BaseResponseChoice{Index: 0, FinishReason: finishReason},
				Text:               token,
			},
		},
	}
}

// createChatCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion
// API response, for chat completion. It sets either role, or token, or tool call info in the message.
func (s *VllmSimulator) createChatCompletionChunk(context *streamingContext, token string, tool *openaiserverapi.ToolCall,
	role string, finishReason *string) openaiserverapi.CompletionRespChunk {
	chunk := openaiserverapi.ChatCompletionRespChunk{
		BaseCompletionResponse: openaiserverapi.BaseCompletionResponse{
			ID:      chatComplIDPrefix + common.GenerateUUIDString(),
			Created: context.creationTime,
			Model:   context.model,
			Object:  chatCompletionChunkObject,
		},
		Choices: []openaiserverapi.ChatRespChunkChoice{
			{
				Delta:              openaiserverapi.Message{},
				BaseResponseChoice: openaiserverapi.BaseResponseChoice{Index: 0, FinishReason: finishReason},
			},
		},
	}

	if len(role) > 0 {
		chunk.Choices[0].Delta.Role = role
	}
	if tool != nil {
		chunk.Choices[0].Delta.ToolCalls = []openaiserverapi.ToolCall{*tool}
	} else if len(token) > 0 {
		chunk.Choices[0].Delta.Content.Raw = token
	}

	return &chunk
}

// sendChunk send a single token chunk in a streamed completion API response,
// receives either a completionRespChunk or a string with the data to send.
func (s *VllmSimulator) sendChunk(w *bufio.Writer, chunk openaiserverapi.CompletionRespChunk, dataString string) error {
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
