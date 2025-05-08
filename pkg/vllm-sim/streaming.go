/*
Copyright 2025 The vLLM-Sim Authors.

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

package vllmsim

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/valyala/fasthttp"
)

// sendStreamingResponse creates and sends a streaming response for completion request of both types (text and chat) as defined by isChatCompletion
// response content is wrapped according SSE format
// First token is send after timeToFirstToken milliseconds, every other token is sent after interTokenLatency milliseconds
// If options IncludeUsage is set, an additional chunk will be streamed before the `data: [DONE]` message.
// The `usage` field on this chunk shows the token usage statistics for the entire
// request, and the `choices` field will always be an empty array.
//
// All other chunks will also include a `usage` field, but with a null value.
// **NOTE:** If the stream is interrupted, you may not receive the final usage
// chunk which contains the total token usage for the request.
func (s *VllmSimulator) sendStreamingResponse(isChatCompletion bool, req completionRequest,
	ctx *fasthttp.RequestCtx, responseTxt string, model string) {
	ctx.SetContentType("text/event-stream")
	ctx.SetStatusCode(fasthttp.StatusOK)

	ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		creationTime := time.Now().Unix()

		tokens := strings.Fields(responseTxt)
		s.logger.Info("Going to send text", "resp body", responseTxt, "tokens num", len(tokens))

		options := req.getStreamOptions()
		var finalUsage *completionUsage
		if options != nil && options.IncludeUsage {
			promptTokens := req.getPromptTokensNumber()
			completionTokens := int64(len(tokens))
			finalUsage = &completionUsage{
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
				TotalTokens:      promptTokens + completionTokens}
		}

		var nullUsage *completionUsage
		if options != nil && options.IncludeUsage {
			// usage should be a null value and not omitted
			nullUsage = &completionUsage{zero: true}
		}
		if len(tokens) > 0 {
			if isChatCompletion {
				// in chat completion first chunk contains the role
				if err := s.sendChunk(true, w, creationTime, model, roleAssistant, "", nullUsage, nil); err != nil {
					ctx.Error("Sending stream first chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
					return
				}
			}

			// time to first token delay
			time.Sleep(time.Duration(s.timeToFirstToken) * time.Millisecond)

			isFirst := true
			for _, token := range tokens {
				if !isFirst {
					time.Sleep(time.Duration(s.interTokenLatency) * time.Millisecond)
				} else {
					isFirst = false
				}

				if err := s.sendChunk(isChatCompletion, w, creationTime, model, "", token, nullUsage, nil); err != nil {
					ctx.Error("Sending stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
					return
				}
			}

			finishReason := stopFinishReason
			if err := s.sendChunk(isChatCompletion, w, creationTime, model, "", "", nullUsage, &finishReason); err != nil {
				ctx.Error("Sending last stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
				return
			}

			if options != nil && options.IncludeUsage {
				// send usage
				if err := s.sendChunk(isChatCompletion, w, creationTime, model, "", "", finalUsage, nil); err != nil {
					ctx.Error("Sending stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
					return
				}
			}
		}

		// finish sse events stream
		_, err := fmt.Fprint(w, "data: [DONE]\n\n")
		if err != nil {
			ctx.Error("fprint failed, "+err.Error(), fasthttp.StatusInternalServerError)
			return
		}
		if err := w.Flush(); err != nil {
			ctx.Error("flush failed, "+err.Error(), fasthttp.StatusInternalServerError)
			return
		}

		s.responseSentCallback(model)
	})
}

// createCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion API response,
// supports both modes (text and chat)
// creationTime time when this response was started
// token the token to send
// model the model
// role this message role, relevant to chat API only
// usage response
// finishReason - a pointer to string that represents finish reason, can be nil or stop or length, ...
func (s *VllmSimulator) createCompletionChunk(isChatCompletion bool, creationTime int64,
	token string, model string, role string, usage *completionUsage, finishReason *string) completionRespChunk {
	complId := ""
	if isChatCompletion {
		complId = chatComplIdPrefix
	} else {
		complId = textComplIdPrefix
	}
	baseChunk := baseCompletionResponse{
		ID:      complId + uuid.NewString(),
		Created: creationTime,
		Model:   model,
		Usage:   usage,
	}
	baseChoice := baseResponseChoice{Index: 0, FinishReason: finishReason}

	if isChatCompletion {
		var chunk chatCompletionRespChunk
		if usage != nil && !usage.IsZero() {
			// this should be the final usage results if it exists
			// choices should be empty array
			chunk = chatCompletionRespChunk{
				baseCompletionResponse: baseChunk,
				Choices:                []chatRespChunkChoice{},
			}
		} else {
			chunk = chatCompletionRespChunk{
				baseCompletionResponse: baseChunk,
				Choices:                []chatRespChunkChoice{{Delta: message{}, baseResponseChoice: baseChoice}},
			}
			if len(role) > 0 {
				chunk.Choices[0].Delta.Role = role
			}
			if len(token) > 0 {
				chunk.Choices[0].Delta.Content = token
			}
		}
		return &chunk
	}
	var chunk textCompletionResponse
	if usage != nil && !usage.IsZero() {
		// this should be the final usage results if it exists
		// choices should be empty array
		chunk = textCompletionResponse{
			baseCompletionResponse: baseChunk,
		}
	} else {
		chunk = textCompletionResponse{
			baseCompletionResponse: baseChunk,
			Choices:                []textRespChoice{{baseResponseChoice: baseChoice, Text: token}},
		}
	}
	return &chunk
}

// sendChunk send a single token chunk in a streamed completion API response
func (s *VllmSimulator) sendChunk(isChatCompletion bool, w *bufio.Writer, creationTime int64,
	model string, role string, token string, usage *completionUsage, finishReason *string) error {
	chunk := s.createCompletionChunk(isChatCompletion, creationTime, token, model, role, usage, finishReason)
	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	if err != nil {
		return err
	}
	err = w.Flush()
	if err != nil {
		return err
	}

	return nil
}
