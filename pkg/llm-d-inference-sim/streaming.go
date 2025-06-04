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

// sendStreamingResponse creates and sends a streaming response for completion request of both types (text and chat) as defined by isChatCompletion
// response content is wrapped according SSE format
// First token is send after timeToFirstToken milliseconds, every other token is sent after interTokenLatency milliseconds
func (s *VllmSimulator) sendStreamingResponse(isChatCompletion bool, ctx *fasthttp.RequestCtx, responseTxt string, model string,
	finishReason string, includeUsage bool, promptTokens int) {
	ctx.SetContentType("text/event-stream")
	ctx.SetStatusCode(fasthttp.StatusOK)

	ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		creationTime := time.Now().Unix()

		tokens := strings.Fields(responseTxt)
		s.logger.Info("Going to send text", "resp body", responseTxt, "tokens num", len(tokens))

		if len(tokens) > 0 {
			if isChatCompletion {
				// in chat completion first chunk contains the role
				err := s.sendChunk(true, w, creationTime, model, roleAssistant, "", nil, 0, 0)
				if err != nil {
					ctx.Error("Sending stream first chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
					return
				}
			}

			// time to first token delay
			time.Sleep(time.Duration(s.timeToFirstToken) * time.Millisecond)

			for i, token := range tokens {
				if i != 0 {
					time.Sleep(time.Duration(s.interTokenLatency) * time.Millisecond)
					token = " " + token
				}

				var err error

				if i == len(tokens)-1 && finishReason == lengthFinishReason {
					err = s.sendChunk(isChatCompletion, w, creationTime, model, "", token, &finishReason, 0, 0)
				} else {
					err = s.sendChunk(isChatCompletion, w, creationTime, model, "", token, nil, 0, 0)
				}
				if err != nil {
					ctx.Error("Sending stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
					return
				}
			}

			// send the last chunk if finish reason is stop
			if finishReason == stopFinishReason {
				err := s.sendChunk(isChatCompletion, w, creationTime, model, "", "", &finishReason, 0, 0)
				if err != nil {
					ctx.Error("Sending last stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
					return
				}
			}
		}

		// send usage
		if includeUsage {
			err := s.sendChunk(isChatCompletion, w, creationTime, model, "", "", nil, promptTokens, len(tokens))
			if err != nil {
				ctx.Error("Sending usage chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
				return
			}
		}

		// finish sse events stream
		_, err := fmt.Fprint(w, "data: [DONE]\n\n")
		if err != nil {
			ctx.Error("fprint failed, "+err.Error(), fasthttp.StatusInternalServerError)
			return
		}
		err = w.Flush()
		if err != nil {
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
// role this message role, relevenat to chat API only
// finishReason - a pointer to string that represents finish reason, can be nil or stop or length, ...
func (s *VllmSimulator) createCompletionChunk(isChatCompletion bool, creationTime int64, token string, model string, role string,
	finishReason *string, promptTokens int, completionTokens int) completionRespChunk {
	baseChunk := baseCompletionResponse{
		ID:      chatComplIDPrefix + uuid.NewString(),
		Created: creationTime,
		Model:   model,
	}
	if isChatCompletion {
		baseChunk.Object = chatCompletionChunkObject
	} else {
		baseChunk.Object = textCompletionObject
	}
	if completionTokens != 0 {
		baseChunk.Usage = &usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		}
		if isChatCompletion {
			return &chatCompletionResponse{
				baseCompletionResponse: baseChunk,
				Choices:                []chatRespChoice{},
			}
		}
		return &textCompletionResponse{
			baseCompletionResponse: baseChunk,
			Choices:                []textRespChoice{},
		}
	}
	baseChoice := baseResponseChoice{Index: 0, FinishReason: finishReason}

	if isChatCompletion {
		chunk := chatCompletionRespChunk{
			baseCompletionResponse: baseChunk,
			Choices:                []chatRespChunkChoice{{Delta: message{}, baseResponseChoice: baseChoice}},
		}

		if len(role) > 0 {
			chunk.Choices[0].Delta.Role = role
		}
		if len(token) > 0 {
			chunk.Choices[0].Delta.Content = token
		}

		return &chunk
	}

	return &textCompletionResponse{
		baseCompletionResponse: baseChunk,
		Choices:                []textRespChoice{{baseResponseChoice: baseChoice, Text: token}},
	}
}

// sendChunk send a single token chunk in a streamed completion API response
func (s *VllmSimulator) sendChunk(isChatCompletion bool, w *bufio.Writer, creationTime int64, model string, role string, token string,
	finishReason *string, promptTokens int, completionTokens int) error {
	chunk := s.createCompletionChunk(isChatCompletion, creationTime, token, model, role, finishReason, promptTokens, completionTokens)
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
