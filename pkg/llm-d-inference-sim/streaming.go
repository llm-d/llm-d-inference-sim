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
	"strconv"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

func (s *VllmSimulator) sendStream(ctx *fasthttp.RequestCtx, reqCtx requestContext, channel chan *responseInfo) {
	ctx.SetContentType("text/event-stream")
	ctx.SetStatusCode(fasthttp.StatusOK)

	// Add pod and namespace information to response headers for testing/debugging
	if s.context.pod != "" {
		ctx.Response.Header.Add(podHeader, s.context.pod)
		ctx.Response.Header.Add(portHeader, strconv.Itoa(s.context.config.Port))
	}
	if s.context.namespace != "" {
		ctx.Response.Header.Add(namespaceHeader, s.context.namespace)
	}
	if s.context.config.EnableRequestIDHeaders {
		ctx.Response.Header.Add(requestIDHeader, reqCtx.request().GetRequestID())
	}

	ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		first := true
		var respCtx responseContext
		var lastToolCall *openaiserverapi.ToolCall
		var toolCallIndex int
		for response := range channel {
			if response.err != nil {
				ctx.Error(response.err.Message, response.err.Code)
				return
			}
			if first {
				respCtx = response.respCtx
				respCtx.setCreationTime(time.Now().Unix())
			}

			if response.tokenStrs != nil {
				// in chat completion first chunk contains the role
				if first {
					chunk := respCtx.createFirstCompletionChunk()
					if chunk != nil {
						if err := s.sendChunk(w, chunk, ""); err != nil {
							s.chunkSendFailed(ctx, respCtx, "Sending first stream chunk failed, ", err)
							return
						}
					}
				}
				if response.toolCall != nil {
					if lastToolCall != response.toolCall {
						toolCallIndex = 0
					} else {
						toolCallIndex++
					}
					if ok := s.sendTools(respCtx, ctx, w, response.tokenStrs[0], response.toolCall, toolCallIndex); !ok {
						return
					}
					lastToolCall = response.toolCall
				} else {
					chunk := respCtx.createCompletionChunk(response.tokenStrs[0], nil, "", nil)
					if err := s.sendChunk(w, chunk, ""); err != nil {
						s.chunkSendFailed(ctx, respCtx, "Sending stream chunk failed, ", err)
						return
					}
				}
			} else if respCtx.finishReason() != nil && *respCtx.finishReason() == common.CacheThresholdFinishReason {
				// No tokens to stream but we still need to emit a finish chunk for cache_threshold
				chunk := respCtx.createCompletionChunk("", nil, "", respCtx.finishReason())
				if err := s.sendChunk(w, chunk, ""); err != nil {
					s.chunkSendFailed(ctx, respCtx, "Sending finish chunk failed, ", err)
					return
				}
			}
		}

		// send the last chunk if finish reason is stop
		if *respCtx.finishReason() == common.StopFinishReason {
			chunk := respCtx.createCompletionChunk("", nil, "", respCtx.finishReason())
			if err := s.sendChunk(w, chunk, ""); err != nil {
				s.chunkSendFailed(ctx, respCtx, "Sending last stream chunk failed, ", err)
				return
			}
		}

		// send usage
		if respCtx.sendUsageData() {
			chunk := respCtx.createUsageChunk()
			if err := s.sendChunk(w, chunk, ""); err != nil {
				s.chunkSendFailed(ctx, respCtx, "Sending usage chunk failed, ", err)
				return
			}
		}

		// finish sse events stream
		if err := s.sendChunk(w, nil, "[DONE]"); err != nil {
			s.chunkSendFailed(ctx, respCtx, "Sending last stream chunk failed, ", err)
			return
		}
		s.responseSentCallback(respCtx.requestContext(), respCtx.displayModel())
		respCtx.done()
	})
}

func (s *VllmSimulator) chunkSendFailed(ctx *fasthttp.RequestCtx, respCtx responseContext, msg string, err error) {
	ctx.Error(msg+err.Error(), fasthttp.StatusInternalServerError)
	respCtx.done()
}

func (s *VllmSimulator) sendTools(respCtx responseContext, ctx *fasthttp.RequestCtx, w *bufio.Writer, token string,
	tc *openaiserverapi.ToolCall, index int) bool {
	toolChunkInsert := &openaiserverapi.ToolCall{
		ID:    tc.ID,
		Type:  tc.Type,
		Index: tc.Index,
		Function: openaiserverapi.FunctionCall{
			Arguments: token,
		},
	}
	if index == 0 {
		toolChunkInsert.Function.Name = tc.Function.Name
	}

	var chunk openaiserverapi.CompletionRespChunk
	var finishReasonToSend *string
	if index == tc.Function.TokenizedArguments().Length()-1 && (*respCtx.finishReason() == common.LengthFinishReason ||
		*respCtx.finishReason() == common.ToolsFinishReason ||
		*respCtx.finishReason() == common.CacheThresholdFinishReason) {
		finishReasonToSend = respCtx.finishReason()
	}
	chunk = respCtx.createCompletionChunk(token, toolChunkInsert, "", finishReasonToSend)
	if err := s.sendChunk(w, chunk, ""); err != nil {
		ctx.Error("Sending stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
		return false
	}
	return true
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
