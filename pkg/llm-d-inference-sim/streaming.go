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
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

// sendStreamingResponse creates and sends a streaming response for completion requests of both types (text and chat)
// response content is wrapped according SSE format
// First token is send after timeToFirstToken milliseconds, every other token is sent after interTokenLatency milliseconds
func (h *httpResponseSender) sendStreamingResponse(respCtx responseContext) {
	ctx := h.ctx
	ctx.SetContentType("text/event-stream")
	ctx.SetStatusCode(fasthttp.StatusOK)

	// Add pod and namespace information to response headers for testing/debugging
	if h.sim.pod != "" {
		ctx.Response.Header.Add(podHeader, h.sim.pod)
		ctx.Response.Header.Add(portHeader, strconv.Itoa(h.sim.config.Port))
	}
	if h.sim.namespace != "" {
		ctx.Response.Header.Add(namespaceHeader, h.sim.namespace)
	}
	if h.sim.config.EnableRequestIDHeaders {
		ctx.Response.Header.Add(requestIDHeader, respCtx.requestID())
	}

	ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		respCtx.setCreationTime(time.Now().Unix())

		if (respCtx.responseTokens() != nil && respCtx.responseTokens().Length() > 0) || len(respCtx.toolCalls()) > 0 {
			// in chat completion first chunk contains the role
			chunk := respCtx.createFirstCompletionChunk()
			if chunk != nil {
				if err := h.sendChunk(w, chunk, ""); err != nil {
					ctx.Error("Sending stream first chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
					return
				}
			}
			if len(respCtx.toolCalls()) > 0 {
				h.sim.logger.V(logging.TRACE).Info("Going to send tools calls")
				for _, tc := range respCtx.toolCalls() {
					h.sendTokenChunks(respCtx, ctx, w, tc.Function.TokenizedArguments.Strings, &tc)
				}
			} else {
				respTokens := respCtx.responseTokens()
				h.sim.logger.V(logging.TRACE).Info("Going to send text", "number of tokens", respTokens.Length())
				h.sendTokenChunks(respCtx, ctx, w, respTokens.Strings, nil)
				h.sim.logger.V(4).Info("Finished sending text", "number of tokens", respTokens.Length())
			}
		} else if respCtx.finishReason() != nil && *respCtx.finishReason() == common.CacheThresholdFinishReason {
			// No tokens to stream but we still need to emit a finish chunk for cache_threshold
			chunk := respCtx.createCompletionChunk("", nil, "", respCtx.finishReason())
			if err := h.sendChunk(w, chunk, ""); err != nil {
				ctx.Error("Sending finish chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
				return
			}
		}

		// send usage
		if respCtx.sendUsageData() {
			chunk := respCtx.createUsageChunk()
			if err := h.sendChunk(w, chunk, ""); err != nil {
				ctx.Error("Sending usage chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
				return
			}
		}

		// finish sse events stream
		if err := h.sendChunk(w, nil, "[DONE]"); err != nil {
			ctx.Error("Sending last stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
			return
		}
		h.responseSentCallback(respCtx.requestContext(), respCtx.displayModel())
		respCtx.done()
	})
}

// sendTokenChunks creates and sends response chunks
func (h *httpResponseSender) sendTokenChunks(respCtx responseContext, ctx *fasthttp.RequestCtx, w *bufio.Writer, genTokens []string,
	tc *openaiserverapi.ToolCall) {
	// Skip delays if finish reason is cache_threshold (immediate return)
	isCacheThresholdFinishReason := respCtx.finishReason() != nil && *respCtx.finishReason() == common.CacheThresholdFinishReason
	if !isCacheThresholdFinishReason {
		h.sim.simulateTTFT(respCtx)
	}
	startDecode := time.Now()
	for i, token := range genTokens {
		if i != 0 && !isCacheThresholdFinishReason {
			h.sim.simulateInterTokenLatency()
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
		if i == len(genTokens)-1 && (*respCtx.finishReason() == common.LengthFinishReason ||
			*respCtx.finishReason() == common.ToolsFinishReason ||
			*respCtx.finishReason() == common.CacheThresholdFinishReason) {
			finishReasonToSend = respCtx.finishReason()
		}
		chunk = respCtx.createCompletionChunk(token, toolChunkInsert, "", finishReasonToSend)
		if err := h.sendChunk(w, chunk, ""); err != nil {
			ctx.Error("Sending stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
			return
		}
	}

	common.WriteToChannel(h.sim.metrics.reqDecodeTimeChan, time.Since(startDecode).Seconds(), h.sim.logger, "metrics.reqDecodeTimeChan")

	// send the last chunk if finish reason is stop
	var chunk openaiserverapi.CompletionRespChunk
	if *respCtx.finishReason() == common.StopFinishReason {
		chunk = respCtx.createCompletionChunk("", nil, "", respCtx.finishReason())
		if err := h.sendChunk(w, chunk, ""); err != nil {
			ctx.Error("Sending last stream chunk failed, "+err.Error(), fasthttp.StatusInternalServerError)
			return
		}
	}
}

// sendChunk send a single token chunk in a streamed completion API response,
// receives either a completionRespChunk or a string with the data to send.
func (h *httpResponseSender) sendChunk(w *bufio.Writer, chunk openaiserverapi.CompletionRespChunk, dataString string) error {
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
