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

package communication

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/buaazp/fasthttprouter"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

const (
	PodHeader                        = "x-inference-pod"
	PortHeader                       = "x-inference-port"
	NamespaceHeader                  = "x-inference-namespace"
	RequestIDHeader                  = "X-Request-Id"
	CacheThresholdFinishReasonHeader = "X-Cache-Threshold-Finish-Reason"
	XReturnErrorHeader               = "X-Return-Error"

	maxHTTPLogBodyBytes = 512 * 1024
)

func (c *Communication) newListener() (net.Listener, error) {
	listener, err := net.Listen("tcp4", fmt.Sprintf(":%d", c.simulator.Context.Config.Port))
	if err != nil {
		return nil, err
	}
	return listener, nil
}

// startHTTPServer builds and starts the HTTP server, returning the server instance and an error channel.
// It does not handle shutdown — callers are responsible for calling server.Shutdown().
func (c *Communication) startHTTPServer(listener net.Listener) (*fasthttp.Server, <-chan error, error) {
	r := fasthttprouter.New()

	// support completion APIs
	r.POST("/v1/chat/completions", c.HandleChatCompletions)
	r.POST("/v1/completions", c.HandleTextCompletions)
	r.POST("/v1/responses", c.HandleResponses)
	if !c.simulator.Context.Config.MMEncoderOnly {
		r.POST("/v1/embeddings", c.HandleEmbeddings)
	}
	// supports /models API
	r.GET("/v1/models", c.HandleModels)
	// support load/unload of lora adapter
	r.POST("/v1/load_lora_adapter", c.HandleLoadLora)
	r.POST("/v1/unload_lora_adapter", c.HandleUnloadLora)
	// supports /metrics prometheus API
	r.GET("/metrics", fasthttpadaptor.NewFastHTTPHandler(promhttp.HandlerFor(c.simulator.Context.MetricsRegistry(), promhttp.HandlerOpts{})))
	r.POST("/fake_metrics", c.HandleFakeMetrics)
	// supports standard Kubernetes health and readiness checks
	r.GET("/health", c.HandleHealth)
	r.GET("/health/ready", c.HandleHealthReady)
	r.GET("/ready", c.HandleReady)
	r.POST("/tokenize", c.HandleTokenize)
	r.POST("/sleep", c.HandleSleep)
	r.POST("/wake_up", c.HandleWakeUp)
	r.GET("/is_sleeping", c.HandleIsSleeping)

	handler := r.Handler
	if c.simulator.Context.Config.LogHTTP {
		handler = c.logHTTPMiddleware(handler)
	}

	server := &fasthttp.Server{
		ErrorHandler: c.HandleError,
		Handler:      handler,
		Logger:       c,
	}

	if err := c.configureSSL(server); err != nil {
		return nil, nil, err
	}

	errCh := make(chan error, 1)
	go func() {
		if c.simulator.Context.Config.SSLEnabled() {
			c.logger.V(logging.INFO).Info("Server starting", "protocol", "HTTPS", "port", c.simulator.Context.Config.Port)
			errCh <- server.ServeTLS(listener, "", "")
		} else {
			c.logger.V(logging.INFO).Info("Server starting", "protocol", "HTTP", "port", c.simulator.Context.Config.Port)
			errCh <- server.Serve(listener)
		}
	}()

	return server, errCh, nil
}

// getRequestID retrieves the request ID from the X-Request-Id header or generates a new one if not present
func (c *Communication) getRequestID(ctx *fasthttp.RequestCtx) string {
	if c.simulator.Context.Config.EnableRequestIDHeaders {
		requestID := string(ctx.Request.Header.Peek(RequestIDHeader))
		if requestID != "" {
			return requestID
		}
	}
	return c.simulator.Context.Random.GenerateUUIDString()
}

// HandleChatCompletions http handler for /v1/chat/completions
func (c *Communication) HandleChatCompletions(ctx *fasthttp.RequestCtx) {
	c.handleHTTP(&vllmsim.ChatCompletionsRequest{}, &chatComplHTTPRespBuilder{}, ctx)
}

// HandleTextCompletions http handler for /v1/completions
func (c *Communication) HandleTextCompletions(ctx *fasthttp.RequestCtx) {
	c.handleHTTP(&vllmsim.TextCompletionsRequest{}, &textComplHTTPRespBuilder{}, ctx)
}

// HandleResponses http handler for /v1/responses
func (c *Communication) HandleResponses(ctx *fasthttp.RequestCtx) {
	c.handleHTTP(&vllmsim.ResponsesRequest{}, &responsesHTTPRespBuilder{}, ctx)
}

// addResponseHeaders adds optional pod/port/namespace/request-id headers to the response for testing/debugging.
func (c *Communication) addResponseHeaders(ctx *fasthttp.RequestCtx, requestID string) {
	if c.simulator.Context.Config.PodName != "" {
		ctx.Response.Header.Add(PodHeader, c.simulator.Context.Config.PodName)
		ctx.Response.Header.Add(PortHeader, strconv.Itoa(c.simulator.Context.Config.Port))
	}
	if c.simulator.Context.Config.PodNameSpace != "" {
		ctx.Response.Header.Add(NamespaceHeader, c.simulator.Context.Config.PodNameSpace)
	}
	if c.simulator.Context.Config.EnableRequestIDHeaders {
		ctx.Response.Header.Add(RequestIDHeader, requestID)
	}
}

func (c *Communication) handleHTTP(req vllmsim.Request, respBuilder responseBuilder, ctx *fasthttp.RequestCtx) {
	if c.stopping.Load() {
		ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
		return
	}

	if err := req.Unmarshal(ctx.Request.Body()); err != nil {
		c.logger.Error(err, "failed to read and parse request body")
		errToSend := openaiserverapi.NewError("Failed to read and parse request body, "+err.Error(), fasthttp.StatusBadRequest, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}

	requestID := c.getRequestID(ctx)
	req.SetRequestID(requestID)

	// Check for X-Return-Error header - deterministic error trigger
	if errCodeStr := string(ctx.Request.Header.Peek(XReturnErrorHeader)); errCodeStr != "" {
		code, err := strconv.Atoi(errCodeStr)
		if err != nil {
			errToSend := openaiserverapi.NewError(
				fmt.Sprintf("Invalid X-Return-Error header value %q: must be an integer", errCodeStr),
				fasthttp.StatusBadRequest, nil)
			c.sendError(ctx, &errToSend, false)
			return
		}
		errToSend := openaiserverapi.NewError(
			fmt.Sprintf("Simulated error triggered by X-Return-Error header (code %d)", code),
			code, nil)
		c.sendError(ctx, &errToSend, true)
		return
	}

	// Check for cache threshold finish reason header - this forces a cache_threshold finish reason
	headerValue := string(ctx.Request.Header.Peek(CacheThresholdFinishReasonHeader))
	if parsedValue, err := strconv.ParseBool(headerValue); err == nil {
		req.SetCacheThresholdFinishReason(parsedValue)
	}

	isStream, channel, err, errInjected := c.simulator.HandleRequest(req)
	if err != nil {
		c.sendError(ctx, err, errInjected)
		return
	}

	c.logger.V(logging.DEBUG).Info("Received", "new HTTP", req.AsString())

	ctx.SetStatusCode(fasthttp.StatusOK)

	c.addResponseHeaders(ctx, req.GetRequestID())

	if isStream {
		ctx.SetContentType("text/event-stream")
		c.sendStream(ctx, *channel, respBuilder)
	} else {
		ctx.SetContentType("application/json")
		c.sendNonStream(ctx, *channel, respBuilder)
	}
}

func (c *Communication) sendNonStream(ctx *fasthttp.RequestCtx, channel common.Channel[*vllmsim.ResponseInfo],
	respBuilder responseBuilder) {
	tokens := openaiserverapi.Tokenized{
		Tokens:  make([]uint32, 0),
		Strings: make([]string, 0),
	}

	var respCtx vllmsim.ResponseContext
	for response := range channel.Channel {
		if response.Err != nil {
			c.sendError(ctx, response.Err, false)
			return
		}

		if response.Tokens != nil {
			tokens.Append(*response.Tokens)
		}
		respCtx = response.RespCtx
	}

	defer c.onResponseSendFinished(respCtx)
	resp := respBuilder.createResponse(respCtx, &tokens)
	data, err := json.Marshal(resp)
	if err != nil {
		err := openaiserverapi.NewError("Response body creation failed, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &err, false)
		return
	}
	ctx.Response.SetBody(data)
}

func (c *Communication) sendStream(ctx *fasthttp.RequestCtx, channel common.Channel[*vllmsim.ResponseInfo],
	respBuilder responseBuilder) {
	pr, pw := io.Pipe()

	go func() {
		w := bufio.NewWriter(pw)
		first := true
		firstTokens := true
		var respCtx vllmsim.ResponseContext

		defer func() {
			w.Flush()  //nolint:errcheck
			pw.Close() //nolint:errcheck
			c.onResponseSendFinished(respCtx)
		}()

		var lastToolCall *openaiserverapi.ToolCall
		var toolCallIndex int
		for response := range channel.Channel {
			if response.Err != nil {
				ctx.Error(response.Err.Message, response.Err.Code)
				return
			}
			if first {
				respCtx = response.RespCtx
				respCtx.SetCreationTime(time.Now().Unix())
				first = false
				if response.Status == vllmsim.ResponseStatusCreated {
					if chunk := respBuilder.createInitialChunk(respCtx); chunk != nil {
						if err := c.sendChunk(w, chunk); err != nil {
							c.chunkSendFailed(ctx, "Sending first stream chunk failed, ", err)
							return
						}
					}
					continue
				}
			}

			// nolint
			if response.Tokens != nil {
				// in chat completion first chunk contains the role
				if firstTokens {
					if chunk := respBuilder.createFirstChunk(respCtx); chunk != nil {
						if err := c.sendChunk(w, chunk); err != nil {
							c.chunkSendFailed(ctx, "Sending first stream chunk failed, ", err)
							return
						}
					}
					firstTokens = false
				}
				if response.ToolCall != nil {
					if lastToolCall != response.ToolCall {
						toolCallIndex = 0
					} else {
						toolCallIndex++
					}
					if err := c.sendStreamedTools(respCtx, respBuilder, w, response.Tokens.Strings, response.ToolCall,
						toolCallIndex); err != nil {
						c.chunkSendFailed(ctx, "Sending tools chunk failed, ", err)
						return
					}
					lastToolCall = response.ToolCall
				} else {
					chunk := respBuilder.createChunk(respCtx, response.Tokens, nil, "", nil)
					if err := c.sendChunk(w, chunk); err != nil {
						c.chunkSendFailed(ctx, "Sending stream chunk failed, ", err)
						return
					}
				}
			} else if respCtx.FinishReason() != nil && *respCtx.FinishReason() == common.CacheThresholdFinishReason {
				// No tokens to stream but we still need to emit a finish chunk for cache_threshold
				if chunk := respBuilder.createChunk(respCtx, nil, nil, "", respCtx.FinishReason()); chunk != nil {
					if err := c.sendChunk(w, chunk); err != nil {
						c.chunkSendFailed(ctx, "Sending finish chunk failed, ", err)
						return
					}
				}
				break
			} else {
				ctx.Error("unexpected response part in streaming", fasthttp.StatusInternalServerError)
				return
			}
		}

		// each builder decides whether to produce a last chunk based on the finish reason
		if chunk := respBuilder.createLastChunk(respCtx, *respCtx.FinishReason()); chunk != nil {
			if err := c.sendChunk(w, chunk); err != nil {
				c.chunkSendFailed(ctx, "Sending last stream chunk failed, ", err)
				return
			}
		}

		// send usage — builder returns nil when it has nothing to emit
		if chunk := respBuilder.createUsageChunk(respCtx); chunk != nil {
			if err := c.sendChunk(w, chunk); err != nil {
				c.chunkSendFailed(ctx, "Sending usage chunk failed, ", err)
				return
			}
		}

		// finish sse events stream — builder returns nil to omit [DONE]
		if chunk := respBuilder.createDoneChunk(); chunk != nil {
			if err := c.sendChunk(w, chunk); err != nil {
				c.chunkSendFailed(ctx, "Sending [DONE] chunk failed, ", err)
				return
			}
		}
	}()

	ctx.Response.SetBodyStream(pr, -1)
}

func (c *Communication) chunkSendFailed(ctx *fasthttp.RequestCtx, msg string, err error) {
	message := msg
	if err != nil {
		message += err.Error()
	}
	ctx.Error(message, fasthttp.StatusInternalServerError)
}

func (c *Communication) sendStreamedTools(respCtx vllmsim.ResponseContext, respBuilder responseBuilder,
	w *bufio.Writer, tokens []string, tc *openaiserverapi.ToolCall, index int) error {
	tokensStr := strings.Join(tokens, "")

	toolChunkInsert := &openaiserverapi.ToolCall{
		ID:    tc.ID,
		Type:  tc.Type,
		Index: tc.Index,
		Function: openaiserverapi.FunctionCall{
			Arguments: tokensStr,
		},
	}
	if index == 0 {
		toolChunkInsert.Function.Name = tc.Function.Name
	}

	var finishReasonToSend *string
	if index == tc.Function.TokenizedArguments().Length()-1 && (*respCtx.FinishReason() == common.LengthFinishReason ||
		*respCtx.FinishReason() == common.ToolsFinishReason ||
		*respCtx.FinishReason() == common.CacheThresholdFinishReason) {
		finishReasonToSend = respCtx.FinishReason()
	}
	return c.sendChunk(w, respBuilder.createChunk(respCtx, nil, toolChunkInsert, "", finishReasonToSend))
}

func (c *Communication) sendChunk(w *bufio.Writer, chunk sseChunk) error {
	b, err := chunk.SSEBytes()
	if err != nil {
		return err
	}
	if _, err = w.Write(b); err != nil {
		return err
	}
	return w.Flush()
}

func (c *Communication) sendError(ctx *fasthttp.RequestCtx, err *openaiserverapi.Error, isInjected bool) {
	if isInjected {
		c.logger.V(logging.TRACE).Info("Injecting failure", "type", err.Type, "message", err.Message)
	} else {
		c.logger.Error(nil, err.Message)
	}

	errorResp := openaiserverapi.ErrorResponse{
		Error: *err,
	}

	data, jsonErr := json.Marshal(errorResp)
	if jsonErr != nil {
		ctx.Error(jsonErr.Error(), fasthttp.StatusInternalServerError)
	} else {
		ctx.SetContentType("application/json")
		ctx.SetStatusCode(err.Code)
		ctx.SetBody(data)
	}
}

// readTokenizeRequest reads and parses data from the body of the given request
func (c *Communication) readTokenizeRequest(ctx *fasthttp.RequestCtx) (*vllmapi.TokenizeRequest, error) {
	var tokenizeReq vllmapi.TokenizeRequest
	if err := json.Unmarshal(ctx.Request.Body(), &tokenizeReq); err != nil {
		c.logger.Error(err, "failed to unmarshal tokenize request body")
		return nil, err
	}
	return &tokenizeReq, nil
}

// HandleEmbeddings http handler for /v1/embeddings (OpenAI-compatible).
// Supports input: string, []string, []number (token ids), [][]number; encoding_format: "float" or "base64".
func (c *Communication) HandleEmbeddings(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("Embeddings request received")
	var req openaiserverapi.EmbeddingRequest
	if err := json.Unmarshal(ctx.Request.Body(), &req); err != nil {
		c.logger.Error(err, "failed to unmarshal embeddings request body")
		errToSend := openaiserverapi.NewError("Failed to read and parse request body, "+err.Error(), fasthttp.StatusBadRequest, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}
	if req.Input.Len() == 0 {
		errToSend := openaiserverapi.NewError("input is required and must be a non-empty string or array", fasthttp.StatusBadRequest, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}
	model := req.Model
	if model == "" {
		model = c.simulator.Context.Config.Model
	}
	dim := c.simulator.Context.Config.DefaultEmbeddingDimensions
	if req.Dimensions != nil {
		if *req.Dimensions < 1 {
			errToSend := openaiserverapi.NewError("dimensions must be at least 1", fasthttp.StatusBadRequest, nil)
			c.sendError(ctx, &errToSend, false)
			return
		}
		dim = *req.Dimensions
	}
	useBase64 := req.EncodingFormat == "base64"

	var data []openaiserverapi.EmbeddingDataItem
	var totalTokens int

	if req.Input.IsTokenInput() {
		for i, tokIDs := range req.Input.TokenInputs() {
			tokens := make([]uint32, len(tokIDs))
			for j, id := range tokIDs {
				if id < 0 {
					id = 0
				}
				tokens[j] = uint32(id)
			}
			totalTokens += len(tokens)
			embedding := common.BuildStubEmbedding(tokens, dim)
			item := openaiserverapi.EmbeddingDataItem{Object: "embedding", Index: i}
			if useBase64 {
				item.Embedding = openaiserverapi.EncodeEmbeddingBase64(embedding)
			} else {
				item.Embedding = embedding
			}
			data = append(data, item)
		}
	} else {
		for i, text := range req.Input.TextInputs() {
			if text == "" {
				errToSend := openaiserverapi.NewError("input cannot be an empty string", fasthttp.StatusBadRequest, nil)
				c.sendError(ctx, &errToSend, false)
				return
			}
			tokens, _, err := c.simulator.Context.Tokenizer.RenderText(text)
			if err != nil {
				c.logger.Error(err, "failed to tokenize embedding input")
				ctx.Error("Failed to tokenize input, "+err.Error(), fasthttp.StatusInternalServerError)
				return
			}
			totalTokens += len(tokens)
			embedding := common.BuildStubEmbedding(tokens, dim)
			item := openaiserverapi.EmbeddingDataItem{Object: "embedding", Index: i}
			if useBase64 {
				item.Embedding = openaiserverapi.EncodeEmbeddingBase64(embedding)
			} else {
				item.Embedding = embedding
			}
			data = append(data, item)
		}
	}

	resp := openaiserverapi.EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage: openaiserverapi.EmbeddingResponseUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}
	out, err := json.Marshal(resp)
	if err != nil {
		c.logger.Error(err, "failed to marshal embeddings response")
		ctx.Error("Response body creation failed, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	c.addResponseHeaders(ctx, c.getRequestID(ctx))
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(out)
}

// HandleTokenize http handler for /tokenize
func (c *Communication) HandleTokenize(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("Tokenize request received")
	req, err := c.readTokenizeRequest(ctx)
	if err != nil {
		c.logger.Error(err, "failed to read and parse tokenize request body")
		ctx.Error("Failed to read and parse tokenize request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	// Check that the request has only one input to tokenize
	if req.Prompt != "" && req.Messages != nil {
		err := openaiserverapi.NewError("both prompt and messages fields in tokenize request",
			fasthttp.StatusBadRequest, nil)
		c.sendError(ctx, &err, false)
		return
	}

	var tokens []uint32
	if req.Prompt != "" {
		tokens, _, err = c.simulator.Context.Tokenizer.RenderText(req.Prompt)
	} else {
		// has messages
		tokens, _, _, err = c.simulator.Context.Tokenizer.RenderChatCompletion(req.Messages)
	}
	if err != nil {
		c.logger.Error(err, "failed to tokenize")
		ctx.Error("Failed to tokenize, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	resp := vllmapi.TokenizeResponse{
		Count:       len(tokens),
		Tokens:      tokens,
		MaxModelLen: c.simulator.Context.Config.MaxModelLen,
	}
	data, err := json.Marshal(resp)
	if err != nil {
		ctx.Error("Response body creation failed, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

func (c *Communication) HandleLoadLora(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.DEBUG).Info("Load lora request received")
	c.simulator.Context.LoadLoraAdaptor(ctx)
}

func (c *Communication) HandleUnloadLora(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.DEBUG).Info("Unload lora request received")
	c.simulator.Context.UnloadLoraAdaptor(ctx)
}

// HandleModels handles /v1/models request according the data stored in the simulator
func (c *Communication) HandleModels(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("/models request received")
	modelsResp := c.simulator.Context.CreateModelsResponse()

	data, err := json.Marshal(modelsResp)
	if err != nil {
		c.logger.Error(err, "failed to marshal models response")
		ctx.Error("Failed to marshal models response, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

func (c *Communication) HandleError(_ *fasthttp.RequestCtx, err error) {
	c.logger.Error(err, "vLLM server error")
}

// HandleHealth http handler for /health
func (c *Communication) HandleHealth(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("Health request received")
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
}

// HandleHealth http handler for /health/ready
func (c *Communication) HandleHealthReady(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("Health ready request received")
	ctx.Response.Header.SetContentType("application/json")
	if d := c.simulator.Context.Config.StartupDuration; d > 0 && time.Since(c.startTime) < d {
		ctx.Response.Header.SetStatusCode(fasthttp.StatusServiceUnavailable)
		return
	}
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
}

// HandleReady http handler for /ready
func (c *Communication) HandleReady(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("Readiness request received")
	if !c.deprecatedLogged {
		c.deprecatedLogged = true
		c.logger.V(logging.INFO).Info("/ready endpoint is deprecated and will be removed in a future release; please use /health/ready instead")
	}
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
}

// HandleIsSleeping handles /is_sleeping request according
func (c *Communication) HandleIsSleeping(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("/is_sleeping request received")

	c.sleepMutex.RLock()
	defer c.sleepMutex.RUnlock()
	data, err := json.Marshal(map[string]bool{"is_sleeping": c.simulator.IsSleeping})
	if err != nil {
		c.logger.Error(err, "failed to marshal isSleeping response")
		ctx.Error("Failed to marshal isSleeping response, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

// HandleSleep http handler for /sleep
func (c *Communication) HandleSleep(ctx *fasthttp.RequestCtx) {
	if c.simulator.Context.Config.EnableSleepMode && c.simulator.Context.Config.VllmDevMode {
		c.logger.V(logging.INFO).Info("Sleep request received")
		c.sleepMutex.Lock()
		defer c.sleepMutex.Unlock()

		c.simulator.IsSleeping = true
		if c.simulator.Context.Config.EnableKVCache {
			c.simulator.DiscardKVCache()
		}
	} else {
		c.logger.V(logging.INFO).Info("Sleep request received, skipped since simulator not in dev mode or sleep support is not enabled")
	}

	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
}

// HandleWakeUp http handler for /wake_up
func (c *Communication) HandleWakeUp(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.INFO).Info("Wake up request received")

	var wakeUpKVCache bool
	tags := ctx.QueryArgs().Peek("tags")
	if tags != nil {
		if string(tags) == "kv_cache" {
			wakeUpKVCache = true
		}
	} else {
		wakeUpKVCache = true
	}

	c.sleepMutex.Lock()
	defer c.sleepMutex.Unlock()

	// Activate the kv cache if either the tags are "kv_cache" or there are no tags
	if c.simulator.Context.Config.EnableKVCache && wakeUpKVCache {
		c.simulator.ActivateKVCache()
	}

	c.simulator.IsSleeping = false

	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
}

func formatRequestHeaders(h *fasthttp.RequestHeader) string {
	var b strings.Builder
	for key, value := range h.All() {
		b.Write(key)
		b.WriteString(": ")
		b.Write(value)
		b.WriteByte('\n')
	}
	return b.String()
}

func formatResponseHeaders(h *fasthttp.ResponseHeader) string {
	var b strings.Builder
	for key, value := range h.All() {
		b.Write(key)
		b.WriteString(": ")
		b.Write(value)
		b.WriteByte('\n')
	}
	return b.String()
}

func truncateBodyForLog(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	if len(body) <= maxHTTPLogBodyBytes {
		return string(body)
	}
	return string(body[:maxHTTPLogBodyBytes]) + fmt.Sprintf(" ... [truncated, total %d bytes]", len(body))
}

func (c *Communication) logHTTPMiddleware(next fasthttp.RequestHandler) fasthttp.RequestHandler {
	return func(ctx *fasthttp.RequestCtx) {
		c.logHTTPRequest(ctx)
		next(ctx)
		c.logHTTPResponse(ctx)
	}
}

func (c *Communication) logHTTPRequest(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.INFO).Info("HTTP request",
		"method", string(ctx.Method()),
		"requestURI", string(ctx.RequestURI()),
		"remoteAddr", ctx.RemoteAddr().String(),
		"headers", formatRequestHeaders(&ctx.Request.Header),
		"body", truncateBodyForLog(ctx.Request.Body()),
	)
}

func (c *Communication) logHTTPResponse(ctx *fasthttp.RequestCtx) {
	resp := &ctx.Response
	if resp.BodyStream() != nil {
		c.logger.V(logging.INFO).Info("HTTP response",
			"statusCode", resp.StatusCode(),
			"headers", formatResponseHeaders(&resp.Header),
			"body", "<streamed response body not logged>",
		)
		return
	}
	c.logger.V(logging.INFO).Info("HTTP response",
		"statusCode", resp.StatusCode(),
		"headers", formatResponseHeaders(&resp.Header),
		"body", truncateBodyForLog(resp.Body()),
	)
}

// HandleFakeMetrics HTTP handler for /fake_metrics
func (c *Communication) HandleFakeMetrics(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.INFO).Info("Fake metrics update received")

	if c.simulator.Context.Config.FakeMetrics == nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		ctx.SetBodyString("The simulator is reporting real metrics; fake metrics cannot be updated.")
		return
	}

	oldFakeMetrics, fmMap, err := c.simulator.Context.Config.FakeMetrics.UnmarshalUpdateJSON(ctx.Request.Body())
	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		ctx.SetBodyString("Failed to unmarshal the payload: " + err.Error())
		return
	}

	if err := c.simulator.Context.UpdateFakeMetrics(fmMap, oldFakeMetrics); err != nil {
		ctx.SetStatusCode(fasthttp.StatusInternalServerError)
		ctx.SetBodyString("Failed to update fake metrics: " + err.Error())
		return
	}

	ctx.Response.Header.SetStatusCode(fasthttp.StatusNoContent)
}
