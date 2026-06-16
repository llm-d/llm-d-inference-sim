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
	listener, err := net.Listen("tcp4", fmt.Sprintf(":%d", c.simulator.Context.Config().Port))
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
	r.POST("/v1/chat/completions/render", c.HandleChatCompletionsRender)
	r.POST("/v1/completions/render", c.HandleTextCompletionsRender)
	r.POST("/v1/responses", c.HandleResponses)
	r.POST("/v1/messages", c.HandleMessages)
	r.POST("/inference/v1/generate", c.HandleGenerate)
	if !c.simulator.Context.Config().MMEncoderOnly {
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
	// emulates vLLM's Mooncake bootstrap endpoint on the prefill pod; the routing sidecar queries it to resolve remote engine ids
	r.GET("/query", c.HandleMooncakeQuery)
	r.GET("/ready", c.HandleReady)
	r.POST("/tokenize", c.HandleTokenize)
	r.POST("/sleep", c.HandleSleep)
	r.POST("/wake_up", c.HandleWakeUp)
	r.GET("/is_sleeping", c.HandleIsSleeping)
	r.GET("/admin/config", c.HandleGetAdminConfig)
	r.POST("/admin/config", c.HandlePostAdminConfig)

	handler := r.Handler
	if c.simulator.Context.Config().LogHTTP {
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
		if c.simulator.Context.Config().SSLEnabled() {
			c.logger.V(logging.INFO).Info("Server starting", "protocol", "HTTPS", "port", c.simulator.Context.Config().Port)
			errCh <- server.ServeTLS(listener, "", "")
		} else {
			c.logger.V(logging.INFO).Info("Server starting", "protocol", "HTTP", "port", c.simulator.Context.Config().Port)
			errCh <- server.Serve(listener)
		}
	}()

	return server, errCh, nil
}

// getRequestID retrieves the request ID from the X-Request-Id header or generates a new one if not present
func (c *Communication) getRequestID(ctx *fasthttp.RequestCtx) string {
	if c.simulator.Context.Config().EnableRequestIDHeaders {
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
	c.handleHTTP(&vllmsim.TextCompletionsParsedRequest{}, &textComplHTTPRespBuilder{}, ctx)
}

// HandleResponses http handler for /v1/responses
func (c *Communication) HandleResponses(ctx *fasthttp.RequestCtx) {
	c.handleHTTP(&vllmsim.ResponsesRequest{}, &responsesHTTPRespBuilder{}, ctx)
}

// HandleMessages http handler for /v1/messages (Anthropic Messages API)
func (c *Communication) HandleMessages(ctx *fasthttp.RequestCtx) {
	c.handleHTTP(&vllmsim.MessagesRequest{}, &messagesHTTPRespBuilder{}, ctx)
}

// HandleGenerate http handler for /inference/v1/generate
func (c *Communication) HandleGenerate(ctx *fasthttp.RequestCtx) {
	c.handleHTTP(&vllmsim.GenerateRequest{}, &generateHTTPRespBuilder{}, ctx)
}

// HandleChatCompletionsRender http handler for /v1/chat/completions/render
func (c *Communication) HandleChatCompletionsRender(ctx *fasthttp.RequestCtx) {
	c.handleRender(&vllmsim.ChatCompletionsRequest{}, &chatComplHTTPRespBuilder{}, ctx)
}

// HandleTextCompletionsRender http handler for /v1/completions/render
func (c *Communication) HandleTextCompletionsRender(ctx *fasthttp.RequestCtx) {
	c.handleRender(&vllmsim.TextCompletionsParsedRequest{}, &textComplHTTPRespBuilder{}, ctx)
}

func (c *Communication) handleRender(req vllmsim.RenderableRequest, respBuilder responseBuilder, ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("Render request received", "endpoint", string(ctx.Path()))
	if err := req.Unmarshal(ctx.Request.Body()); err != nil {
		c.logger.Error(err, "failed to read and parse render request body")
		errToSend := openaiserverapi.NewError("Failed to read and parse request body, "+err.Error(), fasthttp.StatusBadRequest, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}
	if err := req.ValidateBody(); err != nil {
		c.sendError(ctx, err, false)
		return
	}
	if err := c.simulator.ValidateBaseModel(req.GetModel()); err != nil {
		c.sendError(ctx, err, false)
		return
	}
	tokens, features, err := req.Render(c.simulator.Context.Tokenizer)
	if err != nil {
		c.logger.Error(err, "render failed")
		errToSend := openaiserverapi.NewError("Render failed, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}
	respBody, err := json.Marshal(respBuilder.createRenderResponse(tokens, features))
	if err != nil {
		c.logger.Error(err, "render response marshal failed")
		errToSend := openaiserverapi.NewError("Render failed, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}
	c.addResponseHeaders(ctx, c.getRequestID(ctx))
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(respBody)
}

// addResponseHeaders adds optional pod/port/namespace/request-id headers to the response for testing/debugging.
func (c *Communication) addResponseHeaders(ctx *fasthttp.RequestCtx, requestID string) {
	if c.simulator.Context.Config().PodName != "" {
		ctx.Response.Header.Add(PodHeader, c.simulator.Context.Config().PodName)
		ctx.Response.Header.Add(PortHeader, strconv.Itoa(c.simulator.Context.Config().Port))
	}
	if c.simulator.Context.Config().PodNameSpace != "" {
		ctx.Response.Header.Add(NamespaceHeader, c.simulator.Context.Config().PodNameSpace)
	}
	if c.simulator.Context.Config().EnableRequestIDHeaders {
		ctx.Response.Header.Add(RequestIDHeader, requestID)
	}
}

func (c *Communication) handleHTTP(req vllmsim.Request, respBuilder responseBuilder, ctx *fasthttp.RequestCtx) {
	if c.stopping.Load() {
		errToSend := openaiserverapi.NewError("server is shutting down", fasthttp.StatusServiceUnavailable, nil)
		c.sendError(ctx, &errToSend, false)
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

	numChoices, isStream, channel, err, errInjected := c.simulator.HandleRequest(req)
	if err != nil {
		c.sendError(ctx, err, errInjected)
		return
	}

	c.logger.V(logging.DEBUG).Info("Received", "new HTTP", req.AsString())

	ctx.SetStatusCode(fasthttp.StatusOK)

	c.addResponseHeaders(ctx, req.GetRequestID())

	if isStream {
		ctx.SetContentType("text/event-stream")
		c.sendStream(ctx, *channel, respBuilder, numChoices)
	} else {
		ctx.SetContentType("application/json")
		c.sendNonStream(ctx, *channel, respBuilder, numChoices)
	}
}

func (c *Communication) sendNonStream(ctx *fasthttp.RequestCtx, channel common.Channel[*vllmsim.ResponseInfo],
	respBuilder responseBuilder, numChoices int) {
	tokens := make([]openaiserverapi.Tokenized, numChoices)
	for i := range tokens {
		tokens[i] = openaiserverapi.Tokenized{
			Tokens:  make([]uint32, 0),
			Strings: make([]string, 0),
		}
	}
	respCtxPerChoice := make([]vllmsim.ResponseContext, numChoices)
	for response := range channel.Channel {
		if response.Err != nil {
			// Fail-fast: abort the whole request. Drain remaining responses in the
			// background so producers don't fill the buffer and drop messages.
			go drainResponseChannel(channel)
			c.sendError(ctx, response.Err, false)
			return
		}
		if response.Tokens != nil {
			tokens[response.ChoiceIdx].Append(*response.Tokens)
		}
		if respCtxPerChoice[response.ChoiceIdx] == nil {
			respCtxPerChoice[response.ChoiceIdx] = response.RespCtx
		}
	}

	resp := respBuilder.createResponse(respCtxPerChoice, tokens)
	data, err := json.Marshal(resp)
	if err != nil {
		err := openaiserverapi.NewError("Response body creation failed, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &err, false)
		return
	}
	ctx.Response.SetBody(data)
}

// drainResponseChannel reads and discards responses until the channel closes.
// Used after a fail-fast abort so in-flight producers can finish cleanly.
func drainResponseChannel(channel common.Channel[*vllmsim.ResponseInfo]) {
	for range channel.Channel { //nolint:revive
	}
}

// sendStreamErrorAndDone writes a single error SSE frame followed by the [DONE]
// marker to the streaming writer. Errors from the write are ignored — the client
// pipe may already be gone.
func (c *Communication) sendStreamErrorAndDone(w *bufio.Writer, err *openaiserverapi.Error) {
	errResp := openaiserverapi.ErrorResponse{Error: *err}
	_ = c.sendChunk(w, &jsonDataChunk{data: errResp})
	_ = c.sendChunk(w, &doneMarker{})
}

// streamState holds per-choice streaming state, sized once at the start of
// the stream from the known number of choices, plus stream-global flags
// tracking whether the first response and the initial chunk have been seen.
type streamState struct {
	first            bool
	initialSent      bool
	firstTokens      []bool
	lastToolCall     []*openaiserverapi.ToolCall
	toolCallIndex    []int
	respCtxPerChoice []vllmsim.ResponseContext
}

// newStreamState allocates per-choice slices for numChoices choices. The
// firstTokens slice starts true for every choice (no token has been emitted
// yet); the rest start at their zero values.
func newStreamState(numChoices int) streamState {
	firstTokens := make([]bool, numChoices)
	for i := range firstTokens {
		firstTokens[i] = true
	}
	return streamState{
		first:            true,
		firstTokens:      firstTokens,
		lastToolCall:     make([]*openaiserverapi.ToolCall, numChoices),
		toolCallIndex:    make([]int, numChoices),
		respCtxPerChoice: make([]vllmsim.ResponseContext, numChoices),
	}
}

// sendOrFail writes chunk to w, reporting a chunk-send failure on err. A nil
// chunk is a no-op. Returns true on success; the caller should return when it
// sees false (the failure has already been reported on ctx).
func (c *Communication) sendOrFail(ctx *fasthttp.RequestCtx, w *bufio.Writer, chunk sseChunk, failMsg string) bool {
	if chunk == nil {
		return true
	}
	if err := c.sendChunk(w, chunk); err != nil {
		c.chunkSendFailed(ctx, failMsg, err)
		return false
	}
	return true
}

func (c *Communication) sendStream(ctx *fasthttp.RequestCtx, channel common.Channel[*vllmsim.ResponseInfo],
	respBuilder responseBuilder, numChoices int) {
	pr, pw := io.Pipe()

	go func() {
		w := bufio.NewWriter(pw)
		var respCtx vllmsim.ResponseContext
		state := newStreamState(numChoices)

		defer func() {
			w.Flush()  //nolint:errcheck
			pw.Close() //nolint:errcheck
		}()

		for response := range channel.Channel {
			if response.Err != nil {
				// Fail-fast: previously streamed chunks remain sent; emit a single error
				// frame followed by [DONE] and stop reading from the other prompts.
				c.sendStreamErrorAndDone(w, response.Err)
				go drainResponseChannel(channel)
				return
			}
			// Set respCtx once from the first response seen across all choices.
			if state.first {
				respCtx = response.RespCtx
				respCtx.SetCreationTime(time.Now().Unix())
				state.first = false
			}

			choiceIdx := response.ChoiceIdx
			// Capture per-choice respCtx so the final usage chunk can aggregate
			// across all sub-requests and send separate finish reason for each choices.
			if state.respCtxPerChoice[choiceIdx] == nil {
				state.respCtxPerChoice[choiceIdx] = response.RespCtx
			}
			// Every choice emits a Created status as its first message. Emit the initial
			// chunk once globally and skip the response — Created has no tokens.
			if response.Status == vllmsim.ResponseStatusCreated {
				if !state.initialSent {
					if !c.sendOrFail(ctx, w, respBuilder.createInitialChunk(respCtx), "Sending first stream chunk failed, ") {
						return
					}
					state.initialSent = true
				}
				continue
			}

			ok, stop := c.emitResponseChunks(ctx, w, respBuilder, response, respCtx, &state, response.Status == vllmsim.ResponseEndOfTokens)
			if !ok {
				go drainResponseChannel(channel)
				return
			}
			if stop {
				break
			}
		}

		c.finalizeStream(ctx, w, respBuilder, &state)
	}()

	ctx.Response.SetBodyStream(pr, -1)
}

// emitResponseChunks handles a single non-error, non-Created response. Returns
// (ok, stop): ok=false means the caller should return (a send failed and was
// already reported via ctx); stop=true means the stream is complete and the main
// loop should break out to finalize.
func (c *Communication) emitResponseChunks(ctx *fasthttp.RequestCtx, w *bufio.Writer, respBuilder responseBuilder,
	response *vllmsim.ResponseInfo, respCtx vllmsim.ResponseContext, state *streamState, lastTokensChunk bool) (ok bool, stop bool) {
	choiceIdx := response.ChoiceIdx

	if response.Tokens != nil {
		// In chat completion the first chunk contains the role.
		if state.firstTokens[choiceIdx] {
			if !c.sendOrFail(ctx, w, respBuilder.createFirstChunk(respCtx, choiceIdx), "Sending first stream chunk failed, ") {
				return false, false
			}
			state.firstTokens[choiceIdx] = false
		}
		if response.ToolCall != nil {
			if state.lastToolCall[choiceIdx] != response.ToolCall {
				state.toolCallIndex[choiceIdx] = 0
			} else {
				state.toolCallIndex[choiceIdx]++
			}
			if err := c.sendStreamedTools(respCtx, respBuilder, w, response.Tokens.Strings, response.ToolCall,
				state.toolCallIndex[choiceIdx], choiceIdx); err != nil {
				c.chunkSendFailed(ctx, "Sending tools chunk failed, ", err)
				return false, false
			}
			state.lastToolCall[choiceIdx] = response.ToolCall
			return true, false
		}

		var finishReason *string
		if lastTokensChunk && respBuilder.sendFinishReasonWithTokens() {
			finishReason = respCtx.FinishReason()
		}
		return c.sendOrFail(ctx, w, respBuilder.createChunk(respCtx, response.Tokens, nil, "", finishReason, choiceIdx),
			"Sending stream chunk failed, "), false
	}

	if respCtx.FinishReason() != nil && *respCtx.FinishReason() == common.CacheThresholdFinishReason {
		// No tokens to stream but we still need to emit a finish chunk for cache_threshold.
		return c.sendOrFail(ctx, w, respBuilder.createChunk(respCtx, nil, nil, "", respCtx.FinishReason(), choiceIdx),
			"Sending finish chunk failed, "), true
	}

	errToSend := openaiserverapi.NewError("unexpected response part in streaming", fasthttp.StatusInternalServerError, nil)
	c.sendError(ctx, &errToSend, false)
	return false, false
}

// finalizeStream emits the post-loop SSE frames: a last chunk per choice (if the
// builder wants one for the finish reason), the usage chunk, and [DONE].
func (c *Communication) finalizeStream(ctx *fasthttp.RequestCtx, w *bufio.Writer, respBuilder responseBuilder,
	state *streamState) {
	for i, rc := range state.respCtxPerChoice {
		if !c.sendOrFail(ctx, w, respBuilder.createLastChunk(rc, *rc.FinishReason(), i),
			"Sending last stream chunk failed, ") {
			return
		}
	}
	if !c.sendOrFail(ctx, w, respBuilder.createUsageChunk(state.respCtxPerChoice), "Sending usage chunk failed, ") {
		return
	}
	c.sendOrFail(ctx, w, respBuilder.createDoneChunk(), "Sending [DONE] chunk failed, ")
}

func (c *Communication) chunkSendFailed(ctx *fasthttp.RequestCtx, msg string, err error) {
	message := msg
	if err != nil {
		message += err.Error()
	}
	errToSend := openaiserverapi.NewError(message, fasthttp.StatusInternalServerError, nil)
	c.sendError(ctx, &errToSend, false)
}

func (c *Communication) sendStreamedTools(respCtx vllmsim.ResponseContext, respBuilder responseBuilder,
	w *bufio.Writer, tokens []string, tc *openaiserverapi.ToolCall, index int, choiceIdx int) error {
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
	return c.sendChunk(w, respBuilder.createChunk(respCtx, nil, toolChunkInsert, "", finishReasonToSend, choiceIdx))
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
		ctx.SetStatusCode(fasthttp.StatusInternalServerError)
		ctx.SetContentType("application/json")
		ctx.SetBodyString(`{"error":{"message":"internal error","type":"server_error","param":null,"code":"500"}}`)
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
		model = c.simulator.Context.Config().Model
	}
	dim := c.simulator.Context.Config().DefaultEmbeddingDimensions
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
				errToSend := openaiserverapi.NewError("Failed to tokenize input, "+err.Error(), fasthttp.StatusInternalServerError, nil)
				c.sendError(ctx, &errToSend, false)
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
		errToSend := openaiserverapi.NewError("Response body creation failed, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
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
		errToSend := openaiserverapi.NewError("Failed to read and parse tokenize request body, "+err.Error(), fasthttp.StatusBadRequest, nil)
		c.sendError(ctx, &errToSend, false)
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
		tokens, _, _, err = c.simulator.Context.Tokenizer.RenderMessages(req.Messages)
	}

	if err != nil {
		c.logger.Error(err, "failed to tokenize")
		errToSend := openaiserverapi.NewError("Failed to tokenize, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}

	resp := vllmapi.TokenizeResponse{
		Count:       len(tokens),
		Tokens:      tokens,
		MaxModelLen: c.simulator.Context.Config().MaxModelLen,
	}
	data, err := json.Marshal(resp)
	if err != nil {
		errToSend := openaiserverapi.NewError("Response body creation failed, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

func (c *Communication) HandleLoadLora(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.DEBUG).Info("Load lora request received")
	if err := c.simulator.Context.LoadLoraAdaptor(ctx.Request.Body()); err != nil {
		errToSend := openaiserverapi.NewError(err.Error(), fasthttp.StatusBadRequest, nil)
		c.sendError(ctx, &errToSend, false)
	}
}

func (c *Communication) HandleUnloadLora(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.DEBUG).Info("Unload lora request received")
	if err := c.simulator.Context.UnloadLoraAdaptor(ctx.Request.Body()); err != nil {
		errToSend := openaiserverapi.NewError(err.Error(), fasthttp.StatusBadRequest, nil)
		c.sendError(ctx, &errToSend, false)
	}
}

// HandleModels handles /v1/models request according the data stored in the simulator
func (c *Communication) HandleModels(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("/models request received")
	modelsResp := c.simulator.Context.CreateModelsResponse()

	data, err := json.Marshal(modelsResp)
	if err != nil {
		c.logger.Error(err, "failed to marshal models response")
		errToSend := openaiserverapi.NewError("Failed to marshal models response, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
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
	if d := c.simulator.Context.Config().StartupDuration; d > 0 && time.Since(c.startTime) < d {
		ctx.Response.Header.SetStatusCode(fasthttp.StatusServiceUnavailable)
		return
	}
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
}

// HandleReady http handler for /ready
func (c *Communication) HandleReady(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("Readiness request received")
	if !c.readyDeprecatedLogged {
		c.readyDeprecatedLogged = true
		c.logger.V(logging.INFO).Info("/ready endpoint is deprecated and will be removed in a future release; please use /health/ready instead")
	}
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
}

// mooncakeEngineMap returns the dp_rank -> {engine_id} map served by /query, generating
// it once so the engine ids stay stable for the simulator's lifetime.
func (c *Communication) mooncakeEngineMap() map[string]map[string]string {
	c.mooncakeEnginesOnce.Do(func() {
		dpSize := c.simulator.Context.Config().DPSize
		engines := make(map[string]map[string]string, dpSize)
		for rank := 0; rank < dpSize; rank++ {
			engines[strconv.Itoa(rank)] = map[string]string{
				"engine_id": c.simulator.Context.Random.GenerateUUIDString(),
			}
		}
		c.mooncakeEngines = engines
	})
	return c.mooncakeEngines
}

// HandleMooncakeQuery emulates vLLM's Mooncake bootstrap endpoint (/query) served on
// the prefill pod. The routing sidecar calls it to resolve the remote engine id used
// for KV transfer, receiving a dp_rank -> {engine_id} map. The engine ids are
// placeholders generated once per simulator lifetime; a real vLLM prefill pod reports
// the ids of its running KV engines.
func (c *Communication) HandleMooncakeQuery(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("/query request received")

	data, err := json.Marshal(c.mooncakeEngineMap())
	if err != nil {
		c.logger.Error(err, "failed to marshal mooncake query response")
		errToSend := openaiserverapi.NewError("Failed to marshal mooncake query response, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}

	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

// HandleIsSleeping handles /is_sleeping request according
func (c *Communication) HandleIsSleeping(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("/is_sleeping request received")

	c.sleepMutex.RLock()
	defer c.sleepMutex.RUnlock()
	data, err := json.Marshal(map[string]bool{"is_sleeping": c.simulator.IsSleeping})
	if err != nil {
		c.logger.Error(err, "failed to marshal isSleeping response")
		errToSend := openaiserverapi.NewError("Failed to marshal isSleeping response, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}

	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

// HandleSleep http handler for /sleep
func (c *Communication) HandleSleep(ctx *fasthttp.RequestCtx) {
	if c.simulator.Context.Config().EnableSleepMode && c.simulator.Context.Config().VllmDevMode {
		c.logger.V(logging.INFO).Info("Sleep request received")
		c.sleepMutex.Lock()
		defer c.sleepMutex.Unlock()

		c.simulator.IsSleeping = true
		if c.simulator.Context.Config().EnableKVCache {
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
	if c.simulator.Context.Config().EnableKVCache && wakeUpKVCache {
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

// HandleGetAdminConfig http handler for GET /admin/config — returns the full configuration.
func (c *Communication) HandleGetAdminConfig(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.TRACE).Info("Get admin config request received")
	c.writeAdminConfigResponse(ctx)
}

// HandlePostAdminConfig http handler for POST /admin/config — updates the
// admin-configurable subset of fields. A "fake-metrics" key in the body is
// dispatched (by the simulator context) to ApplyFakeMetricsBody, which has
// the same partial-update semantics as the (deprecated) /fake_metrics
// endpoint.
func (c *Communication) HandlePostAdminConfig(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.INFO).Info("Update admin config request received")

	if err := c.simulator.Context.ApplyConfigUpdate(ctx.Request.Body()); err != nil {
		errToSend := openaiserverapi.NewError(err.Error(), fasthttp.StatusBadRequest, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}
	c.writeAdminConfigResponse(ctx)
}

func (c *Communication) writeAdminConfigResponse(ctx *fasthttp.RequestCtx) {
	data, err := c.simulator.Context.Config().MarshalCleaned()
	if err != nil {
		c.logger.Error(err, "failed to marshal admin config response")
		errToSend := openaiserverapi.NewError("Failed to marshal admin config response, "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

// HandleFakeMetrics HTTP handler for /fake_metrics.
//
// Deprecated: superseded by POST /admin/config with a "fake-metrics" field;
// scheduled for removal in release v0.12.0.
func (c *Communication) HandleFakeMetrics(ctx *fasthttp.RequestCtx) {
	c.logger.V(logging.INFO).Info("Fake metrics update received")

	if !c.fakeMetricsDeprecatedLogged {
		c.fakeMetricsDeprecatedLogged = true
		c.logger.V(logging.INFO).Info("/fake_metrics endpoint is deprecated and will be removed in release v0.12.0; please use POST /admin/config with a 'fake-metrics' field instead")
	}

	if err := c.simulator.Context.UpdateFakeMetricsFromBody(ctx.Request.Body()); err != nil {
		errToSend := openaiserverapi.NewError("Failed to update fake metrics: "+err.Error(), fasthttp.StatusInternalServerError, nil)
		c.sendError(ctx, &errToSend, false)
		return
	}

	ctx.Response.Header.SetStatusCode(fasthttp.StatusNoContent)
}
