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

// Package vllmsim implements the vLLM simulator.
package llmdinferencesim

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strconv"

	"github.com/buaazp/fasthttprouter"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"

	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

func (s *VllmSimulator) newListener() (net.Listener, error) {
	listener, err := net.Listen("tcp4", fmt.Sprintf(":%d", s.context.config.Port))
	if err != nil {
		return nil, err
	}
	return listener, nil
}

// startServer starts http/https server on port defined in command line
func (s *VllmSimulator) startServer(ctx context.Context, listener net.Listener) error {
	r := fasthttprouter.New()

	// support completion APIs
	r.POST("/v1/chat/completions", s.HandleChatCompletions)
	r.POST("/v1/completions", s.HandleTextCompletions)
	// supports /models API
	r.GET("/v1/models", s.HandleModels)
	// support load/unload of lora adapter
	r.POST("/v1/load_lora_adapter", s.HandleLoadLora)
	r.POST("/v1/unload_lora_adapter", s.HandleUnloadLora)
	// supports /metrics prometheus API
	r.GET("/metrics", fasthttpadaptor.NewFastHTTPHandler(promhttp.HandlerFor(s.context.metrics.registry, promhttp.HandlerOpts{})))
	// supports standard Kubernetes health and readiness checks
	r.GET("/health", s.HandleHealth)
	r.GET("/ready", s.HandleReady)
	r.POST("/tokenize", s.HandleTokenize)
	r.POST("/sleep", s.HandleSleep)
	r.POST("/wake_up", s.HandleWakeUp)
	r.GET("/is_sleeping", s.HandleIsSleeping)

	server := &fasthttp.Server{
		ErrorHandler: s.HandleError,
		Handler:      r.Handler,
		Logger:       s,
	}

	if err := s.configureSSL(server); err != nil {
		return err
	}

	// Start server in a goroutine
	serverErr := make(chan error, 1)
	go func() {
		if s.context.config.SSLEnabled() {
			s.context.logger.V(logging.INFO).Info("Server starting", "protocol", "HTTPS", "port", s.context.config.Port)
			serverErr <- server.ServeTLS(listener, "", "")
		} else {
			s.context.logger.V(logging.INFO).Info("Server starting", "protocol", "HTTP", "port", s.context.config.Port)
			serverErr <- server.Serve(listener)
		}
	}()

	// Wait for either context cancellation or server error
	select {
	case <-ctx.Done():
		s.context.logger.V(logging.INFO).Info("Shutdown signal received, shutting down server gracefully")

		// Gracefully shutdown the server
		if err := server.Shutdown(); err != nil {
			s.context.logger.Error(err, "error during server shutdown")
			return err
		}

		s.context.logger.V(logging.INFO).Info("Server stopped")
		return nil

	case err := <-serverErr:
		if err != nil {
			s.context.logger.Error(err, "server failed")
		}
		return err
	}
}

// getRequestID retrieves the request ID from the X-Request-Id header or generates a new one if not present
func (s *VllmSimulator) getRequestID(ctx *fasthttp.RequestCtx) string {
	if s.context.config.EnableRequestIDHeaders {
		requestID := string(ctx.Request.Header.Peek(requestIDHeader))
		if requestID != "" {
			return requestID
		}
	}
	return s.context.random.GenerateUUIDString()
}

// HandleChatCompletions http handler for /v1/chat/completions
func (s *VllmSimulator) HandleChatCompletions(ctx *fasthttp.RequestCtx) {
	s.handleRequest(&chatCompletionRequest{}, ctx)
}

// HandleTextCompletions http handler for /v1/completions
func (s *VllmSimulator) HandleTextCompletions(ctx *fasthttp.RequestCtx) {
	s.handleRequest(&textCompletionRequest{}, ctx)
}

// readTokenizeRequest reads and parses data from the body of the given request
func (s *VllmSimulator) readTokenizeRequest(ctx *fasthttp.RequestCtx) (*vllmapi.TokenizeRequest, error) {
	var tokenizeReq vllmapi.TokenizeRequest
	if err := json.Unmarshal(ctx.Request.Body(), &tokenizeReq); err != nil {
		s.context.logger.Error(err, "failed to unmarshal tokenize request body")
		return nil, err
	}
	return &tokenizeReq, nil
}

// HandleTokenize http handler for /tokenize
func (s *VllmSimulator) HandleTokenize(ctx *fasthttp.RequestCtx) {
	s.context.logger.V(logging.TRACE).Info("Tokenize request received")
	req, err := s.readTokenizeRequest(ctx)
	if err != nil {
		s.context.logger.Error(err, "failed to read and parse tokenize request body")
		ctx.Error("Failed to read and parse tokenize request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	// Check that the request has only one input to tokenize
	if req.Prompt != "" && req.Messages != nil {
		s.sendError(ctx, openaiserverapi.NewError("both prompt and messages fields in tokenize request",
			fasthttp.StatusBadRequest, nil), false)
		return
	}
	// Model is optional, if not set, the model from the configuration will be used
	tokens, err := s.context.tokenizer.Encode(req.GetPrompt(), req.Model)
	if err != nil {
		s.context.logger.Error(err, "failed to tokenize")
		ctx.Error("Failed to tokenize, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}
	resp := vllmapi.TokenizeResponse{
		Count:       len(tokens),
		Tokens:      tokens,
		MaxModelLen: s.context.config.MaxModelLen,
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

func (s *VllmSimulator) HandleLoadLora(ctx *fasthttp.RequestCtx) {
	s.context.logger.V(logging.DEBUG).Info("Load lora request received")
	s.context.loadLoraAdaptor(ctx)
}

func (s *VllmSimulator) HandleUnloadLora(ctx *fasthttp.RequestCtx) {
	s.context.logger.V(logging.DEBUG).Info("Unload lora request received")
	s.context.unloadLoraAdaptor(ctx)
}

// sendCompletionResponse sends a completion response
func (s *VllmSimulator) sendCompletionResponse(ctx *fasthttp.RequestCtx, resp openaiserverapi.CompletionResponse) {
	data, err := json.Marshal(resp)
	if err != nil {
		ctx.Error("Response body creation failed, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	// Add pod and namespace information to response headers for testing/debugging
	if s.pod != "" {
		ctx.Response.Header.Add(podHeader, s.pod)
		ctx.Response.Header.Add(portHeader, strconv.Itoa(s.context.config.Port))
	}
	if s.namespace != "" {
		ctx.Response.Header.Add(namespaceHeader, s.namespace)
	}
	if s.context.config.EnableRequestIDHeaders {
		if requestID := resp.GetRequestID(); requestID != "" {
			ctx.Response.Header.Add(requestIDHeader, requestID)
		}
	}
	ctx.Response.SetBody(data)
}

// sendError sends an error response for the current request
// isInjected indicates if this is an injected failure for logging purposes
func (s *VllmSimulator) sendError(ctx *fasthttp.RequestCtx, compErr openaiserverapi.Error, isInjected bool) {
	if isInjected {
		s.context.logger.V(logging.TRACE).Info("Injecting failure", "type", compErr.Type, "message", compErr.Message)
	} else {
		s.context.logger.Error(nil, compErr.Message)
	}

	errorResp := openaiserverapi.ErrorResponse{
		Error: compErr,
	}

	data, err := json.Marshal(errorResp)
	if err != nil {
		ctx.Error(err.Error(), fasthttp.StatusInternalServerError)
	} else {
		ctx.SetContentType("application/json")
		ctx.SetStatusCode(compErr.Code)
		ctx.SetBody(data)
	}
}

func (s *VllmSimulator) setBadRequestError(ctx *fasthttp.RequestCtx, message string) {
	ctx.Error(message, fasthttp.StatusBadRequest)
}

// HandleModels handles /v1/models request according the data stored in the simulator
func (s *VllmSimulator) HandleModels(ctx *fasthttp.RequestCtx) {
	s.context.logger.V(logging.TRACE).Info("/models request received")
	modelsResp := s.createModelsResponse()

	data, err := json.Marshal(modelsResp)
	if err != nil {
		s.context.logger.Error(err, "failed to marshal models response")
		ctx.Error("Failed to marshal models response, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

func (s *VllmSimulator) HandleError(_ *fasthttp.RequestCtx, err error) {
	s.context.logger.Error(err, "vLLM server error")
}

// HandleHealth http handler for /health
func (s *VllmSimulator) HandleHealth(ctx *fasthttp.RequestCtx) {
	s.context.logger.V(logging.TRACE).Info("Health request received")
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody([]byte("{}"))
}

// HandleReady http handler for /ready
func (s *VllmSimulator) HandleReady(ctx *fasthttp.RequestCtx) {
	s.context.logger.V(logging.TRACE).Info("Readiness request received")
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody([]byte("{}"))
}

// HandleIsSleeping handles /is_sleeping request according
func (s *VllmSimulator) HandleIsSleeping(ctx *fasthttp.RequestCtx) {
	s.context.logger.V(logging.TRACE).Info("/is_sleeping request received")

	s.sleepMutex.RLock()
	defer s.sleepMutex.RUnlock()
	data, err := json.Marshal(map[string]bool{"is_sleeping": s.isSleeping})
	if err != nil {
		s.context.logger.Error(err, "failed to marshal isSleeping response")
		ctx.Error("Failed to marshal isSleeping response, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

// HandleSleep http handler for /sleep
func (s *VllmSimulator) HandleSleep(ctx *fasthttp.RequestCtx) {
	if s.context.config.EnableSleepMode && s.isInDevMode {
		s.context.logger.V(logging.INFO).Info("Sleep request received")
		s.sleepMutex.Lock()
		defer s.sleepMutex.Unlock()

		s.isSleeping = true
		if s.context.config.EnableKVCache {
			s.context.kvcacheHelper.Discard()
		}
	} else {
		s.context.logger.V(logging.INFO).Info("Sleep request received, skipped since simulator not in dev mode or sleep support is not enabled")
	}

	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
}

// HandleWakeUp http handler for /wake_up
func (s *VllmSimulator) HandleWakeUp(ctx *fasthttp.RequestCtx) {
	s.context.logger.V(logging.INFO).Info("Wake up request received")

	var wakeUpKVCache bool
	tags := ctx.QueryArgs().Peek("tags")
	if tags != nil {
		if string(tags) == "kv_cache" {
			wakeUpKVCache = true
		}
	} else {
		wakeUpKVCache = true
	}

	s.sleepMutex.Lock()
	defer s.sleepMutex.Unlock()

	// Activate the kv cache if either the tags are "kv_cache" or there are no tags
	if s.context.config.EnableKVCache && wakeUpKVCache {
		s.context.kvcacheHelper.Activate()
	}

	s.isSleeping = false

	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
}
