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

	"github.com/buaazp/fasthttprouter"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

func (s *VllmSimulator) newListener() (net.Listener, error) {
	listener, err := net.Listen("tcp4", fmt.Sprintf(":%d", s.config.Port))
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
	r.GET("/metrics", fasthttpadaptor.NewFastHTTPHandler(promhttp.HandlerFor(s.registry, promhttp.HandlerOpts{})))
	// supports standard Kubernetes health and readiness checks
	r.GET("/health", s.HandleHealth)
	r.GET("/ready", s.HandleReady)
	r.POST("/tokenize", s.HandleTokenize)

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
		if s.config.SSLEnabled() {
			s.logger.Info("Server starting", "protocol", "HTTPS", "port", s.config.Port)
			serverErr <- server.ServeTLS(listener, "", "")
		} else {
			s.logger.Info("Server starting", "protocol", "HTTP", "port", s.config.Port)
			serverErr <- server.Serve(listener)
		}
	}()

	// Wait for either context cancellation or server error
	select {
	case <-ctx.Done():
		s.logger.Info("Shutdown signal received, shutting down server gracefully")

		// Gracefully shutdown the server
		if err := server.Shutdown(); err != nil {
			s.logger.Error(err, "Error during server shutdown")
			return err
		}

		s.logger.Info("Server stopped")
		return nil

	case err := <-serverErr:
		if err != nil {
			s.logger.Error(err, "Server failed")
		}
		return err
	}
}

// readRequest reads and parses data from the body of the given request according the type defined by isChatCompletion
func (s *VllmSimulator) readRequest(ctx *fasthttp.RequestCtx, isChatCompletion bool) (openaiserverapi.CompletionRequest, error) {
	requestID := common.GenerateUUIDString()

	if isChatCompletion {
		var req openaiserverapi.ChatCompletionRequest

		err := json.Unmarshal(ctx.Request.Body(), &req)
		if err != nil {
			s.logger.Error(err, "failed to unmarshal request body")
			return nil, err
		}

		for _, tool := range req.Tools {
			toolJson, err := json.Marshal(tool.Function)
			if err != nil {
				s.logger.Error(err, "failed to marshal request tools")
				return nil, err
			}
			err = s.toolsValidator.ValidateTool(toolJson)
			if err != nil {
				s.logger.Error(err, "tool validation failed")
				return nil, err
			}
		}
		req.RequestID = requestID

		return &req, nil
	}

	var req openaiserverapi.TextCompletionRequest
	err := json.Unmarshal(ctx.Request.Body(), &req)

	req.RequestID = requestID

	return &req, err
}

// HandleChatCompletions http handler for /v1/chat/completions
func (s *VllmSimulator) HandleChatCompletions(ctx *fasthttp.RequestCtx) {
	s.logger.Info("chat completion request received")
	s.handleCompletions(ctx, true)
}

// HandleTextCompletions http handler for /v1/completions
func (s *VllmSimulator) HandleTextCompletions(ctx *fasthttp.RequestCtx) {
	s.logger.Info("completion request received")
	s.handleCompletions(ctx, false)
}

// readTokenizeRequest reads and parses data from the body of the given request
func (s *VllmSimulator) readTokenizeRequest(ctx *fasthttp.RequestCtx) (*vllmapi.TokenizeRequest, error) {
	var tokenizeReq vllmapi.TokenizeRequest
	if err := json.Unmarshal(ctx.Request.Body(), &tokenizeReq); err != nil {
		s.logger.Error(err, "failed to unmarshal tokenize request body")
		return nil, err
	}
	return &tokenizeReq, nil
}

// HandleTokenize http handler for /tokenize
func (s *VllmSimulator) HandleTokenize(ctx *fasthttp.RequestCtx) {
	s.logger.Info("tokenize request received")
	req, err := s.readTokenizeRequest(ctx)
	if err != nil {
		s.logger.Error(err, "failed to read and parse tokenize request body")
		ctx.Error("Failed to read and parse tokenize request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	// Check that the request has only one input to tokenize
	if req.Prompt != "" && req.Messages != nil {
		s.sendCompletionError(ctx, openaiserverapi.NewCompletionError("both prompt and messages fields in tokenize request",
			fasthttp.StatusBadRequest, nil), false)
		return
	}
	// Model is optional, if not set, the model from the configuration will be used
	model := req.Model
	if model == "" {
		model = s.config.Model
	}

	tokens, _, err := s.tokenizer.Encode(req.GetPrompt(), model)
	if err != nil {
		s.logger.Error(err, "failed to tokenize")
		ctx.Error("Failed to tokenize, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}
	resp := vllmapi.TokenizeResponse{
		Count:       len(tokens),
		Tokens:      tokens,
		MaxModelLen: s.config.MaxModelLen,
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
	s.logger.Info("load lora request received")
	s.loadLora(ctx)
}

func (s *VllmSimulator) HandleUnloadLora(ctx *fasthttp.RequestCtx) {
	s.logger.Info("unload lora request received")
	s.unloadLora(ctx)
}

func (s *VllmSimulator) validateRequest(req openaiserverapi.CompletionRequest) (string, int) {
	if !s.isValidModel(req.GetModel()) {
		return fmt.Sprintf("The model `%s` does not exist.", req.GetModel()), fasthttp.StatusNotFound
	}

	if req.GetMaxCompletionTokens() != nil && *req.GetMaxCompletionTokens() <= 0 {
		return "Max completion tokens and max tokens should be positive", fasthttp.StatusBadRequest
	}

	if req.IsDoRemoteDecode() && req.IsStream() {
		return "Prefill does not support streaming", fasthttp.StatusBadRequest
	}

	if req.GetIgnoreEOS() && req.GetMaxCompletionTokens() == nil {
		return "Ignore_eos is true but max_completion_tokens (or max_tokens) is not set", fasthttp.StatusBadRequest
	}

	// Validate context window constraints
	promptTokens := req.GetNumberOfPromptTokens()
	completionTokens := req.GetMaxCompletionTokens()
	isValid, actualCompletionTokens, totalTokens := common.ValidateContextWindow(promptTokens, completionTokens, s.config.MaxModelLen)
	if !isValid {
		message := fmt.Sprintf("This model's maximum context length is %d tokens. However, you requested %d tokens (%d in the messages, %d in the completion). Please reduce the length of the messages or completion",
			s.config.MaxModelLen, totalTokens, promptTokens, actualCompletionTokens)
		return message, fasthttp.StatusBadRequest
	}
	return "", fasthttp.StatusOK
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
	}
	if s.namespace != "" {
		ctx.Response.Header.Add(namespaceHeader, s.namespace)
	}
	ctx.Response.SetBody(data)
}

// sendCompletionError sends an error response for the current completion request
// isInjected indicates if this is an injected failure for logging purposes
func (s *VllmSimulator) sendCompletionError(ctx *fasthttp.RequestCtx,
	compErr openaiserverapi.CompletionError, isInjected bool) {
	if isInjected {
		s.logger.Info("Injecting failure", "type", compErr.Type, "message", compErr.Message)
	} else {
		s.logger.Error(nil, compErr.Message)
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

// HandleModels handles /v1/models request according the data stored in the simulator
func (s *VllmSimulator) HandleModels(ctx *fasthttp.RequestCtx) {
	modelsResp := s.createModelsResponse()

	data, err := json.Marshal(modelsResp)
	if err != nil {
		s.logger.Error(err, "Failed to marshal models response")
		ctx.Error("Failed to marshal models response, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)
}

func (s *VllmSimulator) HandleError(_ *fasthttp.RequestCtx, err error) {
	s.logger.Error(err, "VLLM server error")
}

// HandleHealth http handler for /health
func (s *VllmSimulator) HandleHealth(ctx *fasthttp.RequestCtx) {
	s.logger.V(4).Info("health request received")
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody([]byte("{}"))
}

// HandleReady http handler for /ready
func (s *VllmSimulator) HandleReady(ctx *fasthttp.RequestCtx) {
	s.logger.V(4).Info("readiness request received")
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody([]byte("{}"))
}
