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
	"os"
	"strings"
	"sync"
	"time"

	"github.com/buaazp/fasthttprouter"
	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"
	"golang.org/x/sync/errgroup"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

const (
	chatComplIDPrefix         = "chatcmpl-"
	textCompletionObject      = "text_completion"
	chatCompletionObject      = "chat.completion"
	chatCompletionChunkObject = "chat.completion.chunk"

	podHeader       = "x-inference-pod"
	namespaceHeader = "x-inference-namespace"
	podNameEnv      = "POD_NAME"
	podNsEnv        = "POD_NAMESPACE"

	maxNumberOfRequests = 1000
)

type loraUsageState int

const (
	waitingUsageState loraUsageState = iota
	runningUsageState
	doneUsageState
)

type loraUsage struct {
	// the lora adapter name
	name string
	// state of the lora usage - waiting/running/done
	state loraUsageState
}

// VllmSimulator simulates vLLM server supporting OpenAI API
type VllmSimulator struct {
	// logger is used for information and errors logging
	logger logr.Logger
	// config is the simulator's configuration
	config *common.Configuration
	// loraAdaptors contains list of LoRA available adaptors
	loraAdaptors sync.Map
	// runningLoras is a collection of running loras,
	// the key is lora's name, the value is the number of running requests using this lora
	runningLoras sync.Map
	// waitingLoras is a collection of waiting loras,
	// the key is lora's name, the value is the number of waiting requests using this lora
	waitingLoras sync.Map
	// lorasChan is a channel to update waitingLoras and runningLoras
	lorasChan chan loraUsage
	// nRunningReqs is the number of inference requests that are currently being processed
	nRunningReqs int64
	// runReqChan is a channel to update nRunningReqs
	runReqChan chan int64
	// nWaitingReqs is the number of inference requests that are waiting to be processed
	nWaitingReqs int64
	// waitingReqChan is a channel to update nWaitingReqs
	waitingReqChan chan int64
	// registry is a Prometheus registry
	registry *prometheus.Registry
	// loraInfo is prometheus gauge
	loraInfo *prometheus.GaugeVec
	// runningRequests is prometheus gauge
	runningRequests *prometheus.GaugeVec
	// waitingRequests is prometheus gauge for number of queued requests
	waitingRequests *prometheus.GaugeVec
	// kvCacheUsagePercentage is prometheus gauge
	kvCacheUsagePercentage *prometheus.GaugeVec
	// channel for requeasts to be passed to workers
	reqChan chan *openaiserverapi.CompletionReqCtx
	// schema validator for tools parameters
	toolsValidator *openaiserverapi.Validator
	// kv cache functionality
	kvcacheHelper *kvcache.KVCacheHelper
	// namespace where simulator is running
	namespace string
	// pod name of simulator
	pod string
}

// New creates a new VllmSimulator instance with the given logger
func New(logger logr.Logger) (*VllmSimulator, error) {
	toolsValidator, err := openaiserverapi.CreateValidator()
	if err != nil {
		return nil, fmt.Errorf("failed to create tools validator: %s", err)
	}

	return &VllmSimulator{
		logger:         logger,
		reqChan:        make(chan *openaiserverapi.CompletionReqCtx, maxNumberOfRequests),
		toolsValidator: toolsValidator,
		kvcacheHelper:  nil, // kvcache helper will be created only if required after reading configuration
		namespace:      os.Getenv(podNsEnv),
		pod:            os.Getenv(podNameEnv),
		runReqChan:     make(chan int64, maxNumberOfRequests),
		waitingReqChan: make(chan int64, maxNumberOfRequests),
		lorasChan:      make(chan loraUsage, maxNumberOfRequests),
	}, nil
}

// Start starts the simulator
func (s *VllmSimulator) Start(ctx context.Context) error {
	var err error
	// parse command line parameters
	s.config, err = common.ParseCommandParamsAndLoadConfig()
	if err != nil {
		return err
	}

	err = s.showConfig(s.config.DPSize > 1)
	if err != nil {
		return err
	}

	// For Data Parallel, start data-parallel-size - 1 additional simulators
	if s.config.DPSize > 1 {
		g, ctx := errgroup.WithContext(context.Background())
		for i := 2; i <= s.config.DPSize; i++ {
			newConfig, err := s.config.Copy()
			if err != nil {
				return err
			}
			dpRank := i - 1
			newConfig.Port = s.config.Port + dpRank
			newSim, err := New(klog.LoggerWithValues(s.logger, "rank", dpRank))
			if err != nil {
				return err
			}
			newSim.config = newConfig
			g.Go(func() error {
				return newSim.startSim(ctx)
			})
		}
		if err := g.Wait(); err != nil {
			return err
		}
		s.logger = klog.LoggerWithValues(s.logger, "rank", 0)
	}
	return s.startSim(ctx)
}

func (s *VllmSimulator) startSim(ctx context.Context) error {
	for _, lora := range s.config.LoraModules {
		s.loraAdaptors.Store(lora.Name, "")
	}

	common.InitRandom(s.config.Seed)

	// initialize prometheus metrics
	err := s.createAndRegisterPrometheus()
	if err != nil {
		return err
	}

	if s.config.EnableKVCache {
		s.kvcacheHelper, err = kvcache.NewKVCacheHelper(s.config, s.logger)
		if err != nil {
			return err
		}

		go s.kvcacheHelper.Run(ctx)
	}

	// run request processing workers
	for i := 1; i <= s.config.MaxNumSeqs; i++ {
		go s.reqProcessingWorker(ctx, i)
	}

	s.startMetricsUpdaters(ctx)

	listener, err := s.newListener()
	if err != nil {
		return err
	}

	// start the http server with context support
	return s.startServer(ctx, listener)
}

func (s *VllmSimulator) newListener() (net.Listener, error) {
	s.logger.Info("Server starting", "port", s.config.Port)
	listener, err := net.Listen("tcp4", fmt.Sprintf(":%d", s.config.Port))
	if err != nil {
		return nil, err
	}
	return listener, nil
}

// startServer starts http server on port defined in command line
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

	server := fasthttp.Server{
		ErrorHandler: s.HandleError,
		Handler:      r.Handler,
		Logger:       s,
	}

	// Start server in a goroutine
	serverErr := make(chan error, 1)
	go func() {
		s.logger.Info("HTTP server starting")
		serverErr <- server.Serve(listener)
	}()

	// Wait for either context cancellation or server error
	select {
	case <-ctx.Done():
		s.logger.Info("Shutdown signal received, shutting down HTTP server gracefully")

		// Gracefully shutdown the server
		if err := server.Shutdown(); err != nil {
			s.logger.Error(err, "Error during server shutdown")
			return err
		}

		s.logger.Info("HTTP server stopped")
		return nil

	case err := <-serverErr:
		if err != nil {
			s.logger.Error(err, "HTTP server failed")
		}
		return err
	}
}

// Print prints to a log, implementation of fasthttp.Logger
func (s *VllmSimulator) Printf(format string, args ...interface{}) {
	s.logger.Info("Server error", "msg", fmt.Sprintf(format, args...))
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

	return "", fasthttp.StatusOK
}

// isValidModel checks if the given model is the base model or one of "loaded" LoRAs
func (s *VllmSimulator) isValidModel(model string) bool {
	for _, name := range s.config.ServedModelNames {
		if model == name {
			return true
		}
	}
	for _, lora := range s.getLoras() {
		if model == lora {
			return true
		}
	}

	return false
}

// isLora returns true if the given model name is one of loaded LoRAs
func (s *VllmSimulator) isLora(model string) bool {
	for _, lora := range s.getLoras() {
		if model == lora {
			return true
		}
	}

	return false
}

// handleCompletions general completion requests handler, support both text and chat completion APIs
func (s *VllmSimulator) handleCompletions(ctx *fasthttp.RequestCtx, isChatCompletion bool) {
	// Check if we should inject a failure
	if shouldInjectFailure(s.config) {
		failure := getRandomFailure(s.config)
		s.sendCompletionError(ctx, failure, true)
		return
	}

	vllmReq, err := s.readRequest(ctx, isChatCompletion)
	if err != nil {
		s.logger.Error(err, "failed to read and parse request body")
		ctx.Error("Failed to read and parse request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	errMsg, errCode := s.validateRequest(vllmReq)
	if errMsg != "" {
		s.sendCompletionError(ctx, openaiserverapi.NewCompletionError(errMsg, errCode, nil), false)
		return
	}

	defer func() {
		if s.config.EnableKVCache && !isChatCompletion {
			err := s.kvcacheHelper.OnRequestEnd(vllmReq)
			if err != nil {
				s.logger.Error(err, "kv cache failed to process request end")
			}
		}
	}()
	if s.config.EnableKVCache && !isChatCompletion {
		// kv cache is currently supported for /completion API only
		err = s.kvcacheHelper.OnRequestStart(vllmReq)
		if err != nil {
			s.sendCompletionError(ctx, openaiserverapi.NewCompletionError(err.Error(), fasthttp.StatusInternalServerError, nil), false)
		}
	}

	// Validate context window constraints
	promptTokens := vllmReq.GetNumberOfPromptTokens()
	completionTokens := vllmReq.GetMaxCompletionTokens()
	isValid, actualCompletionTokens, totalTokens := common.ValidateContextWindow(promptTokens, completionTokens, s.config.MaxModelLen)
	if !isValid {
		message := fmt.Sprintf("This model's maximum context length is %d tokens. However, you requested %d tokens (%d in the messages, %d in the completion). Please reduce the length of the messages or completion",
			s.config.MaxModelLen, totalTokens, promptTokens, actualCompletionTokens)
		s.sendCompletionError(ctx, openaiserverapi.NewCompletionError(message, fasthttp.StatusBadRequest, nil), false)
		return
	}

	var wg sync.WaitGroup
	wg.Add(1)
	reqCtx := &openaiserverapi.CompletionReqCtx{
		CompletionReq:    vllmReq,
		HTTPReqCtx:       ctx,
		IsChatCompletion: isChatCompletion,
		Wg:               &wg,
	}
	// increment the waiting requests metric
	s.waitingReqChan <- 1
	if s.isLora(reqCtx.CompletionReq.GetModel()) {
		// update loraInfo metrics with the new waiting request
		s.lorasChan <- loraUsage{reqCtx.CompletionReq.GetModel(), waitingUsageState}
	}
	// send the request to the waiting queue (channel)
	s.reqChan <- reqCtx
	wg.Wait()
}

func (s *VllmSimulator) reqProcessingWorker(ctx context.Context, id int) {
	for {
		select {
		case <-ctx.Done():
			s.logger.Info("reqProcessingWorker stopped:", "worker id", id)
			return
		case reqCtx, ok := <-s.reqChan:
			if !ok {
				s.logger.Info("reqProcessingWorker worker exiting: reqChan closed")
				return
			}

			req := reqCtx.CompletionReq
			model := req.GetModel()
			displayModel := s.getDisplayedModelName(model)

			// decriment waiting and increment running requests count
			s.waitingReqChan <- -1
			s.runReqChan <- 1

			if s.isLora(model) {
				// update loraInfo metric to reflect that
				// the request has changed its status from waiting to running
				s.lorasChan <- loraUsage{model, runningUsageState}
			}

			var responseTokens []string
			var finishReason string
			var err error
			var toolCalls []openaiserverapi.ToolCall
			var completionTokens int
			if reqCtx.IsChatCompletion &&
				req.GetToolChoice() != openaiserverapi.ToolChoiceNone &&
				req.GetTools() != nil {
				toolCalls, finishReason, completionTokens, err =
					openaiserverapi.CreateToolCalls(req.GetTools(), req.GetToolChoice(), s.config)
			}
			if toolCalls == nil && err == nil {
				// Either no tool calls were defined, or we randomly chose not to create tool calls,
				// so we generate a response text.
				responseTokens, finishReason, completionTokens, err = req.CreateResponseText(s.config.Mode)
			}
			if err != nil {
				prefix := ""
				if reqCtx.IsChatCompletion {
					prefix = "failed to create chat response"
				} else {
					prefix = "failed to create text response"
				}
				s.logger.Error(err, prefix)
				reqCtx.HTTPReqCtx.Error(prefix+err.Error(), fasthttp.StatusBadRequest)
			} else {
				usageData := openaiserverapi.Usage{
					PromptTokens:     req.GetNumberOfPromptTokens(),
					CompletionTokens: completionTokens,
					TotalTokens:      req.GetNumberOfPromptTokens() + completionTokens,
				}
				if req.IsStream() {
					var usageDataToSend *openaiserverapi.Usage
					if req.IncludeUsage() {
						usageDataToSend = &usageData
					}
					s.sendStreamingResponse(
						&streamingContext{
							ctx:                 reqCtx.HTTPReqCtx,
							isChatCompletion:    reqCtx.IsChatCompletion,
							model:               displayModel,
							doRemotePrefill:     req.IsDoRemotePrefill(),
							nPromptTokens:       usageData.PromptTokens,
							nCachedPromptTokens: reqCtx.CompletionReq.GetNumberOfCachedPromptTokens(),
						},
						responseTokens, toolCalls, finishReason, usageDataToSend,
					)
				} else {
					if req.IsDoRemoteDecode() {
						// in case this is prefill pod processing, return special finish reason
						finishReason = common.RemoteDecodeFinishReason
					}

					s.sendResponse(reqCtx, responseTokens, toolCalls, displayModel, finishReason, &usageData)
				}
			}
			reqCtx.Wg.Done()
		}
	}
}

// decrease model usage reference number
func (s *VllmSimulator) responseSentCallback(model string) {
	// decriment running requests count
	s.runReqChan <- -1

	if s.isLora(model) {
		// update loraInfo metrics to reflect that the request processing has been finished
		s.lorasChan <- loraUsage{model, doneUsageState}
	}
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

// createCompletionResponse creates the response for completion requests, supports both completion request types (text and chat)
// as defined by isChatCompletion
// respTokens - tokenized content to be sent in the response
// toolCalls - tool calls to be sent in the response
// finishReason - a pointer to string that represents finish reason, can be nil or stop or length, ...
// usageData - usage (tokens statistics) for this response
// modelName - display name returned to the client and used in metrics. It is either the first alias
// from --served-model-name (for a base-model request) or the LoRA adapter name (for a LoRA request).
func (s *VllmSimulator) createCompletionResponse(isChatCompletion bool, respTokens []string, toolCalls []openaiserverapi.ToolCall,
	finishReason *string, usageData *openaiserverapi.Usage, modelName string, doRemoteDecode bool) openaiserverapi.CompletionResponse {
	baseResp := openaiserverapi.BaseCompletionResponse{
		ID:      chatComplIDPrefix + common.GenerateUUIDString(),
		Created: time.Now().Unix(),
		Model:   modelName,
		Usage:   usageData,
	}

	if doRemoteDecode {
		// add special fields related to the prefill pod special behavior
		baseResp.DoRemoteDecode = true
		baseResp.DoRemotePrefill = false
		// currently remote prefill information is hard-coded
		baseResp.RemoteBlockIds = []string{"DUMMY_ID"}
		baseResp.RemoteEngineId = "DUMMY_ID"
		baseResp.RemoteHost = "DUMMY"
		baseResp.RemotePort = 1234
	}

	baseChoice := openaiserverapi.BaseResponseChoice{Index: 0, FinishReason: finishReason}

	respText := strings.Join(respTokens, "")
	if isChatCompletion {
		baseResp.Object = chatCompletionObject

		message := openaiserverapi.Message{Role: openaiserverapi.RoleAssistant}
		if toolCalls != nil {
			message.ToolCalls = toolCalls
		} else {
			message.Content = openaiserverapi.Content{Raw: respText}
		}
		return &openaiserverapi.ChatCompletionResponse{
			BaseCompletionResponse: baseResp,
			Choices:                []openaiserverapi.ChatRespChoice{{Message: message, BaseResponseChoice: baseChoice}},
		}
	}

	baseResp.Object = textCompletionObject
	return &openaiserverapi.TextCompletionResponse{
		BaseCompletionResponse: baseResp,
		Choices:                []openaiserverapi.TextRespChoice{{BaseResponseChoice: baseChoice, Text: respText}},
	}
}

// sendResponse sends response for completion API, supports both completions (text and chat)
// according the value of isChatCompletion in reqCtx
// respTokens - tokenized content to be sent in the response
// toolCalls - tool calls to be sent in the response
// modelName - display name returned to the client and used in metrics. It is either the first alias
// from --served-model-name (for a base-model request) or the LoRA adapter name (for a LoRA request).
// finishReason - a pointer to string that represents finish reason, can be nil, stop, length, or tools
// usageData - usage (tokens statistics) for this response
func (s *VllmSimulator) sendResponse(reqCtx *openaiserverapi.CompletionReqCtx, respTokens []string, toolCalls []openaiserverapi.ToolCall,
	modelName string, finishReason string, usageData *openaiserverapi.Usage) {
	resp := s.createCompletionResponse(reqCtx.IsChatCompletion, respTokens, toolCalls, &finishReason, usageData, modelName,
		reqCtx.CompletionReq.IsDoRemoteDecode())

	ctx := reqCtx.HTTPReqCtx
	data, err := json.Marshal(resp)
	if err != nil {
		ctx.Error("Response body creation failed, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	// calculate how long to wait before returning the response, time is based on number of tokens
	nPromptTokens := usageData.PromptTokens
	nCachedPromptTokens := reqCtx.CompletionReq.GetNumberOfCachedPromptTokens()
	nGenTokens := usageData.CompletionTokens
	ttft := s.getTimeToFirstToken(nPromptTokens, nCachedPromptTokens, reqCtx.CompletionReq.IsDoRemotePrefill())

	time.Sleep(time.Duration(ttft) * time.Microsecond)
	for range nGenTokens - 1 {
		perTokenLatency := s.getInterTokenLatency()
		time.Sleep(time.Duration(perTokenLatency) * time.Microsecond)
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

	s.responseSentCallback(modelName)
}

// returns time to first token based on the current request's doRemotePrefill
func (s *VllmSimulator) getTimeToFirstToken(nPromptTokens int, nCachedPromptTokens int, doRemotePrefill bool) int {
	if doRemotePrefill {
		if s.config.KVCacheTransferLatency == 0 && s.config.KVCacheTransferLatencyStdDev == 0 {
			// is disaggregated PD and ttft is calculated using number of prompt tokens
			kvCacheTransT := s.config.KVCacheTransferTimePerToken * nPromptTokens
			return common.RandomNorm(kvCacheTransT, s.config.KVCacheTransferTimeStdDev)
		}
		// is disaggregated PD and *not* using number of prompt tokens
		return common.RandomNorm(s.config.KVCacheTransferLatency, s.config.KVCacheTransferLatencyStdDev)
	}
	if s.config.TimeToFirstToken == 0 && s.config.TimeToFirstTokenStdDev == 0 {
		// is aggregated PD and ttft is calculated using number of prompt tokens that are not in kv cache
		prefillTime := s.config.GetPrefillOverhead(s.nRunningReqs) + (nPromptTokens-nCachedPromptTokens)*s.config.GetPrefillTimePerToken(s.nRunningReqs)
		return common.RandomNorm(prefillTime, s.config.PrefillTimeStdDev)
	}
	// is aggregated PD and *not* using number of prompt tokens
	return common.RandomNorm(s.config.GetTimeToFirstToken(s.nRunningReqs), s.config.TimeToFirstTokenStdDev)
}

// returns inter token latency
func (s *VllmSimulator) getInterTokenLatency() int {
	return common.RandomNorm(s.config.GetInterTokenLatency(s.nRunningReqs), s.config.InterTokenLatencyStdDev)
}

// createModelsResponse creates and returns ModelResponse for the current state, returned array of models contains the base model + LoRA adapters if exist
func (s *VllmSimulator) createModelsResponse() *vllmapi.ModelsResponse {
	modelsResp := vllmapi.ModelsResponse{Object: "list", Data: []vllmapi.ModelsResponseModelInfo{}}

	// Advertise every public model alias
	for _, alias := range s.config.ServedModelNames {
		modelsResp.Data = append(modelsResp.Data, vllmapi.ModelsResponseModelInfo{
			ID:      alias,
			Object:  vllmapi.ObjectModel,
			Created: time.Now().Unix(),
			OwnedBy: "vllm",
			Root:    alias,
			Parent:  nil,
		})
	}

	// add LoRA adapter's info
	parent := s.config.ServedModelNames[0]
	for _, lora := range s.getLoras() {
		modelsResp.Data = append(modelsResp.Data, vllmapi.ModelsResponseModelInfo{
			ID:      lora,
			Object:  vllmapi.ObjectModel,
			Created: time.Now().Unix(),
			OwnedBy: "vllm",
			Root:    lora,
			Parent:  &parent,
		})
	}

	return &modelsResp
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

// getDisplayedModelName returns the model name that must appear in API
// responses.  LoRA adapters keep their explicit name, while all base-model
// requests are surfaced as the first alias from --served-model-name.
func (s *VllmSimulator) getDisplayedModelName(reqModel string) string {
	if s.isLora(reqModel) {
		return reqModel
	}
	return s.config.ServedModelNames[0]
}

func (s *VllmSimulator) showConfig(dp bool) error {
	cfgJSON, err := json.Marshal(s.config)
	if err != nil {
		return fmt.Errorf("failed to marshal configuration to JSON: %w", err)
	}

	var m map[string]interface{}
	err = json.Unmarshal(cfgJSON, &m)
	if err != nil {
		return fmt.Errorf("failed to unmarshal JSON to map: %w", err)
	}
	if dp {
		// remove the port
		delete(m, "port")
	}
	// clean LoraModulesString field
	m["lora-modules"] = m["LoraModules"]
	delete(m, "LoraModules")
	delete(m, "LoraModulesString")

	// clean fake-metrics field
	if field, ok := m["fake-metrics"].(map[string]interface{}); ok {
		delete(field, "LorasString")
	}

	// show in JSON
	cfgJSON, err = json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal configuration to JSON: %w", err)
	}
	s.logger.Info("Configuration:", "", string(cfgJSON))
	return nil
}
