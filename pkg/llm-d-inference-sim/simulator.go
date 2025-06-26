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
	"errors"
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/buaazp/fasthttprouter"
	"github.com/go-logr/logr"
	"github.com/google/uuid"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/spf13/pflag"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"
)

const (
	vLLMDefaultPort           = 8000
	modeRandom                = "random"
	modeEcho                  = "echo"
	chatComplIDPrefix         = "chatcmpl-"
	stopFinishReason          = "stop"
	lengthFinishReason        = "length"
	toolsFinishReason         = "tool_calls"
	roleAssistant             = "assistant"
	roleUser                  = "user"
	textCompletionObject      = "text_completion"
	chatCompletionObject      = "chat.completion"
	chatCompletionChunkObject = "chat.completion.chunk"
	toolChoiceNone            = "none"
	toolChoiceAuto            = "auto"
	toolChoiceRequired        = "required"
)

// VllmSimulator simulates vLLM server supporting OpenAI API
type VllmSimulator struct {
	// logger is used for information and errors logging
	logger logr.Logger
	// timeToFirstToken time before the first token will be returned, in milliseconds
	timeToFirstToken int
	// interTokenLatency time between generated tokens, in milliseconds
	interTokenLatency int
	// port defines on which port the simulator runs
	port int
	// mode defines the simulator response generation mode, valid values: echo, random
	mode string
	// model defines the current base model name
	model string
	// loraAdaptors contains list of LoRA available adaptors
	loraAdaptors sync.Map
	// maxLoras defines maximum number of loaded loras
	maxLoras int
	// maxCPULoras defines maximum number of loras to store in CPU memory
	maxCPULoras int
	// runningLoras is a collection of running loras, key of lora's name, value is number of requests using this lora
	runningLoras sync.Map
	// waitingLoras will represent collection of loras defined in requests in the queue - Not implemented yet
	waitingLoras sync.Map
	// maxRunningReqs defines the maximum number of inference requests that could be processed at the same time
	maxRunningReqs int64
	// nRunningReqs is the number of inference requests that are currently being processed
	nRunningReqs int64
	// nWaitingReqs is the number of inference requests that are waiting to be processed
	nWaitingReqs int64
	// loraInfo is prometheus gauge
	loraInfo *prometheus.GaugeVec
	// runningRequests is prometheus gauge
	runningRequests *prometheus.GaugeVec
	// waitingRequests is prometheus gauge for number of queued requests
	waitingRequests *prometheus.GaugeVec
	// kvCacheUsagePercentage is prometheus gauge
	kvCacheUsagePercentage *prometheus.GaugeVec
	// channel for requeasts to be passed to workers
	reqChan chan *completionReqCtx
	// schema validator for tools parameters
	toolsValidator *validator
}

// New creates a new VllmSimulator instance with the given logger
func New(logger logr.Logger) (*VllmSimulator, error) {
	toolsValidtor, err := createValidator()
	if err != nil {
		return nil, fmt.Errorf("failed to create tools validator: %s", err)
	}
	return &VllmSimulator{
		logger:         logger,
		reqChan:        make(chan *completionReqCtx, 1000),
		toolsValidator: toolsValidtor,
	}, nil
}

// Start starts the simulator
func (s *VllmSimulator) Start(ctx context.Context) error {
	// parse command line parameters
	err := s.parseCommandParams()
	if err != nil {
		return err
	}

	// initialize prometheus metrics
	err = s.createAndRegisterPrometheus()
	if err != nil {
		return err
	}

	// run request processing workers
	for i := 1; i <= int(s.maxRunningReqs); i++ {
		go s.reqProcessingWorker(ctx, i)
	}
	listener, err := s.newListener()
	if err != nil {
		return err
	}

	// start the http server
	return s.startServer(listener)
}

// parseCommandParams parses and validates command line parameters
func (s *VllmSimulator) parseCommandParams() error {
	f := pflag.NewFlagSet("llm-d-inference-sim flags", pflag.ExitOnError)
	f.StringVar(&s.mode, "mode", "random", "Simulator mode, echo - returns the same text that was sent in the request, for chat completion returns the last message, random - returns random sentence from a bank of pre-defined sentences")
	f.IntVar(&s.port, "port", vLLMDefaultPort, "Port")
	f.IntVar(&s.interTokenLatency, "inter-token-latency", 0, "Time to generate one token (in milliseconds)")
	f.IntVar(&s.timeToFirstToken, "time-to-first-token", 0, "Time to first token (in milliseconds)")
	f.StringVar(&s.model, "model", "", "Currently 'loaded' model")
	var lorasStr string
	f.StringVar(&lorasStr, "lora", "", "List of LoRA adapters, separated by comma")
	f.IntVar(&s.maxLoras, "max-loras", 1, "Maximum number of LoRAs in a single batch")
	f.IntVar(&s.maxCPULoras, "max-cpu-loras", 0, "Maximum number of LoRAs to store in CPU memory")
	f.Int64Var(&s.maxRunningReqs, "max-running-requests", 5, "Maximum number of inference requests that could be processed at the same time (parameter to simulate requests waiting queue)")

	if err := f.Parse(os.Args[1:]); err != nil {
		return err
	}

	loras := strings.Split(lorasStr, ",")
	for _, lora := range loras {
		s.loraAdaptors.Store(lora, "")
	}

	// validate parsed values
	if s.model == "" {
		return errors.New("model parameter is empty")
	}
	if s.mode != modeEcho && s.mode != modeRandom {
		return fmt.Errorf("invalid mode '%s', valid values are 'random' and 'echo'", s.mode)
	}
	if s.port <= 0 {
		return fmt.Errorf("invalid port '%d'", s.port)
	}
	if s.interTokenLatency < 0 {
		return errors.New("inter token latency cannot be negative")
	}
	if s.timeToFirstToken < 0 {
		return errors.New("time to first token cannot be negative")
	}
	if s.maxLoras < 1 {
		return errors.New("max loras cannot be less than 1")
	}
	if s.maxCPULoras == 0 {
		// max cpu loras by default is same as max loras
		s.maxCPULoras = s.maxLoras
	}
	if s.maxCPULoras < 1 {
		return errors.New("max CPU loras cannot be less than 1")
	}

	// just to suppress not used lint error for now
	_ = &s.waitingLoras

	return nil
}

func (s *VllmSimulator) newListener() (net.Listener, error) {
	s.logger.Info("Server starting", "port", s.port)
	listener, err := net.Listen("tcp4", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return nil, err
	}
	return listener, nil
}

// startServer starts http server on port defined in command line
func (s *VllmSimulator) startServer(listener net.Listener) error {
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
	r.GET("/metrics", fasthttpadaptor.NewFastHTTPHandler(promhttp.Handler()))
	// supports standard Kubernetes health and readiness checks
	r.GET("/health", s.HandleHealth)
	r.GET("/ready", s.HandleReady)

	server := fasthttp.Server{
		ErrorHandler: s.HandleError,
		Handler:      r.Handler,
		Logger:       s,
	}

	defer func() {
		if err := listener.Close(); err != nil {
			s.logger.Error(err, "server listener close failed")
		}
	}()

	return server.Serve(listener)
}

// Print prints to a log, implementation of fasthttp.Logger
func (s *VllmSimulator) Printf(format string, args ...interface{}) {
	s.logger.Info("Server error", "msg", fmt.Sprintf(format, args...))
}

// readRequest reads and parses data from the body of the given request according the type defined by isChatCompletion
func (s *VllmSimulator) readRequest(ctx *fasthttp.RequestCtx, isChatCompletion bool) (completionRequest, error) {
	if isChatCompletion {
		var req chatCompletionRequest

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
			err = s.toolsValidator.validateTool(toolJson)
			if err != nil {
				s.logger.Error(err, "tool validation failed")
				return nil, err
			}
		}

		return &req, nil
	}
	var req textCompletionRequest

	err := json.Unmarshal(ctx.Request.Body(), &req)
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
	s.logger.Info("load lora request received")
	s.unloadLora(ctx)
}

// isValidModel checks if the given model is the base model or one of "loaded" LoRAs
func (s *VllmSimulator) isValidModel(model string) bool {
	if model == s.model {
		return true
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
	vllmReq, err := s.readRequest(ctx, isChatCompletion)
	if err != nil {
		s.logger.Error(err, "failed to read and parse request body")
		ctx.Error("Failed to read and parse request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	model := vllmReq.getModel()

	if !s.isValidModel(model) {
		s.sendCompletionError(ctx, fmt.Sprintf("The model `%s` does not exist.", vllmReq.getModel()),
			"NotFoundError", fasthttp.StatusNotFound)
		return
	}

	var wg sync.WaitGroup
	wg.Add(1)
	reqCtx := &completionReqCtx{
		completionReq:    vllmReq,
		httpReqCtx:       ctx,
		isChatCompletion: isChatCompletion,
		wg:               &wg,
	}
	s.reqChan <- reqCtx
	atomic.StoreInt64(&(s.nWaitingReqs), int64(len(s.reqChan)))
	s.reportWaitingRequests()
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
			atomic.StoreInt64(&(s.nWaitingReqs), int64(len(s.reqChan)))
			s.reportWaitingRequests()

			req := reqCtx.completionReq
			model := req.getModel()
			if s.isLora(model) {
				// if current request's model is LoRA, add it to the list of running loras
				value, ok := s.runningLoras.Load(model)
				intValue := 0

				if !ok {
					s.logger.Info("Create reference counter", "model", model)
					intValue = 0
				} else {
					intValue = value.(int)
				}
				s.runningLoras.Store(model, intValue+1)
				s.logger.Info("Update LoRA reference counter", "model", model, "old value", intValue, "new value", intValue+1)

				// TODO - check if this request went to the waiting queue - add it to waiting map
				s.reportLoras()
			}
			atomic.AddInt64(&(s.nRunningReqs), 1)
			s.reportRunningRequests()

			var responseTokens []string
			var finishReason string
			var err error
			var toolCalls []toolCall
			var completionTokens int
			if reqCtx.isChatCompletion && req.getToolChoice() != toolChoiceNone && req.getTools() != nil {
				toolCalls, finishReason, completionTokens, err = createToolCalls(req.getTools(), req.getToolChoice())
			}
			if toolCalls == nil && err == nil {
				// Either no tool calls were defined, or we randomly chose not to create tool calls,
				// so we generate a response text.
				responseTokens, finishReason, completionTokens, err = req.createResponseText(s.mode)
			}
			if err != nil {
				prefix := ""
				if reqCtx.isChatCompletion {
					prefix = "failed to create chat response"
				} else {
					prefix = "failed to create text response"
				}
				s.logger.Error(err, prefix)
				reqCtx.httpReqCtx.Error(prefix+err.Error(), fasthttp.StatusBadRequest)
			} else {
				usageData := usage{
					PromptTokens:     req.getNumberOfPromptTokens(),
					CompletionTokens: completionTokens,
					TotalTokens:      req.getNumberOfPromptTokens() + completionTokens,
				}
				if req.isStream() {
					var usageDataToSend *usage
					if req.includeUsage() {
						usageDataToSend = &usageData
					}
					s.sendStreamingResponse(
						&streamingContext{ctx: reqCtx.httpReqCtx, isChatCompletion: reqCtx.isChatCompletion, model: model},
						responseTokens, toolCalls, finishReason, usageDataToSend)
				} else {
					s.sendResponse(reqCtx.isChatCompletion, reqCtx.httpReqCtx, responseTokens, toolCalls, model, finishReason,
						&usageData)
				}
			}
			reqCtx.wg.Done()
		}
	}
}

// decrease model usage reference number
func (s *VllmSimulator) responseSentCallback(model string) {

	atomic.AddInt64(&(s.nRunningReqs), -1)
	s.reportRunningRequests()

	if model == s.model {
		// this is the base model - do not continue
		return
	}

	value, ok := s.runningLoras.Load(model)

	if !ok {
		s.logger.Info("Error: nil reference counter", "model", model)
		s.logger.Error(nil, "Zero model reference", "model", model)
	} else {
		intValue := value.(int)
		if intValue > 1 {
			s.runningLoras.Store(model, intValue-1)
			s.logger.Info("Update LoRA reference counter", "model", model, "prev value", intValue, "new value", intValue-1)
		} else {
			// last lora instance stopped its execution - remove from the map
			s.runningLoras.Delete(model)
			s.logger.Info("Remove LoRA from set of running loras", "model", model)
		}
	}

	s.reportLoras()

}

// sendCompletionError sends an error response for the current completion request
func (s *VllmSimulator) sendCompletionError(ctx *fasthttp.RequestCtx, msg string, errType string, code int) {
	compErr := completionError{
		Object:  "error",
		Message: msg,
		Type:    errType,
		Code:    code,
		Param:   nil,
	}
	s.logger.Error(nil, compErr.Message)

	data, err := json.Marshal(compErr)
	if err != nil {
		ctx.Error(err.Error(), fasthttp.StatusInternalServerError)
	} else {
		ctx.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusNotFound)
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
// model - model name
// finishReason - a pointer to string that represents finish reason, can be nil or stop or length, ...
// usageData - usage (tokens statistics) for this response
func (s *VllmSimulator) createCompletionResponse(isChatCompletion bool, respTokens []string, toolCalls []toolCall, model string,
	finishReason *string, usageData *usage) completionResponse {
	baseResp := baseCompletionResponse{
		ID:      chatComplIDPrefix + uuid.NewString(),
		Created: time.Now().Unix(),
		Model:   model,
		Usage:   usageData,
	}
	baseChoice := baseResponseChoice{Index: 0, FinishReason: finishReason}

	respText := strings.Join(respTokens, "")
	if isChatCompletion {
		baseResp.Object = chatCompletionObject

		message := message{Role: roleAssistant}
		if toolCalls != nil {
			message.ToolCalls = toolCalls
		} else {
			message.Content = content{Raw: respText}
		}
		return &chatCompletionResponse{
			baseCompletionResponse: baseResp,
			Choices:                []chatRespChoice{{Message: message, baseResponseChoice: baseChoice}},
		}
	}

	baseResp.Object = textCompletionObject
	return &textCompletionResponse{
		baseCompletionResponse: baseResp,
		Choices:                []textRespChoice{{baseResponseChoice: baseChoice, Text: respText}},
	}
}

// sendResponse sends response for completion API, supports both completions (text and chat)
// according the value of isChatCompletion
// respTokens - tokenized content to be sent in the response
// toolCalls - tool calls to be sent in the response
// model - model name
// finishReason - a pointer to string that represents finish reason, can be nil, stop, length, or tools
// usageData - usage (tokens statistics) for this response
func (s *VllmSimulator) sendResponse(isChatCompletion bool, ctx *fasthttp.RequestCtx, respTokens []string, toolCalls []toolCall,
	model string, finishReason string, usageData *usage) {
	resp := s.createCompletionResponse(isChatCompletion, respTokens, toolCalls, model, &finishReason, usageData)

	data, err := json.Marshal(resp)
	if err != nil {
		ctx.Error("Response body creation failed, "+err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	// calculate how long to wait before returning the response, time is based on number of tokens
	numOfTokens := usageData.CompletionTokens
	totalMillisToWait := s.timeToFirstToken + (numOfTokens-1)*s.interTokenLatency
	time.Sleep(time.Duration(totalMillisToWait) * time.Millisecond)

	// TODO - maybe add pod id to response header for testing
	ctx.Response.Header.SetContentType("application/json")
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.SetBody(data)

	s.responseSentCallback(model)
}

// createModelsResponse creates and returns ModelResponse for the current state, returned array of models contains the base model + LoRA adapters if exist
func (s *VllmSimulator) createModelsResponse() *vllmapi.ModelsResponse {
	modelsResp := vllmapi.ModelsResponse{Object: "list", Data: []vllmapi.ModelsResponseModelInfo{}}

	// add base model's info
	modelsResp.Data = append(modelsResp.Data, vllmapi.ModelsResponseModelInfo{
		ID:      s.model,
		Object:  vllmapi.ObjectModel,
		Created: time.Now().Unix(),
		OwnedBy: "vllm",
		Root:    s.model,
		Parent:  nil,
	})

	// add LoRA adapter's info
	for _, lora := range s.getLoras() {
		modelsResp.Data = append(modelsResp.Data, vllmapi.ModelsResponseModelInfo{
			ID:      lora,
			Object:  vllmapi.ObjectModel,
			Created: time.Now().Unix(),
			OwnedBy: "vllm",
			Root:    lora,
			Parent:  &s.model,
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
