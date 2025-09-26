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
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/valyala/fasthttp"
	"golang.org/x/sync/errgroup"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization"
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
	// kvCacheUsageChan is a channel to update kvCacheUsagePercentage
	kvCacheUsageChan chan float64
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
	// tokenizer is currently used in kv-cache and in /tokenize
	tokenizer tokenization.Tokenizer
	// dataset is used for token generation in responses
	dataset dataset.Dataset
}

// New creates a new VllmSimulator instance with the given logger
func New(logger logr.Logger) (*VllmSimulator, error) {
	toolsValidator, err := openaiserverapi.CreateValidator()
	if err != nil {
		return nil, fmt.Errorf("failed to create tools validator: %s", err)
	}

	return &VllmSimulator{
		logger:           logger,
		reqChan:          make(chan *openaiserverapi.CompletionReqCtx, maxNumberOfRequests),
		toolsValidator:   toolsValidator,
		kvcacheHelper:    nil, // kvcache helper will be created only if required after reading configuration
		namespace:        os.Getenv(podNsEnv),
		pod:              os.Getenv(podNameEnv),
		runReqChan:       make(chan int64, maxNumberOfRequests),
		waitingReqChan:   make(chan int64, maxNumberOfRequests),
		lorasChan:        make(chan loraUsage, maxNumberOfRequests),
		kvCacheUsageChan: make(chan float64, maxNumberOfRequests),
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
	g, ctx := errgroup.WithContext(ctx)
	if s.config.DPSize > 1 {
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
		s.logger = klog.LoggerWithValues(s.logger, "rank", 0)
	}
	g.Go(func() error {
		return s.startSim(ctx)
	})
	if err := g.Wait(); err != nil {
		return err
	}
	return nil
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

	tokenizationConfig := tokenization.DefaultConfig()
	if s.config.TokenizersCacheDir != "" {
		tokenizationConfig.TokenizersCacheDir = s.config.TokenizersCacheDir
	}
	s.tokenizer, err = tokenization.NewCachedHFTokenizer(tokenizationConfig.HFTokenizerConfig)
	if err != nil {
		return fmt.Errorf("failed to create tokenizer: %w", err)
	}

	if s.config.EnableKVCache {
		s.kvcacheHelper, err = kvcache.NewKVCacheHelper(s.config, s.logger, s.kvCacheUsageChan, s.tokenizer)
		if err != nil {
			return err
		}

		go s.kvcacheHelper.Run(ctx)
	}

	err = s.initDataset(ctx)
	if err != nil {
		return fmt.Errorf("dataset initialization error: %w", err)
	}

	// run request processing workers
	for i := 1; i <= s.config.MaxNumSeqs; i++ {
		go s.reqProcessingWorker(ctx, i)
	}

	s.startMetricsUpdaters(ctx)

	listener, err := s.newListener()
	if err != nil {
		s.logger.Error(err, "Failed to create listener")
		return fmt.Errorf("listener creation error: %w", err)
	}

	// start the http server with context support
	return s.startServer(ctx, listener)
}

func (s *VllmSimulator) initDataset(ctx context.Context) error {
	randDataset := &dataset.BaseDataset{}
	err := randDataset.Init(ctx, s.logger, "", "", false)
	if err != nil {
		return fmt.Errorf("failed to initialize random dataset: %w", err)
	}

	if s.config.DatasetPath == "" && s.config.DatasetURL == "" {
		s.logger.Info("No dataset path or URL provided, using random text for responses")
		s.dataset = randDataset
		return nil
	}

	custDataset := &dataset.CustomDataset{}
	err = custDataset.Init(ctx, s.logger, s.config.DatasetPath, s.config.DatasetURL, s.config.DatasetInMemory)

	if err == nil {
		s.dataset = custDataset
		return nil
	}

	if strings.HasPrefix(err.Error(), "database is locked") {
		s.logger.Info("Database is locked by another process, will use preset text for responses instead")
		s.dataset = randDataset
		return nil
	}

	return err
}

// Print prints to a log, implementation of fasthttp.Logger
func (s *VllmSimulator) Printf(format string, args ...interface{}) {
	s.logger.Info("Server error", "msg", fmt.Sprintf(format, args...))
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

			// decrement waiting and increment running requests count
			s.waitingReqChan <- -1
			s.runReqChan <- 1

			if s.isLora(model) {
				// update loraInfo metric to reflect that
				// the request has changed its status from waiting to running
				s.lorasChan <- loraUsage{model, runningUsageState}
			}

			if s.config.EnableKVCache && !reqCtx.IsChatCompletion {
				// kv cache is currently supported for /completion API only
				if err := s.kvcacheHelper.OnRequestStart(req); err != nil {
					s.sendCompletionError(reqCtx.HTTPReqCtx, openaiserverapi.NewCompletionError(err.Error(), fasthttp.StatusInternalServerError, nil), false)
				}
			}

			var responseTokens []string
			var finishReason string
			var err error
			var toolCalls []openaiserverapi.ToolCall
			var completionTokens int
			if reqCtx.IsChatCompletion &&
				req.GetToolChoice() != openaiserverapi.ToolChoiceNone &&
				req.GetTools() != nil {
				toolCalls, completionTokens, err =
					openaiserverapi.CreateToolCalls(req.GetTools(), req.GetToolChoice(), s.config)
				finishReason = dataset.ToolsFinishReason
			}
			if toolCalls == nil && err == nil {
				// Either no tool calls were defined, or we randomly chose not to create tool calls,
				// so we generate a response text.
				responseTokens, finishReason, err = s.dataset.GetTokens(req, s.config.Mode)
				completionTokens += len(responseTokens)
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
						finishReason = dataset.RemoteDecodeFinishReason
					}

					s.sendResponse(reqCtx, responseTokens, toolCalls, displayModel, finishReason, &usageData)
				}
			}
			reqCtx.Wg.Done()
		}
	}
}

// request processing finished
func (s *VllmSimulator) responseSentCallback(model string, isChatCompletion bool, requestID string) {
	// decriment running requests count
	s.runReqChan <- -1

	if s.isLora(model) {
		// update loraInfo metrics to reflect that the request processing has been finished
		s.lorasChan <- loraUsage{model, doneUsageState}
	}

	if s.config.EnableKVCache && !isChatCompletion {
		if err := s.kvcacheHelper.OnRequestEnd(requestID); err != nil {
			s.logger.Error(err, "kv cache failed to process request end")
		}
	}
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

	// calculate how long to wait before returning the response, time is based on number of tokens
	nCachedPromptTokens := reqCtx.CompletionReq.GetNumberOfCachedPromptTokens()
	ttft := s.getWaitTimeToFirstToken(usageData.PromptTokens, nCachedPromptTokens, reqCtx.CompletionReq.IsDoRemotePrefill())
	time.Sleep(time.Duration(ttft) * time.Millisecond)
	for range usageData.CompletionTokens - 1 {
		perTokenLatency := s.getInterTokenLatency()
		time.Sleep(time.Duration(perTokenLatency) * time.Millisecond)
	}

	s.sendCompletionResponse(reqCtx.HTTPReqCtx, resp)

	s.responseSentCallback(modelName, reqCtx.IsChatCompletion, reqCtx.CompletionReq.GetRequestID())
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
