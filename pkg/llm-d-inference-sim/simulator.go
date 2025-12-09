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
	"container/list"
	"context"
	"errors"
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
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization"
)

const (
	chatComplIDPrefix         = "chatcmpl-"
	textComplIDPrefix         = "cmpl-"
	textCompletionObject      = "text_completion"
	chatCompletionObject      = "chat.completion"
	chatCompletionChunkObject = "chat.completion.chunk"

	podHeader       = "x-inference-pod"
	portHeader      = "x-inference-port"
	namespaceHeader = "x-inference-namespace"
	requestIDHeader = "X-Request-Id"
	podNameEnv      = "POD_NAME"
	podNsEnv        = "POD_NAMESPACE"
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

// Prometheus metrics
type metricsData struct {
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
	// requestSuccessChan is a channel to update requestSuccessReqs
	requestSuccessChan chan requestSuccessEvent
	// nWaitingReqs is the number of inference requests that are waiting to be processed
	nWaitingReqs int64
	// waitingReqChan is a channel to update nWaitingReqs
	waitingReqChan chan int64
	// ttftChan is a channel to update time to first token
	ttftChan chan float64
	// tpotChan is a channel to update time per output token
	tpotChan chan float64
	// e2eReqLatencyChan is a channel to update request e2e latency
	e2eReqLatencyChan chan float64
	// reqQueueTimeChan is a channel to update request queue time
	reqQueueTimeChan chan float64
	// reqInferenceTimeChan is a channel to update request inference time
	reqInferenceTimeChan chan float64
	// reqPrefillTimeChan is a channel to update request prefill time
	reqPrefillTimeChan chan float64
	// reqDecodeTimeChan is a channel to update request decode time
	reqDecodeTimeChan chan float64
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
	// ttft is prometheus histogram for time to first token in seconds
	ttft *prometheus.HistogramVec
	// tpot is prometheus histogram for time per output token in seconds (deprecated since vLLM 0.11
	tpot *prometheus.HistogramVec
	// interTokenLatency is prometheus histogram for inter-token latency in seconds (replaces tpot since vLLM 0.11)
	interTokenLatency *prometheus.HistogramVec
	// e2eReqLatency is prometheus histogram of end to end request latency in seconds
	e2eReqLatency *prometheus.HistogramVec
	// reqQueueTime is prometheus histogram of request queue time in seconds
	reqQueueTime *prometheus.HistogramVec
	// reqInferenceTime is prometheus histogram of request inference time in seconds
	reqInferenceTime *prometheus.HistogramVec
	// reqPrefillTime is prometheus histogram of request prefill time in seconds
	reqPrefillTime *prometheus.HistogramVec
	// reqDecodeTime is prometheus histogram of request decode time in seconds
	reqDecodeTime *prometheus.HistogramVec
	// kvCacheUsagePercentage is prometheus gauge
	kvCacheUsagePercentage *prometheus.GaugeVec
	// requestPromptTokens is prometheus histogram for number of input (prompt) tokens in request
	requestPromptTokens *prometheus.HistogramVec
	// requestGenerationTokens is prometheus histogram for number of generated tokens in request
	requestGenerationTokens *prometheus.HistogramVec
	// maxNumGenerationTokens is prometheus histogram for maximum number of generated tokens in request
	maxNumGenerationTokens *prometheus.HistogramVec
	// requestParamsMaxTokens is prometheus histogram for 'max_tokens' parameter in request
	requestParamsMaxTokens *prometheus.HistogramVec
	// requestSuccessTotal is prometheus counter for total number of successful requests
	requestSuccessTotal *prometheus.CounterVec
}

// LoRAs usage info for requests execution
type lorasUsageInfo struct {
	mux sync.RWMutex
	// lora adapter name -> reference count (number of currently running requests)
	loadedLoras map[string]int
	// channel for "there is a LoRA that can be removed" event
	loraRemovable chan int
	// maximum number of LoRAs that can be used simultaneously
	maxLoras int
}

type requestCompleted struct {
	worker *worker
	model  string
}

type waitingQueueItem struct {
	reqCtx      *openaiserverapi.CompletionReqCtx
	enqueueTime time.Time
}

// VllmSimulator simulates vLLM server supporting OpenAI API
type VllmSimulator struct {
	// logger is used for information and errors logging
	logger logr.Logger
	// config is the simulator's configuration
	config *common.Configuration
	// loraAdaptors contains list of LoRA available adaptors
	loraAdaptors sync.Map
	// schema validator for tools parameters
	toolsValidator *common.ToolsValidator
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
	// metrics contains all Prometheus metrics related data
	metrics metricsData
	// loras contains information about which LoRAs are in use
	loras *lorasUsageInfo
	// rand with a configurable seed to generate reproducible random responses
	random *common.Random

	// indication whether the simulator is sleeping
	isSleeping bool
	// indication whether the simulator is in development mode, set by environment
	// variable VLLM_SERVER_DEV_MODE
	isInDevMode bool
	// a mutex for sleep-wake up
	sleepMutex sync.RWMutex

	// a channel for free workers
	freeWorkers chan *worker
	// a channel to indicate that a worker finished working on a request
	workerFinished chan *requestCompleted
	// waiting requests queue mutex
	queueLock sync.Mutex
	// bi-directional list of *openaiserverapi.CompletionReqCtx
	waitingQueue *list.List
	// the max capacity of the waiting requests queue
	queueCapacity int
	// a channel for incoming requests
	newRequests chan *openaiserverapi.CompletionReqCtx
}

// New creates a new VllmSimulator instance with the given logger
func New(logger logr.Logger) (*VllmSimulator, error) {
	toolsValidator, err := common.CreateToolsValidator()
	if err != nil {
		return nil, fmt.Errorf("failed to create tools validator: %s", err)
	}

	return &VllmSimulator{
		logger:         logger,
		toolsValidator: toolsValidator,
		kvcacheHelper:  nil, // kvcache helper will be created only if required after reading configuration
		namespace:      os.Getenv(podNsEnv),
		pod:            os.Getenv(podNameEnv),
		isInDevMode:    os.Getenv("VLLM_SERVER_DEV_MODE") == "1",
		loras: &lorasUsageInfo{
			loadedLoras: make(map[string]int),
		},
		waitingQueue: list.New(),
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

	if s.config.DatasetURL != "" {
		// if should use remote responses dataset, download it first (it can take time)
		downloader := dataset.NewDsDownloader(s.logger)
		if err := downloader.DownloadDataset(ctx, s.config.DatasetURL, s.config.DatasetPath); err != nil {
			return err
		}
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
	if err := s.initializeSim(ctx); err != nil {
		return err
	}

	listener, err := s.newListener()
	if err != nil {
		s.logger.Error(err, "failed to create listener")
		return fmt.Errorf("listener creation error: %w", err)
	}

	// start the http server with context support
	return s.startServer(ctx, listener)
}

func (s *VllmSimulator) initializeSim(ctx context.Context) error {
	s.random = common.NewRandom(s.config.Seed, s.config.Port)

	for _, lora := range s.config.LoraModules {
		s.loraAdaptors.Store(lora.Name, "")
	}
	s.loras.maxLoras = s.config.MaxLoras
	s.loras.loraRemovable = make(chan int, s.config.MaxNumSeqs)

	s.queueCapacity = s.config.MaxWaitingQueueLength

	maxNumberOfRequests := s.config.MaxNumSeqs + s.config.MaxWaitingQueueLength
	s.metrics.runReqChan = make(chan int64, maxNumberOfRequests)
	s.metrics.waitingReqChan = make(chan int64, maxNumberOfRequests)
	s.metrics.lorasChan = make(chan loraUsage, maxNumberOfRequests)
	s.metrics.kvCacheUsageChan = make(chan float64, maxNumberOfRequests)
	s.metrics.ttftChan = make(chan float64, maxNumberOfRequests)
	s.metrics.tpotChan = make(chan float64, maxNumberOfRequests)
	s.metrics.e2eReqLatencyChan = make(chan float64, maxNumberOfRequests)
	s.metrics.reqQueueTimeChan = make(chan float64, maxNumberOfRequests)
	s.metrics.reqInferenceTimeChan = make(chan float64, maxNumberOfRequests)
	s.metrics.reqPrefillTimeChan = make(chan float64, maxNumberOfRequests)
	s.metrics.reqDecodeTimeChan = make(chan float64, maxNumberOfRequests)
	s.metrics.requestSuccessChan = make(chan requestSuccessEvent, maxNumberOfRequests)

	s.newRequests = make(chan *openaiserverapi.CompletionReqCtx, maxNumberOfRequests)

	// initialize prometheus metrics
	err := s.createAndRegisterPrometheus()
	if err != nil {
		return err
	}

	tokenizationConfig, err := tokenization.DefaultConfig()
	if err != nil {
		return fmt.Errorf("failed to create default tokenization configuration: %w", err)
	}

	if s.config.TokenizersCacheDir != "" {
		if tokenizationConfig.HFTokenizerConfig == nil {
			tokenizationConfig.HFTokenizerConfig = &tokenization.HFTokenizerConfig{}
		}
		tokenizationConfig.HFTokenizerConfig.TokenizersCacheDir = s.config.TokenizersCacheDir
	}

	s.tokenizer, err = tokenization.NewCachedHFTokenizer(tokenizationConfig.HFTokenizerConfig)
	if err != nil {
		return fmt.Errorf("failed to create hf tokenizer: %w", err)
	}

	if s.config.EnableKVCache {
		s.kvcacheHelper, err = kvcache.NewKVCacheHelper(s.config, s.logger, s.metrics.kvCacheUsageChan, s.tokenizer)
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
	s.freeWorkers = make(chan *worker, s.config.MaxNumSeqs)
	s.workerFinished = make(chan *requestCompleted, s.config.MaxNumSeqs)
	for i := 1; i <= s.config.MaxNumSeqs; i++ {
		worker := &worker{
			id:           i,
			ctx:          ctx,
			logger:       s.logger,
			finishedChan: s.workerFinished,
			reqChan:      make(chan *openaiserverapi.CompletionReqCtx, 1),
			processor:    s,
		}
		go worker.waitForRequests()
		s.freeWorkers <- worker
	}

	s.startMetricsUpdaters(ctx)

	go s.processing(ctx)
	return nil
}

func (s *VllmSimulator) initDataset(ctx context.Context) error {
	if s.config.DatasetPath == "" && s.config.DatasetURL == "" {
		// use predefined sentences as responses
		randDataset := &dataset.DefaultDataset{}
		err := randDataset.Init(ctx, s.logger, s.random, s.config.MaxModelLen)
		if err != nil {
			return fmt.Errorf("failed to initialize random dataset: %w", err)
		}
		s.logger.V(logging.INFO).Info("No dataset path or URL provided, using random text for responses")
		s.dataset = randDataset
		return nil
	}

	// use dataset containing responses
	custDataset := &dataset.CustomDataset{}
	err := custDataset.Init(ctx, s.logger, s.random, s.config.DatasetPath, s.config.DatasetInMemory, s.config.MaxModelLen)

	if err == nil {
		s.dataset = custDataset
		return nil
	}

	return err
}

// Print prints to a log, implementation of fasthttp.Logger
func (s *VllmSimulator) Printf(format string, args ...interface{}) {
	s.logger.V(logging.WARN).Info("Server error", "msg", fmt.Sprintf(format, args...))
}

func (s *VllmSimulator) processing(ctx context.Context) {
	s.logger.V(logging.INFO).Info("Start processing routine")

	for {
		select {
		case <-ctx.Done():
			s.logger.V(logging.INFO).Info("Request processing done")
			return
		case completedReq := <-s.workerFinished:
			worker := completedReq.worker
			s.logger.V(logging.TRACE).Info("Worker finished", "worker", worker.id)
			s.decrementLora(completedReq.model)
			// there is a free worker - find a request for it and send this request for
			// processing with this worker
			s.findRequestAndSendToProcess(worker)
		case <-s.loras.loraRemovable:
			// there is a LoRA that can be removed, go through availbale workers
			// and queued requests and find requests that can run now,
			// stop if there are no free workers, or no requests
			s.logger.V(logging.TRACE).Info("LoRA can be removed")
			for {
				// check if there is a free worker
				worker := s.getFreeWorker()
				if worker == nil {
					break
				}
				// check if there is a request that can run and send this request for
				// processing with this worker
				requestFound := s.findRequestAndSendToProcess(worker)
				if !requestFound {
					// there are no requests to run (either the queue is empty or maxLoras was reached)
					break
				}
			}
		case reqCtx := <-s.newRequests:
			// A new request was received. Find a free worker, and check that the request can run LoRA wise.
			model := reqCtx.CompletionReq.GetModel()

			worker := s.getFreeWorker()
			if worker == nil {
				s.logger.V(logging.TRACE).Info("No free worker - sending the request to the waiting queue",
					"model", reqCtx.CompletionReq.GetModel(), "req id", reqCtx.CompletionReq.GetRequestID())
				// no free worker, add this request to the waiting queue
				s.addRequestToQueue(reqCtx)
				break
			}

			// check if lora usage allows the request to run
			if s.isLora(model) && !s.loadLora(model) {
				// free the worker
				s.freeWorkers <- worker
				s.logger.V(logging.TRACE).Info("LoRA cannot be loaded - sending the request to the waiting queue",
					"LoRA", model, "req id", reqCtx.CompletionReq.GetRequestID())
				// LoRA max reached, try to enqueue
				s.addRequestToQueue(reqCtx)
				break
			}

			s.logger.V(logging.TRACE).Info("Sending the request to the processing channel", "model", model,
				"req id", reqCtx.CompletionReq.GetRequestID(), "worker", worker.id)
			common.WriteToChannel(worker.reqChan, reqCtx, s.logger, "worker's reqChan")
		}
	}
}

func (s *VllmSimulator) findRequestAndSendToProcess(worker *worker) bool {
	nextReq := s.dequeue()
	if nextReq != nil {
		// send this request for processing in this worker
		s.logger.V(logging.TRACE).Info("Sending request to processing", "model", nextReq.CompletionReq.GetModel(),
			"req", nextReq.CompletionReq.GetRequestID(), "worker", worker.id)
		common.WriteToChannel(worker.reqChan, nextReq, s.logger, "worker's reqChan")
		// decrement waiting requests metric
		common.WriteToChannel(s.metrics.waitingReqChan, -1, s.logger, "metrics.waitingReqChan")
		return true
	}

	// no waiting request, return worker to be free
	s.freeWorkers <- worker
	return false
}

func (s *VllmSimulator) addRequestToQueue(reqCtx *openaiserverapi.CompletionReqCtx) {
	if err := s.enqueue(reqCtx); err != nil {
		s.logger.Error(err, "failed to enqueue request")
		reqCtx.HTTPReqCtx.Error("Failed to enqueue request, "+err.Error(), fasthttp.StatusTooManyRequests)
		reqCtx.Wg.Done()
		return
	}
	// increment the waiting requests metric
	common.WriteToChannel(s.metrics.waitingReqChan, 1, s.logger, "metrics.waitingReqChan")
	// update loraInfo metrics with the new waiting request
	common.WriteToChannel(s.metrics.lorasChan, loraUsage{reqCtx.CompletionReq.GetModel(), waitingUsageState},
		s.logger, "metrics.lorasChan")

}

// handleCompletions general completion requests handler, support both text and chat completion APIs
// Importan note: for requests in streaming mode, this function exists before all chunk are sent to the client
func (s *VllmSimulator) handleCompletions(ctx *fasthttp.RequestCtx, isChatCompletion bool) {
	// Check if we should inject a failure
	if shouldInjectFailure(s.config, s.random) {
		failure := getRandomFailure(s.config, s.random)
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

	s.logger.V(logging.DEBUG).Info("Completion request received", "req id", vllmReq.GetRequestID(), "isChat", isChatCompletion)

	var wg sync.WaitGroup
	wg.Add(1)
	reqCtx := &openaiserverapi.CompletionReqCtx{
		CompletionReq:    vllmReq,
		HTTPReqCtx:       ctx,
		IsChatCompletion: isChatCompletion,
		Wg:               &wg,
		StartProcessing:  time.Now(),
	}
	common.WriteToChannel(s.newRequests, reqCtx, s.logger, "newRequests")
	wg.Wait()
}

// request processing finished
func (s *VllmSimulator) responseSentCallback(model string, isChatCompletion bool, requestID string) {
	// decrement running requests count
	common.WriteToChannel(s.metrics.runReqChan, -1, s.logger, "metrics.runReqChan")

	if s.isLora(model) {
		// update loraInfo metrics to reflect that the request processing has been finished
		common.WriteToChannel(s.metrics.lorasChan, loraUsage{model, doneUsageState},
			s.logger, "metrics.lorasChan")
	}

	if s.config.EnableKVCache && !isChatCompletion {
		if err := s.kvcacheHelper.OnRequestEnd(requestID); err != nil {
			s.logger.Error(err, "kv cache failed to process request end")
		}
	}
}

// createCompletionResponse creates the response for completion requests, supports both completion request types (text and chat)
// as defined by isChatCompletion
// logprobs - nil if no logprobs needed, otherwise number of logprob options to include
// respTokens - tokenized content to be sent in the response
// toolCalls - tool calls to be sent in the response
// finishReason - a pointer to string that represents finish reason, can be nil or stop or length, ...
// usageData - usage (tokens statistics) for this response
// modelName - display name returned to the client and used in metrics. It is either the first alias
// from --served-model-name (for a base-model request) or the LoRA adapter name (for a LoRA request).
func (s *VllmSimulator) createCompletionResponse(logprobs *int, isChatCompletion bool, respTokens []string, toolCalls []openaiserverapi.ToolCall,
	finishReason *string, usageData *openaiserverapi.Usage, modelName string, doRemoteDecode bool, requestID string) openaiserverapi.CompletionResponse {
	var idPrefix string
	if isChatCompletion {
		idPrefix = chatComplIDPrefix
	} else {
		idPrefix = textComplIDPrefix
	}
	baseResp := openaiserverapi.CreateBaseCompletionResponse(idPrefix+requestID,
		time.Now().Unix(), modelName, usageData, requestID)

	if doRemoteDecode {
		baseResp.KVParams = &openaiserverapi.KVTransferParams{}
		// add special fields related to the prefill pod special behavior
		baseResp.KVParams.DoRemoteDecode = false
		baseResp.KVParams.DoRemotePrefill = true
		// currently remote prefill information is hard-coded
		baseResp.KVParams.RemoteBlockIds = []string{"DUMMY_ID"}
		baseResp.KVParams.RemoteEngineId = "DUMMY_ID"
		baseResp.KVParams.RemoteHost = "DUMMY"
		baseResp.KVParams.RemotePort = 1234
		baseResp.KVParams.TPSize = 1
	}

	baseChoice := openaiserverapi.CreateBaseResponseChoice(0, finishReason)

	respText := strings.Join(respTokens, "")
	if isChatCompletion {
		baseResp.Object = chatCompletionObject

		message := openaiserverapi.Message{Role: openaiserverapi.RoleAssistant}
		if toolCalls != nil {
			message.ToolCalls = toolCalls
		} else {
			message.Content = openaiserverapi.Content{Raw: respText}
		}

		choice := openaiserverapi.CreateChatRespChoice(baseChoice, message)

		// Generate logprobs if requested
		if logprobs != nil && toolCalls == nil {
			if logprobsData := common.GenerateChatLogprobs(respTokens, *logprobs); logprobsData != nil && len(logprobsData.Content) > 0 {
				choice.Logprobs = logprobsData
			} else {
				// Set to nil if generation failed or content is empty
				choice.Logprobs = nil
			}
		} else {
			// Explicitly ensure logprobs is nil when not requested
			choice.Logprobs = nil
		}

		return openaiserverapi.CreateChatCompletionResponse(baseResp, []openaiserverapi.ChatRespChoice{choice})
	}

	choice := openaiserverapi.CreateTextRespChoice(baseChoice, respText)

	// Generate logprobs if requested for text completion
	if logprobs != nil && *logprobs > 0 {
		if logprobsData := common.GenerateTextLogprobs(respTokens, *logprobs); logprobsData != nil && len(logprobsData.Tokens) > 0 {
			choice.Logprobs = logprobsData
		} else {
			// Set to nil if generation failed or tokens is empty
			choice.Logprobs = nil
		}
	} else {
		// Explicitly ensure logprobs is nil when not requested
		choice.Logprobs = nil
	}

	baseResp.Object = textCompletionObject
	return openaiserverapi.CreateTextCompletionResponse(baseResp, []openaiserverapi.TextRespChoice{choice})
}

// sendResponse sends response for completion API, supports both completions (text and chat)
// according the value of isChatCompletion in reqCtx
// respTokens - tokenized content to be sent in the response
// toolCalls - tool calls to be sent in the response
// modelName - display name returned to the client and used in metrics. It is either the first alias
// from --served-model-name (for a base-model request) or the LoRA adapter name (for a LoRA request).
// finishReason - a pointer to string that represents finish reason, can be nil, stop, length, or tools
// usageData - usage (tokens statistics) for this response
func (s *VllmSimulator) sendResponse(reqCtx *openaiserverapi.CompletionReqCtx, respTokens []string,
	toolCalls []openaiserverapi.ToolCall, modelName string, finishReason string, usageData *openaiserverapi.Usage) {
	// Extract logprob data from request (unified approach)
	var logprobs *int
	if toolCalls == nil {
		logprobs = reqCtx.CompletionReq.GetLogprobs()
	}
	requestID := reqCtx.CompletionReq.GetRequestID()

	resp := s.createCompletionResponse(logprobs, reqCtx.IsChatCompletion, respTokens, toolCalls, &finishReason, usageData, modelName,
		reqCtx.CompletionReq.IsDoRemoteDecode(), requestID)

	// calculate how long to wait before returning the response, time is based on number of tokens
	nCachedPromptTokens := reqCtx.CompletionReq.GetNumberOfCachedPromptTokens()
	startPrefill := time.Now()
	ttft := s.getWaitTimeToFirstToken(usageData.PromptTokens, nCachedPromptTokens, reqCtx.CompletionReq.IsDoRemotePrefill())
	time.Sleep(time.Duration(ttft) * time.Millisecond)

	// report ttft in seconds
	common.WriteToChannel(s.metrics.ttftChan, (float64(ttft) / 1000), s.logger, "metrics.ttftChan")
	common.WriteToChannel(s.metrics.reqPrefillTimeChan, time.Since(startPrefill).Seconds(), s.logger, "metrics.reqPrefillTimeChan")

	startDecode := time.Now()
	for range usageData.CompletionTokens - 1 {
		perTokenLatency := s.getInterTokenLatency()
		time.Sleep(time.Duration(perTokenLatency) * time.Millisecond)

		// report tpot in seconds
		common.WriteToChannel(s.metrics.tpotChan, (float64(perTokenLatency) / 1000), s.logger, "metrics.tpotChan")
	}
	common.WriteToChannel(s.metrics.reqDecodeTimeChan, time.Since(startDecode).Seconds(), s.logger, "metrics.reqDecodeTimeChan")

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

func (s *VllmSimulator) enqueue(req *openaiserverapi.CompletionReqCtx) error {
	s.queueLock.Lock()
	defer s.queueLock.Unlock()

	if s.waitingQueue.Len() >= s.queueCapacity {
		return errors.New("waiting requests queue is full")
	}
	s.waitingQueue.PushBack(waitingQueueItem{req, time.Now()})
	return nil
}

// go though the queue and find the first request that can be executed, while taking into consideration the max lora limitation
func (s *VllmSimulator) dequeue() *openaiserverapi.CompletionReqCtx {
	s.queueLock.Lock()
	defer s.queueLock.Unlock()

	// Find first request for a loaded LoRA
	for elem := s.waitingQueue.Front(); elem != nil; elem = elem.Next() {
		item, ok := elem.Value.(waitingQueueItem)
		if ok && item.reqCtx != nil && s.loraIsLoaded(item.reqCtx.CompletionReq.GetModel()) {
			s.waitingQueue.Remove(elem)
			s.incrementLora(item.reqCtx.CompletionReq.GetModel())
			common.WriteToChannel(s.metrics.reqQueueTimeChan, time.Since(item.enqueueTime).Seconds(), s.logger, "metrics.reqQueueTimeChan")
			return item.reqCtx
		}
	}

	// All the requests require a LoRA that is not loaded, check if we can load a LoRA
	for elem := s.waitingQueue.Front(); elem != nil; elem = elem.Next() {
		item, ok := elem.Value.(waitingQueueItem)
		if ok && item.reqCtx != nil && s.loadLora(item.reqCtx.CompletionReq.GetModel()) {
			s.waitingQueue.Remove(elem)
			common.WriteToChannel(s.metrics.reqQueueTimeChan, time.Since(item.enqueueTime).Seconds(), s.logger, "metrics.reqQueueTimeChan")
			return item.reqCtx
		}
	}

	return nil
}
