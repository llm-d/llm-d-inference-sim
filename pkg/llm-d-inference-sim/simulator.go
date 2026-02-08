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
	"net"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/soheilhy/cmux"
	"github.com/valyala/fasthttp"
	"golang.org/x/sync/errgroup"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	"github.com/llm-d/llm-d-inference-sim/pkg/grpc/pb"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

const (
	textCompletionObject      = "text_completion"
	chatCompletionObject      = "chat.completion"
	chatCompletionChunkObject = "chat.completion.chunk"

	podHeader                        = "x-inference-pod"
	portHeader                       = "x-inference-port"
	namespaceHeader                  = "x-inference-namespace"
	requestIDHeader                  = "X-Request-Id"
	cacheThresholdFinishReasonHeader = "X-Cache-Threshold-Finish-Reason"
	podNameEnv                       = "POD_NAME"
	podNsEnv                         = "POD_NAMESPACE"
)

type requestCompleted struct {
	worker *worker
	model  string
}

type waitingQueueItem struct {
	reqCtx      requestContext
	enqueueTime time.Time
}

// VllmSimulator simulates vLLM server supporting OpenAI API
type VllmSimulator struct {
	pb.UnimplementedVllmEngineServer

	context simContext
	// schema validator for tools parameters
	toolsValidator *toolsValidator

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
	// bi-directional list of requestContext
	waitingQueue *list.List
	// the max capacity of the waiting requests queue
	queueCapacity int
	// a channel for incoming requests
	newRequests chan requestContext
}

// New creates a new VllmSimulator instance with the given logger
func New(logger logr.Logger) (*VllmSimulator, error) {
	toolsValidator, err := createToolsValidator()
	if err != nil {
		return nil, fmt.Errorf("failed to create tools validator: %s", err)
	}

	return &VllmSimulator{
		toolsValidator: toolsValidator,
		isInDevMode:    os.Getenv("VLLM_SERVER_DEV_MODE") == "1",
		context: simContext{
			logger: logger,
			loras: &lorasUsageInfo{
				loadedLoras: make(map[string]int),
			},
			kvcacheHelper: nil, // kvcache helper will be created only if required after reading configuration
			namespace:     os.Getenv(podNsEnv),
			pod:           os.Getenv(podNameEnv),
		},
		waitingQueue: list.New(),
	}, nil
}

// Start starts the simulator
func (s *VllmSimulator) Start(ctx context.Context) error {
	var err error
	// parse command line parameters
	s.context.config, err = common.ParseCommandParamsAndLoadConfig()
	if err != nil {
		return err
	}

	err = s.showConfig(s.context.config.DPSize > 1)
	if err != nil {
		return err
	}

	if s.context.config.DatasetURL != "" && s.context.config.Model != common.ModeEcho {
		// if should use remote responses dataset, download it first (it can take time)
		downloader := dataset.NewDsDownloader(s.context.logger)
		if err := downloader.DownloadDataset(ctx, s.context.config.DatasetURL, s.context.config.DatasetPath); err != nil {
			return err
		}
	}

	// create tokenizer in advance and use it in all ranks (because of python initialization problem)
	if err := s.context.initTokenizer(); err != nil {
		return err
	}

	// For Data Parallel, start data-parallel-size - 1 additional simulators
	g, ctx := errgroup.WithContext(ctx)
	if s.context.config.DPSize > 1 {
		for i := 2; i <= s.context.config.DPSize; i++ {
			newConfig, err := s.context.config.Copy()
			if err != nil {
				return err
			}
			dpRank := i - 1
			newConfig.Port = s.context.config.Port + dpRank
			newSim, err := New(klog.LoggerWithValues(s.context.logger, "rank", dpRank))
			if err != nil {
				return err
			}
			newSim.context.config = newConfig
			// use the same tokenizer in all ranks
			newSim.context.tokenizer = s.context.tokenizer
			g.Go(func() error {
				return newSim.startSim(ctx)
			})
		}
		s.context.logger = klog.LoggerWithValues(s.context.logger, "rank", 0)
	} else if s.context.config.Rank >= 0 {
		s.context.logger = klog.LoggerWithValues(s.context.logger, "rank", s.context.config.Rank)
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
		s.context.logger.Error(err, "failed to create listener")
		return fmt.Errorf("listener creation error: %w", err)
	}

	m := cmux.New(listener)

	// gRPC uses HTTP/2
	grpcL := m.Match(cmux.HTTP2())
	httpL := m.Match(cmux.Any())

	// start the gRPC server
	errCh := make(chan error, 1)
	go func() {
		errCh <- s.startGRPC(ctx, grpcL)
	}()

	select {
	case err := <-errCh:
		if err != nil {
			return err
		}
	default:
	}

	// start the http server with context support
	errCh = make(chan error, 1)
	go func() {
		errCh <- s.startServer(ctx, httpL)
	}()

	select {
	case err := <-errCh:
		if err != nil {
			return err
		}
	default:
	}

	err = m.Serve()
	if !errors.Is(err, net.ErrClosed) {
		return fmt.Errorf("cmux failed: %w", err)
	}
	return nil
}

func (s *VllmSimulator) initializeSim(ctx context.Context) error {
	if err := s.context.initialize(ctx); err != nil {
		return err
	}

	s.queueCapacity = s.context.config.MaxWaitingQueueLength

	maxNumberOfRequests := s.context.config.MaxNumSeqs + s.context.config.MaxWaitingQueueLength
	s.newRequests = make(chan requestContext, maxNumberOfRequests)

	// run request processing workers
	s.freeWorkers = make(chan *worker, s.context.config.MaxNumSeqs)
	s.workerFinished = make(chan *requestCompleted, s.context.config.MaxNumSeqs)
	for i := 1; i <= s.context.config.MaxNumSeqs; i++ {
		worker := &worker{
			id:           i,
			ctx:          ctx,
			logger:       s.context.logger,
			finishedChan: s.workerFinished,
			reqChan:      make(chan requestContext, 1),
			processor:    s,
		}
		go worker.waitForRequests()
		s.freeWorkers <- worker
	}

	go s.processing(ctx)
	return nil
}

// Print prints to a log, implementation of fasthttp.Logger
func (s *VllmSimulator) Printf(format string, args ...interface{}) {
	s.context.logger.V(logging.WARN).Info("Server error", "msg", fmt.Sprintf(format, args...))
}

func (s *VllmSimulator) processing(ctx context.Context) {
	s.context.logger.V(logging.INFO).Info("Start processing routine")

	for {
		select {
		case <-ctx.Done():
			s.context.logger.V(logging.INFO).Info("Request processing done")
			return
		case completedReq := <-s.workerFinished:
			worker := completedReq.worker
			s.context.logger.V(logging.TRACE).Info("Worker finished", "worker", worker.id)
			s.context.decrementLora(completedReq.model)
			// there is a free worker - find a request for it and send this request for
			// processing with this worker
			s.findRequestAndSendToProcess(worker)
		case <-s.context.loras.loraRemovable:
			// there is a LoRA that can be removed, go through availbale workers
			// and queued requests and find requests that can run now,
			// stop if there are no free workers, or no requests
			s.context.logger.V(logging.TRACE).Info("LoRA can be removed")
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
			model := reqCtx.request().GetModel()

			worker := s.getFreeWorker()
			if worker == nil {
				s.context.logger.V(logging.TRACE).Info("No free worker - sending the request to the waiting queue",
					"model", model, "req id", reqCtx.request().GetRequestID())
				// no free worker, add this request to the waiting queue
				s.addRequestToQueue(reqCtx)
				break
			}

			// check if lora usage allows the request to run
			if s.context.isLora(model) && !s.context.loadLora(model) {
				// free the worker
				s.freeWorkers <- worker
				s.context.logger.V(logging.TRACE).Info("LoRA cannot be loaded - sending the request to the waiting queue",
					"LoRA", model, "req id", reqCtx.request().GetRequestID())
				// LoRA max reached, try to enqueue
				s.addRequestToQueue(reqCtx)
				break
			}

			s.context.logger.V(logging.TRACE).Info("Sending the request to the processing channel", "model", model,
				"req id", reqCtx.request().GetRequestID(), "worker", worker.id)
			common.WriteToChannel(worker.reqChan, reqCtx, s.context.logger, "worker's reqChan")
		}
	}
}

func (s *VllmSimulator) findRequestAndSendToProcess(worker *worker) bool {
	nextReq := s.dequeue()
	if nextReq != nil {
		// send this request for processing in this worker
		s.context.logger.V(logging.TRACE).Info("Sending request to processing", "model", nextReq.request().GetModel(),
			"req", nextReq.request().GetRequestID(), "worker", worker.id)
		common.WriteToChannel(worker.reqChan, nextReq, s.context.logger, "worker's reqChan")
		// decrement waiting requests metric
		common.WriteToChannel(s.context.metrics.waitingReqChan, -1, s.context.logger, "metrics.waitingReqChan")
		return true
	}

	// no waiting request, return worker to be free
	s.freeWorkers <- worker
	return false
}

func (s *VllmSimulator) addRequestToQueue(reqCtx requestContext) {
	if err := s.enqueue(reqCtx); err != nil {
		s.context.logger.Error(err, "failed to enqueue request")
		reqCtx.responseSender().sendError(openaiserverapi.NewError("Failed to enqueue request, "+err.Error(),
			fasthttp.StatusTooManyRequests, nil), false)
		reqCtx.done()
		return
	}
	// increment the waiting requests metric
	common.WriteToChannel(s.context.metrics.waitingReqChan, 1, s.context.logger, "metrics.waitingReqChan")
	// update loraInfo metrics with the new waiting request
	common.WriteToChannel(s.context.metrics.lorasChan, loraUsage{reqCtx.request().GetModel(), waitingUsageState},
		s.context.logger, "metrics.lorasChan")

}

func (s *VllmSimulator) handleHTTPRequest(req request, ctx *fasthttp.RequestCtx) {
	respSender := &httpResponseSender{
		baseResponseSender: baseResponseSender{
			sim: &s.context,
		},
		ctx: ctx,
	}

	if err := req.unmarshal(ctx.Request.Body()); err != nil {
		s.context.logger.Error(err, "failed to read and parse request body")
		ctx.Error("Failed to read and parse request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	requestID := s.getRequestID(ctx)
	req.SetRequestID(requestID)

	// Check for cache threshold finish reason header - this forces a cache_threshold finish reason
	headerValue := string(ctx.Request.Header.Peek(cacheThresholdFinishReasonHeader))
	if parsedValue, err := strconv.ParseBool(headerValue); err == nil {
		req.SetCacheThresholdFinishReason(parsedValue)
	}

	s.handleRequest(req, respSender)
}

// handleRequest is a general requests handler
// Important note: for requests in streaming mode, this function exits before all chunks are sent to the client
func (s *VllmSimulator) handleRequest(req request, respSender responseSender) {
	// Check if we should inject a failure
	if shouldInjectFailure(s.context.config, s.context.random) {
		failure := getRandomFailure(s.context.config, s.context.random)
		respSender.sendError(failure, true)
		return
	}

	if !s.isValidModel(req.GetModel()) {
		respSender.sendError(openaiserverapi.NewError(fmt.Sprintf("The model `%s` does not exist.",
			req.GetModel()), fasthttp.StatusNotFound, nil), false)
		return
	}

	errMsg, errCode := req.validate(s.toolsValidator)
	if errMsg != "" {
		respSender.sendError(openaiserverapi.NewError(errMsg, errCode, nil), false)
		return
	}

	s.context.logger.V(logging.DEBUG).Info("Received", "new", respSender.stringForRequest(req))

	var wg sync.WaitGroup
	wg.Add(1)
	reqCtx := req.buildRequestContext(&s.context, respSender, &wg)
	common.WriteToChannel(s.newRequests, reqCtx, s.context.logger, "newRequests")
	wg.Wait()
}

// sendResponse sends response for completion API, supports both completions (text and chat)
func (s *VllmSimulator) sendResponse(reqCtx requestContext, respCtx responseContext) {
	resp := respCtx.createResponse()

	// Skip delays if finish reason is cache_threshold (immediate return)
	isCacheThresholdFinishReason := respCtx.finishReason() != nil && *respCtx.finishReason() == common.CacheThresholdFinishReason
	if !isCacheThresholdFinishReason {
		s.context.simulateTTFT(respCtx)

		startDecode := time.Now()
		for range respCtx.usageData().CompletionTokens - 1 {
			s.context.simulateInterTokenLatency()
		}
		common.WriteToChannel(s.context.metrics.reqDecodeTimeChan, time.Since(startDecode).Seconds(), s.context.logger, "metrics.reqDecodeTimeChan")
	}

	reqCtx.responseSender().sendResponse(resp, respCtx, reqCtx)
	reqCtx.responseSender().responseSentCallback(reqCtx, respCtx.displayModel())
}

// createModelsResponse creates and returns ModelResponse for the current state, returned array of models contains the base model + LoRA adapters if exist
func (s *VllmSimulator) createModelsResponse() *vllmapi.ModelsResponse {
	modelsResp := vllmapi.ModelsResponse{Object: "list", Data: []vllmapi.ModelsResponseModelInfo{}}

	// Advertise every public model alias
	for _, alias := range s.context.config.ServedModelNames {
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
	parent := s.context.config.ServedModelNames[0]
	for _, lora := range s.context.getLoras() {
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

func (s *VllmSimulator) enqueue(req requestContext) error {
	s.queueLock.Lock()
	defer s.queueLock.Unlock()

	if s.waitingQueue.Len() >= s.queueCapacity {
		return errors.New("waiting requests queue is full")
	}
	s.waitingQueue.PushBack(waitingQueueItem{req, time.Now()})
	return nil
}

// go though the queue and find the first request that can be executed, while taking into consideration the max lora limitation
func (s *VllmSimulator) dequeue() requestContext {
	s.queueLock.Lock()
	defer s.queueLock.Unlock()

	// Find first request for a loaded LoRA
	for elem := s.waitingQueue.Front(); elem != nil; elem = elem.Next() {
		item, ok := elem.Value.(waitingQueueItem)
		if ok && item.reqCtx != nil && s.context.loraIsLoaded(item.reqCtx.request().GetModel()) {
			s.waitingQueue.Remove(elem)
			s.context.incrementLora(item.reqCtx.request().GetModel())
			common.WriteToChannel(s.context.metrics.reqQueueTimeChan, time.Since(item.enqueueTime).Seconds(),
				s.context.logger, "metrics.reqQueueTimeChan")
			return item.reqCtx
		}
	}

	// All the requests require a LoRA that is not loaded, check if we can load a LoRA
	for elem := s.waitingQueue.Front(); elem != nil; elem = elem.Next() {
		item, ok := elem.Value.(waitingQueueItem)
		if ok && item.reqCtx != nil && s.context.loadLora(item.reqCtx.request().GetModel()) {
			s.waitingQueue.Remove(elem)
			common.WriteToChannel(s.context.metrics.reqQueueTimeChan, time.Since(item.enqueueTime).Seconds(),
				s.context.logger, "metrics.reqQueueTimeChan")
			return item.reqCtx
		}
	}

	return nil
}
