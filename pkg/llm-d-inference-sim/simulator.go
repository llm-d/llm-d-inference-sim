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
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/valyala/fasthttp"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
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
	Context SimContext
	// schema validator for tools parameters
	toolsValidator *toolsValidator
	// indication whether the simulator is sleeping
	IsSleeping bool
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
	newRequests common.Channel[requestContext]
	// drainCancel cancels the internal drain context once all open requests finish,
	// allowing internal goroutines (workers, metrics, kvcache) to stop cleanly
	drainCancel context.CancelFunc
}

// New creates a new VllmSimulator instance with the given logger
func New(logger logr.Logger) (*VllmSimulator, error) {
	toolsValidator, err := createToolsValidator()
	if err != nil {
		return nil, fmt.Errorf("failed to create tools validator: %s", err)
	}

	return &VllmSimulator{
		toolsValidator: toolsValidator,
		Context: SimContext{
			logger: logger,
			loras: &lorasUsageInfo{
				loadedLoras: make(map[string]int),
			},
			kvcacheHelper: nil, // kvcache helper will be created only if required after reading configuration
		},
		waitingQueue: list.New(),
	}, nil
}

func Start(ctx context.Context, config *common.Configuration, logger logr.Logger) ([]*VllmSimulator, error) {
	if config.MMEncoderOnly && config.Mode == common.ModeEcho {
		logger.V(logging.WARN).Info("MM encoder-only mode: ignoring echo mode")
	}

	if err := dataset.Init(ctx, config, logger); err != nil {
		logger.Error(err, "failed to initialize dataset")
		return nil, err
	}
	// Use context.Background() so the tokenizer's HTTP connection to the render
	// backend is not tied to the parent context. Workers run on drainCtx (cancelled
	// only after all requests drain) and must be able to call the tokenizer until
	// the very end, after the parent context has already been cancelled.
	tokenizer, err := tokenizer.New(context.Background(), config, logger)
	if err != nil {
		logger.Error(err, "failed to initialize tokenizer")
		return nil, err
	}

	// Create data-parallel-size simulators
	dpSize := config.DPSize
	// If the rank was set, we ignore the data parallel size
	if config.Rank >= 0 {
		dpSize = 1
	}

	sims := make([]*VllmSimulator, dpSize)

	for dpRank := 0; dpRank < dpSize; dpRank++ {
		rankConfig := config
		if dpRank > 0 {
			rankConfig, err = config.Copy()
			if err != nil {
				return nil, err
			}
			rankConfig.Port = config.Port + dpRank
		}

		// Add the rank to the logger if dpSize > 1 or the rank was set,
		// i.e., don't add the rank if there is no data parallel
		loggerToUse := logger
		if config.Rank >= 0 {
			loggerToUse = klog.LoggerWithValues(logger, "rank", config.Rank)
		} else if dpSize != 1 {
			loggerToUse = klog.LoggerWithValues(logger, "rank", dpRank)
		}

		sim, err := New(loggerToUse)
		if err != nil {
			return nil, err
		}
		sim.Context.Config = rankConfig
		// use the same tokenizer in all ranks
		sim.Context.Tokenizer = tokenizer
		sims[dpRank] = sim
	}

	for _, sim := range sims {
		if err := sim.InitializeSim(ctx); err != nil {
			return nil, err
		}
	}
	return sims, nil
}

func (s *VllmSimulator) InitializeSim(ctx context.Context) error {
	// drainCtx lives until all open requests finish after ctx is cancelled.
	// It is passed to all internal goroutines so they keep running through drain.
	drainCtx, drainCancel := context.WithCancel(context.Background())
	s.drainCancel = drainCancel

	if err := s.Context.initialize(drainCtx); err != nil {
		drainCancel()
		return err
	}

	s.queueCapacity = s.Context.Config.MaxWaitingQueueLength

	maxNumberOfRequests := s.Context.Config.MaxNumSeqs + s.Context.Config.MaxWaitingQueueLength
	s.newRequests = common.Channel[requestContext]{
		Channel: make(chan requestContext, maxNumberOfRequests),
		Name:    "newRequests",
	}

	// run request processing workers
	s.freeWorkers = make(chan *worker, s.Context.Config.MaxNumSeqs)
	s.workerFinished = make(chan *requestCompleted, s.Context.Config.MaxNumSeqs)
	for i := 1; i <= s.Context.Config.MaxNumSeqs; i++ {
		worker := &worker{
			id:           i,
			ctx:          drainCtx,
			logger:       s.Context.logger,
			finishedChan: s.workerFinished,
			reqChan: common.Channel[requestContext]{
				Channel: make(chan requestContext, 1),
				Name:    "worker's reqChan",
			},
			processor: s,
		}
		go worker.waitForRequests()
		s.freeWorkers <- worker
	}

	go s.processing(drainCtx)
	return nil
}

// Stop cancels the internal drain context, causing all internal goroutines
// (workers, metrics, kvcache) to stop cleanly. It must be called by the
// communication layer after all open requests have been drained.
func (s *VllmSimulator) Stop() {
	s.drainCancel()
}

func (s *VllmSimulator) processing(ctx context.Context) {
	s.Context.logger.V(logging.INFO).Info("Start processing routine")

	for {
		select {
		case <-ctx.Done():
			s.Context.logger.V(logging.INFO).Info("Request processing done")
			return
		case completedReq := <-s.workerFinished:
			worker := completedReq.worker
			s.Context.logger.V(logging.TRACE).Info("Worker finished", "worker", worker.id)
			s.Context.decrementLora(completedReq.model)
			// there is a free worker - find a request for it and send this request for
			// processing with this worker
			s.findRequestAndSendToProcess(worker)
		case <-s.Context.loras.loraRemovable.Channel:
			// there is a LoRA that can be removed, go through availbale workers
			// and queued requests and find requests that can run now,
			// stop if there are no free workers, or no requests
			s.Context.logger.V(logging.TRACE).Info("LoRA can be removed")
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
		case reqCtx := <-s.newRequests.Channel:
			// A new request was received. Find a free worker, and check that the request can run LoRA wise.
			worker := s.getFreeWorker()
			if worker == nil {
				// use real model name for the logs
				s.Context.logger.V(logging.TRACE).Info("No free worker - sending the request to the waiting queue",
					"model", reqCtx.request().GetModel(), "req id", reqCtx.request().GetRequestID())
				// no free worker, add this request to the waiting queue
				s.addRequestToQueue(reqCtx)
				break
			}

			// check if lora usage allows the request to run
			if s.Context.isLora(reqCtx.request().GetDisplayedModel()) && !s.Context.loadLora(reqCtx.request().GetDisplayedModel()) {
				// free the worker
				s.freeWorkers <- worker
				s.Context.logger.V(logging.TRACE).Info("LoRA cannot be loaded - sending the request to the waiting queue",
					"LoRA", reqCtx.request().GetDisplayedModel(), "req id", reqCtx.request().GetRequestID())
				// LoRA max reached, try to enqueue
				s.addRequestToQueue(reqCtx)
				break
			}

			s.Context.logger.V(logging.TRACE).Info("Sending the request to the processing channel", "model", reqCtx.request().GetModel(),
				"req id", reqCtx.request().GetRequestID(), "worker", worker.id)
			common.WriteToChannel(worker.reqChan, reqCtx, s.Context.logger)
		}
	}
}

func (s *VllmSimulator) findRequestAndSendToProcess(worker *worker) bool {
	nextReq := s.dequeue()
	if nextReq != nil {
		// send this request for processing in this worker
		// use model defined in the request for the logs to present real model name in the logs
		s.Context.logger.V(logging.TRACE).Info("Sending request to processing", "model", nextReq.request().GetModel(),
			"req", nextReq.request().GetRequestID(), "worker", worker.id)
		common.WriteToChannel(worker.reqChan, nextReq, s.Context.logger)
		// decrement waiting requests metric
		common.WriteToChannel(s.Context.metrics.waitingReqChan, common.MetricInfo{Value: -1}, s.Context.logger)
		return true
	}

	// no waiting request, return worker to be free
	s.freeWorkers <- worker
	return false
}

func (s *VllmSimulator) addRequestToQueue(reqCtx requestContext) {
	if err := s.enqueue(reqCtx); err != nil {
		s.Context.logger.Error(err, "failed to enqueue request")
		err := openaiserverapi.NewError("Failed to enqueue request, "+err.Error(),
			fasthttp.StatusTooManyRequests, nil)
		common.WriteToChannel(reqCtx.responseChannel(), &ResponseInfo{Err: &err},
			s.Context.logger)
		return
	}
	// increment the waiting requests metric
	common.WriteToChannel(s.Context.metrics.waitingReqChan, common.MetricInfo{Value: 1}, s.Context.logger)
	// update loraInfo metrics with the new waiting request
	if s.Context.isLora(reqCtx.request().GetDisplayedModel()) {
		common.WriteToChannel(s.Context.metrics.lorasChan, loraUsage{reqCtx.request().GetDisplayedModel(), waitingUsageState},
			s.Context.logger)
	}
}

func (s *VllmSimulator) HandleRequest(req Request) (bool, *common.Channel[*ResponseInfo], *openaiserverapi.Error, bool) {
	// Check if we should inject a failure
	if shouldInjectFailure(s.Context.Config, s.Context.Random) {
		failure := getRandomFailure(s.Context.Config, s.Context.Random)
		return false, nil, &failure, true
	}

	// the model defined in the request should be checked here
	if !s.isValidModel(req.GetModel()) {
		err := openaiserverapi.NewError(fmt.Sprintf("The model `%s` does not exist.",
			req.GetModel()), fasthttp.StatusNotFound, nil)
		return false, nil, &err, false
	}

	// the model is valid, update the displayed model which will be different from the model mentioned in the request only in case of base model aliases
	// in this case the first alias is used, in all other cases the model from the request is used as the displayed model
	req.SetDisplayedModel(s.Context.getDisplayedModelName(req.GetModel()))

	errMsg, errCode := req.validate(s.toolsValidator)
	if errMsg != "" {
		err := openaiserverapi.NewError(errMsg, errCode, nil)
		return false, nil, &err, false
	}

	channel := common.Channel[*ResponseInfo]{
		Channel: make(chan *ResponseInfo, s.Context.Config.MaxModelLen),
		Name:    "responseInfo",
	}
	reqCtx := req.buildRequestContext(&s.Context, channel)
	common.WriteToChannel(s.newRequests, reqCtx, s.Context.logger)
	return req.IsStream(), &channel, nil, false
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
		if ok && item.reqCtx != nil && s.Context.loraIsLoaded(item.reqCtx.request().GetDisplayedModel()) {
			s.waitingQueue.Remove(elem)
			s.Context.incrementLora(item.reqCtx.request().GetDisplayedModel())
			common.WriteToChannel(s.Context.metrics.reqQueueTimeChan, time.Since(item.enqueueTime).Seconds(),
				s.Context.logger)
			return item.reqCtx
		}
	}

	// All the requests require a LoRA that is not loaded, check if we can load a LoRA
	for elem := s.waitingQueue.Front(); elem != nil; elem = elem.Next() {
		item, ok := elem.Value.(waitingQueueItem)
		if ok && item.reqCtx != nil && s.Context.loadLora(item.reqCtx.request().GetDisplayedModel()) {
			s.waitingQueue.Remove(elem)
			common.WriteToChannel(s.Context.metrics.reqQueueTimeChan, time.Since(item.enqueueTime).Seconds(),
				s.Context.logger)
			return item.reqCtx
		}
	}

	return nil
}

func (s *VllmSimulator) simulateResponseProcessing(respCtx ResponseContext) {
	reqCtx := respCtx.RequestContext()
	// Skip delays if finish reason is cache_threshold (immediate return)
	if respCtx.FinishReason() != nil && *respCtx.FinishReason() == common.CacheThresholdFinishReason {
		common.WriteToChannel(reqCtx.responseChannel(), &ResponseInfo{RespCtx: respCtx},
			s.Context.logger)
	} else {
		s.Context.simulateTTFT(respCtx)

		// Response started
		common.WriteToChannel(reqCtx.responseChannel(),
			&ResponseInfo{RespCtx: respCtx, Status: ResponseStatusCreated},
			s.Context.logger)

		startDecode := time.Now()
		if respIsEmpty(respCtx) {
			common.WriteToChannel(reqCtx.responseChannel(),
				&ResponseInfo{RespCtx: respCtx}, s.Context.logger)
		} else {
			if respCtx.responseTokens() != nil {
				for i, token := range respCtx.responseTokens().Tokens {
					if i != 0 {
						s.Context.simulateInterTokenLatency()
					}

					tokens := &openaiserverapi.Tokenized{
						Tokens:  []uint32{token},
						Strings: []string{},
					}
					if respCtx.responseTokens().Strings != nil {
						tokens.Strings = append(tokens.Strings, respCtx.responseTokens().Strings[i])
					}
					common.WriteToChannel(reqCtx.responseChannel(),
						&ResponseInfo{Tokens: tokens, RespCtx: respCtx},
						s.Context.logger)
				}
			} else {
				for _, tc := range respCtx.ToolCalls() {
					// Tool calls are only supported in HTTP at the moment, so we assume that we always
					// have string tokenized arguments
					for i, token := range tc.Function.TokenizedArguments().Tokens {
						if i != 0 {
							s.Context.simulateInterTokenLatency()
						}
						common.WriteToChannel(reqCtx.responseChannel(),
							&ResponseInfo{Tokens: &openaiserverapi.Tokenized{
								Tokens:  []uint32{token},
								Strings: []string{tc.Function.TokenizedArguments().Strings[i]}},
								RespCtx: respCtx, ToolCall: &tc}, s.Context.logger)
					}
				}
			}
		}
		common.WriteToChannel(s.Context.metrics.reqDecodeTimeChan, time.Since(startDecode).Seconds(), s.Context.logger)
	}
	close(reqCtx.responseChannel().Channel)
}

// request processing finished
func (s *VllmSimulator) onResponseProcessingFinished(reqCtx requestContext) {
	// decrement running requests count
	common.WriteToChannel(s.Context.metrics.runReqChan, common.MetricInfo{Value: -1}, s.Context.logger)

	model := reqCtx.request().GetDisplayedModel()
	if s.Context.isLora(model) {
		// update loraInfo metrics to reflect that the request processing has been finished
		common.WriteToChannel(s.Context.metrics.lorasChan, loraUsage{model, doneUsageState},
			s.Context.logger)
	}

	reqCtx.kvCacheOnRequestEnd()
}

// OpenRequests returns the number of requests currently in the system
// (waiting in queue + being processed by workers).
func (s *VllmSimulator) OpenRequests() int64 {
	s.queueLock.Lock()
	waiting := s.waitingQueue.Len()
	s.queueLock.Unlock()
	running := cap(s.freeWorkers) - len(s.freeWorkers)
	return int64(waiting + running)
}

func (s *VllmSimulator) DiscardKVCache() {
	s.Context.kvcacheHelper.Discard()
}

func (s *VllmSimulator) ActivateKVCache() {
	s.Context.kvcacheHelper.Activate()
}
