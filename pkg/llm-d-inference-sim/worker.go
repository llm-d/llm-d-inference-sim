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

package llmdinferencesim

import (
	"context"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
)

// worker runs simulators requests
type worker struct {
	ctx    context.Context
	logger logr.Logger
	// worker's id
	id int
	// a channel for requests
	reqChan chan requestContext
	// a channel to indicate that the worker finished processing a request
	finishedChan chan *requestCompleted
	// the request processor
	processor requestProcessor
}

func (w *worker) waitForRequests() {
	for {
		select {
		case <-w.ctx.Done():
			w.logger.V(logging.TRACE).Info("worker done", "id", w.id)
			return
		case reqCtx := <-w.reqChan:
			w.processor.processRequest(reqCtx, nil)
			w.finishedChan <- &requestCompleted{worker: w, model: reqCtx.request().GetModel()}
		}

	}
}

type requestProcessor interface {
	processRequest(reqCtx requestContext, wg *sync.WaitGroup)
}

func (s *VllmSimulator) processRequest(reqCtx requestContext, _ *sync.WaitGroup) {
	startTime := time.Now()
	wg := sync.WaitGroup{}
	wg.Add(1)
	req := reqCtx.request()
	respCtx, err := reqCtx.handleRequest()
	if err != nil {
		common.WriteToChannel(reqCtx.responseChannel(), &responseInfo{respCtx: respCtx, err: err, injected: false},
			s.context.logger, "responseChannel")
		return
	}

	respCtx.setWG(&wg)

	s.sendResponse(reqCtx, respCtx)

	wg.Wait()

	common.WriteToChannel(s.context.metrics.requestSuccessChan,
		requestSuccessEvent{
			promptTokens:     respCtx.usageData().PromptTokens,
			generationTokens: respCtx.usageData().CompletionTokens,
			// currently only responses with a single choice are supported
			genTokensPerChoice: []int{respCtx.usageData().CompletionTokens},
			maxTokens:          req.GetMaxCompletionTokens(),
			finishReason:       *respCtx.finishReason()},
		s.context.logger, "metrics.requestSuccessChan")

	s.context.logger.V(logging.DEBUG).Info("Finished processing request", "id", req.GetRequestID())

	// calculate inference time and finish e2e latency calculation only when sure that request processing was finished for streaming requests too
	common.WriteToChannel(s.context.metrics.e2eReqLatencyChan, time.Since(reqCtx.startProcessingTime()).Seconds(), s.context.logger, "metrics.e2eReqLatencyChan")
	common.WriteToChannel(s.context.metrics.reqInferenceTimeChan, time.Since(startTime).Seconds(), s.context.logger, "metrics.reqInferenceTimeChan")
}

// getFreeWorker returns a free worker or nil if none are available (non-blocking)
func (s *VllmSimulator) getFreeWorker() *worker {
	select {
	case w := <-s.freeWorkers:
		return w
	default:
		return nil
	}
}
