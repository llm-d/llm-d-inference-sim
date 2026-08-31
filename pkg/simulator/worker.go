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

package simulator

import (
	"context"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/endpoint"
)

// worker runs simulators requests
type worker struct {
	ctx    context.Context
	logger logr.Logger
	// worker's id
	id int
	// a channel for requests
	reqChan common.Channel[endpoint.RequestContext]
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
		case reqCtx := <-w.reqChan.Channel:
			w.processor.processRequest(reqCtx)
			w.finishedChan <- &requestCompleted{worker: w, model: reqCtx.Request().GetDisplayedModel()}
		}

	}
}

type requestProcessor interface {
	processRequest(reqCtx endpoint.RequestContext)
}

func (s *Simulator) processRequest(reqCtx endpoint.RequestContext) {
	defer s.onResponseProcessingFinished(reqCtx)

	startTime := time.Now()
	req := reqCtx.Request()
	respCtx, err := reqCtx.HandleRequest()
	if err != nil {
		common.WriteToChannel(reqCtx.ResponseChannel(),
			&endpoint.ResponseInfo{RespCtx: respCtx, Err: err, ChoiceIdx: reqCtx.ChoiceIndex()},
			s.Context.logger)
		return
	}

	s.simulateResponseProcessing(respCtx)
	s.Context.logger.V(logging.DEBUG).Info("Finished processing request", "id", req.GetRequestID())

	common.WriteToChannel(s.Context.metrics.requestSuccessChan,
		requestSuccessEvent{
			promptTokens:       respCtx.UsageData().PromptTokens,
			generationTokens:   respCtx.UsageData().CompletionTokens,
			genTokensPerChoice: []int{respCtx.UsageData().CompletionTokens},
			maxTokens:          req.GetMaxCompletionTokens(),
			finishReason:       *respCtx.FinishReason()},
		s.Context.logger)

	common.WriteToChannel(s.Context.metrics.e2eReqLatencyChan, time.Since(reqCtx.StartProcessingTime()).Seconds(), s.Context.logger)
	common.WriteToChannel(s.Context.metrics.reqInferenceTimeChan, time.Since(startTime).Seconds(), s.Context.logger)
}

// getFreeWorker returns a free worker or nil if none are available (non-blocking)
func (s *Simulator) getFreeWorker() *worker {
	select {
	case w := <-s.freeWorkers:
		return w
	default:
		return nil
	}
}
