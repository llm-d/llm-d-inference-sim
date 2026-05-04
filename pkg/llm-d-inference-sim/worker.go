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
	reqChan common.Channel[requestContext]
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
			w.finishedChan <- &requestCompleted{worker: w, model: reqCtx.request().GetDisplayedModel()}
		}

	}
}

type requestProcessor interface {
	processRequest(reqCtx requestContext)
}

func (s *VllmSimulator) processRequest(reqCtx requestContext) {
	startTime := time.Now()
	req := reqCtx.request()
	respCtx, err := reqCtx.handleRequest()
	if err != nil {
		common.WriteToChannel(reqCtx.responseChannel(), &ResponseInfo{RespCtx: respCtx, Err: err},
			s.Context.logger)
		s.ResponseSentCallback(reqCtx)
		return
	}

	wg := sync.WaitGroup{}
	wg.Add(1)
	respCtx.setWG(&wg)

	s.sendResponse(respCtx)

	// Wait for the response to be fully sent, with a timeout to prevent
	// the worker from hanging indefinitely when the client doesn't consume the stream.
	doneCh := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneCh)
	}()

	const sendTimeout = 60 * time.Second
	select {
	case <-doneCh:
		// Response sent successfully
	case <-time.After(sendTimeout):
		s.Context.logger.Error(nil, "Timeout waiting for response to be sent, releasing worker",
			"id", req.GetRequestID(), "timeout", sendTimeout)
		s.ResponseSentCallback(reqCtx)
		return
	}

	common.WriteToChannel(s.Context.metrics.requestSuccessChan,
		requestSuccessEvent{
			promptTokens:     respCtx.UsageData().PromptTokens,
			generationTokens: respCtx.UsageData().CompletionTokens,
			// currently only responses with a single choice are supported
			genTokensPerChoice: []int{respCtx.UsageData().CompletionTokens},
			maxTokens:          req.GetMaxCompletionTokens(),
			finishReason:       *respCtx.FinishReason()},
		s.Context.logger)

	s.Context.logger.V(logging.DEBUG).Info("Finished processing request", "id", req.GetRequestID())

	// calculate inference time and finish e2e latency calculation only when sure that request processing was finished for streaming requests too
	common.WriteToChannel(s.Context.metrics.e2eReqLatencyChan, time.Since(reqCtx.startProcessingTime()).Seconds(), s.Context.logger)
	common.WriteToChannel(s.Context.metrics.reqInferenceTimeChan, time.Since(startTime).Seconds(), s.Context.logger)
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
