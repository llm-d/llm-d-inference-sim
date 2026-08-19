/*
Copyright 2026 The llm-d-inference-sim Authors.

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
	"os"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
	"k8s.io/klog/v2"
)

// newHandleRequestTestSim builds a VllmSimulator with a real (echo-mode)
// dataset, tokenizer, latency calculator and metrics -- everything
// HandleRequest/processRequest touch -- via SimContext.initialize, but
// deliberately skips InitializeSim. InitializeSim also starts the
// processing() goroutine and worker pool, which would immediately start
// draining newRequests as fast as items arrive; that makes it effectively
// impossible to deterministically observe newRequests full. Building
// newRequests by hand here, with nothing consuming it, keeps this test in
// full control of the channel's occupancy and lets it drive worker-side
// processing (sim.processRequest) explicitly, one call at a time.
func newHandleRequestTestSim(ctx context.Context, newRequestsCapacity int, extraArgs ...string) *VllmSimulator {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = append([]string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}, extraArgs...)

	config, err := common.ParseCommandParamsAndLoadConfig()
	Expect(err).NotTo(HaveOccurred())

	sim, err := New(klog.Background())
	Expect(err).NotTo(HaveOccurred())
	sim.Context.SetConfig(config)
	sim.Context.Tokenizer = tokenizer.NewSimpleTokenizer()

	Expect(sim.Context.initialize(ctx)).To(Succeed())

	sim.newRequests = common.Channel[requestContext]{
		Channel: make(chan requestContext, newRequestsCapacity),
		Name:    "newRequests",
	}
	return sim
}

func newTextCompletionsRequestWithChoices(id string, n int) *TextCompletionsParsedRequest {
	req := &TextCompletionsParsedRequest{}
	req.RequestID = id
	req.Model = common.TestModelName
	req.Prompt = []api.PromptInput{{Text: "hi"}}
	req.N = ptrInt(n)
	return req
}

var _ = Describe("HandleRequest enqueue failure", func() {
	It("returns a 429 error instead of dropping the request when newRequests is full", func() {
		sim := newHandleRequestTestSim(context.Background(), 1)

		// The first request fills the only slot in newRequests; nothing drains
		// it, so it stays there.
		numChoices, _, fillChan, topErr, _ := sim.HandleRequest(newTextCompletionsRequestWithChoices("fill", 1))
		Expect(topErr).To(BeNil())
		Expect(numChoices).To(Equal(1))
		Expect(fillChan).NotTo(BeNil())
		Expect(sim.newRequests.Channel).To(HaveLen(1))

		// The second request has nowhere to go: its sub-request must come back
		// as a 429 error on its own response channel instead of being silently
		// dropped.
		numChoices, _, overflowChan, topErr, _ := sim.HandleRequest(newTextCompletionsRequestWithChoices("overflow", 1))
		Expect(topErr).To(BeNil())
		Expect(numChoices).To(Equal(1))

		var response *ResponseInfo
		Expect(overflowChan.Channel).To(Receive(&response))
		Expect(response.Err).NotTo(BeNil())
		Expect(response.Err.Code).To(Equal(fasthttp.StatusTooManyRequests))
		Expect(response.Err.Message).To(ContainSubstring("channel is full"))

		// The failed sub-request must still signal done, so its response
		// channel closes rather than leaving the (would-be) HTTP reader
		// blocked forever. The actual close() runs on the background goroutine
		// HandleRequest spawns for this (wg.Wait(); close(...)), so give it a
		// moment rather than asserting synchronously.
		Eventually(overflowChan.Channel).Should(BeClosed())
	})

	It("does not affect sibling choices that already made it into newRequests and are being processed", func() {
		sim := newHandleRequestTestSim(context.Background(), 2)

		// One request, 3 choices, newRequests capacity 2: choices 0 and 1 fit,
		// choice 2 overflows. Nothing drains newRequests concurrently here (no
		// processing() goroutine is running), so this split is deterministic.
		numChoices, _, respChan, topErr, _ := sim.HandleRequest(newTextCompletionsRequestWithChoices("multi", 3))
		Expect(topErr).To(BeNil())
		Expect(numChoices).To(Equal(3))
		Expect(sim.newRequests.Channel).To(HaveLen(2))

		// Simulate the two surviving sub-requests already having been picked up
		// and run by a worker, by draining them from newRequests and running the
		// real worker-side processing on them directly.
		reqCtx1 := <-sim.newRequests.Channel
		reqCtx2 := <-sim.newRequests.Channel
		sim.processRequest(reqCtx1)
		sim.processRequest(reqCtx2)

		// respChan is shared by all 3 choices of this one request; it closes once
		// all 3 have signalled done (2 via real completion above, 1 via the
		// immediate enqueue-failure path).
		var errCount, completedCount int
		var sawErr *api.Error
		for response := range respChan.Channel {
			if response.Err != nil {
				errCount++
				sawErr = response.Err
				continue
			}
			if response.Status == ResponseEndOfTokens {
				completedCount++
			}
		}

		Expect(errCount).To(Equal(1))
		Expect(sawErr.Code).To(Equal(fasthttp.StatusTooManyRequests))
		Expect(sawErr.Message).To(ContainSubstring("channel is full"))
		// The two choices that made it into newRequests must complete normally
		// (one ResponseEndOfTokens marker each), unaffected by their sibling's
		// enqueue failure.
		Expect(completedCount).To(Equal(2))
	})
})
