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

package simulator

import (
	"context"
	"os"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/endpoint"
	"github.com/llm-d/llm-d-inference-sim/pkg/engine/vllm"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/valyala/fasthttp"
	"k8s.io/klog/v2"
)

func ptrInt(v int) *int { return &v }

// newHandleRequestTestSim builds a Simulator with a real (echo-mode)
// dataset, tokenizer, latency calculator and metrics -- everything
// HandleRequest/processRequest touch -- via SimContext.initialize, but
// deliberately skips InitializeSim. InitializeSim also starts the
// processing() goroutine and worker pool, which would immediately start
// draining newRequests as fast as items arrive; that makes it effectively
// impossible to deterministically observe newRequests full. Building
// newRequests by hand here, with nothing consuming it, keeps this test in
// full control of the channel's occupancy and lets it drive worker-side
// processing (sim.processRequest) explicitly, one call at a time.
func newHandleRequestTestSim(ctx context.Context, newRequestsCapacity int, extraArgs ...string) *Simulator {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = append([]string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}, extraArgs...)

	engine := vllm.New()
	config, err := common.ParseCommandParamsAndLoadConfig(engine)
	Expect(err).NotTo(HaveOccurred())

	sim, err := New(klog.Background())
	Expect(err).NotTo(HaveOccurred())
	sim.Context.SetConfig(config)
	sim.Context.Tokenizer = tokenizer.NewSimpleTokenizer()
	sim.Context.Engine = engine

	Expect(sim.Context.initialize(ctx)).To(Succeed())

	sim.newRequests = common.Channel[endpoint.RequestContext]{
		Channel: make(chan endpoint.RequestContext, newRequestsCapacity),
		Name:    "newRequests",
	}
	return sim
}

// useUnreadMetricsBus gives sim a metrics bus that nothing drains, so a test
// can read the events that processing emits.
func useUnreadMetricsBus(sim *Simulator) *metrics.MetricsBus {
	bus, err := metrics.NewMetricsBus(context.Background(), *sim.Context.Config(), prometheus.NewRegistry(),
		klog.Background(), metrics.NewStubAdapter)
	Expect(err).NotTo(HaveOccurred())
	sim.Context.metricsBus = bus
	return bus
}

func newTextCompletionsRequestWithChoices(id string, n int) *endpoint.TextCompletionsParsedRequest {
	req := &endpoint.TextCompletionsParsedRequest{}
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
		numChoices, _, fillChan, topErr, _ := sim.HandleRequest(context.Background(), newTextCompletionsRequestWithChoices("fill", 1))
		Expect(topErr).To(BeNil())
		Expect(numChoices).To(Equal(1))
		Expect(fillChan).NotTo(BeNil())
		Expect(sim.newRequests.Channel).To(HaveLen(1))

		// The second request has nowhere to go: its sub-request must come back
		// as a 429 error on its own response channel instead of being silently
		// dropped.
		numChoices, _, overflowChan, topErr, _ := sim.HandleRequest(context.Background(), newTextCompletionsRequestWithChoices("overflow", 1))
		Expect(topErr).To(BeNil())
		Expect(numChoices).To(Equal(1))

		var response *endpoint.ResponseInfo
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
		numChoices, _, respChan, topErr, _ := sim.HandleRequest(context.Background(), newTextCompletionsRequestWithChoices("multi", 3))
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
			if response.Status == endpoint.ResponseEndOfTokens {
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

var _ = Describe("HandleRequest with n > 1 and the KV cache enabled", func() {
	It("lets the choices of one request hold KV-cache blocks at the same time", func() {
		GinkgoT().Setenv("POD_IP", "localhost")
		sim := newHandleRequestTestSim(context.Background(), 2,
			"--enable-kvcache", "--kv-cache-size", "16", "--block-size", "8")

		_, _, _, topErr, _ := sim.HandleRequest(context.Background(), newTextCompletionsRequestWithChoices("kv", 2))
		Expect(topErr).To(BeNil())
		reqCtx1 := <-sim.newRequests.Channel
		reqCtx2 := <-sim.newRequests.Channel

		// Both choices run concurrently: each starts before either ends.
		_, err := reqCtx1.HandleRequest()
		Expect(err).To(BeNil())
		_, err = reqCtx2.HandleRequest()
		Expect(err).To(BeNil())
		reqCtx1.KVCacheOnRequestEnd()
		reqCtx2.KVCacheOnRequestEnd()
	})
})

var _ = Describe("simulateResponseProcessing", func() {
	// startProcessing runs req up to simulateResponseProcessing on a goroutine
	// and returns its response channel and a channel for its result.
	startProcessing := func(sim *Simulator, ctx context.Context, req endpoint.Request) (
		*common.Channel[*endpoint.ResponseInfo], chan bool) {
		_, _, respChan, topErr, _ := sim.HandleRequest(ctx, req)
		Expect(topErr).To(BeNil())
		reqCtx := <-sim.newRequests.Channel
		DeferCleanup(reqCtx.SignalDone)
		respCtx, err := reqCtx.HandleRequest()
		Expect(err).To(BeNil())

		completed := make(chan bool, 1)
		go func() { completed <- sim.simulateResponseProcessing(respCtx) }()
		return respChan, completed
	}

	It("stops waiting for the first token once the response is abandoned", func() {
		sim := newHandleRequestTestSim(context.Background(), 1, "--time-to-first-token", "10s")
		bus := useUnreadMetricsBus(sim)

		ctx, cancel := context.WithCancel(context.Background())
		DeferCleanup(cancel)
		respChan, completed := startProcessing(sim, ctx, newTextCompletionsRequestWithChoices("ttft", 1))
		Consistently(completed).ShouldNot(Receive())

		cancel()
		Eventually(completed).Should(Receive(BeFalse()))
		Expect(respChan.Channel).To(BeEmpty())
		Expect(bus.PrefillEnded.Channel).To(BeEmpty())
	})

	It("stops waiting for the next token once the response is abandoned", func() {
		sim := newHandleRequestTestSim(context.Background(), 1, "--inter-token-latency", "10s")
		bus := useUnreadMetricsBus(sim)

		req := newTextCompletionsRequestWithChoices("itl", 1)
		req.Prompt = []api.PromptInput{{Text: "hello there, how are you"}}
		ctx, cancel := context.WithCancel(context.Background())
		DeferCleanup(cancel)
		respChan, completed := startProcessing(sim, ctx, req)

		// Once the first token is out, the worker is waiting for the second.
		Eventually(func(g Gomega) {
			var response *endpoint.ResponseInfo
			g.Expect(respChan.Channel).To(Receive(&response))
			g.Expect(response.Tokens).NotTo(BeNil())
		}).Should(Succeed())
		Consistently(completed).ShouldNot(Receive())

		cancel()
		Eventually(completed).Should(Receive(BeFalse()))
		Expect(respChan.Channel).To(BeEmpty())
		Expect(bus.TokenGenerated.Channel).To(BeEmpty())
		Expect(bus.DecodeEnded.Channel).To(BeEmpty())
	})

	It("observes decode metrics before waiting for the image, and stops waiting once the response is abandoned", func() {
		sim := newHandleRequestTestSim(context.Background(), 1, "--omni", "--time-to-generate-image", "10s")
		bus := useUnreadMetricsBus(sim)

		req := &endpoint.ChatCompletionsRequest{}
		Expect(req.Unmarshal([]byte(`{"model":"` + common.TestModelName +
			`","messages":[{"role":"user","content":"hi"}],"stream":true}`))).To(Succeed())
		req.SetRequestID("image")
		req.SetSendImage(true)

		ctx, cancel := context.WithCancel(context.Background())
		DeferCleanup(cancel)
		respChan, completed := startProcessing(sim, ctx, req)

		// Once the last token is out, the worker is waiting for the image.
		Eventually(func(g Gomega) {
			var response *endpoint.ResponseInfo
			g.Expect(respChan.Channel).To(Receive(&response))
			g.Expect(response.Status).To(Equal(endpoint.ResponseEndOfTokens))
		}).Should(Succeed())
		Eventually(bus.DecodeEnded.Channel).Should(HaveLen(1))
		Consistently(completed).ShouldNot(Receive())

		cancel()
		Eventually(completed).Should(Receive(BeFalse()))
	})
})

var _ = Describe("processRequest", func() {
	It("drops a request abandoned while queued without running it", func() {
		sim := newHandleRequestTestSim(context.Background(), 1,
			"--lora-modules", `{"name":"lora1","path":"/path/to/lora1"}`, "--max-loras", "1")
		bus := useUnreadMetricsBus(sim)
		sim.queueCapacity = 1

		req := newTextCompletionsRequestWithChoices("queued", 1)
		req.Model = "lora1"
		ctx, cancel := context.WithCancel(context.Background())
		_, _, respChan, topErr, _ := sim.HandleRequest(ctx, req)
		Expect(topErr).To(BeNil())
		sim.addRequestToQueue(<-sim.newRequests.Channel)

		cancel()
		reqCtx := sim.dequeue()
		Expect(reqCtx).NotTo(BeNil())
		sim.processRequest(reqCtx)

		Expect(sim.Context.nRunningReqs.Load()).To(BeZero())
		Expect(bus.RequestQueued.Channel).To(HaveLen(1))
		Expect(bus.RequestDequeued.Channel).To(HaveLen(1))
		Expect(bus.RequestRunning.Channel).To(BeEmpty())
		Expect(bus.RequestFailed.Channel).To(BeEmpty())
		Expect(bus.RequestSucceeded.Channel).To(BeEmpty())
		Expect(bus.LoRAChanged.Channel).To(Receive(Equal(metrics.LoRAChanged{Model: "lora1", State: metrics.LoRAWaiting})))
		Expect(bus.LoRAChanged.Channel).To(Receive(Equal(metrics.LoRAChanged{Model: "lora1", State: metrics.LoRADropped})))
		Expect(bus.LoRAChanged.Channel).To(BeEmpty())
		Expect(respChan.Channel).To(BeEmpty())
		Eventually(respChan.Channel).Should(BeClosed())
	})
})

var _ = Describe("sleepUnlessDone", func() {
	closed := make(chan struct{})
	close(closed)

	DescribeTable("reports whether it waited without done being closed",
		func(d time.Duration, done <-chan struct{}, expected bool) {
			start := time.Now()
			Expect(sleepUnlessDone(d, done)).To(Equal(expected))
			if expected {
				Expect(time.Since(start)).To(BeNumerically(">=", d))
			} else {
				Expect(time.Since(start)).To(BeNumerically("<", time.Second))
			}
		},
		Entry("waits the full time when done is nil", 50*time.Millisecond, nil, true),
		Entry("returns early once done is closed", 10*time.Second, closed, false),
		Entry("stops a loop without latency once done is closed", time.Duration(0), closed, false),
	)
})
