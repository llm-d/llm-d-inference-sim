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

package communication

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/go-logr/logr/funcr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/endpoint"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
	"github.com/llm-d/llm-d-inference-sim/pkg/simulator"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

// stubEngine is the engine these tests wire into the simulator. The metrics
// stub covers everything but the KV-event encoder, which pkg/metrics cannot
// supply without importing pkg/kvcache and closing an import cycle.
type stubEngine struct {
	metrics.StubAdapter
}

// NewKVEventEncoder is never called: these tests leave the KV cache disabled,
// and the simulator only builds an encoder when it is enabled.
func (stubEngine) NewKVEventEncoder(_ common.Configuration) (kvcache.EventEncoder, error) {
	return nil, nil
}

// newRunningSim builds and starts a real Simulator (echo mode), so
// HandleRequest produces genuine ResponseInfo entries -- including real,
// non-nil RespCtx values -- via the actual worker pool. Built directly via
// common.NewConfig rather than command-line parsing, since pkg/communication
// cannot import pkg/engine/vllm (which itself imports pkg/communication).
func newRunningSim(ctx context.Context) *simulator.Simulator {
	config := common.NewConfig()
	config.Model = common.TestModelName
	config.Mode = common.ModeEcho
	config.ServedModelNames = []string{config.Model}
	config.DisplayModelName = config.Model

	sim, err := simulator.New(klog.Background())
	Expect(err).NotTo(HaveOccurred())
	sim.Context.SetConfig(config)
	sim.Context.Tokenizer = tokenizer.NewSimpleTokenizer()
	sim.Context.Engine = stubEngine{}

	Expect(sim.InitializeSim(ctx)).To(Succeed())
	return sim
}

var _ = Describe("sendNonStream missing-choice guard", func() {
	It("returns a 500 instead of building a response with a nil respCtx when a choice never receives any entry", func() {
		ctx := context.Background()
		sim := newRunningSim(ctx)

		req := &endpoint.TextCompletionsParsedRequest{}
		req.RequestID = "multi"
		req.Model = common.TestModelName
		req.Prompt = []api.PromptInput{{Text: "hi"}}
		n := 2
		req.N = &n

		numChoices, isStream, respChan, topErr, _ := sim.HandleRequest(ctx, req)
		Expect(topErr).To(BeNil())
		Expect(numChoices).To(Equal(2))
		Expect(isStream).To(BeFalse())

		// Relay every real entry for choice 0 into a fresh channel, but drop every
		// entry for choice 1 -- simulating a choice whose respCtx never arrived --
		// then close it once the real channel closes (both choices have completed).
		filtered := common.Channel[*endpoint.ResponseInfo]{
			Channel: make(chan *endpoint.ResponseInfo, 100),
			Name:    "filtered",
		}
		for response := range respChan.Channel {
			if response.ChoiceIdx == 1 {
				continue
			}
			filtered.Channel <- response
		}
		close(filtered.Channel)

		c := &Communication{logger: klog.Background(), transport: fakeTransport{}}
		httpCtx := &fasthttp.RequestCtx{}
		c.sendNonStream(httpCtx, filtered, nil, numChoices)

		Expect(httpCtx.Response.StatusCode()).To(Equal(fasthttp.StatusInternalServerError))
		Expect(string(httpCtx.Response.Body())).To(ContainSubstring("no tokens for choice index: 1"))
	})
})

// The missing-choice guard ends the stream with an error frame, confirming the
// streaming goroutine returns via that check instead of calling
// rc.FinishReason() on a nil ResponseContext, which would panic it (and, since
// nothing recovers a goroutine panic, crash the whole process).
var _ = Describe("sendStream missing-choice guard", func() {
	It("streams an error instead of panicking when a choice never receives any entry", func() {
		ctx := context.Background()
		sim := newRunningSim(ctx)

		req := &endpoint.TextCompletionsParsedRequest{}
		req.RequestID = "multi-stream"
		req.Model = common.TestModelName
		req.Prompt = []api.PromptInput{{Text: "hi"}}
		req.Stream = true
		n := 2
		req.N = &n

		numChoices, isStream, respChan, topErr, _ := sim.HandleRequest(ctx, req)
		Expect(topErr).To(BeNil())
		Expect(numChoices).To(Equal(2))
		Expect(isStream).To(BeTrue())

		// Mirror handleStream: peek the first response before committing to a
		// stream. Drop every remaining entry for whichever choice "first" is NOT,
		// so that choice ends up with zero entries -- simulating a choice whose
		// respCtx never arrived -- while the other choice streams normally.
		first := <-respChan.Channel
		Expect(first).NotTo(BeNil())
		missingChoice := 1 - first.ChoiceIdx

		filtered := common.Channel[*endpoint.ResponseInfo]{
			Channel: make(chan *endpoint.ResponseInfo, 100),
			Name:    "filtered",
		}
		for response := range respChan.Channel {
			if response.ChoiceIdx == missingChoice {
				continue
			}
			filtered.Channel <- response
		}
		close(filtered.Channel)

		c := &Communication{logger: klog.Background(), transport: fakeTransport{}}
		httpCtx := &fasthttp.RequestCtx{}
		httpCtx.SetStatusCode(fasthttp.StatusOK)
		httpCtx.SetContentType("text/event-stream")

		c.sendStream(httpCtx, filtered, &textComplHTTPRespBuilder{transport: c.transport}, numChoices, first, func() {})

		body, err := io.ReadAll(httpCtx.Response.BodyStream())
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(ContainSubstring(fmt.Sprintf("no tokens for choice index: %d", missingChoice)))
		Expect(string(body)).To(ContainSubstring("[DONE]"))
		Expect(httpCtx.Response.StatusCode()).To(Equal(fasthttp.StatusOK))
	})
})

// When the client disconnects, fasthttp closes the body stream and recycles ctx
// for another request, so the stream goroutine must not touch ctx after a
// failed write.
var _ = Describe("sendStream client disconnect", func() {
	It("leaves ctx untouched and calls abort when writing to the stream fails", func() {
		ctx := context.Background()
		sim := newRunningSim(ctx)

		req := &endpoint.TextCompletionsParsedRequest{}
		req.RequestID = "disconnect-stream"
		req.Model = common.TestModelName
		req.Prompt = []api.PromptInput{{Text: "hi"}}
		req.Stream = true

		numChoices, isStream, respChan, topErr, _ := sim.HandleRequest(ctx, req)
		Expect(topErr).To(BeNil())
		Expect(isStream).To(BeTrue())

		first := <-respChan.Channel
		Expect(first).NotTo(BeNil())
		// Buffer the remaining entries so the simulator never blocks on a stream
		// that stops reading.
		buffered := common.Channel[*endpoint.ResponseInfo]{
			Channel: make(chan *endpoint.ResponseInfo, 100),
			Name:    "buffered",
		}
		for response := range respChan.Channel {
			buffered.Channel <- response
		}
		close(buffered.Channel)

		var logMu sync.Mutex
		var logs strings.Builder
		logger := funcr.New(func(_, args string) {
			logMu.Lock()
			defer logMu.Unlock()
			logs.WriteString(args + "\n")
		}, funcr.Options{})

		c := &Communication{logger: logger, transport: fakeTransport{}}
		httpCtx := &fasthttp.RequestCtx{}
		httpCtx.SetStatusCode(fasthttp.StatusOK)
		httpCtx.SetContentType("text/event-stream")

		aborted := make(chan struct{})
		c.sendStream(httpCtx, buffered, &textComplHTTPRespBuilder{transport: c.transport}, numChoices, first,
			func() { close(aborted) })

		// The goroutine is blocked writing the first chunk into the pipe, since
		// nothing reads it. Closing the read end fails that write, as a client
		// disconnect does.
		bodyStream, ok := httpCtx.Response.BodyStream().(io.Closer)
		Expect(ok).To(BeTrue())
		Expect(bodyStream.Close()).To(Succeed())

		Eventually(func() string {
			logMu.Lock()
			defer logMu.Unlock()
			return logs.String()
		}, "1s").Should(ContainSubstring("closed pipe"))
		Eventually(aborted).Should(BeClosed())

		Consistently(func() int {
			return httpCtx.Response.StatusCode()
		}, "200ms").Should(Equal(fasthttp.StatusOK))
		Expect(string(httpCtx.Response.Header.ContentType())).To(Equal("text/event-stream"))
	})
})

// A sub-request error ends the stream while its siblings may still be
// producing, so the stream goroutine aborts without waiting for the channel to
// close.
var _ = Describe("sendStream sub-request error", func() {
	It("ends the stream and calls abort while the channel is still open", func() {
		ctx := context.Background()
		sim := newRunningSim(ctx)

		req := &endpoint.TextCompletionsParsedRequest{}
		req.RequestID = "error-stream"
		req.Model = common.TestModelName
		req.Prompt = []api.PromptInput{{Text: "hi"}}
		req.Stream = true

		numChoices, _, respChan, topErr, _ := sim.HandleRequest(ctx, req)
		Expect(topErr).To(BeNil())
		first := <-respChan.Channel
		Expect(first).NotTo(BeNil())

		// Never closed: a sibling sub-request is still running.
		pending := common.Channel[*endpoint.ResponseInfo]{
			Channel: make(chan *endpoint.ResponseInfo, 1),
			Name:    "pending",
		}
		subReqErr := api.NewError("sub-request failed", fasthttp.StatusBadRequest, nil)
		pending.Channel <- &endpoint.ResponseInfo{Err: &subReqErr}

		c := &Communication{logger: klog.Background(), transport: fakeTransport{}}
		httpCtx := &fasthttp.RequestCtx{}
		aborted := make(chan struct{})
		c.sendStream(httpCtx, pending, &textComplHTTPRespBuilder{transport: c.transport}, numChoices, first,
			func() { close(aborted) })

		body, err := io.ReadAll(httpCtx.Response.BodyStream())
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(ContainSubstring("sub-request failed"))
		Expect(string(body)).To(ContainSubstring("[DONE]"))
		Expect(aborted).To(BeClosed())
	})
})

// abortRecorder records the context handleHTTP passes to HandleRequest, which
// is done once the request is aborted.
type abortRecorder struct {
	Processor
	ctx context.Context
}

func (p *abortRecorder) HandleRequest(ctx context.Context, req endpoint.Request) (int, bool,
	*common.Channel[*endpoint.ResponseInfo], *api.Error, bool) {
	p.ctx = ctx
	return p.Processor.HandleRequest(ctx, req)
}

func newCompletionsRequestCtx(body string) *fasthttp.RequestCtx {
	httpCtx := &fasthttp.RequestCtx{}
	httpCtx.Request.SetRequestURI("/v1/completions")
	httpCtx.Request.Header.SetContentType("application/json")
	httpCtx.Request.SetBodyString(body)
	return httpCtx
}

var _ = Describe("handleHTTP abort", func() {
	var (
		sim      *simulator.Simulator
		recorder *abortRecorder
		c        *Communication
	)

	BeforeEach(func() {
		sim = newRunningSim(context.Background())
		recorder = &abortRecorder{Processor: sim}
		c = New(klog.Background(), recorder, &sim.Context)
		c.transport = fakeTransport{}
	})

	// Exceeds the default context window, so its sub-request fails before
	// producing any content.
	tooLong := strings.Repeat("hi ", 1024)

	DescribeTable("aborts the request when the handler returns",
		func(body string, status int) {
			httpCtx := newCompletionsRequestCtx(body)
			c.HandleTextCompletions(httpCtx)

			Expect(httpCtx.Response.StatusCode()).To(Equal(status))
			Expect(recorder.ctx.Done()).To(BeClosed())
			Eventually(sim.OpenRequests).Should(BeZero())
		},
		Entry("for a request rejected before it is queued",
			`{"model":"unknown","prompt":"hi"}`, fasthttp.StatusNotFound),
		Entry("for a non-streamed response",
			fmt.Sprintf(`{"model":%q,"prompt":"hi"}`, common.TestModelName), fasthttp.StatusOK),
		Entry("for a non-streamed response whose sub-request fails",
			fmt.Sprintf(`{"model":%q,"prompt":["hi",%q]}`, common.TestModelName, tooLong), fasthttp.StatusBadRequest),
		Entry("for a streamed response that fails before its first chunk",
			fmt.Sprintf(`{"model":%q,"prompt":%q,"stream":true}`, common.TestModelName, tooLong), fasthttp.StatusBadRequest),
	)

	It("leaves a streamed response for its stream goroutine to abort", func() {
		httpCtx := newCompletionsRequestCtx(fmt.Sprintf(`{"model":%q,"prompt":"hi","stream":true}`, common.TestModelName))
		c.HandleTextCompletions(httpCtx)
		Expect(httpCtx.Response.StatusCode()).To(Equal(fasthttp.StatusOK))

		// Nothing reads the body yet, so the stream goroutine is blocked on its first write.
		Consistently(recorder.ctx.Done(), "100ms").ShouldNot(BeClosed())

		_, err := io.ReadAll(httpCtx.Response.BodyStream())
		Expect(err).NotTo(HaveOccurred())
		Expect(recorder.ctx.Done()).To(BeClosed())
		Eventually(sim.OpenRequests).Should(BeZero())
	})
})
