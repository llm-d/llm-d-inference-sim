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
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/simulator"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

// newRunningSim builds and starts a real Simulator (echo mode), so
// HandleRequest produces genuine ResponseInfo entries -- including real,
// non-nil RespCtx values -- via the actual worker pool.
func newRunningSim(ctx context.Context) *simulator.Simulator {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}

	config, err := common.ParseCommandParamsAndLoadConfig()
	Expect(err).NotTo(HaveOccurred())

	sim, err := simulator.New(klog.Background())
	Expect(err).NotTo(HaveOccurred())
	sim.Context.SetConfig(config)
	sim.Context.Tokenizer = tokenizer.NewSimpleTokenizer()

	Expect(sim.InitializeSim(ctx)).To(Succeed())
	return sim
}

var _ = Describe("sendNonStream missing-choice guard", func() {
	It("returns a 500 instead of building a response with a nil respCtx when a choice never receives any entry", func() {
		ctx := context.Background()
		sim := newRunningSim(ctx)

		req := &simulator.TextCompletionsParsedRequest{}
		req.RequestID = "multi"
		req.Model = common.TestModelName
		req.Prompt = []api.PromptInput{{Text: "hi"}}
		n := 2
		req.N = &n

		numChoices, isStream, respChan, topErr, _ := sim.HandleRequest(req)
		Expect(topErr).To(BeNil())
		Expect(numChoices).To(Equal(2))
		Expect(isStream).To(BeFalse())

		// Relay every real entry for choice 0 into a fresh channel, but drop every
		// entry for choice 1 -- simulating a choice whose respCtx never arrived --
		// then close it once the real channel closes (both choices have completed).
		filtered := common.Channel[*simulator.ResponseInfo]{
			Channel: make(chan *simulator.ResponseInfo, 100),
			Name:    "filtered",
		}
		for response := range respChan.Channel {
			if response.ChoiceIdx == 1 {
				continue
			}
			filtered.Channel <- response
		}
		close(filtered.Channel)

		c := &Communication{logger: klog.Background()}
		httpCtx := &fasthttp.RequestCtx{}
		c.sendNonStream(httpCtx, filtered, nil, numChoices)

		Expect(httpCtx.Response.StatusCode()).To(Equal(fasthttp.StatusInternalServerError))
		Expect(string(httpCtx.Response.Body())).To(ContainSubstring("no tokens for choice index: 1"))
	})
})

// The comment on handleStream notes that once ctx.Response.SetBodyStream is called,
// a real client has already received the 200/SSE status line, so a real caller
// never sees the finalizeStream nil-check's error response. But sendError mutates
// ctx.Response's status/body fields directly, separately from the piped body
// stream, so this white-box test can still observe it -- and, more importantly,
// this also confirms the streaming goroutine returns via that check instead of
// calling rc.FinishReason() on a nil ResponseContext, which would panic it (and,
// since nothing recovers a goroutine panic, crash the whole process).
var _ = Describe("sendStream missing-choice guard", func() {
	It("reports a 500 instead of panicking when a choice never receives any entry", func() {
		ctx := context.Background()
		sim := newRunningSim(ctx)

		req := &simulator.TextCompletionsParsedRequest{}
		req.RequestID = "multi-stream"
		req.Model = common.TestModelName
		req.Prompt = []api.PromptInput{{Text: "hi"}}
		req.Stream = true
		n := 2
		req.N = &n

		numChoices, isStream, respChan, topErr, _ := sim.HandleRequest(req)
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

		filtered := common.Channel[*simulator.ResponseInfo]{
			Channel: make(chan *simulator.ResponseInfo, 100),
			Name:    "filtered",
		}
		for response := range respChan.Channel {
			if response.ChoiceIdx == missingChoice {
				continue
			}
			filtered.Channel <- response
		}
		close(filtered.Channel)

		c := &Communication{logger: klog.Background()}
		httpCtx := &fasthttp.RequestCtx{}
		httpCtx.SetStatusCode(fasthttp.StatusOK)
		httpCtx.SetContentType("text/event-stream")

		c.sendStream(httpCtx, filtered, &textComplHTTPRespBuilder{}, numChoices, first)

		// sendStream's own goroutine runs concurrently with this one. Hitting the
		// nil-check calls sendError, whose SetBody closes the response's body
		// stream as a side effect of abandoning it in favor of a plain error body
		// -- which unblocks a pending Read on that stream with exactly this error.
		// Waiting for it both confirms the nil-check (not a panic) is what ended
		// the stream, and -- since closing happens right after sendError sets the
		// status/body -- gives a happens-before edge making it safe to read those
		// fields afterward without racing the goroutine that set them.
		_, err := io.ReadAll(httpCtx.Response.BodyStream())
		Expect(err).To(MatchError(io.ErrClosedPipe))

		Expect(httpCtx.Response.StatusCode()).To(Equal(fasthttp.StatusInternalServerError))
		// SetBody writes the actual error bytes right after the Close() call that
		// unblocked the ReadAll above, so there's a brief window where the body
		// isn't visible yet; Eventually absorbs it.
		Eventually(func() string {
			return string(httpCtx.Response.Body())
		}, "1s").Should(ContainSubstring(fmt.Sprintf("no tokens for choice index: %d", missingChoice)))
	})
})
