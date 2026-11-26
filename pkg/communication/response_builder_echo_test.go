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
	"encoding/json"
	"fmt"
	"io"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/endpoint"
)

// The simulator runs in echo mode here, so the generated text of a choice is
// its prompt. The echo request field then shows as the prompt sent twice.
var _ = Describe("text completions echo", func() {
	newRequest := func(body string) (*endpoint.TextCompletionsParsedRequest, *Communication) {
		req := &endpoint.TextCompletionsParsedRequest{}
		Expect(req.Unmarshal([]byte(body))).To(Succeed())
		return req, &Communication{logger: klog.Background(), transport: fakeTransport{}}
	}

	nonStream := func(body string) api.TextCompletionsResponse {
		req, c := newRequest(body)
		ctx := context.Background()
		numChoices, _, respChan, topErr, _ := newRunningSim(ctx).HandleRequest(ctx, req)
		Expect(topErr).To(BeNil())

		httpCtx := &fasthttp.RequestCtx{}
		c.sendNonStream(httpCtx, *respChan, &textComplHTTPRespBuilder{transport: c.transport}, numChoices)
		Expect(httpCtx.Response.StatusCode()).To(Equal(fasthttp.StatusOK))

		var resp api.TextCompletionsResponse
		Expect(json.Unmarshal(httpCtx.Response.Body(), &resp)).To(Succeed())
		return resp
	}

	// stream returns the text of each chunk per choice index and the final usage.
	stream := func(body string) (map[int][]string, *api.Usage) {
		req, c := newRequest(body)
		ctx := context.Background()
		numChoices, _, respChan, topErr, _ := newRunningSim(ctx).HandleRequest(ctx, req)
		Expect(topErr).To(BeNil())
		first := <-respChan.Channel

		httpCtx := &fasthttp.RequestCtx{}
		c.sendStream(httpCtx, *respChan, &textComplHTTPRespBuilder{transport: c.transport}, numChoices, first, func() {})
		raw, err := io.ReadAll(httpCtx.Response.BodyStream())
		Expect(err).NotTo(HaveOccurred())

		texts := map[int][]string{}
		var usage *api.Usage
		for _, frame := range strings.Split(string(raw), "\n\n") {
			data, found := strings.CutPrefix(frame, "data: ")
			if !found || data == "[DONE]" {
				continue
			}
			var chunk api.TextCompletionsResponse
			Expect(json.Unmarshal([]byte(data), &chunk)).To(Succeed())
			for _, choice := range chunk.Choices {
				idx := choice.Index
				texts[idx] = append(texts[idx], choice.Text)
			}
			if len(chunk.Choices) == 0 {
				usage = chunk.Usage
			}
		}
		return texts, usage
	}

	// body is a request for two prompts with two choices each, plus extra fields.
	body := func(extra string) string {
		return fmt.Sprintf(`{"model":%q,"prompt":["alpha","beta"],"n":2%s}`, common.TestModelName, extra)
	}

	It("prepends the prompt to the text of every choice", func() {
		resp := nonStream(body(`,"echo":true`))
		Expect(resp.Choices).To(HaveLen(4))
		for i, want := range []string{"alphaalpha", "alphaalpha", "betabeta", "betabeta"} {
			Expect(resp.Choices[i].Text).To(Equal(want), fmt.Sprintf("choice %d", i))
		}
	})

	It("leaves the usage unchanged", func() {
		withEcho := nonStream(body(`,"echo":true`))
		without := nonStream(body(``))
		Expect(withEcho.Usage).To(Equal(without.Usage))
	})

	It("does not prepend the prompt when echo is false or absent", func() {
		for _, extra := range []string{``, `,"echo":false`, `,"echo":null`} {
			resp := nonStream(body(extra))
			Expect(resp.Choices).To(HaveLen(4))
			for i, want := range []string{"alpha", "alpha", "beta", "beta"} {
				Expect(resp.Choices[i].Text).To(Equal(want), fmt.Sprintf("choice %d, extra %q", i, extra))
			}
		}
	})

	It("streams the prompt once in the first chunk of every choice", func() {
		texts, usage := stream(body(`,"echo":true,"stream":true,"stream_options":{"include_usage":true}`))
		Expect(texts).To(HaveLen(4))
		for i, prompt := range []string{"alpha", "alpha", "beta", "beta"} {
			Expect(texts[i][0]).To(HavePrefix(prompt))
			Expect(strings.Join(texts[i], "")).To(Equal(prompt + prompt))
		}

		_, usageWithoutEcho := stream(body(`,"stream":true,"stream_options":{"include_usage":true}`))
		Expect(usage).To(Equal(usageWithoutEcho))
	})
})
