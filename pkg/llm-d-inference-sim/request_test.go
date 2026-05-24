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
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// ptrInt64 and ptrInt return pointers to their argument; small helpers for
// building request fixtures without leaking temp vars into each test.
func ptrInt64(v int64) *int64 { return &v }
func ptrInt(v int) *int       { return &v }

// newTextCompletionsFixture builds a TextCompletionsParsedRequest populated
// with a cross-section of fields that split must preserve on each sub-request.
// Keep this exhaustive — if you add a field to the parsed type that should
// survive the copy, add it here too.
func newTextCompletionsFixture() *TextCompletionsParsedRequest {
	req := &TextCompletionsParsedRequest{}
	req.RequestID = "req-abc"
	req.Model = "test-model"
	req.DisplayedModel = "test-model-alias"
	req.Stream = true
	req.IgnoreEOS = true
	req.StreamOptions = &openaiserverapi.StreamOptions{IncludeUsage: true}
	threshold := 0.25
	req.CacheHitThreshold = &threshold
	req.Prompt = openaiserverapi.NewStringOrArrayFromSlice([]string{"one", "two"})
	req.MaxTokens = ptrInt64(42)
	req.Logprobs = ptrInt(3)
	return req
}

var _ = Describe("TextCompletionsParsedRequest.split", func() {
	It("returns one sub-request per array element with suffixed RequestIDs", func() {
		orig := newTextCompletionsFixture()

		subs := orig.split()

		Expect(subs).To(HaveLen(2))
		first := subs[0].(*TextCompletionsRequest)
		second := subs[1].(*TextCompletionsRequest)
		Expect(first.GetRequestID()).To(Equal("req-abc-0"))
		Expect(first.Prompt).To(Equal("one"))
		Expect(second.GetRequestID()).To(Equal("req-abc-1"))
		Expect(second.Prompt).To(Equal("two"))
	})

	It("preserves non-prompt fields on each sub-request", func() {
		orig := newTextCompletionsFixture()

		subs := orig.split()
		sub := subs[0].(*TextCompletionsRequest)

		Expect(sub.GetModel()).To(Equal(orig.GetModel()))
		Expect(sub.GetDisplayedModel()).To(Equal(orig.GetDisplayedModel()))
		Expect(sub.IsStream()).To(Equal(orig.IsStream()))
		Expect(sub.IncludeUsage()).To(Equal(orig.IncludeUsage()))
		Expect(sub.GetIgnoreEOS()).To(Equal(orig.GetIgnoreEOS()))
		Expect(sub.ExtractMaxTokens()).To(Equal(orig.ExtractMaxTokens()))
		Expect(sub.GetLogprobs()).To(Equal(orig.GetLogprobs()))
		Expect(sub.GetCacheHitThreshold()).To(Equal(orig.GetCacheHitThreshold()))
	})

	It("returns a single sub-request with the original RequestID for a string prompt", func() {
		orig := &TextCompletionsParsedRequest{}
		orig.RequestID = "req-xyz"
		orig.Model = "test-model"
		orig.Prompt = openaiserverapi.NewStringOrArray("only-prompt")

		subs := orig.split()

		Expect(subs).To(HaveLen(1))
		sub := subs[0].(*TextCompletionsRequest)
		Expect(sub.GetRequestID()).To(Equal("req-xyz"))
		Expect(sub.Prompt).To(Equal("only-prompt"))
	})

	It("does not mutate the parsed request", func() {
		orig := newTextCompletionsFixture()
		origPromptArr := orig.Prompt.Array()
		origID := orig.GetRequestID()

		_ = orig.split()

		Expect(orig.GetRequestID()).To(Equal(origID))
		Expect(orig.Prompt.IsArray()).To(BeTrue())
		Expect(orig.Prompt.Array()).To(Equal(origPromptArr))
	})
})
