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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/endpoint"
)

// cachedPromptRespCtx is a response context for a prompt that was partly read
// from the prefix cache. Methods the Messages builder does not call are left
// to the embedded nil interface.
type cachedPromptRespCtx struct {
	endpoint.ResponseContext
	usage  api.Usage
	cached int
}

func (r *cachedPromptRespCtx) UsageData() *api.Usage                             { return &r.usage }
func (r *cachedPromptRespCtx) NumberCachedPromptTokens() int                     { return r.cached }
func (r *cachedPromptRespCtx) FinishReason() *string                             { s := common.StopFinishReason; return &s }
func (r *cachedPromptRespCtx) ToolCalls() []api.ToolCall                         { return nil }
func (r *cachedPromptRespCtx) DisplayModel() string                              { return common.TestModelName }
func (r *cachedPromptRespCtx) RequestID() string                                 { return "cached" }
func (r *cachedPromptRespCtx) DoRemoteDecode() bool                              { return false }
func (r *cachedPromptRespCtx) ECTransferParams() map[string]api.ECTransferParams { return nil }

var _ = Describe("Messages HTTP usage", func() {
	respCtx := &cachedPromptRespCtx{
		usage: api.Usage{PromptTokens: 27, CompletionTokens: 5, TotalTokens: 32,
			PromptTokensDetails: &api.PromptTokensDetails{CachedTokens: 24}},
		cached: 24,
	}

	It("reports prompt tokens read from the prefix cache in cache_read_input_tokens", func() {
		b := &messagesHTTPRespBuilder{}
		tokens := []api.Tokenized{{Strings: []string{"hi"}}}
		resp, ok := b.createResponse([]endpoint.ResponseContext{respCtx}, tokens).(*api.MessagesResponse)
		Expect(ok).To(BeTrue())
		Expect(resp.Usage).To(Equal(api.MessagesUsage{InputTokens: 3, CacheReadInputTokens: 24, OutputTokens: 5}))
	})

	It("reports them in the message_start event of a stream", func() {
		b := &messagesHTTPRespBuilder{}
		chunk, ok := b.createInitialChunk(respCtx).(*namedEventChunk)
		Expect(ok).To(BeTrue())
		start, ok := chunk.data[0].(api.MessagesMessageStartEvent)
		Expect(ok).To(BeTrue())
		Expect(start.Message.Usage).To(Equal(api.MessagesUsage{InputTokens: 3, CacheReadInputTokens: 24}))
	})
})

// deltaUsageTransport records the full usage the builder hands the engine for
// a message_delta event.
type deltaUsageTransport struct {
	fakeTransport
	full api.MessagesUsage
}

func (t *deltaUsageTransport) DecorateMessagesDeltaUsage(usage api.MessagesStreamUsage, full api.MessagesUsage) api.MessagesStreamUsage {
	t.full = full
	return usage
}

var _ = Describe("Messages HTTP message_delta usage", func() {
	It("hands the engine the full usage, split by the prefix cache", func() {
		respCtx := &cachedPromptRespCtx{
			usage: api.Usage{PromptTokens: 27, CompletionTokens: 5, TotalTokens: 32,
				PromptTokensDetails: &api.PromptTokensDetails{CachedTokens: 24}},
			cached: 24,
		}
		transport := &deltaUsageTransport{}
		b := &messagesHTTPRespBuilder{transport: transport}
		chunk, ok := b.createLastChunk(respCtx, common.StopFinishReason, 0).(*namedEventChunk)
		Expect(ok).To(BeTrue())
		delta, ok := chunk.data[1].(api.MessagesMessageDeltaEvent)
		Expect(ok).To(BeTrue())
		Expect(delta.Usage.OutputTokens).To(Equal(5))
		Expect(transport.full).To(Equal(api.MessagesUsage{InputTokens: 3, CacheReadInputTokens: 24, OutputTokens: 5}))
	})
})
