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
package kvcache

import (
	"context"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("reported prefix cache hits", func() {
	DescribeTable("caps a full hit at one block below the prompt length",
		func(cachedTokens, promptTokens, blockSize, expected int) {
			Expect(reportedCachedPromptTokens(cachedTokens, promptTokens, blockSize)).To(Equal(expected))
		},
		Entry("32-token prompt, block size 16, both blocks resident", 32, 32, 16, 16),
		Entry("32-token prompt, only the first block resident", 16, 32, 16, 16),
		Entry("31-token prompt, first block resident", 16, 31, 16, 16),
		Entry("16-token prompt, single resident block", 16, 16, 16, 0),
		Entry("24-token prompt, three resident blocks of size 8", 24, 24, 8, 16),
		Entry("empty prompt", 0, 0, 16, 0),
	)

	It("reports a 16-token cap for a fully cached 32-token prompt", func() {
		const blockSize = 16
		helper, err := NewKVCacheHelper(context.Background(), &common.Configuration{
			IP: localhost, Model: common.TestModelName,
			KVCache: common.KVCacheConfig{KVCacheSize: 64, TokenBlockSize: blockSize, EventBatchSize: 1},
		}, logr.Discard(), nil, nil, nil)
		Expect(err).NotTo(HaveOccurred())

		tokens := make([]uint32, 32)
		for i := range tokens {
			tokens[i] = uint32(i + 1)
		}

		start := func(id string) int {
			req := &api.ChatCompletionsRequest{}
			req.SetDisplayedModel(common.TestModelName)
			req.SetRequestID(id)
			req.SetTokenizedPrompt(&api.Tokenized{Tokens: tokens})
			stats, started, err := helper.OnRequestStart(req, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(started).To(BeTrue())
			Expect(helper.OnRequestEnd(req.GetRequestID())).To(Succeed())
			return stats.CachedPromptTokens
		}

		Expect(start("first")).To(BeZero())
		Expect(start("second")).To(Equal(len(tokens) - blockSize))
	})

	It("leaves a trailing partial block uncached", func() {
		const blockSize = 16
		helper, err := NewKVCacheHelper(context.Background(), &common.Configuration{
			IP: localhost, Model: common.TestModelName,
			KVCache: common.KVCacheConfig{KVCacheSize: 64, TokenBlockSize: blockSize, EventBatchSize: 1},
		}, logr.Discard(), nil, nil, nil)
		Expect(err).NotTo(HaveOccurred())

		tokens := make([]uint32, 40)
		for i := range tokens {
			tokens[i] = uint32(i + 1)
		}
		for i, id := range []string{"first", "second"} {
			req := &api.ChatCompletionsRequest{}
			req.SetDisplayedModel(common.TestModelName)
			req.SetRequestID(id)
			req.SetTokenizedPrompt(&api.Tokenized{Tokens: tokens})
			stats, _, err := helper.OnRequestStart(req, 0)
			Expect(err).NotTo(HaveOccurred())
			if i == 0 {
				Expect(stats.CachedPromptTokens).To(BeZero())
			} else {
				Expect(stats.CachedPromptTokens).To(Equal(2 * blockSize))
			}
			Expect(helper.OnRequestEnd(req.GetRequestID())).To(Succeed())
		}
	})
})
