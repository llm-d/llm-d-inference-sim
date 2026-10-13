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

package sglang

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

var _ = Describe("ValidateConfig", func() {
	var config *common.Configuration

	BeforeEach(func() {
		config = common.NewConfig()
		New().ApplyDefaults(config)
	})

	It("should accept this engine's own defaults", func() {
		Expect(New().ValidateConfig(config)).To(Succeed())
	})

	It("should reject an enabled KV cache", func() {
		config.KVCache.EnableKVCache = true
		Expect(New().ValidateConfig(config)).To(
			MatchError(ContainSubstring("KV-cache simulation is not implemented by the sglang engine yet")))
	})

	It("should accept fake metrics", func() {
		config.FakeMetrics = &SGLangFakeMetrics{
			RunningRequests: &common.FakeMetricWithFunction{FixedValue: 5},
		}
		Expect(New().ValidateConfig(config)).To(Succeed())
	})

	It("should reject fake metrics with a non-positive refresh interval", func() {
		config.FakeMetrics = &SGLangFakeMetrics{}
		config.FakeMetricsRefreshInterval = 0
		Expect(New().ValidateConfig(config)).To(
			MatchError(ContainSubstring("fake metrics refresh interval must be positive")))
	})

	DescribeTable("should reject an invalid fake-metrics configuration",
		func(fake *SGLangFakeMetrics, expected string) {
			config.FakeMetrics = fake
			Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring(expected)))
		},
		Entry("negative running requests",
			&SGLangFakeMetrics{RunningRequests: &common.FakeMetricWithFunction{FixedValue: -1}},
			"fake metrics running-requests cannot be negative"),
		Entry("negative queued requests",
			&SGLangFakeMetrics{QueuedRequests: &common.FakeMetricWithFunction{FixedValue: -1}},
			"fake metrics queue-requests cannot be negative"),
		Entry("token usage above one",
			&SGLangFakeMetrics{TokenUsage: &common.FakeMetricWithFunction{FixedValue: 1.5}},
			"fake metrics token-usage must be between 0 and 1"),
		Entry("token usage generator leaving the range", &SGLangFakeMetrics{
			TokenUsage: &common.FakeMetricWithFunction{
				IsFunction: true,
				Function: &common.FunctionInfo{
					Name: common.RampFuncName, Start: 0, End: 2, Period: time.Second,
				},
			},
		}, "fake metrics token-usage start and end must be between 0 and 1"),
		Entry("unknown generator function", &SGLangFakeMetrics{
			RunningRequests: &common.FakeMetricWithFunction{
				IsFunction: true,
				Function: &common.FunctionInfo{
					Name: "spiral", Start: 0, End: 2, Period: time.Second,
				},
			},
		}, "invalid fake metrics generation function spiral"),
		Entry("negative counter series", &SGLangFakeMetrics{
			NumRequestsTotal: &StreamSplit{Streaming: int64Ptr(-1)},
		}, "fake metrics num-requests-total cannot be negative"),
		Entry("cached tokens without queried tokens",
			&SGLangFakeMetrics{CachedTokens: int64Ptr(5)},
			"fake metrics cached-tokens and queried-tokens must be specified together"),
		Entry("cached tokens exceeding queried tokens",
			&SGLangFakeMetrics{CachedTokens: int64Ptr(10), QueriedTokens: int64Ptr(5)},
			"fake metrics cached-tokens cannot exceed queried-tokens"),
		Entry("negative histogram samples",
			&SGLangFakeMetrics{QueueTimeBucketValues: []int{1, -1}},
			"fake metrics queue-time-buckets-values cannot contain negative values"),
		Entry("negative samples in a split histogram series", &SGLangFakeMetrics{
			TTFTBucketValues: &StreamSplitBuckets{NonStreaming: []int{-1}},
		}, "fake metrics ttft-buckets-values cannot contain negative values"),
		Entry("more samples than buckets", &SGLangFakeMetrics{
			TTFTBucketValues: &StreamSplitBuckets{Streaming: make([]int, len(ttftBucketsBoundaries)+2)},
		}, "fake metrics ttft-buckets-values is too long"),
	)

	// One sample per bucket, including the +Inf bucket past the last boundary.
	It("should accept a histogram array as long as the bucket count", func() {
		config.FakeMetrics = &SGLangFakeMetrics{
			QueueTimeBucketValues: make([]int, len(queueTimeBucketsBoundaries)+1),
		}
		Expect(New().ValidateConfig(config)).To(Succeed())
	})
})

var _ = Describe("KV events", func() {
	It("should have no encoder", func() {
		encoder, err := New().NewKVEventEncoder(*common.NewConfig())
		Expect(err).To(MatchError(ContainSubstring("does not support KV-cache events")))
		Expect(encoder).To(BeNil())
	})
})

var _ = Describe("strict request validation", func() {
	It("should leave strict validation disabled for the experimental engine", func() {
		validator, err := New().NewRequestValidator()
		Expect(err).NotTo(HaveOccurred())
		Expect(validator).To(BeNil())
	})
})
