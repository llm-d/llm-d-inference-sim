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

// Unit tests for the sglang metrics adapter, covering the paths reached by
// invoking the event handlers directly, without booting the simulator.

package sglang

import (
	"context"

	"github.com/go-logr/logr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
)

const (
	streaming    = "true"
	nonStreaming = "false"
)

func newTestConfig() common.Configuration {
	return common.Configuration{
		Model:                 common.TestModelName,
		ServedModelNames:      []string{common.TestModelName},
		DisplayModelName:      common.TestModelName,
		MaxNumSeqs:            4,
		MaxWaitingQueueLength: 8,
		MaxModelLen:           1024,
		KVCache:               common.KVCacheConfig{KVCacheSize: 128, TokenBlockSize: 16},
	}
}

// kvCapacityTokens is the token capacity newTestConfig's cache geometry adds up
// to.
const kvCapacityTokens = 128 * 16

// newTestAdapter builds a metricsAdapter with a fresh Prometheus registry and a
// canceled-on-cleanup context, which stops all updater goroutines.
func newTestAdapter() (*sglangMetricsAdapter, *prometheus.Registry) {
	ctx, cancel := context.WithCancel(context.Background())
	DeferCleanup(cancel)

	registry := prometheus.NewRegistry()
	adapter, err := newMetricsAdapter(ctx, registry, logr.Discard(), newTestConfig())
	Expect(err).NotTo(HaveOccurred())

	return adapter, registry
}

// gaugeValue reads the model-labeled gauge. Adapter writes are asynchronous
// (channel -> updater goroutine -> Prometheus), so callers wrap this in
// Eventually.
func gaugeValue(g *prometheus.GaugeVec) func() float64 {
	return func() float64 {
		return testutil.ToFloat64(g.WithLabelValues(common.TestModelName))
	}
}

func counterValue(c *prometheus.CounterVec, labelValues ...string) func() float64 {
	return func() float64 {
		return testutil.ToFloat64(c.WithLabelValues(labelValues...))
	}
}

// histogramStats returns the observation count and sum of the series of the
// named histogram carrying the given is_streaming value, and zeroes when no
// such series exists. An empty stream matches the series of a histogram that is
// not split by request kind.
func histogramStats(registry *prometheus.Registry, name string, stream string) (uint64, float64) {
	gathered, err := registry.Gather()
	Expect(err).NotTo(HaveOccurred())

	for _, family := range gathered {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			value := ""
			for _, label := range metric.GetLabel() {
				if label.GetName() == promLabelIsStreaming {
					value = label.GetValue()
				}
			}
			if value == stream {
				return metric.GetHistogram().GetSampleCount(), metric.GetHistogram().GetSampleSum()
			}
		}
	}
	return 0, 0
}

func histogramCount(registry *prometheus.Registry, name string, stream string) func() uint64 {
	return func() uint64 {
		count, _ := histogramStats(registry, name, stream)
		return count
	}
}

func histogramSum(registry *prometheus.Registry, name string, stream string) func() float64 {
	return func() float64 {
		_, sum := histogramStats(registry, name, stream)
		return sum
	}
}

var _ = Describe("Metrics adapter", func() {
	Describe("initial values", func() {
		It("starts the request gauges at zero and leaves the rest unstamped", func() {
			adapter, registry := newTestAdapter()

			Expect(testutil.ToFloat64(adapter.runningRequests.WithLabelValues(common.TestModelName))).To(BeZero())
			Expect(testutil.ToFloat64(adapter.queuedRequests.WithLabelValues(common.TestModelName))).To(BeZero())

			gathered, err := registry.Gather()
			Expect(err).NotTo(HaveOccurred())
			names := make([]string, 0, len(gathered))
			for _, family := range gathered {
				names = append(names, family.GetName())
			}
			Expect(names).To(ConsistOf(SGLangQueueReqsMetricName, SGLangRunningReqsMetricName,
				SGLangContextLenMetricName, SGLangTokenUsageMetricName, SGLangUsedTokensMetricName,
				SGLangKVUsedTokensMetricName, SGLangKVAvailableTokensMetricName,
				SGLangMaxTotalNumTokensMetricName, SGLangPageSizeMetricName, SGLangNumPagesMetricName,
				SGLangCacheHitRateMetricName, SGLangCacheConfigMetricName))
		})

		It("stamps the context length from the configuration", func() {
			adapter, _ := newTestAdapter()

			Expect(testutil.ToFloat64(adapter.contextLen.WithLabelValues(common.TestModelName))).
				To(Equal(float64(newTestConfig().MaxModelLen)))
		})

		It("describes an empty cache of the configured geometry", func() {
			adapter, _ := newTestAdapter()

			Expect(gaugeValue(adapter.tokenUsage)()).To(BeZero())
			Expect(gaugeValue(adapter.usedTokens)()).To(BeZero())
			Expect(gaugeValue(adapter.kvUsedTokens)()).To(BeZero())
			Expect(gaugeValue(adapter.cacheHitRate)()).To(BeZero())
			Expect(gaugeValue(adapter.kvAvailableTokens)()).To(Equal(float64(kvCapacityTokens)))
			Expect(gaugeValue(adapter.maxTotalNumTokens)()).To(Equal(float64(kvCapacityTokens)))
			Expect(gaugeValue(adapter.pageSize)()).To(Equal(float64(16)))
			Expect(gaugeValue(adapter.numPages)()).To(Equal(float64(128)))
		})

		// llm-d-router reads the block size and block count out of this metric's
		// labels, so their spelling is a contract with it.
		It("stamps the cache config the router reads", func() {
			adapter, _ := newTestAdapter()

			Expect(testutil.ToFloat64(adapter.cacheConfig.WithLabelValues("16", "128"))).To(Equal(float64(1)))
		})
	})

	Describe("event handlers", func() {
		It("updates the queue gauge and queue time histogram on queue/dequeue", func() {
			adapter, registry := newTestAdapter()

			adapter.OnRequestQueued(metrics.RequestQueued{})
			adapter.OnRequestQueued(metrics.RequestQueued{})
			Eventually(gaugeValue(adapter.queuedRequests)).Should(Equal(float64(2)))

			adapter.OnRequestDequeued(metrics.RequestDequeued{QueueTime: 0.1})
			Eventually(gaugeValue(adapter.queuedRequests)).Should(Equal(float64(1)))
			Eventually(histogramCount(registry, SGLangQueueTimeMetricName, "")).Should(Equal(uint64(1)))
		})

		It("updates the running gauge on OnRequestRunning", func() {
			adapter, _ := newTestAdapter()

			adapter.OnRequestRunning(metrics.RequestRunning{})
			adapter.OnRequestRunning(metrics.RequestRunning{})
			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(2)))
		})

		It("observes time to first token", func() {
			adapter, registry := newTestAdapter()

			adapter.OnPrefillEnded(metrics.PrefillEnded{PrefillDuration: 0.3, IsStreaming: true})
			Eventually(histogramCount(registry, SGLangTTFTMetricName, streaming)).Should(Equal(uint64(1)))
		})

		It("observes inter-token latency per generated token", func() {
			adapter, registry := newTestAdapter()

			adapter.OnTokenGenerated(metrics.TokenGenerated{InterTokenLatency: 0.01})
			adapter.OnTokenGenerated(metrics.TokenGenerated{InterTokenLatency: 0.02})
			Eventually(histogramCount(registry, SGLangInterTokenLatencyMetricName, "")).Should(Equal(uint64(2)))
		})

		It("observes a per-request time per output token", func() {
			adapter, registry := newTestAdapter()

			adapter.OnDecodeEnded(metrics.DecodeEnded{
				GenerationTokens: 4,
				DecodeDuration:   0.4,
				FinishReason:     common.StopFinishReason,
				IsStreaming:      true,
			})
			Eventually(histogramCount(registry, SGLangReqTPOTMetricName, streaming)).Should(Equal(uint64(1)))
			Eventually(histogramSum(registry, SGLangReqTPOTMetricName, streaming)).Should(BeNumerically("~", 0.1, 1e-9))
		})

		It("skips the per-request time per output token where sglang skips it", func() {
			adapter, registry := newTestAdapter()

			// A single generated token leaves no gap to measure.
			adapter.OnDecodeEnded(metrics.DecodeEnded{
				GenerationTokens: 0, DecodeDuration: 0.4, FinishReason: common.StopFinishReason,
			})
			// A truncated decode phase.
			adapter.OnDecodeEnded(metrics.DecodeEnded{
				GenerationTokens: 4, DecodeDuration: 0.4, FinishReason: common.ToolsFinishReason,
			})
			adapter.OnDecodeEnded(metrics.DecodeEnded{
				GenerationTokens: 4, DecodeDuration: 0.4, FinishReason: common.RemoteDecodeFinishReason,
			})

			Consistently(histogramCount(registry, SGLangReqTPOTMetricName, nonStreaming)).Should(BeZero())
		})

		It("decrements running and records latency and tokens on success", func() {
			adapter, registry := newTestAdapter()

			adapter.OnRequestRunning(metrics.RequestRunning{})
			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(1)))

			adapter.OnRequestSucceeded(metrics.RequestSucceeded{
				PromptTokens:     10,
				GenerationTokens: 20,
				FinishReason:     common.StopFinishReason,
				E2ELatency:       0.5,
			})
			Eventually(gaugeValue(adapter.runningRequests)).Should(BeZero())
			Eventually(counterValue(adapter.promptTokensTotal, common.TestModelName, nonStreaming)).Should(Equal(float64(10)))
			Eventually(counterValue(adapter.generationTokensTotal, common.TestModelName, nonStreaming)).Should(Equal(float64(20)))
			Eventually(counterValue(adapter.requestsTotal, common.TestModelName, nonStreaming)).Should(Equal(float64(1)))
			Eventually(histogramCount(registry, SGLangE2EReqLatencyMetricName, nonStreaming)).Should(Equal(uint64(1)))
			Eventually(histogramCount(registry, SGLangPromptTokensHistMetricName, "")).Should(Equal(uint64(1)))
			Eventually(histogramCount(registry, SGLangGenerationTokensHistMetricName, "")).Should(Equal(uint64(1)))
		})

		It("excludes the cached prefix from the uncached prompt token histogram", func() {
			adapter, registry := newTestAdapter()

			adapter.OnRequestSucceeded(metrics.RequestSucceeded{
				PromptTokens:       1000,
				CachedPromptTokens: 400,
				GenerationTokens:   20,
				FinishReason:       common.StopFinishReason,
			})

			Eventually(histogramSum(registry, SGLangPromptTokensHistMetricName, "")).Should(Equal(float64(1000)))
			Eventually(histogramSum(registry, SGLangUncachedPromptTokensHistMetricName, "")).Should(Equal(float64(600)))
		})

		It("splits the per-request metrics by request kind", func() {
			adapter, registry := newTestAdapter()

			adapter.OnRequestSucceeded(metrics.RequestSucceeded{
				PromptTokens: 10, GenerationTokens: 20, E2ELatency: 0.5, IsStreaming: true,
			})
			adapter.OnRequestSucceeded(metrics.RequestSucceeded{
				PromptTokens: 3, GenerationTokens: 7, E2ELatency: 0.2,
			})

			Eventually(counterValue(adapter.promptTokensTotal, common.TestModelName, streaming)).Should(Equal(float64(10)))
			Eventually(counterValue(adapter.promptTokensTotal, common.TestModelName, nonStreaming)).Should(Equal(float64(3)))
			Eventually(counterValue(adapter.generationTokensTotal, common.TestModelName, streaming)).Should(Equal(float64(20)))
			Eventually(counterValue(adapter.generationTokensTotal, common.TestModelName, nonStreaming)).Should(Equal(float64(7)))
			Eventually(histogramCount(registry, SGLangE2EReqLatencyMetricName, streaming)).Should(Equal(uint64(1)))
			Eventually(histogramCount(registry, SGLangE2EReqLatencyMetricName, nonStreaming)).Should(Equal(uint64(1)))
		})

		It("records only the running gauge for a failed request", func() {
			adapter, registry := newTestAdapter()

			adapter.OnRequestRunning(metrics.RequestRunning{})
			adapter.OnRequestFailed(metrics.RequestFailed{E2ELatency: 0.5, InferenceTime: 0.4})

			Eventually(gaugeValue(adapter.runningRequests)).Should(BeZero())
			Consistently(histogramCount(registry, SGLangE2EReqLatencyMetricName, nonStreaming)).Should(BeZero())
			Consistently(counterValue(adapter.promptTokensTotal, common.TestModelName, nonStreaming)).Should(BeZero())
		})

		// Not reachable while the engine rejects an enabled KV cache, so these
		// cover the handlers against the day it does not.
		It("mirrors the cache utilization onto the token gauges", func() {
			adapter, _ := newTestAdapter()

			adapter.OnKVCacheUsageChanged(metrics.KVCacheUsageChanged{KVCacheUsagePerc: 0.25})

			Eventually(gaugeValue(adapter.tokenUsage)).Should(Equal(0.25))
			Eventually(gaugeValue(adapter.usedTokens)).Should(Equal(float64(kvCapacityTokens) * 0.25))
			Eventually(gaugeValue(adapter.kvUsedTokens)).Should(Equal(float64(kvCapacityTokens) * 0.25))
			Eventually(gaugeValue(adapter.kvAvailableTokens)).Should(Equal(float64(kvCapacityTokens) * 0.75))
		})

		It("accumulates cached tokens and a cumulative hit rate", func() {
			adapter, _ := newTestAdapter()

			adapter.OnPrefixCacheQueried(metrics.PrefixCacheQueried{QueriedTokens: 100, CachedPromptTokens: 40})
			adapter.OnPrefixCacheQueried(metrics.PrefixCacheQueried{QueriedTokens: 100, CachedPromptTokens: 10})

			Eventually(counterValue(adapter.cachedTokensTotal, common.TestModelName, cacheSourceTotal)).
				Should(Equal(float64(50)))
			Eventually(gaugeValue(adapter.cacheHitRate)).Should(Equal(0.25))
		})
	})

	Describe("rejected features", func() {
		It("accepts and drops the events of a feature it does not simulate", func() {
			adapter, _ := newTestAdapter()

			Expect(func() {
				adapter.OnRequestReceived(metrics.RequestReceived{})
				adapter.OnRequestRejected(metrics.RequestRejected{})
				adapter.OnPrefillStarted(metrics.PrefillStarted{})
				adapter.OnDecodeStarted(metrics.DecodeStarted{})
				adapter.OnLoRASetsChanged(metrics.LoRASetsChanged{})
				adapter.ApplyFakeMetricsUpdate(fakeMetrics{})
			}).NotTo(Panic())
		})
	})
})
