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
	"time"

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
	return newTestAdapterWith(nil)
}

// newTestAdapterWith builds an adapter reporting the given fake metrics, Started
// so that the configuration has been applied and the generator ticker is
// running. A nil fake leaves it reporting real metrics.
func newTestAdapterWith(fake *SGLangFakeMetrics) (*sglangMetricsAdapter, *prometheus.Registry) {
	ctx, cancel := context.WithCancel(context.Background())
	DeferCleanup(cancel)

	config := newTestConfig()
	if fake != nil {
		config.FakeMetrics = fake
		config.FakeMetricsRefreshInterval = 10 * time.Millisecond
	}

	registry := prometheus.NewRegistry()
	adapter, err := newMetricsAdapter(ctx, registry, logr.Discard(), config)
	Expect(err).NotTo(HaveOccurred())
	Expect(adapter.Start(ctx)).To(Succeed())
	DeferCleanup(func() { Expect(adapter.Close()).To(Succeed()) })

	return adapter, registry
}

func int64Ptr(v int64) *int64 { return &v }

// counterSeries returns the values of the named counter's series, keyed by their
// is_streaming label. Reading through the collector would create the series it
// asks for, so a series the adapter never stamped has to be read off a scrape.
func counterSeries(registry *prometheus.Registry, name string) func() map[string]float64 {
	return func() map[string]float64 {
		gathered, err := registry.Gather()
		Expect(err).NotTo(HaveOccurred())

		series := map[string]float64{}
		for _, family := range gathered {
			if family.GetName() != name {
				continue
			}
			for _, metric := range family.GetMetric() {
				for _, label := range metric.GetLabel() {
					if label.GetName() == promLabelIsStreaming {
						series[label.GetValue()] = metric.GetCounter().GetValue()
					}
				}
			}
		}
		return series
	}
}

// metricNames returns the names of every metric family the registry serves.
func metricNames(registry *prometheus.Registry) []string {
	gathered, err := registry.Gather()
	Expect(err).NotTo(HaveOccurred())

	names := make([]string, 0, len(gathered))
	for _, family := range gathered {
		names = append(names, family.GetName())
	}
	return names
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
				SGLangCacheHitRateMetricName))
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
				GenerationTokens: 4, DecodeDuration: 0.4, FinishReason: common.RemoteDecodeFinishReason,
			})

			Consistently(histogramCount(registry, SGLangReqTPOTMetricName, nonStreaming)).Should(BeZero())
		})

		It("observes a per-request time per output token for a tool-call finish", func() {
			adapter, registry := newTestAdapter()

			adapter.OnDecodeEnded(metrics.DecodeEnded{
				GenerationTokens: 4, DecodeDuration: 0.4, FinishReason: common.ToolsFinishReason,
			})
			Eventually(histogramCount(registry, SGLangReqTPOTMetricName, nonStreaming)).Should(Equal(uint64(1)))
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

	Describe("fake metrics", func() {
		It("reports the fixed values the configuration names", func() {
			adapter, _ := newTestAdapterWith(&SGLangFakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{FixedValue: 5},
				QueuedRequests:  &common.FakeMetricWithFunction{FixedValue: 30},
			})

			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(5)))
			Eventually(gaugeValue(adapter.queuedRequests)).Should(Equal(float64(30)))
		})

		It("leaves a load metric the configuration does not name absent", func() {
			_, registry := newTestAdapterWith(&SGLangFakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{FixedValue: 5},
			})

			Eventually(func() []string { return metricNames(registry) }).
				Should(ContainElement(SGLangRunningReqsMetricName))
			Expect(metricNames(registry)).NotTo(ContainElement(SGLangQueueReqsMetricName))
			Expect(metricNames(registry)).NotTo(ContainElement(SGLangTokenUsageMetricName))
		})

		// These report the geometry the simulator was given, which fake metrics
		// do not replace.
		It("still describes the configuration", func() {
			adapter, _ := newTestAdapterWith(&SGLangFakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{FixedValue: 5},
			})

			Expect(gaugeValue(adapter.contextLen)()).To(Equal(float64(newTestConfig().MaxModelLen)))
			Expect(gaugeValue(adapter.maxTotalNumTokens)()).To(Equal(float64(kvCapacityTokens)))
			Expect(gaugeValue(adapter.pageSize)()).To(Equal(float64(16)))
			Expect(gaugeValue(adapter.numPages)()).To(Equal(float64(128)))
		})

		It("recalculates a generator-valued metric over time", func() {
			adapter, _ := newTestAdapterWith(&SGLangFakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{
					IsFunction: true,
					Function: &common.FunctionInfo{
						Name:   common.RampFuncName,
						Start:  0,
						End:    10,
						Period: 100 * time.Millisecond,
					},
				},
			})

			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(10)))
		})

		It("derives the token counts from the usage fraction", func() {
			adapter, _ := newTestAdapterWith(&SGLangFakeMetrics{
				TokenUsage: &common.FakeMetricWithFunction{FixedValue: 0.25},
			})

			Eventually(gaugeValue(adapter.tokenUsage)).Should(Equal(0.25))
			Eventually(gaugeValue(adapter.usedTokens)).Should(Equal(float64(kvCapacityTokens) * 0.25))
			Eventually(gaugeValue(adapter.kvUsedTokens)).Should(Equal(float64(kvCapacityTokens) * 0.25))
			Eventually(gaugeValue(adapter.kvAvailableTokens)).Should(Equal(float64(kvCapacityTokens) * 0.75))
		})

		It("derives the hit rate from the prefix cache tallies", func() {
			adapter, _ := newTestAdapterWith(&SGLangFakeMetrics{
				CachedTokens:  int64Ptr(2000),
				QueriedTokens: int64Ptr(8000),
			})

			Eventually(gaugeValue(adapter.cacheHitRate)).Should(Equal(0.25))
			Eventually(counterValue(adapter.cachedTokensTotal, common.TestModelName, cacheSourceTotal)).
				Should(Equal(float64(2000)))
		})

		It("reports each series of a stream-split counter the configuration names", func() {
			_, registry := newTestAdapterWith(&SGLangFakeMetrics{
				NumRequestsTotal: &StreamSplit{
					Streaming:    int64Ptr(412),
					NonStreaming: int64Ptr(1038),
				},
				TotalPromptTokens: &StreamSplit{NonStreaming: int64Ptr(80000)},
			})

			Eventually(counterSeries(registry, SGLangRequestsTotalMetricName)).Should(Equal(
				map[string]float64{streaming: 412, nonStreaming: 1038}))
			// A series the configuration leaves out stays absent rather than
			// reporting a zero nothing asked for.
			Eventually(counterSeries(registry, SGLangPromptTokensTotalMetricName)).Should(Equal(
				map[string]float64{nonStreaming: 80000}))
		})

		It("suppresses the real metrics", func() {
			adapter, registry := newTestAdapterWith(&SGLangFakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{FixedValue: 5},
				QueuedRequests:  &common.FakeMetricWithFunction{FixedValue: 30},
				TokenUsage:      &common.FakeMetricWithFunction{FixedValue: 0.25},
				CachedTokens:    int64Ptr(2000),
				QueriedTokens:   int64Ptr(8000),
				NumRequestsTotal: &StreamSplit{
					Streaming:    int64Ptr(412),
					NonStreaming: int64Ptr(1038),
				},
			})
			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(5)))

			adapter.OnRequestQueued(metrics.RequestQueued{})
			adapter.OnRequestDequeued(metrics.RequestDequeued{QueueTime: 0.1})
			adapter.OnRequestRunning(metrics.RequestRunning{})
			adapter.OnPrefillEnded(metrics.PrefillEnded{PrefillDuration: 0.3, IsStreaming: true})
			adapter.OnTokenGenerated(metrics.TokenGenerated{InterTokenLatency: 0.01})
			adapter.OnDecodeEnded(metrics.DecodeEnded{
				GenerationTokens: 4, DecodeDuration: 0.4, FinishReason: common.StopFinishReason})
			adapter.OnRequestSucceeded(metrics.RequestSucceeded{
				PromptTokens: 10, GenerationTokens: 5, E2ELatency: 1, FinishReason: common.StopFinishReason})
			adapter.OnKVCacheUsageChanged(metrics.KVCacheUsageChanged{KVCacheUsagePerc: 0.9})
			adapter.OnPrefixCacheQueried(metrics.PrefixCacheQueried{CachedPromptTokens: 7, QueriedTokens: 9})

			Consistently(gaugeValue(adapter.runningRequests)).Should(Equal(float64(5)))
			Consistently(gaugeValue(adapter.queuedRequests)).Should(Equal(float64(30)))
			Consistently(gaugeValue(adapter.tokenUsage)).Should(Equal(0.25))
			Consistently(gaugeValue(adapter.cacheHitRate)).Should(Equal(0.25))
			Consistently(counterSeries(registry, SGLangRequestsTotalMetricName)).Should(Equal(
				map[string]float64{streaming: 412, nonStreaming: 1038}))
		})

		// A histogram the configuration names nothing for reports nothing, rather
		// than the observations of real requests.
		It("leaves an unnamed histogram unpopulated", func() {
			adapter, registry := newTestAdapterWith(&SGLangFakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{FixedValue: 5},
			})

			adapter.OnRequestDequeued(metrics.RequestDequeued{QueueTime: 0.1})
			adapter.OnPrefillEnded(metrics.PrefillEnded{PrefillDuration: 0.3, IsStreaming: true})

			Consistently(histogramCount(registry, SGLangQueueTimeMetricName, "")).Should(BeZero())
			Consistently(histogramCount(registry, SGLangTTFTMetricName, streaming)).Should(BeZero())
		})

		// Each sample is observed at its bucket's upper boundary, so the count
		// and sum follow from which buckets the configuration fills.
		It("fills a histogram sglang does not split by request kind", func() {
			_, registry := newTestAdapterWith(&SGLangFakeMetrics{
				QueueTimeBucketValues: []int{0, 2, 3},
			})

			Eventually(histogramCount(registry, SGLangQueueTimeMetricName, "")).Should(Equal(uint64(5)))
			// Two samples at 0.001 and three at 0.005.
			Eventually(histogramSum(registry, SGLangQueueTimeMetricName, "")).
				Should(BeNumerically("~", 2*0.001+3*0.005, 1e-9))
		})

		It("fills each series of a stream-split histogram", func() {
			_, registry := newTestAdapterWith(&SGLangFakeMetrics{
				TTFTBucketValues: &StreamSplitBuckets{
					Streaming:    []int{5, 3, 2},
					NonStreaming: []int{0, 0, 1, 4, 10},
				},
			})

			Eventually(histogramCount(registry, SGLangTTFTMetricName, streaming)).Should(Equal(uint64(10)))
			Eventually(histogramSum(registry, SGLangTTFTMetricName, streaming)).
				Should(BeNumerically("~", 5*0.1+3*0.2+2*0.4, 1e-9))

			Eventually(histogramCount(registry, SGLangTTFTMetricName, nonStreaming)).Should(Equal(uint64(15)))
			Eventually(histogramSum(registry, SGLangTTFTMetricName, nonStreaming)).
				Should(BeNumerically("~", 1*0.4+4*0.6+10*0.8, 1e-9))
		})

		It("leaves a stream-split histogram's unnamed series absent", func() {
			_, registry := newTestAdapterWith(&SGLangFakeMetrics{
				E2EReqLatencyBucketValues: &StreamSplitBuckets{NonStreaming: []int{1, 2}},
			})

			Eventually(histogramCount(registry, SGLangE2EReqLatencyMetricName, nonStreaming)).
				Should(Equal(uint64(3)))
			Consistently(histogramCount(registry, SGLangE2EReqLatencyMetricName, streaming)).Should(BeZero())
		})

		Describe("partial updates", func() {
			It("changes only the metrics the update names", func() {
				adapter, _ := newTestAdapterWith(&SGLangFakeMetrics{
					RunningRequests: &common.FakeMetricWithFunction{FixedValue: 5},
					QueuedRequests:  &common.FakeMetricWithFunction{FixedValue: 30},
				})
				Eventually(gaugeValue(adapter.queuedRequests)).Should(Equal(float64(30)))

				adapter.ApplyFakeMetricsUpdate(&SGLangFakeMetrics{
					QueuedRequests: &common.FakeMetricWithFunction{FixedValue: 7},
				})

				Eventually(gaugeValue(adapter.queuedRequests)).Should(Equal(float64(7)))
				Consistently(gaugeValue(adapter.runningRequests)).Should(Equal(float64(5)))
			})

			It("drops the other series of a stream-split counter", func() {
				adapter, registry := newTestAdapterWith(&SGLangFakeMetrics{
					NumRequestsTotal: &StreamSplit{
						Streaming:    int64Ptr(412),
						NonStreaming: int64Ptr(1038),
					},
				})
				Eventually(counterSeries(registry, SGLangRequestsTotalMetricName)).Should(Equal(
					map[string]float64{streaming: 412, nonStreaming: 1038}))

				adapter.ApplyFakeMetricsUpdate(&SGLangFakeMetrics{
					NumRequestsTotal: &StreamSplit{Streaming: int64Ptr(500)},
				})

				Eventually(counterSeries(registry, SGLangRequestsTotalMetricName)).Should(Equal(
					map[string]float64{streaming: 500}))
			})

			It("replaces the prefix cache tallies rather than accumulating them", func() {
				adapter, _ := newTestAdapterWith(&SGLangFakeMetrics{
					CachedTokens:  int64Ptr(2000),
					QueriedTokens: int64Ptr(8000),
				})
				Eventually(gaugeValue(adapter.cacheHitRate)).Should(Equal(0.25))

				adapter.ApplyFakeMetricsUpdate(&SGLangFakeMetrics{
					CachedTokens:  int64Ptr(3000),
					QueriedTokens: int64Ptr(6000),
				})

				Eventually(gaugeValue(adapter.cacheHitRate)).Should(Equal(0.5))
				Eventually(counterValue(adapter.cachedTokensTotal, common.TestModelName, cacheSourceTotal)).
					Should(Equal(float64(3000)))
			})

			// The updater goroutine's tallies are the record of both, so an
			// update naming one of the pair leaves the other where it was.
			It("keeps the other prefix cache tally", func() {
				adapter, _ := newTestAdapterWith(&SGLangFakeMetrics{
					CachedTokens:  int64Ptr(2000),
					QueriedTokens: int64Ptr(8000),
				})
				Eventually(gaugeValue(adapter.cacheHitRate)).Should(Equal(0.25))

				adapter.ApplyFakeMetricsUpdate(&SGLangFakeMetrics{CachedTokens: int64Ptr(4000)})

				Eventually(gaugeValue(adapter.cacheHitRate)).Should(Equal(0.5))
				Eventually(counterValue(adapter.cachedTokensTotal, common.TestModelName, cacheSourceTotal)).
					Should(Equal(float64(4000)))
			})

			It("drops the other series of a stream-split histogram", func() {
				adapter, registry := newTestAdapterWith(&SGLangFakeMetrics{
					TTFTBucketValues: &StreamSplitBuckets{
						Streaming:    []int{5},
						NonStreaming: []int{0, 0, 3},
					},
				})
				Eventually(histogramCount(registry, SGLangTTFTMetricName, streaming)).Should(Equal(uint64(5)))

				adapter.ApplyFakeMetricsUpdate(&SGLangFakeMetrics{
					TTFTBucketValues: &StreamSplitBuckets{Streaming: []int{1, 1}},
				})

				Eventually(histogramCount(registry, SGLangTTFTMetricName, streaming)).Should(Equal(uint64(2)))
				Eventually(histogramCount(registry, SGLangTTFTMetricName, nonStreaming)).Should(BeZero())
			})

			It("resets collectors in place", func() {
				adapter, registry := newTestAdapterWith(&SGLangFakeMetrics{
					QueueTimeBucketValues: []int{1, 2},
					TTFTBucketValues:      &StreamSplitBuckets{Streaming: []int{5}},
					NumRequestsTotal:      &StreamSplit{Streaming: int64Ptr(10)},
					CachedTokens:          int64Ptr(2000),
					QueriedTokens:         int64Ptr(8000),
				})
				Eventually(histogramCount(registry, SGLangQueueTimeMetricName, "")).Should(Equal(uint64(3)))
				Eventually(histogramCount(registry, SGLangTTFTMetricName, streaming)).Should(Equal(uint64(5)))
				Eventually(counterSeries(registry, SGLangRequestsTotalMetricName)).Should(Equal(
					map[string]float64{streaming: 10}))
				Eventually(gaugeValue(adapter.cacheHitRate)).Should(Equal(0.25))

				queueTime := adapter.queueTime
				ttft := adapter.ttft
				requestsTotal := adapter.requestsTotal
				cachedTokensTotal := adapter.cachedTokensTotal

				adapter.ApplyFakeMetricsUpdate(&SGLangFakeMetrics{
					QueueTimeBucketValues: []int{0, 1},
					TTFTBucketValues:      &StreamSplitBuckets{Streaming: []int{2}},
					NumRequestsTotal:      &StreamSplit{Streaming: int64Ptr(4)},
					CachedTokens:          int64Ptr(3000),
					QueriedTokens:         int64Ptr(6000),
				})
				Eventually(histogramCount(registry, SGLangQueueTimeMetricName, "")).Should(Equal(uint64(1)))
				Eventually(histogramCount(registry, SGLangTTFTMetricName, streaming)).Should(Equal(uint64(2)))
				Eventually(counterSeries(registry, SGLangRequestsTotalMetricName)).Should(Equal(
					map[string]float64{streaming: 4}))
				Eventually(gaugeValue(adapter.cacheHitRate)).Should(Equal(0.5))

				Expect(adapter.queueTime).To(BeIdenticalTo(queueTime))
				Expect(adapter.ttft).To(BeIdenticalTo(ttft))
				Expect(adapter.requestsTotal).To(BeIdenticalTo(requestsTotal))
				Expect(adapter.cachedTokensTotal).To(BeIdenticalTo(cachedTokensTotal))
				Expect(testutil.GatherAndCount(registry, SGLangCachedTokensTotalMetricName)).To(Equal(1))
			})

			It("stops the ticker once no generator is left", func() {
				adapter, _ := newTestAdapterWith(&SGLangFakeMetrics{
					RunningRequests: &common.FakeMetricWithFunction{
						IsFunction: true,
						Function: &common.FunctionInfo{
							Name: common.RampFuncName, Start: 0, End: 10, Period: time.Hour,
						},
					},
				})
				Expect(adapter.generators.Running()).To(BeTrue())

				adapter.ApplyFakeMetricsUpdate(&SGLangFakeMetrics{
					RunningRequests: &common.FakeMetricWithFunction{FixedValue: 3},
				})

				Expect(adapter.generators.Running()).To(BeFalse())
				Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(3)))
			})
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
			}).NotTo(Panic())
		})

		It("drops a fake-metrics update while reporting real metrics", func() {
			adapter, _ := newTestAdapter()

			Expect(func() {
				adapter.ApplyFakeMetricsUpdate(&SGLangFakeMetrics{
					QueuedRequests: &common.FakeMetricWithFunction{FixedValue: 7},
				})
			}).NotTo(Panic())
			Consistently(gaugeValue(adapter.queuedRequests)).Should(BeZero())
		})
	})
})
