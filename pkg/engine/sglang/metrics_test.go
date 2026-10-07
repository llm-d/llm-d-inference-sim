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
	}
}

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

// histogramCount returns the number of observations on the series of the named
// histogram carrying the given is_streaming value, and zero when no such series
// exists. An empty stream matches the series of a histogram that is not split
// by request kind.
func histogramCount(registry *prometheus.Registry, name string, stream string) func() uint64 {
	return func() uint64 {
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
					return metric.GetHistogram().GetSampleCount()
				}
			}
		}
		return 0
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
			Expect(names).To(ConsistOf(SGLangQueueReqsMetricName, SGLangRunningReqsMetricName))
		})
	})

	Describe("event handlers", func() {
		It("updates the queue gauge on queue/dequeue", func() {
			adapter, _ := newTestAdapter()

			adapter.OnRequestQueued(metrics.RequestQueued{})
			adapter.OnRequestQueued(metrics.RequestQueued{})
			Eventually(gaugeValue(adapter.queuedRequests)).Should(Equal(float64(2)))

			adapter.OnRequestDequeued(metrics.RequestDequeued{QueueTime: 0.1})
			Eventually(gaugeValue(adapter.queuedRequests)).Should(Equal(float64(1)))
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

		It("decrements running and records latency and tokens on success", func() {
			adapter, registry := newTestAdapter()

			adapter.OnRequestRunning(metrics.RequestRunning{})
			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(1)))

			adapter.OnRequestSucceeded(metrics.RequestSucceeded{
				PromptTokens:     10,
				GenerationTokens: 20,
				FinishReason:     "stop",
				E2ELatency:       0.5,
			})
			Eventually(gaugeValue(adapter.runningRequests)).Should(BeZero())
			Eventually(counterValue(adapter.promptTokensTotal, common.TestModelName, nonStreaming)).Should(Equal(float64(10)))
			Eventually(counterValue(adapter.generationTokensTotal, common.TestModelName, nonStreaming)).Should(Equal(float64(20)))
			Eventually(histogramCount(registry, SGLangE2EReqLatencyMetricName, nonStreaming)).Should(Equal(uint64(1)))
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
	})

	Describe("rejected features", func() {
		It("accepts and drops the events of a feature it does not simulate", func() {
			adapter, _ := newTestAdapter()

			Expect(func() {
				adapter.OnRequestReceived(metrics.RequestReceived{})
				adapter.OnRequestRejected(metrics.RequestRejected{})
				adapter.OnPrefillStarted(metrics.PrefillStarted{})
				adapter.OnDecodeStarted(metrics.DecodeStarted{})
				adapter.OnDecodeEnded(metrics.DecodeEnded{GenerationTokens: 4, DecodeDuration: 0.4})
				adapter.OnKVCacheUsageChanged(metrics.KVCacheUsageChanged{KVCacheUsagePerc: 0.5})
				adapter.OnPrefixCacheQueried(metrics.PrefixCacheQueried{QueriedTokens: 10, CachedPromptTokens: 4})
				adapter.OnLoRASetsChanged(metrics.LoRASetsChanged{})
				adapter.ApplyFakeMetricsUpdate(fakeMetrics{})
			}).NotTo(Panic())
		})
	})
})
