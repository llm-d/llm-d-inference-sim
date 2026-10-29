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

// Unit tests for VLLMMetricsAdapter that do not duplicate the integration
// coverage under pkg/tests/. These cover paths not otherwise exercised: bus
// nil-safety, adapter initial values, direct event handler invocations,
// fake-mode filtering, and ApplyFakeMetricsUpdate scalar/reset paths reached
// without booting the simulator.

package vllm

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/go-logr/logr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
)

func newTestConfig() common.Configuration {
	return common.Configuration{
		Model:                      common.TestModelName,
		ServedModelNames:           []string{common.TestModelName},
		DisplayModelName:           common.TestModelName,
		Lora:                       common.LoraConfig{MaxLoras: 2},
		MaxNumSeqs:                 4,
		MaxWaitingQueueLength:      8,
		MaxModelLen:                1024,
		KVCache:                    common.KVCacheConfig{KVCacheSize: 128, TokenBlockSize: 16, KVCacheDType: "auto"},
		FakeMetricsRefreshInterval: 20 * time.Millisecond,
	}
}

// newTestAdapter builds a VLLMMetricsAdapter with a fresh Prometheus registry
// and a canceled-on-cleanup context, which stops all updater goroutines and
// the ticker.
func newTestAdapter(cfg common.Configuration) (*VLLMMetricsAdapter, *metrics.MetricsBus) {
	ctx, cancel := context.WithCancel(context.Background())
	DeferCleanup(cancel)

	registry := prometheus.NewRegistry()
	adapter, err := newMetricsAdapter(ctx, registry, logr.Discard(), cfg)
	Expect(err).NotTo(HaveOccurred())

	bus, err := metrics.NewMetricsBus(ctx, cfg, registry, logr.Discard(),
		func(context.Context, *prometheus.Registry, logr.Logger,
			common.Configuration) (metrics.MetricsAdapter, error) {
			return adapter, nil
		})
	Expect(err).NotTo(HaveOccurred())

	return adapter, bus
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

var _ = Describe("MetricsBus", func() {
	It("has a nil-safe ApplyFakeMetricsUpdate", func() {
		var b *metrics.MetricsBus
		Expect(func() { b.ApplyFakeMetricsUpdate(&VLLMFakeMetrics{}) }).NotTo(Panic())
	})

	It("treats a nil fake-metrics update as a no-op on a live bus", func() {
		_, bus := newTestAdapter(newTestConfig())
		Expect(func() { bus.ApplyFakeMetricsUpdate(nil) }).NotTo(Panic())
	})
})

var _ = Describe("VLLMMetricsAdapter", func() {
	Describe("initial values", func() {
		It("starts request and cache gauges at zero and stamps cache_config_info", func() {
			adapter, _ := newTestAdapter(newTestConfig())

			Expect(testutil.ToFloat64(adapter.runningRequests.WithLabelValues(common.TestModelName))).To(BeZero())
			Expect(testutil.ToFloat64(adapter.waitingRequests.WithLabelValues(common.TestModelName))).To(BeZero())
			Expect(testutil.ToFloat64(adapter.kvCacheUsagePercentage.WithLabelValues(common.TestModelName))).To(BeZero())
			// cache_config_info is a one-shot gauge stamped with 1 by setInitialValues.
			Expect(testutil.ToFloat64(adapter.cacheConfig.WithLabelValues("16", "auto", "0", "128"))).To(Equal(float64(1)))
		})
	})

	Describe("event handlers", func() {
		It("updates the waiting gauge on queue/dequeue", func() {
			adapter, _ := newTestAdapter(newTestConfig())

			adapter.OnRequestQueued(metrics.RequestQueued{})
			adapter.OnRequestQueued(metrics.RequestQueued{})
			Eventually(gaugeValue(adapter.waitingRequests)).Should(Equal(float64(2)))

			adapter.OnRequestDequeued(metrics.RequestDequeued{QueueTime: 0.1})
			Eventually(gaugeValue(adapter.waitingRequests)).Should(Equal(float64(1)))
		})

		It("updates the running gauge on OnRequestRunning", func() {
			adapter, _ := newTestAdapter(newTestConfig())

			adapter.OnRequestRunning(metrics.RequestRunning{})
			adapter.OnRequestRunning(metrics.RequestRunning{})
			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(2)))
		})

		It("decrements running and records tokens on success", func() {
			adapter, _ := newTestAdapter(newTestConfig())

			adapter.OnRequestRunning(metrics.RequestRunning{})
			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(1)))

			adapter.OnRequestSucceeded(metrics.RequestSucceeded{
				PromptTokens:       10,
				GenerationTokens:   20,
				GenTokensPerChoice: []int{20},
				FinishReason:       "stop",
				E2ELatency:         0.5,
				InferenceTime:      0.4,
			})
			Eventually(gaugeValue(adapter.runningRequests)).Should(BeZero())
			Eventually(counterValue(adapter.promptTokensTotal, common.TestModelName)).Should(Equal(float64(10)))
			Eventually(counterValue(adapter.generationTokensTotal, common.TestModelName)).Should(Equal(float64(20)))
			Eventually(counterValue(adapter.requestSuccessTotal, common.TestModelName, "stop")).Should(Equal(float64(1)))
		})

		It("exports the max generation tokens histogram under vLLM's name", func() {
			adapter, _ := newTestAdapter(newTestConfig())

			adapter.OnRequestSucceeded(metrics.RequestSucceeded{GenTokensPerChoice: []int{3}, FinishReason: "stop"})
			Eventually(func() (int, error) {
				return testutil.GatherAndCount(adapter.registry, "vllm:request_max_num_generation_tokens")
			}).Should(Equal(1))
		})

		It("labels request_success_total with vLLM's finished_reason", func() {
			adapter, _ := newTestAdapter(newTestConfig())

			adapter.OnRequestSucceeded(metrics.RequestSucceeded{GenTokensPerChoice: []int{1}, FinishReason: "stop"})
			expected := fmt.Sprintf(`
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name=%q} 1
`, common.TestModelName)
			Eventually(func() error {
				return testutil.GatherAndCompare(adapter.registry, strings.NewReader(expected), VLLMSuccessTotalMetricName)
			}).Should(Succeed())
		})

		It("updates the KV-cache usage gauge", func() {
			adapter, _ := newTestAdapter(newTestConfig())

			adapter.OnKVCacheUsageChanged(metrics.KVCacheUsageChanged{KVCacheUsagePerc: 0.75})
			Eventually(gaugeValue(adapter.kvCacheUsagePercentage)).Should(Equal(0.75))
		})

		It("stamps each lora_requests_info series later than the one before it", func() {
			cfg := newTestConfig()
			cfg.Lora.LoraModules = []common.LoraModule{{Name: "lora1", Path: "/lora1"}}
			adapter, _ := newTestAdapter(cfg)
			// label order: max_lora, running_lora_adapters, waiting_lora_adapters
			waiting := func() float64 { return testutil.ToFloat64(adapter.loraInfo.WithLabelValues("2", "", "lora1")) }
			running := func() float64 { return testutil.ToFloat64(adapter.loraInfo.WithLabelValues("2", "lora1", "")) }

			adapter.OnLoRASetsChanged(metrics.LoRASetsChanged{Waiting: map[string]int{"lora1": 1}})
			Eventually(waiting).ShouldNot(BeZero())
			adapter.OnLoRASetsChanged(metrics.LoRASetsChanged{Running: map[string]int{"lora1": 1}})
			Eventually(running).Should(BeNumerically(">", waiting()))
		})

		It("accumulates prefix-cache counters", func() {
			adapter, _ := newTestAdapter(newTestConfig())

			adapter.OnPrefixCacheQueried(metrics.PrefixCacheQueried{QueriedTokens: 100, CachedPromptTokens: 40})
			adapter.OnPrefixCacheQueried(metrics.PrefixCacheQueried{QueriedTokens: 50, CachedPromptTokens: 10})
			Eventually(counterValue(adapter.prefixCacheQueriesTotal, common.TestModelName)).Should(Equal(float64(150)))
			Eventually(counterValue(adapter.prefixCacheHitsTotal, common.TestModelName)).Should(Equal(float64(50)))
		})
	})

	Describe("fake mode", func() {
		It("drops real-path events", func() {
			cfg := newTestConfig()
			cfg.FakeMetrics = &VLLMFakeMetrics{}
			adapter, _ := newTestAdapter(cfg)

			adapter.OnRequestQueued(metrics.RequestQueued{})
			adapter.OnRequestRunning(metrics.RequestRunning{})
			adapter.OnKVCacheUsageChanged(metrics.KVCacheUsageChanged{KVCacheUsagePerc: 0.9})
			adapter.OnPrefixCacheQueried(metrics.PrefixCacheQueried{QueriedTokens: 5, CachedPromptTokens: 3})

			// Nothing should change; Consistently gives the updater goroutines a
			// chance to run and still find zeros.
			Consistently(gaugeValue(adapter.waitingRequests), 50*time.Millisecond).Should(BeZero())
			Expect(gaugeValue(adapter.runningRequests)()).To(BeZero())
			Expect(gaugeValue(adapter.kvCacheUsagePercentage)()).To(BeZero())
			Expect(counterValue(adapter.prefixCacheQueriesTotal, common.TestModelName)()).To(BeZero())
		})

		It("applies fixed scalar values via ApplyUpdate", func() {
			cfg := newTestConfig()
			cfg.FakeMetrics = &VLLMFakeMetrics{}
			adapter, _ := newTestAdapter(cfg)

			kv := 0.4
			upd := &VLLMFakeMetrics{
				RunningRequests:        &common.FakeMetricWithFunction{FixedValue: 3},
				WaitingRequests:        &common.FakeMetricWithFunction{FixedValue: 7},
				KVCacheUsagePercentage: &common.FakeMetricWithFunction{FixedValue: kv},
			}
			adapter.ApplyFakeMetricsUpdate(upd)

			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(3)))
			Eventually(gaugeValue(adapter.waitingRequests)).Should(Equal(float64(7)))
			Eventually(gaugeValue(adapter.kvCacheUsagePercentage)).Should(Equal(kv))
		})

		It("starts and stops the ticker as generators are enabled and cleared", func() {
			cfg := newTestConfig()
			cfg.FakeMetrics = &VLLMFakeMetrics{}
			adapter, _ := newTestAdapter(cfg)

			// The ticker comes up only once the generators are started.
			adapter.generators.Start()

			// Enable a generator: ticker must come up.
			upd := &VLLMFakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{
					IsFunction: true,
					Function: &common.FunctionInfo{
						Name:   common.RampFuncName,
						Start:  0,
						End:    5,
						Period: 100 * time.Millisecond,
					},
				},
			}
			adapter.ApplyFakeMetricsUpdate(upd)

			Expect(adapter.generators.Running()).To(BeTrue(), "expected ticker running after generator enabled")

			// Replace the generator with a fixed value: generator set becomes empty,
			// ticker must stop.
			upd2 := &VLLMFakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{FixedValue: 2},
			}
			adapter.ApplyFakeMetricsUpdate(upd2)

			Expect(adapter.generators.Running()).To(BeFalse(), "expected ticker stopped after generators cleared")
		})

		It("stops the ticker on Close", func() {
			cfg := newTestConfig()
			cfg.FakeMetrics = &VLLMFakeMetrics{}
			adapter, _ := newTestAdapter(cfg)

			adapter.generators.Start()

			upd := &VLLMFakeMetrics{
				WaitingRequests: &common.FakeMetricWithFunction{
					IsFunction: true,
					Function: &common.FunctionInfo{
						Name:   common.RampFuncName,
						Start:  0,
						End:    10,
						Period: 100 * time.Millisecond,
					},
				},
			}
			adapter.ApplyFakeMetricsUpdate(upd)
			Expect(adapter.Close()).To(Succeed())

			Expect(adapter.generators.Running()).To(BeFalse(), "expected ticker stopped after Close")

			adapter.ApplyFakeMetricsUpdate(&VLLMFakeMetrics{
				WaitingRequests: &common.FakeMetricWithFunction{FixedValue: 1},
			})
			adapter.ApplyFakeMetricsUpdate(upd)
			Expect(adapter.generators.Running()).To(BeFalse(), "expected no ticker restart after Close")
		})

		It("resets collectors in place and drops stale series", func() {
			cfg := newTestConfig()
			cfg.FakeMetrics = &VLLMFakeMetrics{}
			adapter, _ := newTestAdapter(cfg)

			ttft := adapter.ttft
			promptTokensTotal := adapter.promptTokensTotal
			requestSuccessTotal := adapter.requestSuccessTotal

			firstTotal, secondTotal := int64(10), int64(4)
			adapter.ApplyFakeMetricsUpdate(&VLLMFakeMetrics{
				TTFTBucketValues:    []int{1, 2},
				TotalPromptTokens:   &firstTotal,
				RequestSuccessTotal: map[string]int64{"stop": 3, "length": 2},
			})
			Eventually(counterValue(adapter.promptTokensTotal, common.TestModelName)).Should(Equal(float64(firstTotal)))
			Eventually(func() int { return testutil.CollectAndCount(adapter.requestSuccessTotal) }).Should(Equal(2))

			adapter.ApplyFakeMetricsUpdate(&VLLMFakeMetrics{
				TTFTBucketValues:    []int{0, 1},
				TotalPromptTokens:   &secondTotal,
				RequestSuccessTotal: map[string]int64{"stop": 1},
			})
			Eventually(counterValue(adapter.promptTokensTotal, common.TestModelName)).Should(Equal(float64(secondTotal)))
			Eventually(func() int { return testutil.CollectAndCount(adapter.requestSuccessTotal) }).Should(Equal(1))
			Expect(counterValue(adapter.requestSuccessTotal, common.TestModelName, "stop")()).To(Equal(float64(1)))

			Expect(adapter.ttft).To(BeIdenticalTo(ttft))
			Expect(adapter.promptTokensTotal).To(BeIdenticalTo(promptTokensTotal))
			Expect(adapter.requestSuccessTotal).To(BeIdenticalTo(requestSuccessTotal))
			Expect(testutil.GatherAndCount(adapter.registry, VLLMTTFTMetricName)).To(Equal(1))
		})
	})

})
