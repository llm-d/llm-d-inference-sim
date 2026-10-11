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

// Unit tests for the bus itself. Adapter-side coverage lives with each
// engine's adapter (vLLM's is in pkg/engine/vllm).

package metrics

import (
	"context"
	"time"

	"github.com/go-logr/logr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

func newBusTestConfig() common.Configuration {
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

var _ = Describe("LoRA ref counting on the bus", func() {
	It("moves adapters through waiting -> running -> done", func() {
		ctx, cancel := context.WithCancel(context.Background())
		DeferCleanup(cancel)

		bus, err := NewMetricsBus(ctx, newBusTestConfig(), prometheus.NewRegistry(), logr.Discard(),
			NewStubAdapter)
		Expect(err).NotTo(HaveOccurred())
		Expect(bus.Start(ctx)).To(Succeed())

		send := func(name string, state LoRAState) {
			common.WriteToChannel(bus.LoRAChanged,
				LoRAChanged{Model: name, State: state},
				logr.Discard())
		}

		send("a", LoRAWaiting)
		send("b", LoRAWaiting)
		Eventually(loraKeys(&bus.waitingLoras)).Should(ConsistOf("a", "b"))

		send("a", LoRARunning)
		Eventually(loraKeys(&bus.waitingLoras)).Should(ConsistOf("b"))
		Eventually(loraKeys(&bus.runningLoras)).Should(ConsistOf("a"))

		send("a", LoRADone)
		send("b", LoRARunning)
		send("b", LoRADone)
		Eventually(loraKeys(&bus.waitingLoras)).Should(BeEmpty())
		Eventually(loraKeys(&bus.runningLoras)).Should(BeEmpty())
	})
})

// loraKeys returns a poller that snapshots the string keys of a sync.Map-like
// container (any type implementing Range).
func loraKeys(m interface {
	Range(func(any, any) bool)
}) func() []string {
	return func() []string {
		var got []string
		m.Range(func(k, _ any) bool {
			if s, ok := k.(string); ok {
				got = append(got, s)
			}
			return true
		})
		return got
	}
}
