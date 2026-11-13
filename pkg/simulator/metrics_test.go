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

package simulator

import (
	"context"
	"os"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/engine/vllm"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

var _ = Describe("Disabled metrics", func() {
	It("exposes no registry and registers no collectors", func() {
		oldArgs := os.Args
		defer func() { os.Args = oldArgs }()
		os.Args = []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}

		engine := vllm.New()
		config, err := common.ParseCommandParamsAndLoadConfig(engine)
		Expect(err).NotTo(HaveOccurred())
		// No engine disables metrics yet, so turn them off by hand.
		config.EnableMetrics = false

		sim, err := New(klog.Background())
		Expect(err).NotTo(HaveOccurred())
		sim.Context.SetConfig(config)
		sim.Context.Tokenizer = tokenizer.NewSimpleTokenizer()
		sim.Context.Engine = engine

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		Expect(sim.Context.initialize(ctx)).To(Succeed())

		Expect(sim.MetricsRegistry()).To(BeNil())
		families, err := sim.Context.prometheusRegistry.Gather()
		Expect(err).NotTo(HaveOccurred())
		Expect(families).To(BeEmpty())
		Expect(sim.Context.metricsBus.Close()).To(Succeed())
	})
})

var _ = Describe("Queue time metrics", func() {
	It("observes zero queue time when a request starts on a free worker", func() {
		oldArgs := os.Args
		defer func() { os.Args = oldArgs }()
		os.Args = []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
			"--max-num-seqs", "1"}

		engine := vllm.New()
		config, err := common.ParseCommandParamsAndLoadConfig(engine)
		Expect(err).NotTo(HaveOccurred())

		sim, err := New(klog.Background())
		Expect(err).NotTo(HaveOccurred())
		sim.Context.SetConfig(config)
		sim.Context.Tokenizer = tokenizer.NewSimpleTokenizer()
		sim.Context.Engine = engine

		ctx, cancel := context.WithCancel(context.Background())
		DeferCleanup(cancel)
		Expect(sim.InitializeSim(ctx)).To(Succeed())
		DeferCleanup(sim.Stop)

		numChoices, _, responses, requestErr, _ := sim.HandleRequest(ctx,
			newTextCompletionsRequestWithChoices("immediate", 1))
		Expect(requestErr).To(BeNil())
		Expect(numChoices).To(Equal(1))
		Expect(responses).NotTo(BeNil())

		completed := make(chan struct{})
		go func() {
			defer GinkgoRecover()
			defer close(completed)
			for response := range responses.Channel {
				Expect(response.Err).To(BeNil())
			}
		}()
		Eventually(completed).Should(BeClosed())

		Eventually(func(g Gomega) {
			families, err := sim.MetricsRegistry().Gather()
			g.Expect(err).NotTo(HaveOccurred())
			queueTimeFound, waitingFound := false, false
			for _, family := range families {
				switch family.GetName() {
				case vllm.VLLMReqQueueTimeMetricName:
					queueTimeFound = true
					g.Expect(family.GetMetric()).To(HaveLen(1))
					histogram := family.GetMetric()[0].GetHistogram()
					g.Expect(histogram.GetSampleCount()).To(Equal(uint64(1)))
					g.Expect(histogram.GetSampleSum()).To(BeZero())
				case vllm.VLLMReqWaitingMetricName:
					waitingFound = true
					g.Expect(family.GetMetric()).To(HaveLen(1))
					g.Expect(family.GetMetric()[0].GetGauge().GetValue()).To(BeZero())
				}
			}
			g.Expect(waitingFound).To(BeTrue(), "waiting-request gauge must be exported")
			g.Expect(queueTimeFound).To(BeTrue(), "queue-time histogram must include an immediately started request")
		}).Should(Succeed())
	})
})
