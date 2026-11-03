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
