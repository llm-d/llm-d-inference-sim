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

	It("should reject a non-positive max-loras-per-batch", func() {
		config.Lora.MaxLoras = 0
		Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring("max_loras_per_batch must be positive")))
	})

	It("should leave LoRA disabled when neither enable-lora nor lora-paths is given", func() {
		Expect(New().ValidateConfig(config)).To(Succeed())
		Expect(config.Lora.EnableLora).To(HaveValue(BeFalse()))
	})

	It("should enable LoRA when lora-paths is given", func() {
		config.Lora.LoraModules = []common.LoraModule{{Name: "lora1", Path: "/p1"}}
		Expect(New().ValidateConfig(config)).To(Succeed())
		Expect(config.Lora.EnableLora).To(HaveValue(BeTrue()))
	})

	It("should drop lora-paths when enable-lora is explicitly false", func() {
		config.Lora.EnableLora = ptr(false)
		config.Lora.LoraModules = []common.LoraModule{{Name: "lora1", Path: "/p1"}}
		Expect(New().ValidateConfig(config)).To(Succeed())
		Expect(config.Lora.LoraModules).To(BeEmpty())
	})

	It("should check the target module choices while LoRA is disabled", func() {
		config.Lora.EnableLora = ptr(false)
		config.Lora.TargetModules = []string{"no_such_proj"}
		Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring(
			"invalid choice for lora-target-modules: 'no_such_proj'")))
	})

	It("should not check the LoRA limits while LoRA is disabled", func() {
		config.Lora.EnableLora = ptr(false)
		config.Lora.MaxCPULoras = config.Lora.MaxLoras - 1
		Expect(New().ValidateConfig(config)).To(Succeed())
	})

	Context("with LoRA enabled", func() {
		BeforeEach(func() {
			config.Lora.EnableLora = ptr(true)
			config.Lora.MaxLoraRank = 16
			config.Lora.TargetModules = []string{"q_proj"}
		})

		It("should require max-lora-rank and lora-target-modules without lora-paths", func() {
			config.Lora.MaxLoraRank = 0
			Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring(
				"you need to specify both --max-lora-rank and --lora-target-modules")))
		})

		It("should not require them with lora-paths", func() {
			config.Lora.MaxLoraRank = 0
			config.Lora.TargetModules = nil
			config.Lora.LoraModules = []common.LoraModule{{Name: "lora1", Path: "/p1"}}
			Expect(New().ValidateConfig(config)).To(Succeed())
		})

		It("should reject an unknown target module", func() {
			config.Lora.TargetModules = []string{"no_such_proj"}
			Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring(
				"invalid choice for lora-target-modules: 'no_such_proj'")))
		})

		It("should reject 'all' alongside other target modules", func() {
			config.Lora.TargetModules = []string{"all", "q_proj"}
			Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring(
				"if 'all' is specified in --lora-target-modules, it should be the only module specified")))
		})

		It("should accept a max-loaded-loras equal to max-loras-per-batch", func() {
			config.Lora.MaxCPULoras = config.Lora.MaxLoras
			Expect(New().ValidateConfig(config)).To(Succeed())
		})

		It("should reject a max-loaded-loras below max-loras-per-batch", func() {
			config.Lora.MaxCPULoras = config.Lora.MaxLoras - 1
			Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring(
				"max_loaded_loras should be greater than or equal to max_loras_per_batch. " +
					"max_loaded_loras=7, max_loras_per_batch=8")))
		})

		It("should reject more LoRA paths than max-loaded-loras", func() {
			config.Lora.MaxLoras = 1
			config.Lora.MaxCPULoras = 1
			config.Lora.LoraModules = []common.LoraModule{{Name: "lora1"}, {Name: "lora2"}}
			Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring(
				"the number of LoRA paths should not exceed max_loaded_loras. max_loaded_loras=1, lora_paths=2")))
		})

		It("should not limit the LoRA paths when max-loaded-loras is unset", func() {
			config.Lora.MaxLoras = 1
			config.Lora.LoraModules = []common.LoraModule{{Name: "lora1"}, {Name: "lora2"}}
			Expect(New().ValidateConfig(config)).To(Succeed())
		})

		It("should reject a LoRA path whose name is already taken", func() {
			config.Lora.LoraModules = []common.LoraModule{{Name: "lora1", Path: "/p1"}, {Name: "lora1", Path: "/p2"}}
			Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring(
				"Failed to load LoRA adapter lora1 because it is already loaded")))
		})

		It("should accept up to max-loras-per-batch - 1 pinned LoRA paths", func() {
			config.Lora.MaxLoras = 2
			config.Lora.LoraModules = []common.LoraModule{{Name: "lora1", Pinned: true}, {Name: "lora2"}}
			Expect(New().ValidateConfig(config)).To(Succeed())
		})

		It("should reject pinning every slot of max-loras-per-batch", func() {
			config.Lora.MaxLoras = 2
			config.Lora.LoraModules = []common.LoraModule{{Name: "lora1", Pinned: true}, {Name: "lora2", Pinned: true}}
			Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring(
				"Failed to load LoRA adapter lora2 as a pinned adapter")))
		})
	})

	It("should reject an enabled KV cache", func() {
		config.KVCache.EnableKVCache = true
		Expect(New().ValidateConfig(config)).To(
			MatchError(ContainSubstring("KV-cache simulation is not implemented by the sglang engine yet")))
	})

	It("should reject fake metrics", func() {
		config.FakeMetrics = fakeMetrics{}
		Expect(New().ValidateConfig(config)).To(
			MatchError(ContainSubstring("fake metrics are not implemented by the sglang engine yet")))
	})
})

var _ = Describe("KV events", func() {
	It("should have no encoder", func() {
		encoder, err := New().NewKVEventEncoder(*common.NewConfig())
		Expect(err).To(MatchError(ContainSubstring("does not support KV-cache events")))
		Expect(encoder).To(BeNil())
	})
})
