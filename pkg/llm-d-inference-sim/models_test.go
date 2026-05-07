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

package llmdinferencesim

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

func findByID(models []vllmapi.ModelsResponseModelInfo, id string) *vllmapi.ModelsResponseModelInfo {
	for i := range models {
		if models[i].ID == id {
			return &models[i]
		}
	}
	return nil
}

var _ = Describe("CreateModelsResponse", func() {
	It("should set root to actual model path, not alias", func() {
		s := &SimContext{
			Config: &common.Configuration{
				Model:            "Qwen/Qwen3-0.6B-Base",
				ServedModelNames: []string{"alias1", "alias2"},
				MaxModelLen:      32768,
			},
		}

		resp := s.CreateModelsResponse()
		Expect(resp.Data).To(HaveLen(2))

		for _, model := range resp.Data {
			Expect(model.Root).To(Equal("Qwen/Qwen3-0.6B-Base"))
			Expect(model.MaxModelLen).To(Equal(32768))
			Expect(model.Parent).To(BeNil())
		}
		Expect(resp.Data[0].ID).To(Equal("alias1"))
		Expect(resp.Data[1].ID).To(Equal("alias2"))
	})

	It("should include max_model_len for LoRA adapters", func() {
		s := &SimContext{
			Config: &common.Configuration{
				Model:            "Qwen/Qwen3-0.6B-Base",
				ServedModelNames: []string{"base-model"},
				MaxModelLen:      4096,
			},
		}
		s.loraAdaptors.Store("lora1", true)
		s.loraAdaptors.Store("lora2", true)

		resp := s.CreateModelsResponse()
		Expect(resp.Data).To(HaveLen(3))

		for _, model := range resp.Data {
			Expect(model.MaxModelLen).To(Equal(4096))
		}

		base := resp.Data[0]
		Expect(base.ID).To(Equal("base-model"))
		Expect(base.Root).To(Equal("Qwen/Qwen3-0.6B-Base"))
		Expect(base.Parent).To(BeNil())

		for _, model := range resp.Data[1:] {
			Expect(model.Parent).ToNot(BeNil())
			Expect(*model.Parent).To(Equal("base-model"))
		}
	})

	It("should use model name as root when no aliases are set", func() {
		s := &SimContext{
			Config: &common.Configuration{
				Model:            "meta-llama/Llama-3-8B",
				ServedModelNames: []string{"meta-llama/Llama-3-8B"},
				MaxModelLen:      1024,
			},
		}
		resp := s.CreateModelsResponse()
		Expect(resp.Data).To(HaveLen(1))
		Expect(resp.Data[0].ID).To(Equal("meta-llama/Llama-3-8B"))
		Expect(resp.Data[0].Root).To(Equal("meta-llama/Llama-3-8B"))
		Expect(resp.Data[0].MaxModelLen).To(Equal(1024))
	})

	It("should set LoRA root to the adapter path when one is recorded", func() {
		s := &SimContext{
			Config: &common.Configuration{
				Model:            "Qwen/Qwen3-0.6B-Base",
				ServedModelNames: []string{"base-model"},
				MaxModelLen:      4096,
			},
		}
		s.loraAdaptors.Store("lora-with-path", "/lora/path/adapter1")
		s.loraAdaptors.Store("lora-without-path", "")

		resp := s.CreateModelsResponse()
		Expect(resp.Data).To(HaveLen(3))

		withPath := findByID(resp.Data, "lora-with-path")
		Expect(withPath).ToNot(BeNil())
		Expect(withPath.Root).To(Equal("/lora/path/adapter1"))

		withoutPath := findByID(resp.Data, "lora-without-path")
		Expect(withoutPath).ToNot(BeNil())
		Expect(withoutPath.Root).To(Equal("lora-without-path"))
	})
})
