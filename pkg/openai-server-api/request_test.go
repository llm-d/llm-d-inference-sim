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

package openaiserverapi

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const (
	model  = "test_model"
	prompt = "Hello, world!"
)

var _ = Describe("render requests", func() {
	It("creates a new text completions render request with correct fields", func() {
		req := NewTextCompletionsRenderRequest(model, prompt)

		Expect(req.Model()).To(Equal(model))
		Expect(req.Prompt).To(Equal(prompt))
		Expect(req.Endpoint()).To(Equal("/v1/completions"))
		Expect(req.IsMultiModal()).To(BeFalse())
	})

	It("creates a new chat completions render request with correct fields", func() {
		messages := []Message{
			{
				Role: RoleUser,
				Content: ChatComplContent{
					Raw: prompt,
				},
			},
		}
		req := NewChatCompletionsRenderRequest(model, messages)

		Expect(req.Model()).To(Equal(model))
		Expect(req.Messages).To(HaveLen(1))
		Expect(req.Endpoint()).To(Equal("/v1/chat/completions"))
	})
})
