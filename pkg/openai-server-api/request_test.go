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
	"encoding/json"

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

var _ = Describe("GetN", func() {
	It("returns 1 when N is nil", func() {
		req := &baseCompletionsRequest{}
		Expect(req.GetN()).To(Equal(1))
	})

	It("returns 1 when N is zero", func() {
		n := 0
		req := &baseCompletionsRequest{N: &n}
		Expect(req.GetN()).To(Equal(1))
	})

	It("returns 1 when N is negative", func() {
		n := -5
		req := &baseCompletionsRequest{N: &n}
		Expect(req.GetN()).To(Equal(1))
	})

	It("returns the value when N is positive", func() {
		n := 3
		req := &baseCompletionsRequest{N: &n}
		Expect(req.GetN()).To(Equal(3))
	})

	It("unmarshals n from JSON", func() {
		jsonData := []byte(`{"n": 5, "prompt": "test"}`)
		var req TextCompletionsParsedRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		Expect(req.GetN()).To(Equal(5))
	})

	It("unmarshals n from chat completions JSON", func() {
		jsonData := []byte(`{"n": 3, "model": "test", "messages": [{"role": "user", "content": "hi"}]}`)
		var req ChatCompletionsRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		Expect(req.GetN()).To(Equal(3))
	})

	It("defaults to 1 when n is absent from JSON", func() {
		jsonData := []byte(`{"model": "test", "messages": [{"role": "user", "content": "hi"}]}`)
		var req ChatCompletionsRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		Expect(req.GetN()).To(Equal(1))
	})
})

var _ = Describe("TextCompletionsParsedRequest prompt", func() {
	Context("UnmarshalJSON", func() {
		It("should unmarshal a string prompt as a one-element slice", func() {
			jsonData := []byte(`{"prompt": "Hello, world!"}`)
			var req TextCompletionsParsedRequest
			err := json.Unmarshal(jsonData, &req)
			Expect(err).NotTo(HaveOccurred())
			Expect(req.Prompt).To(Equal([]PromptInput{{Text: "Hello, world!"}}))
		})

		It("should unmarshal an array prompt", func() {
			jsonData := []byte(`{"prompt": ["Hello", "world"]}`)
			var req TextCompletionsParsedRequest
			err := json.Unmarshal(jsonData, &req)
			Expect(err).NotTo(HaveOccurred())
			Expect(req.Prompt).To(Equal([]PromptInput{{Text: "Hello"}, {Text: "world"}}))
		})

		It("should return error for invalid prompt type", func() {
			jsonData := []byte(`{"prompt": 123}`)
			var req TextCompletionsParsedRequest
			err := json.Unmarshal(jsonData, &req)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("prompt must be a string, an array of strings, an array of token ids, or an array of arrays of token ids"))
		})

		It("should leave Prompt nil when the field is absent or null", func() {
			var req TextCompletionsParsedRequest
			Expect(json.Unmarshal([]byte(`{}`), &req)).To(Succeed())
			Expect(req.Prompt).To(BeNil())

			req = TextCompletionsParsedRequest{}
			Expect(json.Unmarshal([]byte(`{"prompt": null}`), &req)).To(Succeed())
			Expect(req.Prompt).To(BeNil())
		})
	})
})
