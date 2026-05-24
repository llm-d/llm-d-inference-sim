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

var _ = Describe("StringOrArray", func() {
	Context("UnmarshalJSON", func() {
		It("should unmarshal a string prompt", func() {
			jsonData := []byte(`{"prompt": "Hello, world!"}`)
			var req TextCompletionsParsedRequest
			err := json.Unmarshal(jsonData, &req)
			Expect(err).NotTo(HaveOccurred())
			Expect(req.Prompt.IsArray()).To(BeFalse())
			Expect(req.Prompt.String()).To(Equal("Hello, world!"))
		})

		It("should unmarshal an array prompt", func() {
			jsonData := []byte(`{"prompt": ["Hello", "world"]}`)
			var req TextCompletionsParsedRequest
			err := json.Unmarshal(jsonData, &req)
			Expect(err).NotTo(HaveOccurred())
			Expect(req.Prompt.IsArray()).To(BeTrue())
			Expect(req.Prompt.String()).To(Equal("Hello\nworld"))
			arr := req.Prompt.Array()
			Expect(arr).To(HaveLen(2))
			Expect(arr[0]).To(Equal("Hello"))
			Expect(arr[1]).To(Equal("world"))
		})

		It("should return error for invalid prompt type", func() {
			jsonData := []byte(`{"prompt": 123}`)
			var req TextCompletionsParsedRequest
			err := json.Unmarshal(jsonData, &req)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("prompt must be a string or array of strings"))
		})
	})

	Context("MarshalJSON", func() {
		It("should marshal a string prompt", func() {
			req := TextCompletionsParsedRequest{
				Prompt: NewStringOrArray("Hello, world!"),
			}
			data, err := json.Marshal(req)
			Expect(err).NotTo(HaveOccurred())
			var result map[string]interface{}
			err = json.Unmarshal(data, &result)
			Expect(err).NotTo(HaveOccurred())
			prompt, ok := result["prompt"].(string)
			Expect(ok).To(BeTrue())
			Expect(prompt).To(Equal("Hello, world!"))
		})

		It("should marshal an array prompt", func() {
			req := TextCompletionsParsedRequest{
				Prompt: NewStringOrArrayFromSlice([]string{"Hello", "world"}),
			}
			data, err := json.Marshal(req)
			Expect(err).NotTo(HaveOccurred())
			var result map[string]interface{}
			err = json.Unmarshal(data, &result)
			Expect(err).NotTo(HaveOccurred())
			promptArr, ok := result["prompt"].([]interface{})
			Expect(ok).To(BeTrue())
			Expect(promptArr).To(HaveLen(2))
			Expect(promptArr[0].(string)).To(Equal("Hello"))
			Expect(promptArr[1].(string)).To(Equal("world"))
		})
	})

	Context("Helper methods", func() {
		It("should convert string to array with one element", func() {
			s := NewStringOrArray("test")
			arr := s.Array()
			Expect(arr).To(HaveLen(1))
			Expect(arr[0]).To(Equal("test"))
		})

		It("should join array elements with newlines", func() {
			s := NewStringOrArrayFromSlice([]string{"line1", "line2", "line3"})
			Expect(s.String()).To(Equal("line1\nline2\nline3"))
		})

		It("should correctly identify array type", func() {
			str := NewStringOrArray("test")
			arr := NewStringOrArrayFromSlice([]string{"test"})
			Expect(str.IsArray()).To(BeFalse())
			Expect(arr.IsArray()).To(BeTrue())
		})
	})
})
