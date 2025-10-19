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

package openaiserverapi

import (
	"encoding/json"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestOpenaiServerApi(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "OpenaiServerApi Suite")
}

var _ = Describe("Logprobs", func() {
	Context("TextCompletionRequest", func() {
		It("should handle logprobs parameters correctly", func() {
			logprobsCount := 2
			req := &TextCompletionRequest{
				Prompt:   "The capital of France is",
				Logprobs: &logprobsCount,
			}

			Expect(req.GetLogprobs()).NotTo(BeNil())
			Expect(*req.GetLogprobs()).To(Equal(2))
			Expect(req.IncludeLogprobs()).To(BeTrue())
			Expect(req.GetTopLogprobs()).NotTo(BeNil())
			Expect(*req.GetTopLogprobs()).To(Equal(2))
		})
	})

	Context("ChatCompletionRequest", func() {
		It("should handle logprobs parameters correctly", func() {
			topLogprobs := 3
			req := &ChatCompletionRequest{
				Messages: []Message{
					{Role: "user", Content: Content{Raw: "What is 2+2?"}},
				},
				Logprobs:    true,
				TopLogprobs: &topLogprobs,
			}

			Expect(req.GetLogprobs()).To(BeNil())
			Expect(req.IncludeLogprobs()).To(BeTrue())
			Expect(req.GetTopLogprobs()).NotTo(BeNil())
			Expect(*req.GetTopLogprobs()).To(Equal(3))
		})
	})

	Context("GenerateTextLogprobs", func() {
		It("should generate correct text logprobs structure", func() {
			tokens := []string{" Paris", ",", " the", " capital"}
			logprobsCount := 2

			logprobs := GenerateTextLogprobs(tokens, logprobsCount)

			Expect(logprobs).NotTo(BeNil())
			Expect(logprobs.Tokens).To(HaveLen(len(tokens)))
			Expect(logprobs.TokenLogprobs).To(HaveLen(len(tokens)))
			Expect(logprobs.TopLogprobs).To(HaveLen(len(tokens)))
			Expect(logprobs.TextOffset).To(HaveLen(len(tokens)))

			// Check that each top logprobs entry has the expected number of alternatives
			for i, topLogprob := range logprobs.TopLogprobs {
				Expect(topLogprob).To(HaveLen(logprobsCount))
				// Check that the main token is included in the alternatives
				Expect(topLogprob).To(HaveKey(tokens[i]))
			}

			// Check text offsets are calculated correctly (byte-based)
			expectedOffsets := []int{0, 6, 7, 11} // " Paris" - 6, "," - 1, " the" -4, " capital" - 11
			for i, expected := range expectedOffsets {
				Expect(logprobs.TextOffset[i]).To(Equal(expected))
			}

			// Check deterministic logprobs
			expectedLogprob0 := -1.0 // defaultLogprob - float64(0%3)*0.1
			Expect(logprobs.TokenLogprobs[0]).To(Equal(expectedLogprob0))
		})
	})

	Context("GenerateChatLogprobs", func() {
		It("should generate correct chat logprobs structure", func() {
			tokens := []string{"4"}
			topLogprobsCount := 3

			logprobs := GenerateChatLogprobs(tokens, topLogprobsCount)

			Expect(logprobs).NotTo(BeNil())
			Expect(logprobs.Content).To(HaveLen(len(tokens)))

			content := logprobs.Content[0]
			Expect(content.Token).To(Equal(tokens[0]))
			Expect(content.Bytes).To(HaveLen(len(tokens[0])))
			Expect(content.TopLogprobs).To(HaveLen(topLogprobsCount))

			// Check that the main token is the first in top logprobs
			Expect(content.TopLogprobs[0].Token).To(Equal(tokens[0]))

			// Check alternative tokens follow the pattern
			expectedAlt1 := "4_1"
			Expect(content.TopLogprobs[1].Token).To(Equal(expectedAlt1))

			// Check byte conversion
			expectedBytes := []int{52} // byte value of '4'
			for i, expected := range expectedBytes {
				Expect(content.Bytes[i]).To(Equal(expected))
			}

			// Check deterministic logprobs
			expectedLogprob := -1.0 // defaultLogprob - float64(0%3)*0.1
			Expect(content.Logprob).To(Equal(expectedLogprob))
		})
	})

	Context("Request Serialization", func() {
		It("should serialize and deserialize TextCompletionRequest correctly", func() {
			logprobsCount := 2
			req := &TextCompletionRequest{
				Prompt:   "Hello world",
				Logprobs: &logprobsCount,
			}

			jsonData, err := json.Marshal(req)
			Expect(err).NotTo(HaveOccurred())

			var parsed TextCompletionRequest
			err = json.Unmarshal(jsonData, &parsed)
			Expect(err).NotTo(HaveOccurred())

			Expect(parsed.Prompt).To(Equal(req.Prompt))
			Expect(parsed.Logprobs).NotTo(BeNil())
			Expect(*parsed.Logprobs).To(Equal(*req.Logprobs))
		})

		It("should serialize and deserialize ChatCompletionRequest correctly", func() {
			topLogprobs := 3
			req := &ChatCompletionRequest{
				Messages: []Message{
					{Role: "user", Content: Content{Raw: "Hello"}},
				},
				Logprobs:    true,
				TopLogprobs: &topLogprobs,
			}

			jsonData, err := json.Marshal(req)
			Expect(err).NotTo(HaveOccurred())

			var parsed ChatCompletionRequest
			err = json.Unmarshal(jsonData, &parsed)
			Expect(err).NotTo(HaveOccurred())

			Expect(parsed.Logprobs).To(Equal(req.Logprobs))
			Expect(parsed.TopLogprobs).NotTo(BeNil())
			Expect(*parsed.TopLogprobs).To(Equal(*req.TopLogprobs))
		})
	})

	Context("Edge Cases", func() {
		It("should handle empty tokens for text logprobs", func() {
			logprobs := GenerateTextLogprobs([]string{}, 2)

			Expect(logprobs).NotTo(BeNil())
			Expect(logprobs.Tokens).To(BeEmpty())
		})

		It("should handle empty tokens for chat logprobs", func() {
			logprobs := GenerateChatLogprobs([]string{}, 2)

			Expect(logprobs).NotTo(BeNil())
			Expect(logprobs.Content).To(BeEmpty())
		})
	})

	Context("No Limits", func() {
		It("should allow unlimited logprobs count", func() {
			tokens := []string{"test"}

			// Test text completion (no clamping)
			textLogprobs := GenerateTextLogprobs(tokens, 10)
			Expect(textLogprobs.TopLogprobs[0]).To(HaveLen(10))

			// Test chat completion (no clamping)
			chatLogprobs := GenerateChatLogprobs(tokens, 25)
			Expect(chatLogprobs.Content[0].TopLogprobs).To(HaveLen(25))

			// Test high count
			textLogprobs = GenerateTextLogprobs(tokens, 100)
			Expect(textLogprobs.TopLogprobs[0]).To(HaveLen(100))

			chatLogprobs = GenerateChatLogprobs(tokens, 50)
			Expect(chatLogprobs.Content[0].TopLogprobs).To(HaveLen(50))

			// Test minimum (at least 1)
			textLogprobs = GenerateTextLogprobs(tokens, 0)
			Expect(textLogprobs.TopLogprobs[0]).To(HaveLen(1))
		})
	})
})
