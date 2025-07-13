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
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Utils", func() {
	Context("GetRandomResponseText", func() {
		It("should return complete text", func() {
			text, finishReason := getRandomResponseText(nil)
			Expect(text).Should(Equal(getFullTextFromPartialString(text)))
			Expect(finishReason).Should(Equal(stopFinishReason))
		})
		It("should return partial text", func() {
			maxCompletionTokens := int64(2)
			text, finishReason := getRandomResponseText(&maxCompletionTokens)
			Expect(int64(len(strings.Fields(text)))).Should(Equal(maxCompletionTokens))
			Expect(finishReason).Should(Equal(lengthFinishReason))
		})
		It("should return complete text", func() {
			maxCompletionTokens := int64(2000)
			text, finishReason := getRandomResponseText(&maxCompletionTokens)
			Expect(text).Should(Equal(getFullTextFromPartialString(text)))
			Expect(finishReason).Should(Equal(stopFinishReason))
		})
	})

	Context("validateContextWindow", func() {
		It("should pass when total tokens are within limit", func() {
			promptTokens := 100
			maxCompletionTokens := int64(50)
			maxModelLen := 200

			err := validateContextWindow(promptTokens, &maxCompletionTokens, maxModelLen)
			Expect(err).Should(BeNil())
		})

		It("should fail when total tokens exceed limit", func() {
			promptTokens := 150
			maxCompletionTokens := int64(100)
			maxModelLen := 200

			err := validateContextWindow(promptTokens, &maxCompletionTokens, maxModelLen)
			Expect(err).ShouldNot(BeNil())
			Expect(err.Error()).Should(ContainSubstring("This model's maximum context length is 200 tokens"))
			Expect(err.Error()).Should(ContainSubstring("However, you requested 250 tokens"))
			Expect(err.Error()).Should(ContainSubstring("150 in the messages, 100 in the completion"))
		})

		It("should handle nil max completion tokens", func() {
			promptTokens := 100
			maxModelLen := 200

			err := validateContextWindow(promptTokens, nil, maxModelLen)
			Expect(err).Should(BeNil())
		})

		It("should fail when only prompt tokens exceed limit", func() {
			promptTokens := 250
			maxModelLen := 200

			err := validateContextWindow(promptTokens, nil, maxModelLen)
			Expect(err).ShouldNot(BeNil())
			Expect(err.Error()).Should(ContainSubstring("This model's maximum context length is 200 tokens"))
			Expect(err.Error()).Should(ContainSubstring("However, you requested 250 tokens"))
		})
	})
})
