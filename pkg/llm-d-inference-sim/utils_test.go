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
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Utils", Ordered, func() {
	BeforeAll(func() {
		initRandom(time.Now().UnixNano())
	})

	Context("GetRandomResponseText", func() {
		It("should return complete text", func() {
			text, finishReason := getRandomResponseText(nil)
			Expect(int64(len(tokenize(text)))).Should(BeNumerically("<=", ResponseLenMax))
			Expect(finishReason).Should(Equal(stopFinishReason))
		})
		It("should return short text", func() {
			maxCompletionTokens := int64(2)
			text, finishReason := getRandomResponseText(&maxCompletionTokens)
			Expect(int64(len(tokenize(text)))).Should(Equal(maxCompletionTokens))
			Expect([]string{stopFinishReason, lengthFinishReason}).Should(ContainElement(finishReason))
		})
		It("should return long text", func() {
			// return required number of tokens although it is higher than ResponseLenMax
			maxCompletionTokens := int64(ResponseLenMax * 5)
			text, finishReason := getRandomResponseText(&maxCompletionTokens)
			Expect(int64(len(tokenize(text)))).Should(Equal(maxCompletionTokens))
			Expect([]string{stopFinishReason, lengthFinishReason}).Should(ContainElement(finishReason))
		})
	})

	Context("GetResponseText", func() {
		theText := "Give a man a fish and you feed him for a day; teach a man to fish and you feed him for a lifetime"

		It("should return the same text since max tokens is not defined", func() {
			text, finishReason := getResponseText(nil, theText)
			Expect(text).Should(Equal(theText))
			Expect(finishReason).Should(Equal(stopFinishReason))
		})
		It("should return the same text since max tokens is higher than the text length", func() {
			maxCompletionTokens := int64(1000)
			text, finishReason := getResponseText(&maxCompletionTokens, theText)
			Expect(text).Should(Equal(theText))
			Expect(finishReason).Should(Equal(stopFinishReason))
		})
		It("should return partial text", func() {
			maxCompletionTokens := int64(2)
			text, finishReason := getResponseText(&maxCompletionTokens, theText)
			Expect(int64(len(tokenize(text)))).Should(Equal(maxCompletionTokens))
			Expect(finishReason).Should(Equal(lengthFinishReason))
		})
	})

	Context("validateContextWindow", func() {
		It("should pass when total tokens are within limit", func() {
			promptTokens := 100
			maxCompletionTokens := int64(50)
			maxModelLen := 200

			isValid, actualCompletionTokens, totalTokens := validateContextWindow(promptTokens, &maxCompletionTokens, maxModelLen)
			Expect(isValid).Should(BeTrue())
			Expect(actualCompletionTokens).Should(Equal(int64(50)))
			Expect(totalTokens).Should(Equal(int64(150)))
		})

		It("should fail when total tokens exceed limit", func() {
			promptTokens := 150
			maxCompletionTokens := int64(100)
			maxModelLen := 200

			isValid, actualCompletionTokens, totalTokens := validateContextWindow(promptTokens, &maxCompletionTokens, maxModelLen)
			Expect(isValid).Should(BeFalse())
			Expect(actualCompletionTokens).Should(Equal(int64(100)))
			Expect(totalTokens).Should(Equal(int64(250)))
		})

		It("should handle nil max completion tokens", func() {
			promptTokens := 100
			maxModelLen := 200

			isValid, actualCompletionTokens, totalTokens := validateContextWindow(promptTokens, nil, maxModelLen)
			Expect(isValid).Should(BeTrue())
			Expect(actualCompletionTokens).Should(Equal(int64(0)))
			Expect(totalTokens).Should(Equal(int64(100)))
		})

		It("should fail when only prompt tokens exceed limit", func() {
			promptTokens := 250
			maxModelLen := 200

			isValid, actualCompletionTokens, totalTokens := validateContextWindow(promptTokens, nil, maxModelLen)
			Expect(isValid).Should(BeFalse())
			Expect(actualCompletionTokens).Should(Equal(int64(0)))
			Expect(totalTokens).Should(Equal(int64(250)))
		})
	})

	Context("GetRandomText", func() {
		lenArr := []int{5, 20, 50, 150}

		for _, len := range lenArr {
			name := fmt.Sprintf("should return text with %d tokens", len)
			It(name, func() {
				text := getRandomText(len)
				fmt.Printf("Text with %d tokens: '%s'\n", len, text)
				Expect(tokenize(text)).Should(HaveLen(len))
			})
		}
	})
})
