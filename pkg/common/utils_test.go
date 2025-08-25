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

package common

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Utils", Ordered, func() {
	BeforeAll(func() {
		InitRandom(time.Now().UnixNano())
	})

	Context("GetRandomResponseText", func() {
		It("should return complete text", func() {
			text, finishReason := GetRandomResponseText(nil)
			Expect(IsValidText(text)).To(BeTrue())
			Expect(finishReason).Should(Equal(StopFinishReason))
		})
		It("should return short text", func() {
			maxCompletionTokens := int64(2)
			text, finishReason := GetRandomResponseText(&maxCompletionTokens)
			tokensCnt := int64(len(Tokenize(text)))
			Expect(tokensCnt).Should(BeNumerically("<=", maxCompletionTokens))
			if tokensCnt == maxCompletionTokens {
				Expect(finishReason).To(Equal(LengthFinishReason))
			} else {
				Expect(tokensCnt).To(BeNumerically("<", maxCompletionTokens))
				Expect(finishReason).To(Equal(StopFinishReason))
			}
		})
		It("should return long text", func() {
			// return required number of tokens although it is higher than ResponseLenMax
			maxCompletionTokens := int64(ResponseLenMax * 5)
			text, finishReason := GetRandomResponseText(&maxCompletionTokens)
			tokensCnt := int64(len(Tokenize(text)))
			Expect(tokensCnt).Should(BeNumerically("<=", maxCompletionTokens))
			Expect(IsValidText(text)).To(BeTrue())
			if tokensCnt == maxCompletionTokens {
				Expect(finishReason).To(Equal(LengthFinishReason))
			} else {
				Expect(tokensCnt).To(BeNumerically("<", maxCompletionTokens))
				Expect(finishReason).To(Equal(StopFinishReason))
			}
		})
	})

	Context("GetResponseText", func() {
		theText := "Give a man a fish and you feed him for a day; teach a man to fish and you feed him for a lifetime"

		It("should return the same text since max tokens is not defined", func() {
			text, finishReason := GetResponseText(nil, theText)
			Expect(text).Should(Equal(theText))
			Expect(finishReason).Should(Equal(StopFinishReason))
		})
		It("should return the same text since max tokens is higher than the text length", func() {
			maxCompletionTokens := int64(1000)
			text, finishReason := GetResponseText(&maxCompletionTokens, theText)
			Expect(text).Should(Equal(theText))
			Expect(finishReason).Should(Equal(StopFinishReason))
		})
		It("should return partial text", func() {
			maxCompletionTokens := int64(2)
			text, finishReason := GetResponseText(&maxCompletionTokens, theText)
			Expect(int64(len(Tokenize(text)))).Should(Equal(maxCompletionTokens))
			Expect(finishReason).Should(Equal(LengthFinishReason))
		})
	})

	Context("validateContextWindow", func() {
		It("should pass when total tokens are within limit", func() {
			promptTokens := 100
			maxCompletionTokens := int64(50)
			maxModelLen := 200

			isValid, actualCompletionTokens, totalTokens := ValidateContextWindow(promptTokens, &maxCompletionTokens, maxModelLen)
			Expect(isValid).Should(BeTrue())
			Expect(actualCompletionTokens).Should(Equal(int64(50)))
			Expect(totalTokens).Should(Equal(int64(150)))
		})

		It("should fail when total tokens exceed limit", func() {
			promptTokens := 150
			maxCompletionTokens := int64(100)
			maxModelLen := 200

			isValid, actualCompletionTokens, totalTokens := ValidateContextWindow(promptTokens, &maxCompletionTokens, maxModelLen)
			Expect(isValid).Should(BeFalse())
			Expect(actualCompletionTokens).Should(Equal(int64(100)))
			Expect(totalTokens).Should(Equal(int64(250)))
		})

		It("should handle nil max completion tokens", func() {
			promptTokens := 100
			maxModelLen := 200

			isValid, actualCompletionTokens, totalTokens := ValidateContextWindow(promptTokens, nil, maxModelLen)
			Expect(isValid).Should(BeTrue())
			Expect(actualCompletionTokens).Should(Equal(int64(0)))
			Expect(totalTokens).Should(Equal(int64(100)))
		})

		It("should fail when only prompt tokens exceed limit", func() {
			promptTokens := 250
			maxModelLen := 200

			isValid, actualCompletionTokens, totalTokens := ValidateContextWindow(promptTokens, nil, maxModelLen)
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
				text := GetRandomText(len)
				Expect(Tokenize(text)).Should(HaveLen(len))
			})
		}
	})

	Context("IsValidText", func() {
		validTxts := make([]string, 0)
		invalidTxts := make([]string, 0)

		validTxts = append(validTxts, chatCompletionFakeResponses[0][:4])
		validTxts = append(validTxts, chatCompletionFakeResponses[1])
		validTxts = append(validTxts, chatCompletionFakeResponses[1]+" "+chatCompletionFakeResponses[2])

		invalidTxts = append(invalidTxts, (chatCompletionFakeResponses[1] + " " + chatCompletionFakeResponses[2])[3:4])
		invalidTxts = append(invalidTxts, chatCompletionFakeResponses[0][4:])
		invalidTxts = append(invalidTxts, chatCompletionFakeResponses[1]+"-"+chatCompletionFakeResponses[2])
		invalidTxts = append(invalidTxts, chatCompletionFakeResponses[1]+" ")
		invalidTxts = append(invalidTxts, chatCompletionFakeResponses[1]+"   "+chatCompletionFakeResponses[2])

		for _, txt := range validTxts {
			It("text should be valid", func() {
				Expect(IsValidText(txt)).To(BeTrue())
			})
		}

		for _, txt := range invalidTxts {
			It("text should be invalid", func() {
				Expect(IsValidText(txt)).To(BeFalse())
			})
		}
	})

})
