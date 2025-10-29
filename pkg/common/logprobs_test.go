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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Logprobs", func() {

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

	Context("calculateLogprob", func() {
		It("should calculate main token probabilities correctly", func() {
			// Test position cycle behavior (cycle of 3)
			// Position 0: -1.0 - (0 % 3) * 0.1 = -1.0
			result0 := calculateLogprob(0, 0)
			Expect(result0).To(Equal(-1.0))

			// Position 1: -1.0 - (1 % 3) * 0.1 = -1.1
			result1 := calculateLogprob(1, 0)
			Expect(result1).To(Equal(-1.1))

			// Position 2: -1.0 - (2 % 3) * 0.1 = -1.2
			result2 := calculateLogprob(2, 0)
			Expect(result2).To(Equal(-1.2))

			// Position 3: -1.0 - (3 % 3) * 0.1 = -1.0 (cycle repeats)
			result3 := calculateLogprob(3, 0)
			Expect(result3).To(Equal(-1.0))

			// Position 4: -1.0 - (4 % 3) * 0.1 = -1.1 (cycle repeats)
			result4 := calculateLogprob(4, 0)
			Expect(result4).To(Equal(-1.1))
		})

		It("should calculate alternative token probabilities correctly", func() {
			// Test alternative token decrements (0.5 per alternative index)
			tokenPosition := 0 // Start with position 0 (main logprob = -1.0)

			// Alternative 1: -1.0 - 1 * 0.5 = -1.5
			alt1 := calculateLogprob(tokenPosition, 1)
			Expect(alt1).To(Equal(-1.5))

			// Alternative 2: -1.0 - 2 * 0.5 = -2.0
			alt2 := calculateLogprob(tokenPosition, 2)
			Expect(alt2).To(Equal(-2.0))

			// Alternative 3: -1.0 - 3 * 0.5 = -2.5
			alt3 := calculateLogprob(tokenPosition, 3)
			Expect(alt3).To(Equal(-2.5))
		})

		It("should combine position cycle and alternative index correctly", func() {
			// Test with position 1 (main logprob = -1.1)
			tokenPosition := 1

			// Main token: -1.0 - (1 % 3) * 0.1 = -1.1
			main := calculateLogprob(tokenPosition, 0)
			Expect(main).To(Equal(-1.1))

			// Alternative 1: -1.1 - 1 * 0.5 = -1.6
			alt1 := calculateLogprob(tokenPosition, 1)
			Expect(alt1).To(Equal(-1.6))

			// Alternative 2: -1.1 - 2 * 0.5 = -2.1
			alt2 := calculateLogprob(tokenPosition, 2)
			Expect(alt2).To(Equal(-2.1))
		})

		It("should handle large position values correctly", func() {
			// Test with large position values to ensure cycle works
			largePosition := 100

			// Position 100: -1.0 - (100 % 3) * 0.1 = -1.0 - 1 * 0.1 = -1.1
			result := calculateLogprob(largePosition, 0)
			Expect(result).To(Equal(-1.1))

			// With alternative: -1.1 - 1 * 0.5 = -1.6
			resultAlt := calculateLogprob(largePosition, 1)
			Expect(resultAlt).To(Equal(-1.6))
		})

		It("should handle edge cases correctly", func() {
			// Test with zero values
			result := calculateLogprob(0, 0)
			Expect(result).To(Equal(-1.0))

			// Test with large alternative index
			largeAlt := calculateLogprob(0, 10)
			expectedLargeAlt := -1.0 - float64(10)*0.5 // -6.0
			Expect(largeAlt).To(Equal(expectedLargeAlt))
		})
	})

	Context("Other scenarios", func() {
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

		It("should verify probability pattern as token position grows", func() {
			// Test the cycling pattern of probabilities

			// Test first cycle (positions 0-2)
			prob0 := calculateLogprob(0, 0)
			prob1 := calculateLogprob(1, 0)
			prob2 := calculateLogprob(2, 0)

			Expect(prob0).To(Equal(-1.0)) // defaultLogprob
			Expect(prob1).To(Equal(-1.1)) // defaultLogprob - 1*0.1
			Expect(prob2).To(Equal(-1.2)) // defaultLogprob - 2*0.1

			// Test second cycle (positions 3-5) - should repeat the pattern
			prob3 := calculateLogprob(3, 0)
			prob4 := calculateLogprob(4, 0)
			prob5 := calculateLogprob(5, 0)

			Expect(prob3).To(Equal(prob0)) // Should equal position 0
			Expect(prob4).To(Equal(prob1)) // Should equal position 1
			Expect(prob5).To(Equal(prob2)) // Should equal position 2

			// Test third cycle (positions 6-8) - should repeat again
			prob6 := calculateLogprob(6, 0)
			prob7 := calculateLogprob(7, 0)
			prob8 := calculateLogprob(8, 0)

			Expect(prob6).To(Equal(prob0)) // Should equal position 0
			Expect(prob7).To(Equal(prob1)) // Should equal position 1
			Expect(prob8).To(Equal(prob2)) // Should equal position 2

			// Verify the cycling pattern continues for larger positions
			for i := 0; i < 20; i++ {
				expectedProb := defaultLogprob - float64(i%positionCycle)*positionDecrement
				actualProb := calculateLogprob(i, 0)
				Expect(actualProb).To(Equal(expectedProb), "Position %d should have probability %f", i, expectedProb)
			}
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
