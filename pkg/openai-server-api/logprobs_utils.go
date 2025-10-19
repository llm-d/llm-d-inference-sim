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
	"fmt"
)

const (
	// Default logprob value
	defaultLogprob = -1.0
)

// NOTE: These functions produce synthetic data for API shape compatibility.
// The logprobs are deterministic placeholders and have no semantic meaning.

// GenerateTextLogprobs generates synthetic log probabilities for text completion responses
func GenerateTextLogprobs(tokens []string, logprobsCount int) *TextLogprobs {
	// Always return a valid struct, even for empty input
	if len(tokens) == 0 {
		return &TextLogprobs{
			Tokens:        []string{},
			TokenLogprobs: []float64{},
			TopLogprobs:   []map[string]float64{},
			TextOffset:    []int{},
		}
	}

	// Ensure minimum count
	if logprobsCount <= 0 {
		logprobsCount = 1 // Include the main token, at least
	}

	// Avoid reallocations
	numTokens := len(tokens)
	logprobs := &TextLogprobs{
		Tokens:        tokens,
		TokenLogprobs: make([]float64, numTokens),
		TopLogprobs:   make([]map[string]float64, numTokens),
		TextOffset:    make([]int, numTokens),
	}

	offset := 0
	for i, token := range tokens {
		logprobs.TextOffset[i] = offset
		offset += len(token) // Use byte length

		// Simple deterministic logprob (can vary by position)
		mainLogprob := defaultLogprob - float64(i%3)*0.1
		logprobs.TokenLogprobs[i] = mainLogprob

		topLogprobs := make(map[string]float64, logprobsCount)
		topLogprobs[token] = mainLogprob

		// Add basic token pattern
		for j := 1; j < logprobsCount; j++ {
			altToken := fmt.Sprintf("%s_%d", token, j)
			altLogprob := mainLogprob - float64(j)*0.5
			topLogprobs[altToken] = altLogprob
		}

		logprobs.TopLogprobs[i] = topLogprobs
	}

	return logprobs
}

// GenerateChatLogprobs generates synthetic log probabilities for chat completion responses
func GenerateChatLogprobs(tokens []string, topLogprobsCount int) *ChatLogprobs {
	// Always return a valid struct, even for empty input
	if len(tokens) == 0 {
		return &ChatLogprobs{
			Content: []LogprobsContent{},
		}
	}

	numTokens := len(tokens)
	logprobs := &ChatLogprobs{
		Content: make([]LogprobsContent, numTokens),
	}

	for i, token := range tokens {
		// Simple deterministic logprob (varies by position)
		mainLogprob := defaultLogprob - float64(i%3)*0.1

		tokenBytes := stringToIntBytes(token)

		content := LogprobsContent{
			Token:   token,
			Logprob: mainLogprob,
			Bytes:   tokenBytes,
		}

		// Generate top alternatives if requested
		if topLogprobsCount > 0 {
			// Pre-size alternatives slice
			content.TopLogprobs = make([]LogprobsContent, topLogprobsCount)

			// Main token first
			content.TopLogprobs[0] = LogprobsContent{
				Token:   token,
				Logprob: mainLogprob,
				Bytes:   tokenBytes,
			}

			// Simple alternative tokens
			for j := 1; j < topLogprobsCount; j++ {
				altToken := fmt.Sprintf("%s_%d", token, j)
				altLogprob := mainLogprob - float64(j)*0.5
				altBytes := stringToIntBytes(altToken)

				content.TopLogprobs[j] = LogprobsContent{
					Token:   altToken,
					Logprob: altLogprob,
					Bytes:   altBytes,
				}
			}
		}

		logprobs.Content[i] = content
	}

	return logprobs
}

// stringToIntBytes converts a string to []int of byte values inline
func stringToIntBytes(s string) []int {
	if s == "" {
		return nil
	}
	out := make([]int, len(s))
	for i := range out {
		out[i] = int(s[i])
	}
	return out
}
