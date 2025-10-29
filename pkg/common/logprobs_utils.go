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

	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

const (
	// Default logprob value
	defaultLogprob = -1.0
	// Cycle length for position-based variation
	positionCycle = 3
	// Logprob decrement per cycle position
	positionDecrement = 0.1
	// Logprob decrement per alternative token
	alternativeDecrement = 0.5
)

// NOTE: These functions produce synthetic data for API shape compatibility.
// The logprobs are deterministic placeholders and have no semantic meaning.

// calculateLogprob calculates synthetic log probabilities using a deterministic algorithm.
// For the main token (alternativeIndex = 0), it uses a cycle of 3 positions with decreasing probability.
// For alternative tokens, it decreases probability by 0.5 per alternative index.
//
// Algorithm:
// - Main token: defaultLogprob - (tokenPosition % 3) * 0.1
// - Alternative: mainTokenLogprob - alternativeIndex * 0.5
func calculateLogprob(tokenPosition int, alternativeIndex int) float64 {
	// Calculate main token probability based on position cycle
	mainLogprob := defaultLogprob - float64(tokenPosition%positionCycle)*positionDecrement

	// For main token (index 0), return the main probability
	if alternativeIndex == 0 {
		return mainLogprob
	}

	// For alternatives, decrease by alternativeDecrement per index
	return mainLogprob - float64(alternativeIndex)*alternativeDecrement
}

// GenerateSingleTokenChatLogprobs generates logprobs for a single token in chat completion streaming
func GenerateSingleTokenChatLogprobs(token string, tokenPosition int, topLogprobsCount int) *openaiserverapi.LogprobsContent {
	if token == "" {
		return nil
	}

	// Calculate main token probability
	mainLogprob := calculateLogprob(tokenPosition, 0)
	tokenBytes := stringToIntBytes(token)

	content := openaiserverapi.LogprobsContent{
		Token:   token,
		Logprob: mainLogprob,
		Bytes:   tokenBytes,
	}

	// Generate top alternatives if requested
	if topLogprobsCount > 0 {
		// Pre-size alternatives slice
		content.TopLogprobs = make([]openaiserverapi.LogprobsContent, topLogprobsCount)

		// Main token first
		content.TopLogprobs[0] = openaiserverapi.LogprobsContent{
			Token:   token,
			Logprob: mainLogprob,
			Bytes:   tokenBytes,
		}

		// Alternative tokens
		for j := 1; j < topLogprobsCount; j++ {
			altToken := fmt.Sprintf("%s_%d", token, j)
			altLogprob := calculateLogprob(tokenPosition, j)
			altBytes := stringToIntBytes(altToken)

			content.TopLogprobs[j] = openaiserverapi.LogprobsContent{
				Token:   altToken,
				Logprob: altLogprob,
				Bytes:   altBytes,
			}
		}
	}

	return &content
}

// GenerateSingleTokenTextLogprobs generates logprobs for a single token in text completion streaming
func GenerateSingleTokenTextLogprobs(token string, tokenPosition int, logprobsCount int) *openaiserverapi.TextLogprobs {
	if token == "" {
		return nil
	}

	// Ensure minimum count
	if logprobsCount <= 0 {
		logprobsCount = 1 // Include the main token, at a minimum
	}

	logprobs := &openaiserverapi.TextLogprobs{
		Tokens:        []string{token},
		TokenLogprobs: make([]float64, 1),
		TopLogprobs:   make([]map[string]float64, 1),
		TextOffset:    []int{0},
	}

	// Calculate main token probability
	mainLogprob := calculateLogprob(tokenPosition, 0)
	logprobs.TokenLogprobs[0] = mainLogprob

	topLogprobs := make(map[string]float64, logprobsCount)
	topLogprobs[token] = mainLogprob

	// Add alternative tokens
	for j := 1; j < logprobsCount; j++ {
		altToken := fmt.Sprintf("%s_%d", token, j)
		altLogprob := calculateLogprob(tokenPosition, j)
		topLogprobs[altToken] = altLogprob
	}

	logprobs.TopLogprobs[0] = topLogprobs

	return logprobs
}

// GenerateTextLogprobs generates synthetic log probabilities for text completion responses
func GenerateTextLogprobs(tokens []string, logprobsCount int) *openaiserverapi.TextLogprobs {
	// Return empty struct for empty input (not nil)
	if len(tokens) == 0 {
		return &openaiserverapi.TextLogprobs{
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
	logprobs := &openaiserverapi.TextLogprobs{
		Tokens:        tokens,
		TokenLogprobs: make([]float64, numTokens),
		TopLogprobs:   make([]map[string]float64, numTokens),
		TextOffset:    make([]int, numTokens),
	}

	offset := 0
	for i, token := range tokens {
		logprobs.TextOffset[i] = offset
		offset += len(token) // Use byte length

		// Calculate main token probability using helper function
		mainLogprob := calculateLogprob(i, 0)
		logprobs.TokenLogprobs[i] = mainLogprob

		topLogprobs := make(map[string]float64, logprobsCount)
		topLogprobs[token] = mainLogprob

		// Add alternative tokens using helper function
		for j := 1; j < logprobsCount; j++ {
			altToken := fmt.Sprintf("%s_%d", token, j)
			altLogprob := calculateLogprob(i, j)
			topLogprobs[altToken] = altLogprob
		}

		logprobs.TopLogprobs[i] = topLogprobs
	}

	return logprobs
}

// GenerateChatLogprobs generates synthetic log probabilities for chat completion responses
func GenerateChatLogprobs(tokens []string, topLogprobsCount int) *openaiserverapi.ChatLogprobs {
	// Return empty struct for empty input (not nil)
	if len(tokens) == 0 {
		return &openaiserverapi.ChatLogprobs{
			Content: []openaiserverapi.LogprobsContent{},
		}
	}

	numTokens := len(tokens)
	logprobs := &openaiserverapi.ChatLogprobs{
		Content: make([]openaiserverapi.LogprobsContent, numTokens),
	}

	for i, token := range tokens {
		// Calculate main token probability using helper function
		mainLogprob := calculateLogprob(i, 0)

		tokenBytes := stringToIntBytes(token)

		content := openaiserverapi.LogprobsContent{
			Token:   token,
			Logprob: mainLogprob,
			Bytes:   tokenBytes,
		}

		// Generate top alternatives if requested
		if topLogprobsCount > 0 {
			// Pre-size alternatives slice
			content.TopLogprobs = make([]openaiserverapi.LogprobsContent, topLogprobsCount)

			// Main token first
			content.TopLogprobs[0] = openaiserverapi.LogprobsContent{
				Token:   token,
				Logprob: mainLogprob,
				Bytes:   tokenBytes,
			}

			// Alternative tokens using helper function
			for j := 1; j < topLogprobsCount; j++ {
				altToken := fmt.Sprintf("%s_%d", token, j)
				altLogprob := calculateLogprob(i, j)
				altBytes := stringToIntBytes(altToken)

				content.TopLogprobs[j] = openaiserverapi.LogprobsContent{
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
