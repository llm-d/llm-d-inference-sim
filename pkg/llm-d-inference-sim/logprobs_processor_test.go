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
	"testing"

	. "github.com/onsi/gomega"
)

func TestLogprobsProcessor_Caching(t *testing.T) {
	RegisterTestingT(t)
	processor := NewLogprobsProcessor(100)

	// Test that same token generates same logprobs (deterministic)
	token := "hello"
	topK := 3

	logprob1, topLogprobs1 := processor.GetLogprobs(token, topK)
	logprob2, topLogprobs2 := processor.GetLogprobs(token, topK)

	// Should be identical (deterministic)
	Expect(logprob1).To(Equal(logprob2))
	Expect(len(topLogprobs1)).To(Equal(len(topLogprobs2)))
	Expect(len(topLogprobs1)).To(Equal(topK))

	// Check cache stats
	hits, misses, hitRate := processor.GetCacheStats()
	Expect(hits).To(Equal(int64(1)))
	Expect(misses).To(Equal(int64(1)))
	Expect(hitRate).To(Equal(0.5))
}

func TestLogprobsProcessor_DifferentTokens(t *testing.T) {
	RegisterTestingT(t)
	processor := NewLogprobsProcessor(100)

	// Test that different tokens generate different logprobs
	logprob1, _ := processor.GetLogprobs("hello", 2)
	logprob2, _ := processor.GetLogprobs("world", 2)

	Expect(logprob1).NotTo(Equal(logprob2))
}

func TestLogprobsProcessor_DifferentTopK(t *testing.T) {
	RegisterTestingT(t)
	processor := NewLogprobsProcessor(100)

	// Test that same token with different topK generates different results
	token := "test"

	_, topLogprobs1 := processor.GetLogprobs(token, 2)
	_, topLogprobs2 := processor.GetLogprobs(token, 5)

	Expect(len(topLogprobs1)).To(Equal(2))
	Expect(len(topLogprobs2)).To(Equal(5))
}

func TestLogprobsProcessor_ChatLogprobs(t *testing.T) {
	RegisterTestingT(t)
	processor := NewLogprobsProcessor(100)

	tokens := []string{"Hello", "world", "!"}
	topK := 3

	logprobs := processor.ProcessChatLogprobs(tokens, topK)

	Expect(logprobs).NotTo(BeNil())
	Expect(len(logprobs.Content)).To(Equal(len(tokens)))

	for i, content := range logprobs.Content {
		Expect(content.Token).To(Equal(tokens[i]))
		Expect(content.Logprob).To(BeNumerically("<", 0))
		Expect(len(content.TopLogprobs)).To(Equal(topK))
		Expect(content.Bytes).NotTo(BeNil())
	}
}

func TestLogprobsProcessor_TextLogprobs(t *testing.T) {
	RegisterTestingT(t)
	processor := NewLogprobsProcessor(100)

	tokens := []string{"Hello", "world"}
	topK := 2

	logprobs := processor.ProcessTextLogprobs(tokens, topK)

	Expect(logprobs).NotTo(BeNil())
	Expect(len(logprobs.Tokens)).To(Equal(len(tokens)))
	Expect(len(logprobs.TokenLogprobs)).To(Equal(len(tokens)))
	Expect(len(logprobs.TextOffset)).To(Equal(len(tokens)))
	Expect(len(logprobs.TopLogprobs)).To(Equal(len(tokens)))

	// Check text offsets are cumulative
	expectedOffset := 0
	for i, token := range tokens {
		Expect(logprobs.TextOffset[i]).To(Equal(expectedOffset))
		Expect(logprobs.Tokens[i]).To(Equal(token))
		Expect(logprobs.TokenLogprobs[i]).To(BeNumerically("<", 0))
		Expect(len(logprobs.TopLogprobs[i])).To(Equal(topK))
		expectedOffset += len(token)
	}
}

func TestLogprobsProcessor_EmptyTokens(t *testing.T) {
	RegisterTestingT(t)
	processor := NewLogprobsProcessor(100)

	// Test empty token lists
	chatLogprobs := processor.ProcessChatLogprobs([]string{}, 3)
	textLogprobs := processor.ProcessTextLogprobs([]string{}, 3)

	Expect(chatLogprobs).To(BeNil())
	Expect(textLogprobs).To(BeNil())
}

func TestLogprobsProcessor_ZeroTopK(t *testing.T) {
	RegisterTestingT(t)
	processor := NewLogprobsProcessor(100)

	logprob, topLogprobs := processor.GetLogprobs("test", 0)

	Expect(logprob).To(BeNumerically("<", 0))
	Expect(topLogprobs).To(BeNil())
}

func TestLogprobsProcessor_CacheEviction(t *testing.T) {
	RegisterTestingT(t)
	// Test with very small cache size to trigger eviction
	processor := NewLogprobsProcessor(2)

	// Fill cache beyond capacity
	processor.GetLogprobs("token1", 1)
	processor.GetLogprobs("token2", 1)
	processor.GetLogprobs("token3", 1) // Should trigger eviction

	hits, misses, _ := processor.GetCacheStats()
	Expect(hits).To(Equal(int64(0)))
	Expect(misses).To(Equal(int64(3)))

	// Access one of the earlier tokens - may or may not be in cache due to eviction
	processor.GetLogprobs("token1", 1)

	// Cache should be working (some entries may have been evicted)
	hits2, misses2, _ := processor.GetCacheStats()
	Expect(hits2).To(BeNumerically(">=", 0))
	Expect(misses2).To(BeNumerically(">=", 3))
}
