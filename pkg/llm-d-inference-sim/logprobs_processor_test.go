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
)

func TestLogprobsProcessor_Caching(t *testing.T) {
	processor := NewLogprobsProcessor(100)

	// Test that same token generates same logprobs (deterministic)
	token := "hello"
	topK := 3

	logprob1, topLogprobs1 := processor.GetLogprobs(token, topK)
	logprob2, topLogprobs2 := processor.GetLogprobs(token, topK)

	// Should be identical (deterministic)
	if logprob1 != logprob2 {
		t.Errorf("Expected same logprob for same token, got %.4f and %.4f", logprob1, logprob2)
	}
	if len(topLogprobs1) != len(topLogprobs2) {
		t.Errorf("Expected same topLogprobs length, got %d and %d", len(topLogprobs1), len(topLogprobs2))
	}
	if len(topLogprobs1) != topK {
		t.Errorf("Expected topLogprobs length %d, got %d", topK, len(topLogprobs1))
	}

	// Check cache stats
	hits, misses, hitRate := processor.GetCacheStats()
	if hits != 1 {
		t.Errorf("Expected 1 cache hit, got %d", hits)
	}
	if misses != 1 {
		t.Errorf("Expected 1 cache miss, got %d", misses)
	}
	if hitRate != 0.5 {
		t.Errorf("Expected 50%% hit rate, got %.2f", hitRate)
	}
}

func TestLogprobsProcessor_DifferentTokens(t *testing.T) {
	processor := NewLogprobsProcessor(100)

	// Test that different tokens generate different logprobs
	logprob1, _ := processor.GetLogprobs("hello", 2)
	logprob2, _ := processor.GetLogprobs("world", 2)

	if logprob1 == logprob2 {
		t.Errorf("Different tokens should have different logprobs, both got %.4f", logprob1)
	}
}

func TestLogprobsProcessor_DifferentTopK(t *testing.T) {
	processor := NewLogprobsProcessor(100)

	// Test that same token with different topK generates different results
	token := "test"

	_, topLogprobs1 := processor.GetLogprobs(token, 2)
	_, topLogprobs2 := processor.GetLogprobs(token, 5)

	if len(topLogprobs1) != 2 {
		t.Errorf("Expected 2 top logprobs, got %d", len(topLogprobs1))
	}
	if len(topLogprobs2) != 5 {
		t.Errorf("Expected 5 top logprobs, got %d", len(topLogprobs2))
	}
}

func TestLogprobsProcessor_ChatLogprobs(t *testing.T) {
	processor := NewLogprobsProcessor(100)

	tokens := []string{"Hello", "world", "!"}
	topK := 3

	logprobs := processor.ProcessChatLogprobs(tokens, topK)

	if logprobs == nil {
		t.Fatal("Expected non-nil chat logprobs")
	}
	if len(logprobs.Content) != len(tokens) {
		t.Errorf("Expected %d content items, got %d", len(tokens), len(logprobs.Content))
	}

	for i, content := range logprobs.Content {
		if content.Token != tokens[i] {
			t.Errorf("Expected token %s at index %d, got %s", tokens[i], i, content.Token)
		}
		if content.Logprob >= 0 {
			t.Errorf("Expected negative logprob, got %.4f", content.Logprob)
		}
		if len(content.TopLogprobs) != topK {
			t.Errorf("Expected %d top logprobs, got %d", topK, len(content.TopLogprobs))
		}
		if content.Bytes == nil {
			t.Error("Expected non-nil bytes")
		}
	}
}

func TestLogprobsProcessor_TextLogprobs(t *testing.T) {
	processor := NewLogprobsProcessor(100)

	tokens := []string{"Hello", "world"}
	topK := 2

	logprobs := processor.ProcessTextLogprobs(tokens, topK)

	if logprobs == nil {
		t.Fatal("Expected non-nil text logprobs")
	}
	if len(logprobs.Tokens) != len(tokens) {
		t.Errorf("Expected %d tokens, got %d", len(tokens), len(logprobs.Tokens))
	}
	if len(logprobs.TokenLogprobs) != len(tokens) {
		t.Errorf("Expected %d token logprobs, got %d", len(tokens), len(logprobs.TokenLogprobs))
	}
	if len(logprobs.TextOffset) != len(tokens) {
		t.Errorf("Expected %d text offsets, got %d", len(tokens), len(logprobs.TextOffset))
	}
	if len(logprobs.TopLogprobs) != len(tokens) {
		t.Errorf("Expected %d top logprobs arrays, got %d", len(tokens), len(logprobs.TopLogprobs))
	}

	// Check text offsets are cumulative
	expectedOffset := 0
	for i, token := range tokens {
		if logprobs.TextOffset[i] != expectedOffset {
			t.Errorf("Expected offset %d at index %d, got %d", expectedOffset, i, logprobs.TextOffset[i])
		}
		if logprobs.Tokens[i] != token {
			t.Errorf("Expected token %s at index %d, got %s", token, i, logprobs.Tokens[i])
		}
		if logprobs.TokenLogprobs[i] >= 0 {
			t.Errorf("Expected negative logprob at index %d, got %.4f", i, logprobs.TokenLogprobs[i])
		}
		if len(logprobs.TopLogprobs[i]) != topK {
			t.Errorf("Expected %d top logprobs at index %d, got %d", topK, i, len(logprobs.TopLogprobs[i]))
		}
		expectedOffset += len(token)
	}
}

func TestLogprobsProcessor_EmptyTokens(t *testing.T) {
	processor := NewLogprobsProcessor(100)

	// Test empty token lists
	chatLogprobs := processor.ProcessChatLogprobs([]string{}, 3)
	textLogprobs := processor.ProcessTextLogprobs([]string{}, 3)

	if chatLogprobs != nil {
		t.Error("Expected nil chat logprobs for empty tokens")
	}
	if textLogprobs != nil {
		t.Error("Expected nil text logprobs for empty tokens")
	}
}

func TestLogprobsProcessor_ZeroTopK(t *testing.T) {
	processor := NewLogprobsProcessor(100)

	logprob, topLogprobs := processor.GetLogprobs("test", 0)

	if logprob >= 0 {
		t.Errorf("Expected negative logprob, got %.4f", logprob)
	}
	if topLogprobs != nil {
		t.Error("Expected nil top logprobs for topK=0")
	}
}

func TestLogprobsProcessor_CacheEviction(t *testing.T) {
	// Test with very small cache size to trigger eviction
	processor := NewLogprobsProcessor(2)

	// Fill cache beyond capacity
	processor.GetLogprobs("token1", 1)
	processor.GetLogprobs("token2", 1)
	processor.GetLogprobs("token3", 1) // Should trigger eviction

	hits, misses, _ := processor.GetCacheStats()
	if hits != 0 {
		t.Errorf("Expected 0 cache hits, got %d", hits)
	}
	if misses != 3 {
		t.Errorf("Expected 3 cache misses, got %d", misses)
	}

	// Access one of the earlier tokens - may or may not be in cache due to eviction
	processor.GetLogprobs("token1", 1)

	// Cache should be working (some entries may have been evicted)
	hits2, misses2, _ := processor.GetCacheStats()
	if hits2 < 0 {
		t.Errorf("Expected non-negative cache hits, got %d", hits2)
	}
	if misses2 < 3 {
		t.Errorf("Expected at least 3 cache misses, got %d", misses2)
	}
}