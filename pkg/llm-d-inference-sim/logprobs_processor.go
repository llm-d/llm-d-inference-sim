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
	"crypto/md5"
	"fmt"
	"sync"

	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

// LogprobData represents cached logprob information for a token
type LogprobData struct {
	MainLogprob float64                                 `json:"main_logprob"`
	TopLogprobs []openaiserverapi.ChatCompletionLogProb `json:"top_logprobs"`
}

// LogprobsProcessor handles logprobs generation and caching following vLLM architecture
type LogprobsProcessor struct {
	// tokenCache caches logprobs by token content and topK to avoid recomputation
	tokenCache map[string]*LogprobData
	cacheMutex sync.RWMutex

	// cacheHits and cacheMisses for metrics
	cacheHits   int64
	cacheMisses int64

	// maxCacheSize limits memory usage
	maxCacheSize int
}

// NewLogprobsProcessor creates a new LogprobsProcessor following vLLM design patterns
func NewLogprobsProcessor(maxCacheSize int) *LogprobsProcessor {
	if maxCacheSize <= 0 {
		maxCacheSize = 10000 // Default cache size
	}

	return &LogprobsProcessor{
		tokenCache:   make(map[string]*LogprobData),
		maxCacheSize: maxCacheSize,
	}
}

// generateCacheKey creates a deterministic key for caching based on token and topK
func (lp *LogprobsProcessor) generateCacheKey(token string, topK int) string {
	return fmt.Sprintf("%s:%d", token, topK)
}

// generateDeterministicLogprobs creates logprobs with deterministic values based on token content
// This follows vLLM's approach of consistent logprobs for the same token in similar contexts
func (lp *LogprobsProcessor) generateDeterministicLogprobs(token string, topK int) *LogprobData {
	// Use token content to seed deterministic generation (similar to vLLM's approach)
	hash := md5.Sum([]byte(token))
	seed := int64(hash[0])<<24 | int64(hash[1])<<16 | int64(hash[2])<<8 | int64(hash[3])

	// Generate main logprob deterministically based on token
	// Real logprobs are typically negative, with values closer to 0 being more likely
	mainLogprob := -0.1 - (float64(seed%2000) / 1000.0) // Range: -0.1 to -2.1

	if topK <= 0 {
		return &LogprobData{
			MainLogprob: mainLogprob,
			TopLogprobs: nil,
		}
	}

	// Generate top-k alternatives deterministically
	topLogprobs := make([]openaiserverapi.ChatCompletionLogProb, 0, topK)
	for i := 0; i < topK; i++ {
		// Generate deterministic alternative token
		altToken := fmt.Sprintf("alt_%d_%x", i, hash[i%4])

		// Each alternative gets progressively lower probability
		altLogprob := mainLogprob - (float64(i+1) * (0.5 + float64((seed+int64(i))%1500)/1000.0))

		// Convert token to bytes
		bytes := make([]int, len(altToken))
		for j, b := range []byte(altToken) {
			bytes[j] = int(b)
		}

		topLogprobs = append(topLogprobs, openaiserverapi.ChatCompletionLogProb{
			Token:   altToken,
			Logprob: altLogprob,
			Bytes:   bytes,
		})
	}

	return &LogprobData{
		MainLogprob: mainLogprob,
		TopLogprobs: topLogprobs,
	}
}

// GetLogprobs returns logprobs for a token, using cache when possible
func (lp *LogprobsProcessor) GetLogprobs(token string, topK int) (float64, []openaiserverapi.ChatCompletionLogProb) {
	cacheKey := lp.generateCacheKey(token, topK)

	// Check cache first
	lp.cacheMutex.RLock()
	if cached, exists := lp.tokenCache[cacheKey]; exists {
		lp.cacheMutex.RUnlock()
		lp.cacheHits++
		return cached.MainLogprob, cached.TopLogprobs
	}
	lp.cacheMutex.RUnlock()

	// Cache miss - generate new logprobs
	lp.cacheMisses++
	logprobData := lp.generateDeterministicLogprobs(token, topK)

	// Store in cache (with size limit)
	lp.cacheMutex.Lock()
	if len(lp.tokenCache) >= lp.maxCacheSize {
		// Simple eviction: remove oldest entry
		// In production, this could use LRU or other strategies
		for k := range lp.tokenCache {
			delete(lp.tokenCache, k)
			break
		}
	}
	lp.tokenCache[cacheKey] = logprobData
	lp.cacheMutex.Unlock()

	return logprobData.MainLogprob, logprobData.TopLogprobs
}

// ProcessChatLogprobs creates logprobs data for chat completions following vLLM patterns
func (lp *LogprobsProcessor) ProcessChatLogprobs(tokens []string, topK int) *openaiserverapi.ChatCompletionLogProbs {
	if len(tokens) == 0 {
		return nil
	}

	logprobs := &openaiserverapi.ChatCompletionLogProbs{
		Content: make([]openaiserverapi.ChatCompletionLogProbsContent, 0, len(tokens)),
	}

	for _, token := range tokens {
		mainLogprob, topLps := lp.GetLogprobs(token, topK)

		// Convert token to bytes
		bytes := make([]int, len(token))
		for i, b := range []byte(token) {
			bytes[i] = int(b)
		}

		logprobs.Content = append(logprobs.Content, openaiserverapi.ChatCompletionLogProbsContent{
			Token:       token,
			Logprob:     mainLogprob,
			Bytes:       bytes,
			TopLogprobs: topLps,
		})
	}

	return logprobs
}

// ProcessTextLogprobs creates logprobs data for text completions following vLLM patterns
func (lp *LogprobsProcessor) ProcessTextLogprobs(tokens []string, topK int) *openaiserverapi.CompletionLogProbs {
	if len(tokens) == 0 {
		return nil
	}

	logprobs := &openaiserverapi.CompletionLogProbs{
		TextOffset:    make([]int, 0, len(tokens)),
		TokenLogprobs: make([]float64, 0, len(tokens)),
		Tokens:        make([]string, 0, len(tokens)),
	}

	if topK > 0 {
		logprobs.TopLogprobs = make([]map[string]float64, 0, len(tokens))
	}

	textOffset := 0
	for _, token := range tokens {
		mainLogprob, topLps := lp.GetLogprobs(token, topK)

		logprobs.TextOffset = append(logprobs.TextOffset, textOffset)
		logprobs.TokenLogprobs = append(logprobs.TokenLogprobs, mainLogprob)
		logprobs.Tokens = append(logprobs.Tokens, token)

		if topK > 0 {
			topMap := make(map[string]float64, len(topLps))
			for _, lp := range topLps {
				topMap[lp.Token] = lp.Logprob
			}
			logprobs.TopLogprobs = append(logprobs.TopLogprobs, topMap)
		}

		textOffset += len(token)
	}

	return logprobs
}

// GetCacheStats returns cache performance statistics
func (lp *LogprobsProcessor) GetCacheStats() (hits, misses int64, hitRate float64) {
	lp.cacheMutex.RLock()
	defer lp.cacheMutex.RUnlock()

	total := lp.cacheHits + lp.cacheMisses
	hitRate = 0.0
	if total > 0 {
		hitRate = float64(lp.cacheHits) / float64(total)
	}

	return lp.cacheHits, lp.cacheMisses, hitRate
}

// ClearCache clears the logprobs cache
func (lp *LogprobsProcessor) ClearCache() {
	lp.cacheMutex.Lock()
	defer lp.cacheMutex.Unlock()

	lp.tokenCache = make(map[string]*LogprobData)
	lp.cacheHits = 0
	lp.cacheMisses = 0
}
