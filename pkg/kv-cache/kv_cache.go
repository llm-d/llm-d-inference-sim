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
package kvcache

// contains all logic relevant to KV-cache support
import (
	"context"
	"fmt"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization"
)

type KVCacheHelper struct {
	tokenizer       tokenization.Tokenizer
	tokensProcessor kvblock.TokenProcessor // turns tokens to kv block keys
	logger          logr.Logger
	blockCache      *blockCache
	blockSize       int
}

func NewKVCacheHelper(config *common.Configuration, logger logr.Logger, usageChan chan float64,
	tokenizer tokenization.Tokenizer) (*KVCacheHelper, error) {
	tokenProcConfig := kvblock.DefaultTokenProcessorConfig()
	tokenProcConfig.BlockSize = config.TokenBlockSize
	if config.HashSeed != "" {
		tokenProcConfig.HashSeed = config.HashSeed
	}
	tokensProcessor := kvblock.NewChunkedTokenDatabase(tokenProcConfig)

	blockCache, err := newBlockCache(config, logger, usageChan)
	if err != nil {
		return nil, fmt.Errorf("failed to create block cache: %w", err)
	}
	return &KVCacheHelper{
		tokenizer:       tokenizer,
		tokensProcessor: tokensProcessor,
		blockCache:      blockCache,
		logger:          logger,
		blockSize:       config.TokenBlockSize,
	}, nil
}

// Run starts the helper.
func (h *KVCacheHelper) Run(ctx context.Context) {
	h.blockCache.start(ctx)
}

func (h *KVCacheHelper) Discard() {
	h.blockCache.discard()
}

func (h *KVCacheHelper) Activate() {
	h.blockCache.activate()
}

// CacheHitInfo contains information about cache hits for a request
type CacheHitInfo struct {
	CachedBlocks int
	TotalBlocks  int
}

// HitRate calculates the hit rate for the CacheHitInfo
func (c *CacheHitInfo) HitRate() float64 {
	if c.TotalBlocks > 0 {
		return float64(c.CachedBlocks) / float64(c.TotalBlocks)
	}
	return 0
}

// getBlockHashesFromRequest tokenizes the prompt and converts it to block hashes.
// This is a common operation used by both GetCacheHitInfo and OnRequestStart.
func (h *KVCacheHelper) getBlockHashesFromRequest(vllmReq openaiserverapi.Request) ([]uint64, error) {
	prompt := vllmReq.GetPrompt()
	modelName := vllmReq.GetModel()

	// tokenize the input
	tokens, _, err := h.tokenizer.Encode(prompt, modelName)
	if err != nil {
		h.logger.Error(err, "prompt tokenization failed")
		return nil, err
	}

	// get block keys
	blockKeys := h.tokensProcessor.TokensToKVBlockKeys(tokens, modelName)
	h.logger.V(logging.TRACE).Info("Found tokens", "tokens", tokens, "block-keys", blockKeys)

	blockHashes := make([]uint64, len(blockKeys))
	for i, key := range blockKeys {
		blockHashes[i] = key.ChunkHash
	}

	return blockHashes, nil
}

// GetCacheHitInfo returns cache hit information for a request
func (h *KVCacheHelper) GetCacheHitInfo(vllmReq openaiserverapi.Request) (CacheHitInfo, error) {
	blockHashes, err := h.getBlockHashesFromRequest(vllmReq)
	if err != nil {
		return CacheHitInfo{}, err
	}

	totalBlocks := len(blockHashes)
	cachedBlocks := h.blockCache.countCachedBlocks(blockHashes)
	return CacheHitInfo{
		CachedBlocks: cachedBlocks,
		TotalBlocks:  totalBlocks,
	}, nil
}

func (h *KVCacheHelper) OnRequestStart(vllmReq openaiserverapi.Request) error {
	h.logger.V(logging.TRACE).Info("KV cache - process request")

	blockHashes, err := h.getBlockHashesFromRequest(vllmReq)
	if err != nil {
		return err
	}

	requestID := vllmReq.GetRequestID()
	nBlocksAlreadyInCache, err := h.blockCache.startRequest(requestID, blockHashes)
	if err != nil {
		return err
	}

	vllmReq.SetNumberOfCachedPromptTokens(nBlocksAlreadyInCache * h.blockSize)
	return nil
}

func (h *KVCacheHelper) OnRequestEnd(requestID string) error {
	return h.blockCache.finishRequest(requestID)
}
