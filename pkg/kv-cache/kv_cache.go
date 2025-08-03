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
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization/prefixstore"
)

const (
	// TODO move it to configuration
	maxBlocks = 100
)

type KVCacheHelper struct {
	config          *kvcache.Config
	tokenizersPool  *tokenization.Pool
	tokensIndexer   prefixstore.Indexer    // gets tokens for a prompt
	tokensProcessor kvblock.TokenProcessor // turns tokens to kv block keys
	logger          logr.Logger
	blockCache      *blockCache
}

func NewKVCacheHelper(logger logr.Logger) (*KVCacheHelper, error) {
	config := kvcache.NewDefaultConfig()
	tokensIndexer, err := prefixstore.NewLRUTokenStore(config.PrefixStoreConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create prefixstore.Indexer: %w", err)
	}
	tokenizersPool, err := tokenization.NewTokenizationPool(config.TokenizersPoolConfig, tokensIndexer)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizers pool: %w", err)
	}
	tokensProcessor := kvblock.NewChunkedTokenDatabase(config.TokenProcessorConfig)

	return &KVCacheHelper{
		config:          config,
		tokenizersPool:  tokenizersPool,
		tokensIndexer:   tokensIndexer,
		tokensProcessor: tokensProcessor,
		blockCache:      newBlockCache(maxBlocks),
		logger:          logger,
	}, nil
}

// Run starts the helper.
func (h *KVCacheHelper) Run(ctx context.Context) {
	h.tokenizersPool.Run(ctx)
}

func (h *KVCacheHelper) OnRequestStart(vllmReq openaiserverapi.CompletionRequest) error {
	h.logger.Info("KV cache - process request")

	prompt := vllmReq.GetPrompt()
	modelName := vllmReq.GetModel()
	requestID := vllmReq.GetRequestID()

	// 0. add to tokenizers pool
	h.tokenizersPool.AddTask(prompt, modelName)

	// 1. get available tokens of longest prefix
	tokens := h.tokensIndexer.FindLongestContainedTokens(prompt, modelName)
	h.logger.Info(">>> After tokensIndexer.FindLongestContainedTokens", "tokens", len(tokens))
	if len(tokens) == 0 {
		//nolint:nilnil // no need to return an error
		h.logger.Info(">>> tokensIndexer.FindLongestContainedTokens returned 0 tokens", "prompt len", len(prompt), "model", modelName)
		// add request with empty list of blocks
		fmt.Printf("Before start request '%s' with empty list of tokens\n%s\n", requestID, h.blockCache.GetStateStr())
		return h.blockCache.startRequest(requestID, make([]uint64, 0))
	}

	// 2. get block keys
	blockKeys := h.tokensProcessor.TokensToKVBlockKeys(tokens, modelName)
	h.logger.Info("found tokens", "tokens", tokens, "block-keys", blockKeys)

	blockHashes := make([]uint64, len(blockKeys))
	for i, key := range blockKeys {
		blockHashes[i] = key.ChunkHash
	}

	fmt.Printf("Before start request '%s'\n%s\n", requestID, h.blockCache.GetStateStr())
	return h.blockCache.startRequest(requestID, blockHashes)
}

func (h *KVCacheHelper) OnRequestEnd(vllmReq openaiserverapi.CompletionRequest) error {
	return h.blockCache.finishRequest(vllmReq.GetRequestID())
}
