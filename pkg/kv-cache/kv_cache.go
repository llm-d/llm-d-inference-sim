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
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
)

type KVCacheHelper struct {
	tokenizer       tokenizer.Tokenizer
	tokensProcessor kvblock.TokenProcessor // turns tokens to kv block keys
	logger          logr.Logger
	blockCache      *blockCache
	blockSize       int
}

func NewKVCacheHelper(config *common.Configuration, logger logr.Logger, usageChan chan float64,
	tokenizer tokenizer.Tokenizer) (*KVCacheHelper, error) {
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

func (h *KVCacheHelper) OnRequestStart(vllmReq openaiserverapi.Request) (float64, error) {
	h.logger.V(logging.TRACE).Info("KV cache - process request")

	tokens := vllmReq.TokenizedPrompt().Tokens
	modelName := vllmReq.GetModel()

	// get block keys
	blockKeys := h.tokensProcessor.TokensToKVBlockKeys(tokens, modelName)
	h.logger.V(logging.TRACE).Info("Found tokens", "tokens", tokens, "block-keys", blockKeys)

	blockHashes := make([]uint64, len(blockKeys))
	for i, key := range blockKeys {
		blockHashes[i] = key.ChunkHash
	}

	requestID := vllmReq.GetRequestID()
	nBlocksAlreadyInCache, err := h.blockCache.startRequest(requestID, blockHashes)
	if err != nil {
		return 0, err
	}

	vllmReq.SetNumberOfCachedPromptTokens(nBlocksAlreadyInCache * h.blockSize)

	totalBlocks := len(blockHashes)
	cachedBlocks := h.blockCache.countCachedBlockPrefix(blockHashes)

	var hitRate float64
	if totalBlocks > 0 {
		hitRate = float64(cachedBlocks) / float64(totalBlocks)
	}

	return hitRate, nil
}

func (h *KVCacheHelper) OnRequestEnd(requestID string) error {
	return h.blockCache.finishRequest(requestID)
}
