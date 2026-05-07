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
	"errors"
	"fmt"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// PrefixCacheStats holds token-level prefix cache statistics for a single request,
// matching vLLM's PrefixCacheStats semantics where both fields count tokens.
type PrefixCacheStats struct {
	// QueriedTokens is the total number of prompt tokens checked against the cache
	QueriedTokens int
	// CachedTokens is the number of prompt tokens that were already cached
	CachedTokens int
}

type KVCacheHelper struct {
	tokenizer            tokenizer.Tokenizer
	tokensProcessor      kvblock.TokenProcessor // turns tokens to kv block keys
	logger               logr.Logger
	blockCache           *blockCache
	blockSize            int
	prefixCacheStatsChan common.Channel[PrefixCacheStats]
}

func NewKVCacheHelper(ctx context.Context, config *common.Configuration, logger logr.Logger, usageChan common.Channel[common.MetricInfo],
	prefixCacheStatsChan common.Channel[PrefixCacheStats], tokenizer tokenizer.Tokenizer) (*KVCacheHelper, error) {
	if config.IP == "" {
		return nil, errors.New("IP should be defined in the environment (POD_IP) for KV cache to work")
	}

	tokenProcConfig := kvblock.DefaultTokenProcessorConfig()
	tokenProcConfig.BlockSize = config.TokenBlockSize
	if config.HashSeed != "" {
		tokenProcConfig.HashSeed = config.HashSeed
	}
	tokensProcessor, err := kvblock.NewChunkedTokenDatabase(tokenProcConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokens processor: %w", err)
	}

	blockCache, err := newBlockCache(ctx, config, logger, &usageChan)
	if err != nil {
		return nil, fmt.Errorf("failed to create block cache: %w", err)
	}

	return &KVCacheHelper{
		tokenizer:            tokenizer,
		tokensProcessor:      tokensProcessor,
		blockCache:           blockCache,
		logger:               logger,
		blockSize:            config.TokenBlockSize,
		prefixCacheStatsChan: prefixCacheStatsChan,
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

func (h *KVCacheHelper) OnRequestStart(req openaiserverapi.Request) (PrefixCacheStats, error) {
	h.logger.V(logging.TRACE).Info("KV cache - process request")

	tokens := req.TokenizedPrompt().Tokens

	// compute per-block extra features from multimodal metadata (if present).
	var extraFeatures []*kvblock.BlockExtraFeatures
	mmFeatres := req.MMFeatures()

	if mmFeatres != nil {
		extraFeatures = kvblock.ComputeBlockExtraFeatures(
			mmFeatres.MMHashes, h.convertMMPlaceholders(mmFeatres.MMPlaceholders),
			h.blockSize, len(tokens))
	}

	// get block keys
	blockKeys, err := h.tokensProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, req.GetDisplayedModel(), extraFeatures)
	if err != nil {
		return PrefixCacheStats{}, fmt.Errorf("failed to convert tokens to block keys: %w", err)
	}
	h.logger.V(logging.TRACE).Info("Found tokens", "tokens", tokens, "block-keys", blockKeys)

	blockHashes := make([]uint64, len(blockKeys))
	blockTokens := make([][]uint32, len(blockKeys))
	for i, key := range blockKeys {
		blockHashes[i] = uint64(key)
		blockTokens[i] = tokens[i*h.blockSize : i*h.blockSize+h.blockSize]
	}

	nBlocksAlreadyInCache, err := h.blockCache.startRequest(req, blockHashes, blockTokens)
	if err != nil {
		return PrefixCacheStats{}, err
	}

	cachedTokens := nBlocksAlreadyInCache * h.blockSize
	req.SetNumberOfCachedPromptTokens(cachedTokens)

	stats := PrefixCacheStats{
		QueriedTokens: len(tokens),
		CachedTokens:  cachedTokens,
	}
	common.WriteToChannel(h.prefixCacheStatsChan, stats, h.logger)

	return stats, nil
}

func (h *KVCacheHelper) OnRequestEnd(requestID string) error {
	return h.blockCache.finishRequest(requestID)
}

// SetModelLoaded marks a model as loaded, affecting block eviction priority
func (h *KVCacheHelper) SetModelLoaded(model string) {
	h.blockCache.setModelLoaded(model)
}

// SetModelUnloaded marks a model as unloaded, its blocks become low-priority eviction candidates
func (h *KVCacheHelper) SetModelUnloaded(model string) {
	h.blockCache.setModelUnloaded(model)
}

func (h *KVCacheHelper) convertMMPlaceholders(placeholders map[string][]openaiserverapi.RenderPlaceholder) map[string][]kvblock.PlaceholderRange {
	res := make(map[string][]kvblock.PlaceholderRange, len(placeholders))

	for k, prs := range placeholders {
		res[k] = make([]kvblock.PlaceholderRange, len(prs))
		for i, pr := range prs {
			res[k][i] = kvblock.PlaceholderRange{Offset: pr.Offset, Length: pr.Length}
		}
	}
	return res
}
