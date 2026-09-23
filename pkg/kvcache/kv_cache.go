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
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

type KVCacheHelper struct {
	tokenizer       tokenizer.Tokenizer
	tokensProcessor kvblock.TokenProcessor // turns tokens to kv block keys
	logger          logr.Logger
	blockCache      *blockCache
	blockSize       int
	metrics         *metrics.MetricsBus
}

func NewKVCacheHelper(ctx context.Context, config *common.Configuration, logger logr.Logger,
	tokenizer tokenizer.Tokenizer, metrics *metrics.MetricsBus, encoder EventEncoder) (*KVCacheHelper, error) {
	if config.IP == "" {
		return nil, errors.New("IP should be defined in the environment (POD_IP) for KV cache to work")
	}

	tokenProcConfig := kvblock.DefaultTokenProcessorConfig()
	tokenProcConfig.BlockSizeTokens = config.KVCache.TokenBlockSize
	if config.KVCache.HashSeed != "" {
		tokenProcConfig.HashSeed = config.KVCache.HashSeed
	}
	tokensProcessor, err := kvblock.NewChunkedTokenDatabase(tokenProcConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokens processor: %w", err)
	}

	blockCache, err := newBlockCache(ctx, config, logger, metrics, encoder)
	if err != nil {
		return nil, fmt.Errorf("failed to create block cache: %w", err)
	}

	return &KVCacheHelper{
		tokenizer:       tokenizer,
		tokensProcessor: tokensProcessor,
		blockCache:      blockCache,
		logger:          logger,
		blockSize:       config.KVCache.TokenBlockSize,
		metrics:         metrics,
	}, nil
}

// Run starts the helper.
func (h *KVCacheHelper) Run(ctx context.Context) {
	if r := h.blockCache.eventSender.replayer; r != nil {
		if _, err := r.listen(ctx); err != nil {
			h.logger.Error(err, "KV events replayer failed to bind, replay will be unavailable")
		} else {
			go func() {
				if err := r.serve(ctx); err != nil {
					h.logger.Error(err, "KV events replayer stopped with error")
				}
			}()
		}
	}
	h.blockCache.start(ctx)
}

func (h *KVCacheHelper) Discard() {
	h.blockCache.discard()
}

func (h *KVCacheHelper) Activate() {
	h.blockCache.activate()
}

func (h *KVCacheHelper) OnRequestStart(req api.Request) (metrics.PrefixCacheQueried, error) {
	h.logger.V(logging.TRACE).Info("KV cache - process request")

	tokens := req.TokenizedPrompt().Tokens

	// compute per-block extra features from multimodal metadata (if present).
	var extraFeatures []*kvblock.BlockExtraFeatures
	mmFeatures := req.MMFeatures()

	if mmFeatures != nil {
		extraFeatures = kvblock.ComputeBlockExtraFeatures(
			mmFeatures.MMHashes, h.convertMMPlaceholders(mmFeatures.MMPlaceholders),
			h.blockSize, len(tokens))
	}

	// get block keys
	blockKeys, err := h.tokensProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, req.GetDisplayedModel(), extraFeatures)
	if err != nil {
		return metrics.PrefixCacheQueried{}, fmt.Errorf("failed to convert tokens to block keys: %w", err)
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
		return metrics.PrefixCacheQueried{}, err
	}

	cachedTokens := nBlocksAlreadyInCache * h.blockSize
	req.SetNumberOfCachedPromptTokens(cachedTokens)

	stats := metrics.PrefixCacheQueried{
		QueriedTokens:      len(tokens),
		CachedPromptTokens: cachedTokens,
	}
	if h.metrics != nil {
		common.WriteToChannel(h.metrics.PrefixCacheQuery, stats, h.logger)
	}

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

func (h *KVCacheHelper) convertMMPlaceholders(placeholders map[string][]api.RenderPlaceholder) map[string][]kvblock.PlaceholderRange {
	res := make(map[string][]kvblock.PlaceholderRange, len(placeholders))

	for k, prs := range placeholders {
		res[k] = make([]kvblock.PlaceholderRange, len(prs))
		for i, pr := range prs {
			res[k][i] = kvblock.PlaceholderRange{Offset: pr.Offset, Length: pr.Length}
		}
	}
	return res
}
