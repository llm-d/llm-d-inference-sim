package llmdinferencesim

// contains all logic relevant to KV-cache support
import (
	"context"
	"fmt"

	"github.com/go-logr/logr"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization/prefixstore"
)

type KVCacheHelper struct {
	config         *kvcache.Config
	tokenizersPool *tokenization.Pool
	tokensIndexer  prefixstore.Indexer
	logger         logr.Logger
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

	return &KVCacheHelper{
		config:         config,
		tokenizersPool: tokenizersPool,
		tokensIndexer:  tokensIndexer,
		logger:         logger,
	}, nil
}

// Run starts the helper.
func (h *KVCacheHelper) Run(ctx context.Context) {
	h.tokenizersPool.Run(ctx)
}

func (h *KVCacheHelper) ProcessRequest(vllmReq openaiserverapi.CompletionRequest) error {
	prompt := vllmReq.GetPrompt()
	modelName := vllmReq.GetModel()

	// 0. add to tokenizers pool
	h.tokenizersPool.AddTask(prompt, modelName)

	// 1. get available tokens of longest prefix
	tokens := h.tokensIndexer.FindLongestContainedTokens(prompt, modelName)
	h.logger.Info(">>> After tokensIndexer.FindLongestContainedTokens", "tokens", len(tokens))
	if len(tokens) == 0 {
		//nolint:nilnil // no need to return an error
		h.logger.Info(">>> tokensIndexer.FindLongestContainedTokens returned 0 tokens", "prompt len", len(prompt), "model", modelName)
		return nil
	}

	// 2. get block keys
	// blockKeys := h.tokensProcessor.TokensToKVBlockKeys(tokens, modelName)
	// traceLogger.Info("found tokens", "tokens", tokens, "block-keys", blockKeys)

	return nil
}
