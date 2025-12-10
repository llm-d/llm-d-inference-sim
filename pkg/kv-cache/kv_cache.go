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
	"os"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvblock"
	preprocessing "github.com/llm-d/llm-d-kv-cache-manager/pkg/preprocessing/chat_completions"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization"
)

const (
	envHFToken = "HF_TOKEN"
)

type KVCacheHelper struct {
	tokenizer          tokenization.Tokenizer
	tokensProcessor    kvblock.TokenProcessor // turns tokens to kv block keys
	logger             logr.Logger
	blockCache         *blockCache
	blockSize          int
	chatTemplate       string
	chatTemplateKWArgs map[string]interface{}
	baseModel          string
}

func NewKVCacheHelper(ctx context.Context, config *common.Configuration, logger logr.Logger, usageChan chan float64,
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

	templateReq := preprocessing.FetchChatTemplateRequest{
		Model: config.Model,
		Token: os.Getenv(envHFToken),
	}

	chatTemplatingProcessor := preprocessing.NewChatTemplatingProcessor()
	if err := chatTemplatingProcessor.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize chat-templating processor: %w", err)
	}

	chatTemplate, chatTemplateKWArgs, err1 := chatTemplatingProcessor.FetchChatTemplate(ctx, templateReq)
	if err1 != nil {
		logger.Error(err, "failed to get chat template")
		return nil, err1
	}
	logger.V(logging.DEBUG).Info("Chat template loaded", "template", chatTemplate, "params", chatTemplateKWArgs)

	return &KVCacheHelper{
		tokenizer:          tokenizer,
		tokensProcessor:    tokensProcessor,
		blockCache:         blockCache,
		logger:             logger,
		blockSize:          config.TokenBlockSize,
		chatTemplate:       chatTemplate,
		chatTemplateKWArgs: chatTemplateKWArgs,
		baseModel:          config.Model,
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

func (h *KVCacheHelper) OnRequestStart(vllmReq openaiserverapi.CompletionRequest, isChatCompletion bool) error {
	h.logger.V(logging.TRACE).Info("KV cache - process request")

	var prompt string

	if isChatCompletion {
		renderReq := preprocessing.RenderJinjaTemplateRequest{
			Conversations:             make([]preprocessing.ChatMessage, 0),
			Tools:                     make([]interface{}, 0),
			Documents:                 make([]interface{}, 0),
			ReturnAssistantTokensMask: false,
			ContinueFinalMessage:      false,
			AddGenerationPrompt:       false,
			ChatTemplate:              h.chatTemplate,
			ChatTemplateKWArgs:        h.chatTemplateKWArgs,
		}
		// Convert messages to the format expected by the renderer
		for _, msg := range vllmReq.GetMessages() {
			renderReq.Conversations = append(renderReq.Conversations, preprocessing.ChatMessage{
				Role:    msg.Role,
				Content: msg.Content.Raw,
			})
		}

		var err error
		// Don't use vllmReq.GetModel() - it can contain LoRA's name,
		// this call requires the base model name
		prompt, err = h.tokenizer.RenderChatTemplate(h.baseModel, &renderReq)
		if err != nil {
			h.logger.Error(err, "chat template render failed")
			return err
		}
	} else {
		prompt = vllmReq.GetPrompt()
	}

	modelName := vllmReq.GetModel()
	requestID := vllmReq.GetRequestID()

	// tokenize the input
	tokens, _, err := h.tokenizer.Encode(prompt, modelName)
	if err != nil {
		h.logger.Error(err, "prompt tokenization failed")
		return err
	}

	// get block keys
	blockKeys := h.tokensProcessor.TokensToKVBlockKeys(tokens, modelName)
	h.logger.V(logging.TRACE).Info("Found tokens", "tokens", tokens, "block-keys", blockKeys)

	blockHashes := make([]uint64, len(blockKeys))
	for i, key := range blockKeys {
		blockHashes[i] = key.ChunkHash
	}

	nBlocksAlreadyInCache, err := h.blockCache.startRequest(requestID, blockHashes)
	vllmReq.SetNumberOfCachedPromptTokens(nBlocksAlreadyInCache * h.blockSize)
	return err
}

func (h *KVCacheHelper) OnRequestEnd(requestID string) error {
	return h.blockCache.finishRequest(requestID)
}
