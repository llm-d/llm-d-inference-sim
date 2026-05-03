/*
Copyright 2026 The llm-d-inference-sim Authors.

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

package tokenizer

import (
	"context"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	crlog "sigs.k8s.io/controller-runtime/pkg/log"
)

type HFTokenizer struct {
	udsTokenizer *tokenization.UdsTokenizer
	model        string
	ctx          context.Context
	logger       logr.Logger
}

// HF Tokenizer
func NewHFTokenizer(ctx context.Context, logger logr.Logger, udsSocketPath, model string) (*HFTokenizer, error) {
	crlog.SetLogger(logger)
	logger.V(logging.INFO).Info("Connecting to UDS tokenizer", "socket path", udsSocketPath)
	udsTokenizer, err := tokenization.NewUdsTokenizer(ctx,
		&tokenization.UdsTokenizerConfig{SocketFile: udsSocketPath}, model)

	if err != nil {
		logger.Error(err, "failed to connect to UDS tokenizer")
		return nil, err
	}

	logger.V(logging.INFO).Info("Connected to UDS tokenizer")
	return &HFTokenizer{ctx: ctx, model: model, udsTokenizer: udsTokenizer, logger: logger}, nil
}

// Converts input to tokens
func (hft *HFTokenizer) RenderText(input string) ([]uint32, []string, error) {
	tokens, offsets, err := hft.udsTokenizer.Render(input)

	if err != nil {
		return nil, nil, err
	}

	textTokens := make([]string, len(tokens))

	if len(offsets) > 0 {
		for i, offset := range offsets {
			textTokens[i] = input[offset[0]:offset[1]]
		}
	} else {
		// only one token returned, use the whole input as the text token
		textTokens[0] = input
	}

	return tokens, textTokens, err
}

// Converts input to tokens in two steps: templatization and tokenization
func (hft *HFTokenizer) RenderChatCompletion(messages []openaiserverapi.ChatComplMessage) ([]uint32, []string, *tokenization.MultiModalFeatures, error) {
	// Convert messages to conversation format
	conversations := make([]types.Conversation, len(messages))
	for i, msg := range messages {
		conversations[i].Role = msg.Role

		if msg.Content.Structured != nil {
			conversations[i].Content.Structured = make([]types.ContentBlock, len(msg.Content.Structured))

			// copy structured content
			for j, block := range msg.Content.Structured {
				conversations[i].Content.Structured[j] = types.ContentBlock{
					Type: block.Type,
					Text: block.Text,
					ImageURL: types.ImageBlock{
						URL: block.ImageURL.Url,
					},
				}
			}
		} else {
			conversations[i].Content.Raw = msg.Content.Raw
		}
	}

	renderReq := types.RenderChatRequest{
		Conversation:        conversations,
		Tools:               make([]any, 0),
		Documents:           make([]any, 0),
		AddGenerationPrompt: false,
	}

	tokens, mmFeatures, err := hft.udsTokenizer.RenderChat(&renderReq)

	return tokens, []string{}, mmFeatures, err
}
