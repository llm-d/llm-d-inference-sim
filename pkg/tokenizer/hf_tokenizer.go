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
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	crlog "sigs.k8s.io/controller-runtime/pkg/log"
)

type HFTokenizer struct {
	baseTokenizer

	baseModel    string
	ctx          context.Context
	logger       logr.Logger
	renderClient *renderClient
}

// HF Tokenizer
func NewHFTokenizer(ctx context.Context, logger logr.Logger, renderURL, baseModel string,
	timeout, mmTimeout time.Duration) (*HFTokenizer, error) {
	crlog.SetLogger(logger)
	url := strings.TrimRight(renderURL, "/")
	logger.V(logging.INFO).Info("HF tokenizer created", "render URL", url)
	return &HFTokenizer{
		baseTokenizer: newBaseTokenizer(),
		ctx:           ctx,
		baseModel:     baseModel,
		renderClient:  newRenderClient(ctx, logger, renderURL, timeout, mmTimeout),
		logger:        logger,
	}, nil
}

// RenderRequest sends req.RawRequestPayload() to {renderURL}{req.GetEndpoint()}/render
// and returns the token IDs and multimodal features from the response.
func (hft *HFTokenizer) RenderRequest(req openaiserverapi.Request) ([]uint32, []string, *tokenization.MultiModalFeatures, error) {
	endpoint := req.GetEndpoint()
	if endpoint == "" {
		return nil, nil, nil, errors.New("RenderRequest: request type does not support a render endpoint")
	}

	payload := req.RawRequestPayload()
	if len(payload) == 0 {
		return nil, nil, nil, errors.New("RenderRequest: raw request payload is empty")
	}

	tokenIDs, features, err := hft.renderClient.render(endpoint, payload, req.IsMultiModal())
	if err != nil {
		return nil, nil, nil, fmt.Errorf("RenderRequest: %w", err)
	}

	strTokens := hft.splitIntoTokens(req.PlainText(), len(tokenIDs))
	return tokenIDs, strTokens, hft.toKVCacheMM(features), nil
}

func (hft *HFTokenizer) RenderPlainText(text string) ([]uint32, []string, error) {
	req, err := common.CreateRequestForRenderText(hft.baseModel, text)
	if err != nil {
		return nil, nil, err
	}
	tokens, strTokens, _, err := hft.RenderRequest(req)
	return tokens, strTokens, err
}

func (hft *HFTokenizer) toKVCacheMM(f *renderMMFeatures) *tokenization.MultiModalFeatures {
	if f == nil || (len(f.MMHashes) == 0 && len(f.MMPlaceholders) == 0) {
		return nil
	}
	out := &tokenization.MultiModalFeatures{
		MMHashes:       f.MMHashes,
		MMPlaceholders: make(map[string][]kvblock.PlaceholderRange, len(f.MMPlaceholders)),
	}
	for k, prs := range f.MMPlaceholders {
		ranges := make([]kvblock.PlaceholderRange, len(prs))
		for i, pr := range prs {
			ranges[i] = kvblock.PlaceholderRange{Offset: pr.Offset, Length: pr.Length}
		}
		out.MMPlaceholders[k] = ranges
	}
	return out
}
