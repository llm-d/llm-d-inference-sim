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
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
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

func (hft *HFTokenizer) RenderText(text string) ([]uint32, []string, error) {
	req := openaiserverapi.NewTextCompletionsRenderRequest(hft.baseModel, text)

	tokens, strTokens, _, err := hft.renderRequest(&req, text)

	return tokens, strTokens, err
}

func (hft *HFTokenizer) RenderMessages(messages []openaiserverapi.Message) ([]uint32, []string, *openaiserverapi.RenderMMFeatures, error) {
	req := openaiserverapi.NewChatCompletionsRenderRequest(hft.baseModel, messages)

	return hft.renderRequest(&req, FlattenMessages(messages))
}

func (hft *HFTokenizer) renderRequest(req openaiserverapi.RenderRequest, plainText string) ([]uint32, []string, *openaiserverapi.RenderMMFeatures, error) {
	if req.Endpoint() == "" {
		return nil, nil, nil, errors.New("renderRequest: render endpoint is empty")
	}

	payload, err := json.Marshal(req)
	if err != nil {
		return nil, nil, nil, err
	}

	tokenIDs, features, err := hft.renderClient.render(req.Endpoint(), payload, req.IsMultiModal())
	if err != nil {
		return nil, nil, nil, fmt.Errorf("RenderRequest: %w", err)
	}

	strTokens := hft.splitIntoTokens(plainText, len(tokenIDs))
	return tokenIDs, strTokens, features, nil
}
