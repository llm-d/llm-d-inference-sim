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
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	crlog "sigs.k8s.io/controller-runtime/pkg/log"
)

type HFTokenizer struct {
	baseTokenizer

	baseModel    string
	ctx          context.Context
	logger       logr.Logger
	renderClient *renderClient

	// tokenizeMu guards whether the render service serves /tokenize, read
	// from its /version on the first Tokenize call.
	tokenizeMu        sync.Mutex
	tokenizeChecked   bool
	tokenizeSupported bool
}

// minTokenizeVersion is the first vLLM version whose render service serves /tokenize.
var minTokenizeVersion = [3]int{0, 24, 0}

var vllmVersionRe = regexp.MustCompile(`^v?(\d+)\.(\d+)\.(\d+)`)

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
	req := api.NewTextCompletionsRenderRequest(hft.baseModel, text)

	tokens, strTokens, _, err := hft.renderRequest(&req, text)

	return tokens, strTokens, err
}

func (hft *HFTokenizer) RenderOutputText(text string) ([]uint32, []string, error) {
	req := api.NewTextCompletionsRenderRequest(hft.baseModel, text)
	req.AddSpecialTokens = new(false)

	tokens, strTokens, _, err := hft.renderRequest(&req, text)

	return tokens, strTokens, err
}

func (hft *HFTokenizer) RenderMessages(messages []api.Message, tools api.RenderTools, mmPayload string) ([]uint32, []string, *api.RenderMMFeatures, error) {
	req := api.NewChatCompletionsRenderRequest(hft.baseModel, messages, tools, mmPayload)

	return hft.renderRequest(&req, FlattenMessages(messages))
}

// Detokenize decodes token ids through the render service's
// /v1/completions/derender endpoint.
func (hft *HFTokenizer) Detokenize(tokenIDs []uint32) (string, error) {
	payload, err := json.Marshal(api.NewDetokenizeDerenderRequest(hft.baseModel, tokenIDs))
	if err != nil {
		return "", err
	}
	return hft.renderClient.derender(payload)
}

// Tokenize forwards the request to the render service's /tokenize endpoint,
// so the token IDs and strings are the model's own. A render service older
// than minTokenizeVersion renders the request instead.
func (hft *HFTokenizer) Tokenize(req api.TokenizeRequest) ([]uint32, []string, error) {
	if !hft.renderServesTokenize() {
		return tokenizeByRendering(hft, req)
	}
	body := tokenizeRenderRequest{
		Model:           hft.baseModel,
		Prompt:          req.Prompt,
		ReturnTokenStrs: req.ReturnTokenStrs,
	}
	if req.Prompt == nil {
		messages := req.Messages
		if messages == nil {
			messages = []api.Message{}
		}
		body.Messages = &messages
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, nil, err
	}
	return hft.renderClient.tokenize(payload)
}

// renderServesTokenize reports whether the render service's vLLM version
// serves /tokenize. A 4xx or a body that is not a version object counts as an
// older service; a network error or a 5xx is retried on the next call. The lock is not held during the
// read, so concurrent first calls may each read the version.
func (hft *HFTokenizer) renderServesTokenize() bool {
	hft.tokenizeMu.Lock()
	checked, supported := hft.tokenizeChecked, hft.tokenizeSupported
	hft.tokenizeMu.Unlock()
	if checked {
		return supported
	}

	version, err := hft.renderClient.version()
	if err != nil {
		var statusErr *RenderStatusError
		isStatus := errors.As(err, &statusErr) && statusErr.StatusCode < http.StatusInternalServerError
		if !isStatus && !errors.Is(err, errVersionBody) {
			hft.logger.Error(err, "failed to read the render service version, rendering this /tokenize request")
			return false
		}
		hft.logger.V(logging.WARN).Info("render service has no usable /version, rendering /tokenize requests",
			"error", err.Error())
	} else {
		supported = versionAtLeast(version, minTokenizeVersion)
		hft.logger.V(logging.INFO).Info("Render service version read", "version", version,
			"forward /tokenize", supported)
	}

	hft.tokenizeMu.Lock()
	hft.tokenizeChecked, hft.tokenizeSupported = true, supported
	hft.tokenizeMu.Unlock()
	return supported
}

// versionAtLeast reports whether a vLLM version string such as "0.34.0" or
// "0.34.0+cpu" is at least minVersion. A version it cannot parse is not.
func versionAtLeast(version string, minVersion [3]int) bool {
	m := vllmVersionRe.FindStringSubmatch(version)
	if m == nil {
		return false
	}
	for i := range minVersion {
		n, _ := strconv.Atoi(m[i+1])
		if n != minVersion[i] {
			return n > minVersion[i]
		}
	}
	return true
}

// tokenizeRenderRequest is the body sent to the render service's /tokenize.
// vLLM picks the prompt or the chat form by which field is present, so
// messages, even an empty list, are sent only without a prompt.
type tokenizeRenderRequest struct {
	Model           string         `json:"model"`
	Prompt          *string        `json:"prompt,omitempty"`
	Messages        *[]api.Message `json:"messages,omitempty"`
	ReturnTokenStrs bool           `json:"return_token_strs,omitempty"`
}

func (hft *HFTokenizer) renderRequest(req api.RenderRequest, plainText string) ([]uint32, []string, *api.RenderMMFeatures, error) {
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
