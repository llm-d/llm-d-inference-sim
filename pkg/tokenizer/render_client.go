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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
)

// RenderStatusError is a non-2xx answer from the render service.
type RenderStatusError struct {
	StatusCode int
	Body       []byte
}

func (e *RenderStatusError) Error() string {
	return fmt.Sprintf("vLLM render returned status %d: %s", e.StatusCode, e.Body)
}

// Below vLLM's default keep-alive timeout (VLLM_HTTP_TIMEOUT_KEEP_ALIVE, 5s), so pooled connections are closed by the
// client before the server closes them under a request.
const renderIdleConnTimeout = 4 * time.Second

func newRenderTransport() *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.IdleConnTimeout = renderIdleConnTimeout
	return t
}

type renderClient struct {
	ctx       context.Context
	logger    logr.Logger
	renderURL string
	client    *http.Client
	timeout   time.Duration
	mmTimeout time.Duration
}

func newRenderClient(ctx context.Context, logger logr.Logger, renderURL string, timeout, mmTimeout time.Duration) *renderClient {
	url := strings.TrimRight(renderURL, "/")
	logger.V(logging.INFO).Info("Render client created", "render URL", url)
	return &renderClient{
		ctx:       ctx,
		logger:    logger,
		renderURL: url,
		client:    &http.Client{Transport: newRenderTransport()},
		timeout:   timeout,
		mmTimeout: mmTimeout,
	}
}

func (rc *renderClient) render(endpoint string, payload []byte, mm bool) ([]uint32, *api.RenderMMFeatures, error) {
	if endpoint == "" {
		return nil, nil, errors.New("render endpoint is empty")
	}

	if len(payload) == 0 {
		return []uint32{}, nil, nil // return empty tokens and nil features for empty input
	}

	renderPath := endpoint + "/render"
	timeout := rc.timeout
	if mm {
		timeout = rc.mmTimeout
	}
	respBody, err := rc.postRaw(renderPath, payload, timeout)
	if err != nil {
		return nil, nil, err
	}

	tokenIDs, features, err := rc.parseRenderResponse(respBody)
	if err != nil {
		return nil, nil, fmt.Errorf("RenderRequest: %w", err)
	}
	return tokenIDs, features, nil
}

// derender posts a single-response completions derender payload to the render
// service and returns the decoded text of its only choice.
func (rc *renderClient) derender(payload []byte) (string, error) {
	respBody, err := rc.postRaw("/v1/completions/derender", payload, rc.timeout)
	if err != nil {
		return "", err
	}

	var resp api.TextCompletionsResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return "", fmt.Errorf("unmarshal derender response: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("derender returned no choices")
	}
	return resp.Choices[0].Text, nil
}

// parseRenderResponse handles both array (completions) and object (chat/responses) response shapes.
func (rc *renderClient) parseRenderResponse(body []byte) ([]uint32, *api.RenderMMFeatures, error) {
	body = bytes.TrimSpace(body)
	if len(body) == 0 {
		return nil, nil, errors.New("empty response body")
	}

	if body[0] == '[' {
		var arr []api.RenderResponse
		if err := json.Unmarshal(body, &arr); err != nil {
			return nil, nil, fmt.Errorf("unmarshal array response: %w", err)
		}
		if len(arr) == 0 {
			return nil, nil, errors.New("render returned empty array")
		}
		return arr[0].TokenIDs, arr[0].Features, nil
	}

	var single api.RenderResponse
	if err := json.Unmarshal(body, &single); err != nil {
		return nil, nil, fmt.Errorf("unmarshal response: %w", err)
	}
	return single.TokenIDs, single.Features, nil
}

func (rc *renderClient) postRaw(path string, payload []byte, timeout time.Duration) ([]byte, error) {
	reqCtx, cancel := context.WithTimeout(rc.ctx, timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, rc.renderURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	// Render has no side effects. Marking the POST idempotent lets the transport resend it on a new connection when
	// the server closed a reused one; a nil value sends no header.
	httpReq.Header["Idempotency-Key"] = nil

	httpResp, err := rc.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("post %s: %w", path, err)
	}
	defer httpResp.Body.Close() //nolint:errcheck

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, &RenderStatusError{StatusCode: httpResp.StatusCode, Body: respBody}
	}
	return respBody, nil
}
