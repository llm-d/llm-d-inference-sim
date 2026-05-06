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
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	crlog "sigs.k8s.io/controller-runtime/pkg/log"

	testcontainers "github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
)

type TokenizerManager struct {
	testTokenizer Tokenizer
	qwenTokenizer Tokenizer
	mmTokenizer   Tokenizer
	cleanupFunc   func()

	logger logr.Logger
}

// creates a new tokenizer manager, each suite creates it's own manager
func NewTokenizerManager() *TokenizerManager {
	return &TokenizerManager{}
}

func (tm *TokenizerManager) TestTokenizer() Tokenizer {
	return tm.testTokenizer
}

func (tm *TokenizerManager) RealTokenizer() Tokenizer {
	return tm.qwenTokenizer
}

func (tm *TokenizerManager) MMTokenizer() Tokenizer {
	return tm.mmTokenizer
}

func (tm *TokenizerManager) Init(ctx context.Context, logger logr.Logger) error {
	tm.logger = logger
	crlog.SetLogger(logger)

	// no need to start a stand alone tokenizer - it will not be used for the test model
	tm.testTokenizer = tm.newSimpleTokenizer(common.TestModelName)

	// run one container for qwen model
	renderURL, cleanup, err := tm.startRenderContainer(ctx, common.QwenModelName)
	if err != nil {
		return err
	}
	tm.cleanupFunc = cleanup

	// don't start a new container - use an existing one
	// renderURL := "http://localhost:8001/"
	// var err error
	// create tokenizer for Qwen model
	tm.qwenTokenizer, err = tm.newTokenizer(ctx, logger, renderURL, common.QwenModelName, 10*time.Second, 30*time.Second)
	if err != nil {
		cleanup()
		return err
	}

	// create tokenizer for multimodal model
	tm.mmTokenizer, err = tm.newTokenizer(ctx, logger, renderURL, common.QwenModelName, 10*time.Second, 30*time.Second)
	if err != nil {
		cleanup()
		return err
	}

	return nil
}

func (tm *TokenizerManager) Clean() {
	if tm.cleanupFunc != nil {
		tm.cleanupFunc()
	}
}

// starts a docker container which runs cpu vLLM in render mode (vllm serve)
// returns the HTTP base URL (http://host:port), cleanup function and error
func (tm *TokenizerManager) startRenderContainer(ctx context.Context, model string) (string, func(), error) {
	container, err := testcontainers.Run(ctx,
		"vllm/vllm-openai-cpu:v0.19.1",
		testcontainers.WithExposedPorts("8000/tcp"),
		testcontainers.WithEntrypoint("vllm"),
		testcontainers.WithCmd("launch", "render", model, "--port=8000"),
		testcontainers.WithTmpfs(map[string]string{"/.cache": "rw"}),
		testcontainers.WithWaitStrategy(
			wait.ForHTTP("/health").
				WithPort("8000/tcp").
				WithStartupTimeout(10*time.Minute),
		),
	)
	if err != nil {
		return "", nil, fmt.Errorf("failed to start render container: %w", err)
	}

	// get mapped port
	mappedPort, err := container.MappedPort(ctx, "8000")
	if err != nil {
		_ = container.Terminate(context.Background())
		return "", nil, fmt.Errorf("failed to get mapped port: %w", err)
	}

	// get host
	host, err := container.Host(ctx)
	if err != nil {
		_ = container.Terminate(context.Background())
		return "", nil, fmt.Errorf("failed to get container host: %w", err)
	}

	address := fmt.Sprintf("http://%s:%s", host, mappedPort.Port())

	cleanup := func() {
		// use another context for cleanup in case the original ctx was cancelled
		if err := container.Terminate(context.Background()); err != nil {
			fmt.Printf("failed to terminate render container: %s\n", err)
		}
	}

	return address, cleanup, nil
}

func (tm *TokenizerManager) newTokenizer(ctx context.Context, logger logr.Logger, renderURL, model string,
	timeout, mmTimeout time.Duration) (Tokenizer, error) {
	if modelExists(model) {
		// for real model create HF tokenizer
		return tm.newHFTokenizer(ctx, logger, renderURL, model, timeout, mmTimeout)
	} else {
		// for dummy model create simple tokenizer
		return tm.newSimpleTokenizer(model), nil
	}
}

// create Simple Tokenizer
func (tm *TokenizerManager) newSimpleTokenizer(model string) *SimpleTokenizer {
	tm.logger.Info("Model is not a real HF model, using simulated tokenizer", "model", model)
	return NewSimpleTokenizer()
}

// create HF Tokenizer
func (tm *TokenizerManager) newHFTokenizer(ctx context.Context, logger logr.Logger, renderURL, model string,
	timeout, mmTimeout time.Duration) (*HFTokenizer, error) {
	tm.logger.V(logging.DEBUG).Info("Creating HF tokenizer", "renderURL", renderURL)
	return NewHFTokenizer(ctx, logger, renderURL, model, timeout, mmTimeout)
}
