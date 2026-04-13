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

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"

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
// 1 - checks which type of connection should be used: UDS socket or TCP
// 2 - based on the information above creates two tokenizers
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

	// no need to start a stand alone tokenizer - it will not be used for the test model
	tokenizer, err := tm.newTokenizer(ctx, "", common.TestModelName)
	if err != nil {
		return err
	}
	tm.testTokenizer = tokenizer

	// run one container for all real model tokenizers
	address, cleanup, err := tm.startTokenizerContainer(ctx)
	if err != nil {
		return err
	}
	tm.cleanupFunc = cleanup
	// create tokenizer for Qwen model
	tm.qwenTokenizer, err = tm.newTokenizer(ctx, address, common.QwenModelName)
	if err != nil {
		cleanup()
		return err
	}

	// create tokenizer for multimodal model
	tm.mmTokenizer, err = tm.newTokenizer(ctx, address, common.MMModelName)
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

// starts a tokenizer in a docker container
// returns docker address, cleanup function and error
func (tm *TokenizerManager) startTokenizerContainer(ctx context.Context) (string, func(), error) {
	container, err := testcontainers.Run(ctx,
		"ghcr.io/llm-d/llm-d-uds-tokenizer:v0.7.1",
		testcontainers.WithExposedPorts("50051/tcp"),
		testcontainers.WithEnv(map[string]string{
			"GRPC_PORT": "50051",
		}),
		testcontainers.WithWaitStrategy(
			wait.ForListeningPort("50051/tcp"),
		),
	)
	if err != nil {
		return "", nil, fmt.Errorf("failed to start testcontainer: %w", err)
	}

	// get mapped port
	mappedPort, err := container.MappedPort(ctx, "50051")
	if err != nil {
		return "", nil, fmt.Errorf("failed to get mapped port: %w", err)
	}

	// get host
	host, err := container.Host(ctx)
	if err != nil {
		return "", nil, fmt.Errorf("failed to get container host: %w", err)
	}

	address := fmt.Sprintf("%s:%s", host, mappedPort.Port())

	cleanup := func() {
		// use another context for cleanup in case the original ctx was cancelled
		if err := container.Terminate(context.Background()); err != nil {
			fmt.Printf("failed to terminate container: %s\n", err)
		}
	}

	return address, cleanup, nil
}

func (tm *TokenizerManager) newTokenizer(ctx context.Context, address string, model string) (Tokenizer, error) {
	if modelExists(model) {
		// for real model create HF tokenizer
		return tm.newHFTokenizer(ctx, address, model)
	} else {
		// for dummy model create simple tokenizer
		tm.logger.Info("Model is not a real HF model, using simulated tokenizer", "model", model)
		tokenizer := NewSimpleTokenizer()
		return tokenizer, nil
	}
}

// create HF Tokenizer
func (tm *TokenizerManager) newHFTokenizer(ctx context.Context, tokenizerAddress, model string) (*HFTokenizer, error) {
	// in test mode don't use uds socker, use tcp instead
	udsTokenizer, err := tokenization.NewUdsTokenizer(ctx,
		&tokenization.UdsTokenizerConfig{SocketFile: tokenizerAddress, UseTCP: true}, model)
	if err != nil {
		tm.logger.Error(err, "failed to connect to tokenizer using TCP")
		return nil, err
	}

	tm.logger.V(logging.DEBUG).Info("Connected to tokenizer using TCP", "address", tokenizerAddress)
	return &HFTokenizer{ctx: ctx, model: model, udsTokenizer: udsTokenizer, logger: tm.logger}, nil
}
