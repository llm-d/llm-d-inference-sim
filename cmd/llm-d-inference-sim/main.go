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

// VLLM server simulator
// supports /v1/chat/completions, /v1/completions, /models, and /metrics (TODO - add it)
package main

import (
	"context"

	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/cmd/signals"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
)

func main() {
	// setup logger and context with graceful shutdown
	logger := klog.Background()
	ctx := klog.NewContext(context.Background(), logger)
	ctx = signals.SetupSignalHandler(ctx)

	logger.Info("Starting vLLM simulator")

	vllmSim, err := vllmsim.New(logger)
	if err != nil {
		logger.Error(err, "Failed to create vLLM simulator")
		return
	}
	if err := vllmSim.Start(ctx); err != nil {
		logger.Error(err, "vLLM simulator failed")
	}
}
