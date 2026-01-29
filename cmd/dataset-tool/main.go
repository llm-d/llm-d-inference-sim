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

package main

import (
	"context"

	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/cmd/signals"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
)

func main() {
	// setup logger and context with graceful shutdown
	logger := klog.Background()
	ctx := klog.NewContext(context.Background(), logger)
	ctx = signals.SetupSignalHandler(ctx)

	logger.V(logging.INFO).Info("Starting dataset creation tool")

	config := dataset.NewDefaultDSToolConfiguration()
	if err := config.LoadConfig(); err != nil {
		logger.Error(err, "invalid configuration")
		return
	}

	tool, err := dataset.NewDatasetTool(config, logger)
	if err != nil {
		logger.Error(err, "failed to create dataset creation tool")
		return
	}
	err = tool.Run(ctx)
	if err != nil {
		logger.Error(err, "failed to run dataset creation tool")
		return
	}
	logger.V(logging.INFO).Info("Dataset creation tool finished")
}
