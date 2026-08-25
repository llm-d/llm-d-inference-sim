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

// Inference server simulator
package main

import (
	"context"

	"golang.org/x/sync/errgroup"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/cmd/signals"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/engine"
	"github.com/llm-d/llm-d-inference-sim/pkg/simulator"
)

func main() {
	// setup logger and context with graceful shutdown
	logger := klog.Background()
	ctx := klog.NewContext(context.Background(), logger)
	ctx = signals.SetupSignalHandler(ctx)

	// the engine must be resolved before the rest of the configuration, since it
	// determines which configuration defaults apply
	eng, err := engine.Resolve()
	if err != nil {
		logger.Error(err, "failed to resolve engine")
		return
	}

	logger.V(logging.INFO).Info("Starting inference simulator", "engine", eng.Name())

	// parse command line parameters
	config, err := common.ParseCommandParamsAndLoadConfig(eng.NewConfiguration())
	if err != nil {
		logger.Error(err, "failed to read configuration")
		return
	}
	if err := config.Show(logger); err != nil {
		logger.Error(err, "failed to show configuration")
		return
	}

	simulators, err := simulator.Start(ctx, eng, config, logger)
	if err != nil {
		logger.Error(err, "failed to create inference simulator")
		return
	}

	g := new(errgroup.Group)
	for _, sim := range simulators {
		g.Go(func() error {
			return communication.Start(ctx, logger, sim)
		})
	}
	if err := g.Wait(); err != nil {
		logger.Error(err, "failed to start communication layer")
		return
	}

}
