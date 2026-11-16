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

	engineName, err := common.ResolveEngineName()
	if err != nil {
		logger.Error(err, "failed to resolve engine")
		exitOnStartupError(ctx)
	}
	eng, err := engine.Select(engineName)
	if err != nil {
		logger.Error(err, "failed to select engine")
		exitOnStartupError(ctx)
	}

	// parse command line parameters
	config, err := common.ParseCommandParamsAndLoadConfig(eng)
	if err != nil {
		logger.Error(err, "failed to read configuration")
		exitOnStartupError(ctx)
	}

	// klog's default verbosity (0) is only raised to INFO by ParseCommandParamsAndLoadConfig
	// above, so this must run after it to actually be visible at the default verbosity.
	logger.V(logging.INFO).Info("Starting inference simulator", "engine", eng.Name())
	if eng.Name() == "sglang" {
		logger.V(logging.WARN).Info("================================================================")
		logger.V(logging.WARN).Info("WARNING: SGLang engine support is experimental and incomplete.")
		logger.V(logging.WARN).Info("It is not ready for production use, see docs/engine-backends.md.")
		logger.V(logging.WARN).Info("================================================================")
	}

	if err := config.Show(logger); err != nil {
		logger.Error(err, "failed to show configuration")
		exitOnStartupError(ctx)
	}

	simulators, err := simulator.Start(ctx, config, logger, eng)
	if err != nil {
		logger.Error(err, "failed to create inference simulator")
		exitOnStartupError(ctx)
	}

	// A rank that fails stops the others, so the process exits instead of serving on some ranks only.
	g, rankCtx := errgroup.WithContext(ctx)
	for _, sim := range simulators {
		comm := communication.New(logger, sim, &sim.Context)
		g.Go(func() error {
			return comm.Start(rankCtx, eng)
		})
	}
	if err := g.Wait(); err != nil {
		logger.Error(err, "failed to start communication layer")
		exitOnStartupError(ctx)
	}

}

// exitOnStartupError ends the process after a startup error. A stop signal received before the error means the stop
// was requested, so the process exits 0 whatever the error says; any other startup error exits 1.
func exitOnStartupError(ctx context.Context) {
	code := 1
	if ctx.Err() != nil {
		code = 0
	}
	klog.FlushAndExit(klog.ExitFlushTimeout, code)
}
