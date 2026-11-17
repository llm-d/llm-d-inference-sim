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
	"sync/atomic"

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
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	eng, err := engine.Select(engineName)
	if err != nil {
		logger.Error(err, "failed to select engine")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	// parse command line parameters
	config, err := common.ParseCommandParamsAndLoadConfig(eng)
	if err != nil {
		logger.Error(err, "failed to read configuration")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
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
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	simulators, err := simulator.Start(ctx, config, logger, eng)
	if err != nil {
		stopped := ctx.Err() != nil
		logger.Error(err, "failed to create inference simulator")
		exitAfterError(stopped)
	}

	starts := make([]func(context.Context) error, 0, len(simulators))
	for _, sim := range simulators {
		comm := communication.New(logger, sim, &sim.Context)
		starts = append(starts, func(rankCtx context.Context) error { return comm.Start(rankCtx, eng) })
	}
	if stopped, err := runRanks(ctx, starts); err != nil {
		logger.Error(err, "failed to start communication layer")
		exitAfterError(stopped)
	}
}

// runRanks runs every rank and waits for all of them; a rank that fails stops the others, so the process does not keep
// serving on some ranks only.
// stopped reports whether ctx was cancelled before any rank failure was observed: a rank that returns an error while
// ctx is live counts as failed, even if a stop signal arrives while the others shut down.
func runRanks(ctx context.Context, starts []func(context.Context) error) (stopped bool, err error) {
	var failedBeforeStop atomic.Bool
	g, rankCtx := errgroup.WithContext(ctx)
	for _, start := range starts {
		g.Go(func() error {
			err := start(rankCtx)
			if err != nil && ctx.Err() == nil {
				failedBeforeStop.Store(true)
			}
			return err
		})
	}
	err = g.Wait()
	return ctx.Err() != nil && !failedBeforeStop.Load(), err
}

// exitAfterError ends the process after an error from simulator start-up or a rank.
// stopRequested reports whether a stop signal arrived before that error: the process then exits 0, as a graceful
// shutdown does; otherwise it exits 1.
// Errors before simulator start-up exit 1 without consulting it, since a stop signal does not cause them.
func exitAfterError(stopRequested bool) {
	code := 1
	if stopRequested {
		code = 0
	}
	klog.FlushAndExit(klog.ExitFlushTimeout, code)
}
