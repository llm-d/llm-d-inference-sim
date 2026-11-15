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
	"time"

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
		return
	}
	eng, err := engine.Select(engineName)
	if err != nil {
		logger.Error(err, "failed to select engine")
		exitOnStartupError(ctx)
		return
	}

	// parse command line parameters
	config, err := common.ParseCommandParamsAndLoadConfig(eng)
	if err != nil {
		logger.Error(err, "failed to read configuration")
		exitOnStartupError(ctx)
		return
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
		return
	}

	simulators, err := simulator.Start(ctx, config, logger, eng)
	if err != nil {
		logger.Error(err, "failed to create inference simulator")
		exitOnStartupError(ctx)
		return
	}

	g := new(errgroup.Group)
	for _, sim := range simulators {
		comm := communication.New(logger, sim, &sim.Context)
		g.Go(func() error {
			return comm.Start(ctx, eng)
		})
	}
	if err := g.Wait(); err != nil {
		logger.Error(err, "failed to start communication layer")
		exitOnStartupError(ctx)
		return
	}

}

// exitOnStartupError ends the process with status 1 unless the signal handler cancelled ctx: a stop requested while
// the simulator starts is not a startup failure, so main returns and the process exits 0.
func exitOnStartupError(ctx context.Context) {
	if ctx.Err() == nil {
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	// Bounded as FlushAndExit is, so a stalled log file cannot keep the process from stopping.
	flushed := make(chan struct{})
	go func() {
		klog.Flush()
		close(flushed)
	}()
	select {
	case <-flushed:
	case <-time.After(klog.ExitFlushTimeout):
	}
}
