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
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"k8s.io/klog/v2"
)

func TestStartupErrorExitsNonZero(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "llm-d-inference-sim")
	buildCtx, cancelBuild := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancelBuild()
	if out, err := exec.CommandContext(buildCtx, "go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}

	runCtx, cancelRun := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancelRun()
	cmd := exec.CommandContext(runCtx, bin, "--engine=vllm", "--model=test-model", "--kv-cache-size=256 --enable-request-id-headers")
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "SIM_ENGINE=") {
			cmd.Env = append(cmd.Env, kv)
		}
	}
	out, err := cmd.CombinedOutput()
	if runCtx.Err() != nil {
		t.Fatalf("simulator did not exit within 30s on an invalid flag\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("simulator exited with %v on an invalid flag, want exit status 1\n%s", err, out)
	}
	if !strings.Contains(string(out), "failed to read configuration") {
		t.Fatalf("exit status 1 came from another error than the invalid flag\n%s", out)
	}
}

func TestStartupErrorAfterAStopSignalReturns(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	returned := make(chan struct{})
	go func() {
		// Exiting here would end the test binary with status 1 and fail the run.
		exitOnStartupError(ctx)
		close(returned)
	}()
	select {
	case <-returned:
	case <-time.After(klog.ExitFlushTimeout + 5*time.Second):
		t.Fatal("exitOnStartupError did not return after a stop signal")
	}
}
