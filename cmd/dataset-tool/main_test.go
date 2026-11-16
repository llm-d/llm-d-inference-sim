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
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

var toolBinary string

// buildBinary builds the package in its own process group, so a build that times out is stopped with every compiler
// process it started.
func buildBinary(out string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "build", "-o", out, ".")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = 10 * time.Second
	return cmd.CombinedOutput()
}

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "dataset-tool-test")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	toolBinary = filepath.Join(dir, "dataset-tool")
	if out, err := buildBinary(toolBinary); err != nil {
		fmt.Fprintf(os.Stderr, "build: %v\n%s", err, out)
		_ = os.RemoveAll(dir)
		os.Exit(1)
	}
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

func TestConfigurationErrorExitsNonZero(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, toolBinary).CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("dataset-tool did not exit in time\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("dataset-tool exited with %v without --model, want exit status 1\n%s", err, out)
	}
	if !strings.Contains(string(out), "--model is not defined") {
		t.Fatalf("exit status 1 came from another error than the missing --model\n%s", out)
	}
}

func TestRunErrorExitsNonZero(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dir := t.TempDir()
	out, err := exec.CommandContext(ctx, toolBinary, "--model=test-model", "--local-path="+filepath.Join(dir, "missing"),
		"--input-file=input.json", "--render-url=http://127.0.0.1:1", "--output-path="+dir).CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("dataset-tool did not exit in time\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("dataset-tool exited with %v on a missing input file, want exit status 1\n%s", err, out)
	}
	if !strings.Contains(string(out), "failed to run dataset creation tool") ||
		!strings.Contains(string(out), filepath.Join(dir, "missing", "input.json")) {
		t.Fatalf("exit status 1 came from another error than the failed run\n%s", out)
	}
}

func TestStopSignalDuringRunExitsNonZero(t *testing.T) {
	dir := t.TempDir()
	input := `[{"id": "1", "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}]`
	if err := os.WriteFile(filepath.Join(dir, "input.json"), []byte(input), 0o600); err != nil {
		t.Fatal(err)
	}
	requested, release := make(chan struct{}), make(chan struct{})
	var requestedOnce sync.Once
	// The render service holds every request, so the run is still busy when the signal arrives.
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		requestedOnce.Do(func() { close(requested) })
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	// Cleanups run in reverse order: the child is killed, then held requests are released, then the server closes.
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(release) })

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	var out bytes.Buffer
	cmd := exec.CommandContext(ctx, toolBinary, "--model=test-model", "--local-path="+dir, "--input-file=input.json",
		"--render-url="+server.URL, "--render-timeout=60s", "--output-path="+dir)
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	t.Cleanup(func() {
		cancel()
		<-done
	})
	select {
	case <-requested:
	case err := <-done:
		done <- err
		t.Fatalf("dataset-tool exited with %v before rendering\n%s", err, out.String())
	case <-ctx.Done():
		t.Fatal("dataset-tool never requested a render")
	}
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	var err error
	select {
	case err = <-done:
		done <- err
	case <-time.After(10 * time.Second):
		t.Fatal("dataset-tool did not exit within 10s of the stop signal")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("dataset-tool exited with %v after a stop signal during the run, want exit status 1\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "failed to run dataset creation tool") ||
		!strings.Contains(out.String(), "context canceled") {
		t.Fatalf("exit status 1 came from another error than the cancelled run\n%s", out.String())
	}
}
