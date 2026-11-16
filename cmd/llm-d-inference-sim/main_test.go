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
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func buildSimulator(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "llm-d-inference-sim")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, "go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	return bin
}

// simulatorCommand runs the binary with the vllm engine, whatever SIM_ENGINE the environment sets.
func simulatorCommand(ctx context.Context, bin string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, bin, append([]string{"--engine=vllm", "--model=test-model"}, args...)...)
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "SIM_ENGINE=") {
			cmd.Env = append(cmd.Env, kv)
		}
	}
	return cmd
}

func exitCode(t *testing.T, ctx context.Context, err error, out []byte) int {
	t.Helper()
	if ctx.Err() != nil {
		t.Fatalf("the simulator did not exit in time\n%s", out)
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode()
	}
	if err != nil {
		t.Fatalf("run: %v\n%s", err, out)
	}
	return 0
}

func TestStartupErrorExitsNonZero(t *testing.T) {
	bin := buildSimulator(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := simulatorCommand(ctx, bin, "--kv-cache-size=256 --enable-request-id-headers").CombinedOutput()
	if code := exitCode(t, ctx, err, out); code != 1 {
		t.Fatalf("exit status %d on an invalid flag, want 1\n%s", code, out)
	}
	if !strings.Contains(string(out), "--kv-cache-size") || !strings.Contains(string(out), "must be an integer") {
		t.Fatalf("exit status 1 came from another error than the invalid flag\n%s", out)
	}
}

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = l.Close() }()
	return l.Addr().(*net.TCPAddr).Port
}

func TestFailedRankExitsNonZero(t *testing.T) {
	bin := buildSimulator(t)
	port := freePort(t)
	busy, err := net.Listen("tcp", fmt.Sprintf(":%d", port+1))
	if err != nil {
		t.Skipf("port %d is not free: %v", port+1, err)
	}
	defer func() { _ = busy.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	out, err := simulatorCommand(ctx, bin, fmt.Sprintf("--port=%d", port), "--data-parallel-size=2").CombinedOutput()
	if code := exitCode(t, ctx, err, out); code != 1 {
		t.Fatalf("exit status %d with the second rank's port taken, want 1\n%s", code, out)
	}
}

func TestStopSignalDuringStartupExitsZero(t *testing.T) {
	bin := buildSimulator(t)
	requested := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "100000000")
		w.WriteHeader(http.StatusOK)
		close(requested)
		for {
			if _, err := w.Write(bytes.Repeat([]byte{0}, 1024)); err != nil {
				return
			}
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
				return
			case <-time.After(100 * time.Millisecond):
			}
		}
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	var out bytes.Buffer
	cmd := simulatorCommand(ctx, bin, fmt.Sprintf("--port=%d", freePort(t)),
		"--dataset-path="+filepath.Join(t.TempDir(), "dataset.sqlite3"), "--dataset-url="+server.URL)
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-requested:
	case <-ctx.Done():
		t.Fatalf("the simulator never requested the dataset\n%s", out.String())
	}
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	err := cmd.Wait()
	if code := exitCode(t, ctx, err, out.Bytes()); code != 0 {
		t.Fatalf("exit status %d after a stop signal during startup, want 0\n%s", code, out.String())
	}
}
