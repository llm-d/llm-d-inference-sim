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
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

var simBinary string

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
	dir, err := os.MkdirTemp("", "llm-d-inference-sim-test")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	simBinary = filepath.Join(dir, "llm-d-inference-sim")
	if out, err := buildBinary(simBinary); err != nil {
		fmt.Fprintf(os.Stderr, "build: %v\n%s", err, out)
		_ = os.RemoveAll(dir)
		os.Exit(1)
	}
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

func freePort(t *testing.T) int {
	t.Helper()
	port, err := common.FreeTCPPort()
	if err != nil {
		t.Fatal(err)
	}
	return port
}

type simProcess struct {
	cmd    *exec.Cmd
	out    bytes.Buffer
	done   chan struct{}
	err    error
	cancel context.CancelFunc
}

func startSimulator(t *testing.T, args ...string) *simProcess {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	s := &simProcess{done: make(chan struct{}), cancel: cancel}
	s.cmd = exec.CommandContext(ctx, simBinary, append([]string{"--engine=vllm", "--model=test-model"}, args...)...)
	s.cmd.Stdout, s.cmd.Stderr = &s.out, &s.out
	if err := s.cmd.Start(); err != nil {
		cancel()
		t.Fatal(err)
	}
	go func() {
		s.err = s.cmd.Wait()
		close(s.done)
	}()
	// Every path out of the test waits for the process, so its output is never read while it is still written.
	t.Cleanup(func() {
		cancel()
		<-s.done
	})
	return s
}

func (s *simProcess) exitCode(t *testing.T) int {
	t.Helper()
	select {
	case <-s.done:
	case <-time.After(60 * time.Second):
		s.cancel()
		<-s.done
		t.Fatalf("the simulator did not exit in time\n%s", s.out.String())
	}
	var exitErr *exec.ExitError
	if errors.As(s.err, &exitErr) {
		return exitErr.ExitCode()
	}
	if s.err != nil {
		t.Fatalf("run: %v\n%s", s.err, s.out.String())
	}
	return 0
}

func TestStartupErrorExitsNonZero(t *testing.T) {
	s := startSimulator(t, "--kv-cache-size=abc")
	if code := s.exitCode(t); code != 1 {
		t.Fatalf("exit status %d on an invalid flag, want 1\n%s", code, s.out.String())
	}
	if out := s.out.String(); !strings.Contains(out, "--kv-cache-size") || !strings.Contains(out, "must be an integer") {
		t.Fatalf("exit status 1 came from another error than the invalid flag\n%s", out)
	}
}

// reservePortPair returns a free port whose successor is held by a listener, so the second rank cannot bind it.
func reservePortPair(t *testing.T) (int, net.Listener) {
	t.Helper()
	for range 20 {
		port := freePort(t)
		busy, err := net.Listen("tcp", fmt.Sprintf(":%d", port+1))
		if err == nil {
			return port, busy
		}
	}
	t.Fatal("no free pair of adjacent ports found")
	return 0, nil
}

func TestFailedRankExitsNonZero(t *testing.T) {
	port, busy := reservePortPair(t)
	defer func() { _ = busy.Close() }()
	s := startSimulator(t, fmt.Sprintf("--port=%d", port), "--data-parallel-size=2")
	if code := s.exitCode(t); code != 1 {
		t.Fatalf("exit status %d with the second rank's port taken, want 1\n%s", code, s.out.String())
	}
	out := s.out.String()
	if !strings.Contains(out, fmt.Sprintf(":%d", port+1)) || !strings.Contains(out, "address already in use") {
		t.Fatalf("exit status 1 came from another error than the second rank's port\n%s", out)
	}
}

func waitReady(t *testing.T, s *simProcess, port int) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	client := &http.Client{Timeout: 2 * time.Second}
	for ctx.Err() == nil {
		select {
		case <-s.done:
			t.Fatalf("the simulator exited before it was ready\n%s", s.out.String())
		default:
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/health", port), nil)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatal("the simulator was not ready within 30s")
}

func TestStopSignalWhileServingExitsZero(t *testing.T) {
	port := freePort(t)
	s := startSimulator(t, fmt.Sprintf("--port=%d", port))
	waitReady(t, s, port)
	if err := s.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	if code := s.exitCode(t); code != 0 {
		t.Fatalf("exit status %d after a stop signal while serving, want 0\n%s", code, s.out.String())
	}
}

func TestRunRanksKeepsAFailureThatPrecedesTheStop(t *testing.T) {
	ctx, signal := context.WithCancel(context.Background())
	draining, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	releaseRank := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(func() {
		signal()
		releaseRank()
	})
	starts := []func(context.Context) error{
		func(context.Context) error { return errors.New("bind: address already in use") },
		func(rankCtx context.Context) error {
			<-rankCtx.Done()
			close(draining)
			<-release
			return rankCtx.Err()
		},
	}
	type outcome struct {
		stopped bool
		err     error
	}
	result := make(chan outcome, 1)
	go func() {
		stopped, err := runRanks(ctx, starts)
		result <- outcome{stopped, err}
	}()
	select {
	case <-draining:
	case <-time.After(10 * time.Second):
		t.Fatal("the failed rank did not stop the other rank")
	}
	signal() // the stop signal arrives while the other rank drains
	releaseRank()
	select {
	case o := <-result:
		if o.err == nil {
			t.Fatal("runRanks returned no error after a rank failed")
		}
		if o.stopped {
			t.Fatal("a rank failure followed by a stop signal was reported as a requested stop")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("runRanks did not return")
	}
}

func TestRunRanksReportsAStopThatPrecedesTheError(t *testing.T) {
	// Any error observed after the stop counts as part of the stop, whether or not it comes from the cancellation.
	for _, rankErr := range []error{errors.New("download cancelled"), errors.New("bind: address already in use")} {
		t.Run(rankErr.Error(), func(t *testing.T) {
			ctx, signal := context.WithCancel(context.Background())
			defer signal()
			running := make(chan struct{})
			result := make(chan error, 1)
			var stopped bool
			go func() {
				var err error
				stopped, err = runRanks(ctx, []func(context.Context) error{
					func(rankCtx context.Context) error {
						close(running)
						<-rankCtx.Done()
						return rankErr
					},
				})
				result <- err
			}()
			select {
			case <-running:
			case <-time.After(10 * time.Second):
				t.Fatal("the rank did not start")
			}
			signal()
			select {
			case err := <-result:
				if !errors.Is(err, rankErr) || !stopped {
					t.Fatalf("stopped %v err %v, want a requested stop with the rank's error", stopped, err)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("runRanks did not return")
			}
		})
	}
}

func TestRunRanksReturnsNilWhenEveryRankStops(t *testing.T) {
	ctx, signal := context.WithCancel(context.Background())
	signal()
	stopped, err := runRanks(ctx, []func(context.Context) error{
		func(rankCtx context.Context) error { <-rankCtx.Done(); return nil },
		func(rankCtx context.Context) error { <-rankCtx.Done(); return nil },
	})
	if err != nil || !stopped {
		t.Fatalf("stopped %v err %v, want a requested stop and no error", stopped, err)
	}
}
