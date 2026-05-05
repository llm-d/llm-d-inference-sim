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

package communication

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	"github.com/soheilhy/cmux"
	"google.golang.org/grpc"
)

type Communication struct {
	logger    logr.Logger
	simulator *vllmsim.VllmSimulator

	// set to 1 during graceful shutdown; new requests are rejected while draining
	stopping atomic.Bool

	// a mutex for sleep-wake up
	sleepMutex sync.RWMutex

	pb.UnimplementedVllmEngineServer

	deprecatedLogged bool

	// startTime records when the server started, used for startup-duration readiness check
	startTime time.Time
}

func New(logger logr.Logger, simulator *vllmsim.VllmSimulator) *Communication {
	return &Communication{logger: logger, simulator: simulator, startTime: time.Now()}
}

func Start(ctx context.Context, logger logr.Logger, simulator *vllmsim.VllmSimulator) error {
	c := Communication{logger: logger, simulator: simulator, startTime: time.Now()}
	c.logger.V(logging.INFO).Info("Starting communication layer")
	return c.start(ctx)
}

func (c *Communication) start(ctx context.Context) error {
	listener, err := c.newListener()
	if err != nil {
		c.logger.Error(err, "failed to create listener")
		return fmt.Errorf("listener creation error: %w", err)
	}

	m := cmux.New(listener)

	// Timeout idle connections during protocol matching to avoid blocking shutdown.
	m.SetReadTimeout(5 * time.Second)

	var grpcServer *grpc.Server
	var grpcErrCh <-chan error
	if !c.simulator.Context.Config.MMEncoderOnly {
		// gRPC uses HTTP/2
		grpcL := m.Match(cmux.HTTP2())
		grpcServer, grpcErrCh = c.startGRPC(grpcL)
		// Check for an immediate startup error.
		select {
		case err := <-grpcErrCh:
			if err != nil {
				return err
			}
		default:
		}
	}

	httpL := m.Match(cmux.Any())
	httpServer, httpErrCh, err := c.startHTTPServer(httpL)
	if err != nil {
		return err
	}
	// Check for an immediate startup error.
	select {
	case err := <-httpErrCh:
		if err != nil {
			return err
		}
	default:
	}

	// Run cmux in a goroutine so the select below can coordinate shutdown.
	cmuxErrCh := make(chan error, 1)
	go func() {
		if err := m.Serve(); !errors.Is(err, net.ErrClosed) {
			cmuxErrCh <- err
		} else {
			cmuxErrCh <- nil
		}
	}()

	// Centralized wait: all shutdown is handled here, not inside individual servers.
	select {
	case <-ctx.Done():
		c.logger.V(logging.INFO).Info("Shutdown signal received, shutting down servers gracefully")

		c.stopping.Store(true)

		// Wait for all in-flight requests to finish before tearing down the servers.
		const drainTimeout = 30 * time.Second
		drainDeadline := time.Now().Add(drainTimeout)
		c.logger.V(logging.INFO).Info("Waiting for all in-flight requests to finish")
		for c.simulator.OpenRequests() > 0 {
			if time.Now().After(drainDeadline) {
				c.logger.V(logging.INFO).Info("Drain timed out, proceeding with shutdown",
					"open_requests", c.simulator.OpenRequests())
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		c.simulator.Stop()

		// Shut down HTTP first: grpcServer.Stop() closes grpcL which causes cmux to stop
		// routing to httpL, making the HTTP server's Serve() return and set s.ln = nil.
		// If gRPC is stopped before HTTP, fasthttp finds s.ln == nil and returns immediately
		// without waiting for active connections.
		const shutdownTimeout = 5 * time.Second
		done := make(chan error, 1)
		go func() {
			done <- httpServer.Shutdown()
		}()
		select {
		case err := <-done:
			// Ignore closed-listener errors from cmux shutting down first.
			if err != nil && !errors.Is(err, net.ErrClosed) {
				c.logger.Error(err, "error during HTTP server shutdown")
			}
		case <-time.After(shutdownTimeout):
			c.logger.V(logging.INFO).Info("HTTP shutdown timed out, forcing close")
		}
		c.logger.V(logging.INFO).Info("HTTP server stopped")

		if grpcServer != nil {
			c.logger.V(logging.INFO).Info("Shutting down gRPC server")
			grpcServer.Stop()
			c.logger.V(logging.INFO).Info("gRPC server stopped")
		}

		listener.Close() //nolint:errcheck
		return nil

	case err := <-grpcErrCh: // nil channel if gRPC disabled — never fires
		if err != nil {
			c.logger.Error(err, "gRPC server failed")
		}
		return err

	case err := <-httpErrCh:
		if err != nil {
			c.logger.Error(err, "HTTP server failed")
		}
		return err

	case err := <-cmuxErrCh:
		if err != nil {
			return fmt.Errorf("cmux failed: %w", err)
		}
		return nil
	}
}

// Print prints to a log, implementation of fasthttp.Logger
func (c *Communication) Printf(format string, args ...interface{}) {
	c.logger.V(logging.WARN).Info("Server error", "msg", fmt.Sprintf(format, args...))
}
