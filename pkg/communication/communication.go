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

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	"github.com/soheilhy/cmux"
)

type Communication struct {
	logger    logr.Logger
	simulator *vllmsim.VllmSimulator

	// a mutex for sleep-wake up
	sleepMutex sync.RWMutex

	pb.UnimplementedVllmEngineServer
}

func New(logger logr.Logger, simulator *vllmsim.VllmSimulator) *Communication {
	return &Communication{logger: logger, simulator: simulator}
}

func Start(ctx context.Context, logger logr.Logger, simulator *vllmsim.VllmSimulator) error {
	c := Communication{logger: logger, simulator: simulator}
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

	if !c.simulator.Context.Config.MMEncoderOnly {
		// gRPC uses HTTP/2
		grpcL := m.Match(cmux.HTTP2())

		// start the gRPC server
		errCh := make(chan error, 1)
		go func() {
			errCh <- c.startGRPC(ctx, grpcL)
		}()

		select {
		case err := <-errCh:
			if err != nil {
				return err
			}
		default:
		}
	}
	httpL := m.Match(cmux.Any())

	// start the http server with context support
	errCh := make(chan error, 1)
	go func() {
		errCh <- c.StartHTTPServer(ctx, httpL)
	}()

	select {
	case err := <-errCh:
		if err != nil {
			return err
		}
	default:
	}

	err = m.Serve()
	if !errors.Is(err, net.ErrClosed) {
		return fmt.Errorf("cmux failed: %w", err)
	}
	return nil
}

// Print prints to a log, implementation of fasthttp.Logger
func (c *Communication) Printf(format string, args ...interface{}) {
	c.logger.V(logging.WARN).Info("Server error", "msg", fmt.Sprintf(format, args...))
}

func (c *Communication) onResponseSendFinished(respCtx vllmsim.ResponseContext) {
	if respCtx != nil {
		c.simulator.ResponseSentCallback(respCtx.RequestContext())
		respCtx.Done()
	}
}
