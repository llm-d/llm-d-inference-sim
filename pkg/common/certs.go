/*
Copyright 2025 The Kubernetes Authors.

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

package common

import (
	"context"
	"crypto/tls"
	"fmt"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
)

const debounceDelay = 250 * time.Millisecond

type CertReloader struct {
	cert *atomic.Pointer[tls.Certificate]
}

func NewCertReloader(ctx context.Context, certFile, keyFile string, init *tls.Certificate) (*CertReloader, error) {
	certPtr := &atomic.Pointer[tls.Certificate]{}
	certPtr.Store(init)

	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("failed to create cert watcher: %w", err)
	}

	logger := logr.FromContextOrDiscard(ctx).
		WithName("cert-reloader").
		WithValues("certFile", certFile, "keyFile", keyFile)
	traceLogger := logger.V(logging.TRACE)

	certDir := filepath.Dir(certFile)
	keyDir := filepath.Dir(keyFile)

	if err := w.Add(certDir); err != nil {
		_ = w.Close()
		return nil, fmt.Errorf("failed to watch %q: %w", certDir, err)
	}
	if keyDir != certDir {
		if err := w.Add(keyDir); err != nil {
			_ = w.Close()
			return nil, fmt.Errorf("failed to watch %q: %w", keyDir, err)
		}
	}

	go func() {
		defer func() { _ = w.Close() }()

		var debounceTimer *time.Timer

		for {
			select {
			case ev := <-w.Events:
				traceLogger.Info("Cert changed", "event", ev)

				if ev.Op&(fsnotify.Write|fsnotify.Create) == 0 {
					continue
				}

				if debounceTimer != nil {
					debounceTimer.Stop()
				}

				debounceTimer = time.AfterFunc(debounceDelay, func() {
					cert, err := tls.LoadX509KeyPair(certFile, keyFile)
					if err != nil {
						logger.Error(err, "Failed to reload TLS certificate")
						return
					}
					certPtr.Store(&cert)
					traceLogger.Info("Reloaded TLS certificate")
				})

			case err := <-w.Errors:
				if err != nil {
					logger.Error(err, "cert watcher failed")
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	return &CertReloader{cert: certPtr}, nil
}

func (r *CertReloader) Get() *tls.Certificate {
	return r.cert.Load()
}
