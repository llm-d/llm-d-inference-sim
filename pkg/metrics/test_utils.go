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
package metrics

import (
	"context"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// StubAdapter records nothing and exposes no collectors, for tests that assert
// on state outside any engine's metric surface.
type StubAdapter struct{}

// NewStubAdapter is an AdapterFactory building a StubAdapter.
func NewStubAdapter(context.Context, *prometheus.Registry, logr.Logger,
	common.Configuration) (MetricsAdapter, error) {
	return StubAdapter{}, nil
}

func (StubAdapter) Start(context.Context) error { return nil }
func (StubAdapter) Close() error                { return nil }

func (StubAdapter) OnRequestReceived(RequestReceived)         {}
func (StubAdapter) OnRequestQueued(RequestQueued)             {}
func (StubAdapter) OnRequestDequeued(RequestDequeued)         {}
func (StubAdapter) OnRequestRunning(RequestRunning)           {}
func (StubAdapter) OnPrefillStarted(PrefillStarted)           {}
func (StubAdapter) OnPrefillEnded(PrefillEnded)               {}
func (StubAdapter) OnDecodeStarted(DecodeStarted)             {}
func (StubAdapter) OnTokenGenerated(TokenGenerated)           {}
func (StubAdapter) OnDecodeEnded(DecodeEnded)                 {}
func (StubAdapter) OnRequestSucceeded(RequestSucceeded)       {}
func (StubAdapter) OnRequestFailed(RequestFailed)             {}
func (StubAdapter) OnRequestRejected(RequestRejected)         {}
func (StubAdapter) OnKVCacheUsageChanged(KVCacheUsageChanged) {}
func (StubAdapter) OnPrefixCacheQueried(PrefixCacheQueried)   {}
func (StubAdapter) OnLoRASetsChanged(LoRASetsChanged)         {}

func (StubAdapter) ApplyFakeMetricsUpdate(common.FakeMetrics) {}

func (StubAdapter) NewFakeMetrics() common.FakeMetrics { return nil }

func (StubAdapter) ValidateConfig(*common.Configuration) error { return nil }

func (StubAdapter) NewMetricsAdapter(context.Context, *prometheus.Registry, logr.Logger,
	common.Configuration) (MetricsAdapter, error) {
	return StubAdapter{}, nil
}
