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

package sglang

import (
	"context"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
)

// metricsAdapter registers no collectors and records nothing, so /metrics
// serves an empty body for this engine. sglang's own sglang:-prefixed metric
// set replaces it.
type metricsAdapter struct{}

// NewMetricsAdapter returns this engine's metrics adapter. It registers nothing
// on registry, so the simulator's state-change events are accepted and dropped.
func (Engine) NewMetricsAdapter(context.Context, *prometheus.Registry, logr.Logger,
	common.Configuration) (metrics.MetricsAdapter, error) {
	return metricsAdapter{}, nil
}

func (metricsAdapter) Start(context.Context) error { return nil }
func (metricsAdapter) Close() error                { return nil }

func (metricsAdapter) OnRequestReceived(metrics.RequestReceived)         {}
func (metricsAdapter) OnRequestQueued(metrics.RequestQueued)             {}
func (metricsAdapter) OnRequestDequeued(metrics.RequestDequeued)         {}
func (metricsAdapter) OnRequestRunning(metrics.RequestRunning)           {}
func (metricsAdapter) OnPrefillStarted(metrics.PrefillStarted)           {}
func (metricsAdapter) OnPrefillEnded(metrics.PrefillEnded)               {}
func (metricsAdapter) OnDecodeStarted(metrics.DecodeStarted)             {}
func (metricsAdapter) OnTokenGenerated(metrics.TokenGenerated)           {}
func (metricsAdapter) OnDecodeEnded(metrics.DecodeEnded)                 {}
func (metricsAdapter) OnRequestSucceeded(metrics.RequestSucceeded)       {}
func (metricsAdapter) OnRequestFailed(metrics.RequestFailed)             {}
func (metricsAdapter) OnRequestRejected(metrics.RequestRejected)         {}
func (metricsAdapter) OnKVCacheUsageChanged(metrics.KVCacheUsageChanged) {}
func (metricsAdapter) OnPrefixCacheQueried(metrics.PrefixCacheQueried)   {}
func (metricsAdapter) OnLoRASetsChanged(metrics.LoRASetsChanged)         {}

// ApplyFakeMetricsUpdate is unreachable: ValidateConfig rejects a non-nil
// FakeMetrics, so no update can be enqueued for this engine.
func (metricsAdapter) ApplyFakeMetricsUpdate(common.FakeMetrics) {}
