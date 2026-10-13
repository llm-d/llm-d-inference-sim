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

// The channel payloads an engine's MetricsAdapter carries from its event
// handlers to its per-metric updater goroutines, and the collector reset those
// updaters perform.
//
// Each payload is a two-variant union: a value a real event produced, or the
// absolute target state of a fake-metrics configuration. Both engines need the
// same three shapes, so they live here rather than once per adapter. A payload
// whose variants carry engine-specific labels stays with its engine.

package metrics

import (
	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
)

// GaugeUpdate is a gauge-channel payload. Exactly one variant must be non-nil.
//
// Add is the event-sourced variant: a delta for a gauge the adapter mirrors
// from a local counter, and the absolute value for a utilization gauge it sets
// directly. Which of the two it is belongs to the updater receiving it, not to
// the payload.
type GaugeUpdate struct {
	Add   *float64
	Reset *float64
}

// HistogramUpdate is a histogram-channel payload: either a single Observe or a
// Reset to a target bucket state. Exactly one variant must be non-nil.
type HistogramUpdate struct {
	Observe *float64
	Reset   *HistogramReset
}

// HistogramReset carries the target state for a reset-shaped histogram
// update: unregister the current collector, recreate it, then Observe once
// per bucket according to Samples over the Buckets boundaries.
type HistogramReset struct {
	Buckets []float64
	Samples []int
}

// CounterUpdate is a counter-channel payload: either an Add delta or a
// Reset to a specific value. Exactly one variant must be non-nil.
type CounterUpdate struct {
	Add   *float64
	Reset *CounterReset
}

// CounterReset carries the target state for a reset-shaped counter update:
// unregister the current collector, recreate it, then Add Value. A nil
// Value leaves the recreated counter with no series stamped, so it is
// absent from /metrics rather than reporting an explicit zero.
type CounterReset struct {
	Value *float64
}

// GaugeAdd wraps a real event's value for a gauge channel.
func GaugeAdd(v float64) GaugeUpdate {
	return GaugeUpdate{Add: &v}
}

// GaugeReset wraps an absolute fake-metrics value for a gauge channel.
func GaugeReset(v float64) GaugeUpdate {
	return GaugeUpdate{Reset: &v}
}

// Observation wraps a single Observe value for a histogram channel.
func Observation(v float64) HistogramUpdate {
	return HistogramUpdate{Observe: &v}
}

// CounterAdd wraps a real event's delta for a counter channel.
func CounterAdd(v float64) CounterUpdate {
	return CounterUpdate{Add: &v}
}

// CounterResetTo wraps an absolute fake-metrics value for a counter channel. A
// nil value resets the counter to no series at all.
func CounterResetTo(v *float64) CounterUpdate {
	return CounterUpdate{Reset: &CounterReset{Value: v}}
}

// ResetCollector unregisters current from registry, recreates it via recreate,
// then calls populate to stamp the recreated collector's series. A collector
// that fails to re-register is logged under errMsg and left unpopulated, so a
// caller cannot observe the failure; call only from an updater goroutine.
//
// Recreating is what makes a reset absolute: a Prometheus collector's value can
// only grow for a counter and only be stamped per series for any of them, so a
// fake-metrics configuration replacing the current state has to replace the
// collector behind it.
func ResetCollector(registry *prometheus.Registry, logger logr.Logger, current prometheus.Collector,
	recreate func() error, errMsg string, populate func()) {
	registry.Unregister(current)
	if err := recreate(); err != nil {
		logger.Error(err, errMsg)
		return
	}
	populate()
}
