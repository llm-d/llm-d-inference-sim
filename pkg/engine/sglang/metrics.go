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

// sglang Prometheus implementation of metrics.MetricsAdapter.

package sglang

import (
	"context"
	"strconv"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
)

const (
	SGLangRunningReqsMetricName           = "sglang:num_running_reqs"
	SGLangQueueReqsMetricName             = "sglang:num_queue_reqs"
	SGLangTTFTMetricName                  = "sglang:time_to_first_token_seconds"
	SGLangInterTokenLatencyMetricName     = "sglang:inter_token_latency_seconds"
	SGLangE2EReqLatencyMetricName         = "sglang:e2e_request_latency_seconds"
	SGLangPromptTokensTotalMetricName     = "sglang:prompt_tokens_total"
	SGLangGenerationTokensTotalMetricName = "sglang:generation_tokens_total"
)

// promLabelIsStreaming splits the metrics sglang reports separately for
// streaming and non-streaming requests.
const promLabelIsStreaming = "is_streaming"

var (
	modelLabel       = []string{api.PromLabelModelName}
	streamModelLabel = []string{api.PromLabelModelName, promLabelIsStreaming}
)

// Bucket boundaries of the three request-latency histograms, matching the
// defaults of sglang's --bucket-time-to-first-token,
// --bucket-inter-token-latency and --bucket-e2e-request-latency.
//
// Reference: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py
var (
	ttftBucketsBoundaries = []float64{0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400}

	interTokenLatencyBucketsBoundaries = []float64{0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025,
		0.030, 0.035, 0.040, 0.060, 0.080, 0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 2.000, 4.000, 6.000, 8.000}

	e2eReqLatencyBucketsBoundaries = []float64{0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80,
		100, 200, 400, 600, 1200, 1800, 2400}
)

// gaugeDelta is a gauge-channel payload: a delta to apply to the adapter-local
// counter mirrored onto the gauge.
type gaugeDelta float64

// observation is a histogram-channel payload for a metric sglang reports
// without splitting it by request kind.
type observation float64

// streamObservation is a payload carrying the observed value together with the
// kind of request it came from, for a metric labeled by is_streaming.
type streamObservation struct {
	value       float64
	isStreaming bool
}

// sglangMetricsAdapter implements metrics.MetricsAdapter and produces the sglang
// Prometheus surface. The bus dispatches drained events to the On<Event>
// handlers, which fan out to per-metric channels, written to Prometheus by one
// updater goroutine per metric.
//
// The surface is a subset of sglang's own: the metrics whose feature this
// engine simulates, carrying the labels the simulator can fill. See
// docs/metrics.md for what is left out and why.
type sglangMetricsAdapter struct {
	logger   logr.Logger
	config   common.Configuration
	registry *prometheus.Registry

	// gauges
	runningRequests *prometheus.GaugeVec
	queuedRequests  *prometheus.GaugeVec

	// histograms
	ttft              *prometheus.HistogramVec
	interTokenLatency *prometheus.HistogramVec
	e2eReqLatency     *prometheus.HistogramVec

	// counters
	promptTokensTotal     *prometheus.CounterVec
	generationTokensTotal *prometheus.CounterVec

	// Channels: one per Prometheus metric family. Handlers push here; the
	// updater goroutines drain each and perform the actual Prometheus mutation.
	runReqChan   common.Channel[gaugeDelta]
	queueReqChan common.Channel[gaugeDelta]

	ttftChan              common.Channel[streamObservation]
	interTokenLatencyChan common.Channel[observation]
	e2eReqLatencyChan     common.Channel[streamObservation]

	promptTokensTotalChan     common.Channel[streamObservation]
	generationTokensTotalChan common.Channel[streamObservation]

	// nQueuedReqs / nRunningReqs are the adapter-local counters mirrored onto
	// the num_queue_reqs / num_running_reqs gauges.
	nQueuedReqs  int64
	nRunningReqs int64
}

// NewMetricsAdapter returns this engine's metrics adapter.
func (Engine) NewMetricsAdapter(ctx context.Context, registry *prometheus.Registry,
	logger logr.Logger, config common.Configuration) (metrics.MetricsAdapter, error) {
	return newMetricsAdapter(ctx, registry, logger, config)
}

// newMetricsAdapter registers the Prometheus collectors on registry, stamps
// initial values, and spawns the per-metric updater goroutines. ctx must match
// the one passed to metrics.NewMetricsBus. Returns an error if any collector
// fails to register.
func newMetricsAdapter(ctx context.Context, registry *prometheus.Registry,
	logger logr.Logger, config common.Configuration) (*sglangMetricsAdapter, error) {
	m := &sglangMetricsAdapter{
		logger:   logger,
		registry: registry,
		config:   config,
	}

	if err := m.buildMetrics(); err != nil {
		return nil, err
	}
	m.createAndStartPrometheusChannels(ctx)
	m.setInitialValues()

	return m, nil
}

func (m *sglangMetricsAdapter) Start(context.Context) error { return nil }
func (m *sglangMetricsAdapter) Close() error                { return nil }

// -- Event handlers  -------------------

func (m *sglangMetricsAdapter) OnRequestReceived(metrics.RequestReceived) {
	// State marker; no exposed metric today.
}

func (m *sglangMetricsAdapter) OnRequestRejected(metrics.RequestRejected) {
	// State marker; no exposed metric today.
}

// request queued
// - update number of queued requests
func (m *sglangMetricsAdapter) OnRequestQueued(metrics.RequestQueued) {
	common.WriteToChannel(m.queueReqChan, gaugeDelta(1), m.logger)
}

// request dequeued
// - update number of queued requests
func (m *sglangMetricsAdapter) OnRequestDequeued(metrics.RequestDequeued) {
	common.WriteToChannel(m.queueReqChan, gaugeDelta(-1), m.logger)
}

// request running
// - update number of running requests
func (m *sglangMetricsAdapter) OnRequestRunning(metrics.RequestRunning) {
	common.WriteToChannel(m.runReqChan, gaugeDelta(1), m.logger)
}

func (m *sglangMetricsAdapter) OnPrefillStarted(metrics.PrefillStarted) {
	// State marker.
}

// prefill step ended
// - update TTFT histogram
func (m *sglangMetricsAdapter) OnPrefillEnded(ev metrics.PrefillEnded) {
	common.WriteToChannel(m.ttftChan,
		streamObservation{value: ev.PrefillDuration, isStreaming: ev.IsStreaming}, m.logger)
}

func (m *sglangMetricsAdapter) OnDecodeStarted(metrics.DecodeStarted) {
	// State marker.
}

// token generated
// - update inter-token latency histogram
func (m *sglangMetricsAdapter) OnTokenGenerated(ev metrics.TokenGenerated) {
	common.WriteToChannel(m.interTokenLatencyChan, observation(ev.InterTokenLatency), m.logger)
}

func (m *sglangMetricsAdapter) OnDecodeEnded(metrics.DecodeEnded) {
	// State marker; sglang's per-request TPOT metric is not exposed yet.
}

// request processing finished successfully - update all relevant metrics
func (m *sglangMetricsAdapter) OnRequestSucceeded(ev metrics.RequestSucceeded) {
	common.WriteToChannel(m.promptTokensTotalChan,
		streamObservation{value: float64(ev.PromptTokens), isStreaming: ev.IsStreaming}, m.logger)
	common.WriteToChannel(m.generationTokensTotalChan,
		streamObservation{value: float64(ev.GenerationTokens), isStreaming: ev.IsStreaming}, m.logger)
	common.WriteToChannel(m.e2eReqLatencyChan,
		streamObservation{value: ev.E2ELatency, isStreaming: ev.IsStreaming}, m.logger)

	m.finishRunning()
}

// request processing failed
//
// Only the running-request gauge is updated: sglang observes a request's
// latency and token counts when it finishes, so a request that never did
// contributes to neither.
func (m *sglangMetricsAdapter) OnRequestFailed(metrics.RequestFailed) {
	m.finishRunning()
}

func (m *sglangMetricsAdapter) OnKVCacheUsageChanged(metrics.KVCacheUsageChanged) {
	// Unreachable: ValidateConfig rejects a KV-cache configuration.
}

func (m *sglangMetricsAdapter) OnPrefixCacheQueried(metrics.PrefixCacheQueried) {
	// Unreachable: ValidateConfig rejects a KV-cache configuration.
}

func (m *sglangMetricsAdapter) OnLoRASetsChanged(metrics.LoRASetsChanged) {
	// This engine registers no LoRA flags, so no adapter can be requested.
}

// ApplyFakeMetricsUpdate is unreachable: ValidateConfig rejects a non-nil
// FakeMetrics, so no update can be enqueued for this engine.
func (m *sglangMetricsAdapter) ApplyFakeMetricsUpdate(common.FakeMetrics) {}

func (m *sglangMetricsAdapter) finishRunning() {
	common.WriteToChannel(m.runReqChan, gaugeDelta(-1), m.logger)
}

func (m *sglangMetricsAdapter) createAndStartPrometheusChannels(ctx context.Context) {
	maxNumberOfRunningRequests, maxNumberOfWaitingRequests, _, maxNumberOfTokens := metrics.ChannelCapacities(m.config)

	m.runReqChan = common.NewChannel[gaugeDelta]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.runReqChan, m.updateRunningRequests)

	m.queueReqChan = common.NewChannel[gaugeDelta]("sglang", maxNumberOfWaitingRequests, ctx.Done())
	go common.Subscribe(ctx, m.queueReqChan, m.updateQueuedRequests)

	m.ttftChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.ttftChan, m.updateTTFT)

	m.interTokenLatencyChan = common.NewChannel[observation]("sglang", maxNumberOfTokens, ctx.Done())
	go common.Subscribe(ctx, m.interTokenLatencyChan, m.updateInterTokenLatency)

	m.e2eReqLatencyChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.e2eReqLatencyChan, m.updateE2EReqLatency)

	m.promptTokensTotalChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.promptTokensTotalChan, m.updatePromptTokensTotal)

	m.generationTokensTotalChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.generationTokensTotalChan, m.updateGenerationTokensTotal)
}

// -- Per-metric updaters --------------------------------------------------

func (m *sglangMetricsAdapter) updateRunningRequests(upd gaugeDelta) {
	m.nRunningReqs += int64(upd)
	m.runningRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nRunningReqs))
}

func (m *sglangMetricsAdapter) updateQueuedRequests(upd gaugeDelta) {
	m.nQueuedReqs += int64(upd)
	m.queuedRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nQueuedReqs))
}

func (m *sglangMetricsAdapter) updateTTFT(upd streamObservation) {
	m.observeByStream(m.ttft, upd)
}

func (m *sglangMetricsAdapter) updateInterTokenLatency(upd observation) {
	m.interTokenLatency.WithLabelValues(m.config.DisplayModelName).Observe(float64(upd))
}

func (m *sglangMetricsAdapter) updateE2EReqLatency(upd streamObservation) {
	m.observeByStream(m.e2eReqLatency, upd)
}

func (m *sglangMetricsAdapter) updatePromptTokensTotal(upd streamObservation) {
	m.addByStream(m.promptTokensTotal, upd)
}

func (m *sglangMetricsAdapter) updateGenerationTokensTotal(upd streamObservation) {
	m.addByStream(m.generationTokensTotal, upd)
}

// observeByStream records upd on the series of hist matching the kind of
// request it came from.
func (m *sglangMetricsAdapter) observeByStream(hist *prometheus.HistogramVec, upd streamObservation) {
	hist.WithLabelValues(m.config.DisplayModelName, strconv.FormatBool(upd.isStreaming)).Observe(upd.value)
}

// addByStream adds upd to the series of counter matching the kind of request it
// came from.
func (m *sglangMetricsAdapter) addByStream(counter *prometheus.CounterVec, upd streamObservation) {
	counter.WithLabelValues(m.config.DisplayModelName, strconv.FormatBool(upd.isStreaming)).Add(upd.value)
}

// -- Prometheus wiring ------------------------------------------------------

// buildMetrics constructs and registers all Prometheus collectors. It is called once during adapter construction.
func (m *sglangMetricsAdapter) buildMetrics() error {
	if err := m.createAndRegisterRunningRequestsGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterQueuedRequestsGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterTTFTHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterInterTokenLatencyHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterE2EReqLatencyHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterPromptTokensTotalCounter(); err != nil {
		return err
	}
	return m.createAndRegisterGenerationTokensTotalCounter()
}

// register registers c with the bus's Prometheus registry, logging errMsg
// on failure.
func (m *sglangMetricsAdapter) register(c prometheus.Collector, errMsg string) error {
	if err := m.registry.Register(c); err != nil {
		m.logger.Error(err, errMsg)
		return err
	}
	return nil
}

func (m *sglangMetricsAdapter) createAndRegisterRunningRequestsGauge() error {
	m.runningRequests = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangRunningReqsMetricName,
		Help: "The number of running requests.",
	}, modelLabel)
	return m.register(m.runningRequests, "prometheus number of running requests gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterQueuedRequestsGauge() error {
	m.queuedRequests = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangQueueReqsMetricName,
		Help: "The number of requests in the waiting queue.",
	}, modelLabel)
	return m.register(m.queuedRequests, "prometheus number of requests in queue gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterTTFTHistogram() error {
	m.ttft = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    SGLangTTFTMetricName,
		Help:    "Histogram of time to first token in seconds.",
		Buckets: ttftBucketsBoundaries,
	}, streamModelLabel)
	return m.register(m.ttft, "prometheus time to first token histogram register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterInterTokenLatencyHistogram() error {
	m.interTokenLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    SGLangInterTokenLatencyMetricName,
		Help:    "Histogram of inter-token latency in seconds.",
		Buckets: interTokenLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.interTokenLatency, "prometheus inter-token latency histogram register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterE2EReqLatencyHistogram() error {
	m.e2eReqLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    SGLangE2EReqLatencyMetricName,
		Help:    "Histogram of End-to-end request latency in seconds",
		Buckets: e2eReqLatencyBucketsBoundaries,
	}, streamModelLabel)
	return m.register(m.e2eReqLatency, "prometheus e2e request latency histogram register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterPromptTokensTotalCounter() error {
	m.promptTokensTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: SGLangPromptTokensTotalMetricName,
		Help: "Number of prefill tokens processed.",
	}, streamModelLabel)
	return m.register(m.promptTokensTotal, "prometheus prompt_tokens_total counter register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterGenerationTokensTotalCounter() error {
	m.generationTokensTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: SGLangGenerationTokensTotalMetricName,
		Help: "Number of generation tokens processed.",
	}, streamModelLabel)
	return m.register(m.generationTokensTotal, "prometheus generation_tokens_total counter register failed")
}

// setInitialValues zeroes the request gauges so the first scrape has a
// consistent baseline. The histograms and counters stay unstamped until a
// request is observed, since their series are per request kind.
func (m *sglangMetricsAdapter) setInitialValues() {
	m.runningRequests.WithLabelValues(m.config.DisplayModelName).Set(0)
	m.queuedRequests.WithLabelValues(m.config.DisplayModelName).Set(0)
}
