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
	SGLangRunningReqsMetricName              = "sglang:num_running_reqs"
	SGLangQueueReqsMetricName                = "sglang:num_queue_reqs"
	SGLangContextLenMetricName               = "sglang:context_len"
	SGLangQueueTimeMetricName                = "sglang:queue_time_seconds"
	SGLangTTFTMetricName                     = "sglang:time_to_first_token_seconds"
	SGLangInterTokenLatencyMetricName        = "sglang:inter_token_latency_seconds"
	SGLangReqTPOTMetricName                  = "sglang:request_time_per_output_token_seconds"
	SGLangE2EReqLatencyMetricName            = "sglang:e2e_request_latency_seconds"
	SGLangPromptTokensTotalMetricName        = "sglang:prompt_tokens_total"
	SGLangGenerationTokensTotalMetricName    = "sglang:generation_tokens_total"
	SGLangRequestsTotalMetricName            = "sglang:num_requests_total"
	SGLangPromptTokensHistMetricName         = "sglang:prompt_tokens_histogram"
	SGLangUncachedPromptTokensHistMetricName = "sglang:uncached_prompt_tokens_histogram"
	SGLangGenerationTokensHistMetricName     = "sglang:generation_tokens_histogram"
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

// queueTimeBucketsBoundaries are the boundaries of sglang:queue_time_seconds,
// which takes no override flag.
var queueTimeBucketsBoundaries = []float64{0.000, 0.001, 0.005, 0.010, 0.050, 0.100, 0.200, 0.500,
	1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800,
	900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000}

// tokenBucketsBoundaries are the boundaries shared by the token-length
// histograms, sglang's defaults for them. They do not scale with
// --max-model-len the way the vLLM engine's token buckets do.
var tokenBucketsBoundaries = []float64{100, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000,
	7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 35000, 40000,
	60000, 80000, 100000, 200000, 300000, 400000, 600000, 800000, 1000000, 1100000}

// tpotFinishReasons are the finish reasons sglang records a per-request TPOT
// for. A request that ended any other way had a truncated decode phase, which
// would report a time per output token the engine never spent. sglang's engine
// finishes a tool-calling request with stop; tool_calls is set only above it.
var tpotFinishReasons = map[string]bool{
	common.StopFinishReason:   true,
	common.LengthFinishReason: true,
	common.ToolsFinishReason:  true,
}

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
	logger logr.Logger
	config common.Configuration

	registry *prometheus.Registry

	// gauges
	runningRequests *prometheus.GaugeVec
	queuedRequests  *prometheus.GaugeVec
	contextLen      *prometheus.GaugeVec

	// histograms
	queueTime                *prometheus.HistogramVec
	ttft                     *prometheus.HistogramVec
	interTokenLatency        *prometheus.HistogramVec
	reqTpot                  *prometheus.HistogramVec
	e2eReqLatency            *prometheus.HistogramVec
	promptTokensHist         *prometheus.HistogramVec
	uncachedPromptTokensHist *prometheus.HistogramVec
	generationTokensHist     *prometheus.HistogramVec

	// counters
	promptTokensTotal     *prometheus.CounterVec
	generationTokensTotal *prometheus.CounterVec
	requestsTotal         *prometheus.CounterVec

	// Channels: one per Prometheus metric family. Handlers push here; the
	// updater goroutines drain each and perform the actual Prometheus mutation.
	runReqChan   common.Channel[metrics.GaugeUpdate]
	queueReqChan common.Channel[metrics.GaugeUpdate]

	queueTimeChan                common.Channel[metrics.HistogramUpdate]
	ttftChan                     common.Channel[streamObservation]
	interTokenLatencyChan        common.Channel[metrics.HistogramUpdate]
	reqTpotChan                  common.Channel[streamObservation]
	e2eReqLatencyChan            common.Channel[streamObservation]
	promptTokensHistChan         common.Channel[metrics.HistogramUpdate]
	uncachedPromptTokensHistChan common.Channel[metrics.HistogramUpdate]
	generationTokensHistChan     common.Channel[metrics.HistogramUpdate]

	promptTokensTotalChan     common.Channel[streamObservation]
	generationTokensTotalChan common.Channel[streamObservation]
	requestsTotalChan         common.Channel[streamObservation]

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
	m.createPrometheusChannels(ctx)
	m.setInitialValues()

	return m, nil
}

func (m *sglangMetricsAdapter) Start(ctx context.Context) error {
	m.startPrometheusChannels(ctx)
	return nil
}

func (m *sglangMetricsAdapter) Close() error { return nil }

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
	common.WriteToChannel(m.queueReqChan, metrics.GaugeAdd(1), m.logger)
}

// request dequeued
// - update number of queued requests
// - update queue time histogram
func (m *sglangMetricsAdapter) OnRequestDequeued(ev metrics.RequestDequeued) {
	common.WriteToChannel(m.queueReqChan, metrics.GaugeAdd(-1), m.logger)
	common.WriteToChannel(m.queueTimeChan, metrics.Observation(ev.QueueTime), m.logger)
}

// request running
// - update number of running requests
func (m *sglangMetricsAdapter) OnRequestRunning(metrics.RequestRunning) {
	common.WriteToChannel(m.runReqChan, metrics.GaugeAdd(1), m.logger)
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
	common.WriteToChannel(m.interTokenLatencyChan, metrics.Observation(ev.InterTokenLatency), m.logger)
}

// decode ended
// - update the per-request TPOT histogram
//
// A request that generated a single token has no time per output token to
// report, and one that ended for any other reason stopped mid-decode, so
// neither contributes, as neither does upstream.
func (m *sglangMetricsAdapter) OnDecodeEnded(ev metrics.DecodeEnded) {
	if ev.GenerationTokens <= 0 || !tpotFinishReasons[ev.FinishReason] {
		return
	}
	common.WriteToChannel(m.reqTpotChan, streamObservation{
		value:       ev.DecodeDuration / float64(ev.GenerationTokens),
		isStreaming: ev.IsStreaming,
	}, m.logger)
}

// request processing finished successfully - update all relevant metrics
func (m *sglangMetricsAdapter) OnRequestSucceeded(ev metrics.RequestSucceeded) {
	common.WriteToChannel(m.promptTokensTotalChan,
		streamObservation{value: float64(ev.PromptTokens), isStreaming: ev.IsStreaming}, m.logger)
	common.WriteToChannel(m.generationTokensTotalChan,
		streamObservation{value: float64(ev.GenerationTokens), isStreaming: ev.IsStreaming}, m.logger)
	common.WriteToChannel(m.e2eReqLatencyChan,
		streamObservation{value: ev.E2ELatency, isStreaming: ev.IsStreaming}, m.logger)
	common.WriteToChannel(m.requestsTotalChan,
		streamObservation{value: 1, isStreaming: ev.IsStreaming}, m.logger)

	common.WriteToChannel(m.promptTokensHistChan, metrics.Observation(float64(ev.PromptTokens)), m.logger)
	common.WriteToChannel(m.uncachedPromptTokensHistChan,
		metrics.Observation(float64(ev.PromptTokens-ev.CachedPromptTokens)), m.logger)
	common.WriteToChannel(m.generationTokensHistChan, metrics.Observation(float64(ev.GenerationTokens)), m.logger)

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
	// Not reached: ValidateConfig rejects an enabled KV cache.
}

func (m *sglangMetricsAdapter) OnPrefixCacheQueried(metrics.PrefixCacheQueried) {
	// Not reached: ValidateConfig rejects an enabled KV cache.
}

func (m *sglangMetricsAdapter) OnLoRASetsChanged(metrics.LoRASetsChanged) {
	// This engine registers no LoRA flags, so no adapter can be requested.
}

// ApplyFakeMetricsUpdate is unreachable: ValidateConfig rejects a non-nil
// FakeMetrics, so no update can be enqueued for this engine.
func (m *sglangMetricsAdapter) ApplyFakeMetricsUpdate(common.FakeMetrics) {}

func (m *sglangMetricsAdapter) finishRunning() {
	common.WriteToChannel(m.runReqChan, metrics.GaugeAdd(-1), m.logger)
}

func (m *sglangMetricsAdapter) startPrometheusChannels(ctx context.Context) {
	go common.Subscribe(ctx, m.runReqChan, m.updateRunningRequests)
	go common.Subscribe(ctx, m.queueReqChan, m.updateQueuedRequests)
	go common.Subscribe(ctx, m.queueTimeChan, m.updateQueueTime)
	go common.Subscribe(ctx, m.ttftChan, m.updateTTFT)
	go common.Subscribe(ctx, m.interTokenLatencyChan, m.updateInterTokenLatency)
	go common.Subscribe(ctx, m.reqTpotChan, m.updateReqTpot)
	go common.Subscribe(ctx, m.e2eReqLatencyChan, m.updateE2EReqLatency)
	go common.Subscribe(ctx, m.promptTokensHistChan, m.updatePromptTokensHist)
	go common.Subscribe(ctx, m.generationTokensHistChan, m.updateGenerationTokensHist)
	go common.Subscribe(ctx, m.promptTokensTotalChan, m.updatePromptTokensTotal)
	go common.Subscribe(ctx, m.generationTokensTotalChan, m.updateGenerationTokensTotal)
	go common.Subscribe(ctx, m.uncachedPromptTokensHistChan, m.updateUncachedPromptTokensHist)
	go common.Subscribe(ctx, m.requestsTotalChan, m.updateRequestsTotal)
}

func (m *sglangMetricsAdapter) createPrometheusChannels(ctx context.Context) {
	maxNumberOfRunningRequests, maxNumberOfWaitingRequests, _, maxNumberOfTokens := metrics.ChannelCapacities(m.config)

	m.runReqChan = common.NewChannel[metrics.GaugeUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	m.queueReqChan = common.NewChannel[metrics.GaugeUpdate]("sglang", maxNumberOfWaitingRequests, ctx.Done())
	m.queueTimeChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfWaitingRequests, ctx.Done())
	m.ttftChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	m.interTokenLatencyChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfTokens, ctx.Done())
	m.reqTpotChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	m.e2eReqLatencyChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	m.promptTokensHistChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	m.uncachedPromptTokensHistChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	m.generationTokensHistChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	m.promptTokensTotalChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	m.generationTokensTotalChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	m.requestsTotalChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
}

// -- Per-metric updaters --------------------------------------------------

func (m *sglangMetricsAdapter) updateRunningRequests(upd metrics.GaugeUpdate) {
	if upd.Add == nil {
		return
	}
	m.nRunningReqs += int64(*upd.Add)
	m.runningRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nRunningReqs))
}

func (m *sglangMetricsAdapter) updateQueuedRequests(upd metrics.GaugeUpdate) {
	if upd.Add == nil {
		return
	}
	m.nQueuedReqs += int64(*upd.Add)
	m.queuedRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nQueuedReqs))
}

func (m *sglangMetricsAdapter) updateQueueTime(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.queueTime, upd)
}

func (m *sglangMetricsAdapter) updateTTFT(obs streamObservation) {
	m.updateStreamHistogram(m.ttft, obs)
}

func (m *sglangMetricsAdapter) updateInterTokenLatency(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.interTokenLatency, upd)
}

func (m *sglangMetricsAdapter) updateReqTpot(obs streamObservation) {
	m.updateStreamHistogram(m.reqTpot, obs)
}

func (m *sglangMetricsAdapter) updateE2EReqLatency(obs streamObservation) {
	m.updateStreamHistogram(m.e2eReqLatency, obs)
}

func (m *sglangMetricsAdapter) updatePromptTokensHist(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.promptTokensHist, upd)
}

func (m *sglangMetricsAdapter) updateGenerationTokensHist(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.generationTokensHist, upd)
}

func (m *sglangMetricsAdapter) updatePromptTokensTotal(obs streamObservation) {
	m.updateStreamCounter(m.promptTokensTotal, obs)
}

func (m *sglangMetricsAdapter) updateGenerationTokensTotal(obs streamObservation) {
	m.updateStreamCounter(m.generationTokensTotal, obs)
}

func (m *sglangMetricsAdapter) updateUncachedPromptTokensHist(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.uncachedPromptTokensHist, upd)
}

func (m *sglangMetricsAdapter) updateRequestsTotal(obs streamObservation) {
	m.updateStreamCounter(m.requestsTotal, obs)
}

// updateHistogram records an observation on the single series of a histogram
// sglang does not split by request kind.
func (m *sglangMetricsAdapter) updateHistogram(hist *prometheus.HistogramVec, upd metrics.HistogramUpdate) {
	if upd.Observe == nil {
		return
	}
	hist.WithLabelValues(m.config.DisplayModelName).Observe(*upd.Observe)
}

// updateStreamHistogram records an observation on the series matching the kind
// of request it came from.
func (m *sglangMetricsAdapter) updateStreamHistogram(hist *prometheus.HistogramVec, obs streamObservation) {
	hist.WithLabelValues(m.config.DisplayModelName, strconv.FormatBool(obs.isStreaming)).Observe(obs.value)
}

// updateStreamCounter adds an event's value to the series matching the kind of
// request it came from.
func (m *sglangMetricsAdapter) updateStreamCounter(counter *prometheus.CounterVec, obs streamObservation) {
	counter.WithLabelValues(m.config.DisplayModelName, strconv.FormatBool(obs.isStreaming)).Add(obs.value)
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
	if err := m.createAndRegisterContextLenGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterQueueTimeHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterTTFTHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterInterTokenLatencyHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqTpotHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterE2EReqLatencyHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterPromptTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterUncachedPromptTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterGenerationTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterPromptTokensTotalCounter(); err != nil {
		return err
	}
	if err := m.createAndRegisterGenerationTokensTotalCounter(); err != nil {
		return err
	}
	return m.createAndRegisterRequestsTotalCounter()
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

func (m *sglangMetricsAdapter) createAndRegisterContextLenGauge() error {
	m.contextLen = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangContextLenMetricName,
		Help: "Maximum context length.",
	}, modelLabel)
	return m.register(m.contextLen, "prometheus context length gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterQueueTimeHistogram() error {
	m.queueTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    SGLangQueueTimeMetricName,
		Help:    "Histogram of queueing time in seconds.",
		Buckets: queueTimeBucketsBoundaries,
	}, modelLabel)
	return m.register(m.queueTime, "prometheus queue time histogram register failed")
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

func (m *sglangMetricsAdapter) createAndRegisterReqTpotHistogram() error {
	m.reqTpot = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    SGLangReqTPOTMetricName,
		Help:    "Per-request TPOT in seconds: (finished_time - first_token_time) / (completion_tokens - 1).",
		Buckets: interTokenLatencyBucketsBoundaries,
	}, streamModelLabel)
	return m.register(m.reqTpot, "prometheus per-request time per output token histogram register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterE2EReqLatencyHistogram() error {
	m.e2eReqLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    SGLangE2EReqLatencyMetricName,
		Help:    "Histogram of End-to-end request latency in seconds",
		Buckets: e2eReqLatencyBucketsBoundaries,
	}, streamModelLabel)
	return m.register(m.e2eReqLatency, "prometheus e2e request latency histogram register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterPromptTokensHistogram() error {
	m.promptTokensHist = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    SGLangPromptTokensHistMetricName,
		Help:    "Histogram of prompt token length.",
		Buckets: tokenBucketsBoundaries,
	}, modelLabel)
	return m.register(m.promptTokensHist, "prometheus prompt_tokens_histogram register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterUncachedPromptTokensHistogram() error {
	m.uncachedPromptTokensHist = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    SGLangUncachedPromptTokensHistMetricName,
		Help:    "Histogram of uncached (compute) prompt token length.",
		Buckets: tokenBucketsBoundaries,
	}, modelLabel)
	return m.register(m.uncachedPromptTokensHist, "prometheus uncached_prompt_tokens_histogram register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterGenerationTokensHistogram() error {
	m.generationTokensHist = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    SGLangGenerationTokensHistMetricName,
		Help:    "Histogram of generation token length.",
		Buckets: tokenBucketsBoundaries,
	}, modelLabel)
	return m.register(m.generationTokensHist, "prometheus generation_tokens_histogram register failed")
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

func (m *sglangMetricsAdapter) createAndRegisterRequestsTotalCounter() error {
	m.requestsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: SGLangRequestsTotalMetricName,
		Help: "Number of requests processed.",
	}, streamModelLabel)
	return m.register(m.requestsTotal, "prometheus num_requests_total counter register failed")
}

// setInitialValues zeroes the request gauges and stamps the context length, so
// the first scrape has a consistent baseline. The histograms and the
// per-request counters stay unstamped until a request is observed, since their
// series are per request kind.
func (m *sglangMetricsAdapter) setInitialValues() {
	model := m.config.DisplayModelName

	m.runningRequests.WithLabelValues(model).Set(0)
	m.queuedRequests.WithLabelValues(model).Set(0)
	m.contextLen.WithLabelValues(model).Set(float64(m.config.MaxModelLen))
}
