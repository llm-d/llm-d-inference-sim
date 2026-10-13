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
	"errors"
	"fmt"
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
	SGLangTokenUsageMetricName               = "sglang:token_usage"
	SGLangUsedTokensMetricName               = "sglang:num_used_tokens"
	SGLangKVUsedTokensMetricName             = "sglang:kv_used_tokens"
	SGLangKVAvailableTokensMetricName        = "sglang:kv_available_tokens"
	SGLangMaxTotalNumTokensMetricName        = "sglang:max_total_num_tokens"
	SGLangPageSizeMetricName                 = "sglang:page_size"
	SGLangNumPagesMetricName                 = "sglang:num_pages"
	SGLangCacheHitRateMetricName             = "sglang:cache_hit_rate"
	SGLangCachedTokensTotalMetricName        = "sglang:cached_tokens_total"
)

// promLabelCacheSource names the tier a cached prompt token came from.
const promLabelCacheSource = "cache_source"

// cacheSourceTotal is the cache_source value sglang falls back to when it has
// no per-tier breakdown, which is the only shape the simulator can report: its
// block cache has a single tier.
const cacheSourceTotal = "total"

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

// tokenBucketsBoundaries are the boundaries shared by the three token-length
// histograms, sglang's defaults for them. They do not scale with
// --max-model-len the way the vLLM engine's token buckets do.
var tokenBucketsBoundaries = []float64{100, 300, 500, 700, 1000, 1500, 2000, 80000, 100000, 200000,
	300000, 400000, 600000, 800000, 1000000, 1100000}

// tpotFinishReasons are the finish reasons sglang records a per-request TPOT
// for. A request that ended any other way had a truncated decode phase, which
// would report a time per output token the engine never spent.
var tpotFinishReasons = map[string]bool{
	common.StopFinishReason:   true,
	common.LengthFinishReason: true,
}

// streamObservation is a payload carrying the observed value together with the
// kind of request it came from, for a metric labeled by is_streaming.
type streamObservation struct {
	value       float64
	isStreaming bool
}

// streamHistogramUpdate is a histogram-channel payload for a histogram labeled
// by is_streaming: one observation from a real request, or a reset to the bucket
// state of both series. Exactly one is non-nil.
type streamHistogramUpdate struct {
	observe *streamObservation
	reset   *streamHistogramReset
}

// streamHistogramReset is the absolute target bucket state of a stream-split
// histogram. A nil series leaves that series as it was, since a partial
// fake-metrics update may name one of the two without the other.
type streamHistogramReset struct {
	buckets      []float64
	streaming    []int
	nonStreaming []int
}

// merge folds upd's named series into s, leaving a series upd does not name at
// its current value. The boundaries are the metric's own, so they always arrive
// unchanged.
func (s *streamHistogramReset) merge(upd streamHistogramReset) {
	s.buckets = upd.buckets
	if upd.streaming != nil {
		s.streaming = upd.streaming
	}
	if upd.nonStreaming != nil {
		s.nonStreaming = upd.nonStreaming
	}
}

// streamCounterUpdate is a counter-channel payload for a counter labeled by
// is_streaming: an event's addition to one series, or a reset to the absolute
// values of both. Exactly one is non-nil.
type streamCounterUpdate struct {
	add   *streamObservation
	reset *streamCounterReset
}

// streamCounterReset is the absolute target state of a stream-split counter. A
// nil series leaves that series as it was, since a partial fake-metrics update
// may name one of the two without the other.
type streamCounterReset struct {
	streaming    *float64
	nonStreaming *float64
}

// merge folds upd's named series into s, leaving a series upd does not name at
// its current value.
func (s *streamCounterReset) merge(upd streamCounterReset) {
	if upd.streaming != nil {
		s.streaming = upd.streaming
	}
	if upd.nonStreaming != nil {
		s.nonStreaming = upd.nonStreaming
	}
}

// prefixCacheUpdate is the prefix-cache channel payload: the token counts of one
// cache lookup, or a reset to absolute cumulative tallies.
type prefixCacheUpdate struct {
	query *prefixCacheQuery
	reset *prefixCacheReset
}

// prefixCacheReset is the absolute target state of the prefix cache's tallies. A
// nil tally is left as it was, for the same reason streamCounterReset's series
// are: a partial update may name one of the pair without the other.
type prefixCacheReset struct {
	cached  *int64
	queried *int64
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
	// fake is config.FakeMetrics narrowed to this engine's concrete type, nil
	// when fake metrics are off or another engine owns the configuration.
	fake *SGLangFakeMetrics

	registry *prometheus.Registry

	// generators holds the fake-metric generators this adapter has registered
	// and the ticker that re-evaluates them.
	generators *metrics.FakeGenerators

	// gauges
	runningRequests   *prometheus.GaugeVec
	queuedRequests    *prometheus.GaugeVec
	contextLen        *prometheus.GaugeVec
	tokenUsage        *prometheus.GaugeVec
	usedTokens        *prometheus.GaugeVec
	kvUsedTokens      *prometheus.GaugeVec
	kvAvailableTokens *prometheus.GaugeVec
	maxTotalNumTokens *prometheus.GaugeVec
	pageSize          *prometheus.GaugeVec
	numPages          *prometheus.GaugeVec
	cacheHitRate      *prometheus.GaugeVec

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
	cachedTokensTotal     *prometheus.CounterVec

	// Channels: one per Prometheus metric family. Handlers push here; the
	// updater goroutines drain each and perform the actual Prometheus mutation.
	runReqChan   common.Channel[metrics.GaugeUpdate]
	queueReqChan common.Channel[metrics.GaugeUpdate]

	queueTimeChan                common.Channel[metrics.HistogramUpdate]
	ttftChan                     common.Channel[streamHistogramUpdate]
	interTokenLatencyChan        common.Channel[metrics.HistogramUpdate]
	reqTpotChan                  common.Channel[streamHistogramUpdate]
	e2eReqLatencyChan            common.Channel[streamHistogramUpdate]
	promptTokensHistChan         common.Channel[metrics.HistogramUpdate]
	uncachedPromptTokensHistChan common.Channel[metrics.HistogramUpdate]
	generationTokensHistChan     common.Channel[metrics.HistogramUpdate]

	promptTokensTotalChan     common.Channel[streamCounterUpdate]
	generationTokensTotalChan common.Channel[streamCounterUpdate]
	requestsTotalChan         common.Channel[streamCounterUpdate]

	kvUsageChan     common.Channel[metrics.GaugeUpdate]
	prefixCacheChan common.Channel[prefixCacheUpdate]

	// nQueuedReqs / nRunningReqs are the adapter-local counters mirrored onto
	// the num_queue_reqs / num_running_reqs gauges.
	nQueuedReqs  int64
	nRunningReqs int64

	// Cumulative prefix-cache tallies behind cache_hit_rate, mutated only by
	// the prefix-cache updater goroutine.
	nCachedTokens  int64
	nQueriedTokens int64

	// Series a fake-metrics configuration last set on each is_streaming-labeled
	// counter, each mutated only by that counter's own updater goroutine.
	// Applying a reset recreates the collector, which discards every series it
	// does not then stamp, so an update naming one series restamps the other
	// from here.
	fakePromptTokens     streamCounterReset
	fakeGenerationTokens streamCounterReset
	fakeRequestsTotal    streamCounterReset

	// The same, for each is_streaming-labeled histogram.
	fakeTTFT          streamHistogramReset
	fakeReqTpot       streamHistogramReset
	fakeE2EReqLatency streamHistogramReset

	// kvCapacityTokens is the KV cache's capacity in tokens, the denominator of
	// the token-usage gauges.
	kvCapacityTokens int
}

// prefixCacheQuery is the prefix-cache channel payload: the token counts of one
// cache lookup.
type prefixCacheQuery struct {
	cached  int
	queried int
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
		logger:           logger,
		registry:         registry,
		config:           config,
		generators:       metrics.NewFakeGenerators(ctx, config.FakeMetricsRefreshInterval),
		kvCapacityTokens: config.KVCache.KVCacheSize * config.KVCache.TokenBlockSize,
	}
	if fake, ok := config.FakeMetrics.(*SGLangFakeMetrics); ok {
		m.fake = fake
	}

	if err := m.buildMetrics(); err != nil {
		return nil, err
	}
	m.createAndStartPrometheusChannels(ctx)
	m.setInitialValues()

	return m, nil
}

// Start applies the initial fake-metrics configuration and starts the generator
// ticker. The bus has already subscribed the On<Event> handlers to its channels
// by the time this runs.
func (m *sglangMetricsAdapter) Start(context.Context) error {
	if m.fake != nil {
		m.applyFakeMetrics(m.fake)
	}
	m.generators.Start()

	return nil
}

func (m *sglangMetricsAdapter) Close() error {
	m.generators.Stop()

	return nil
}

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
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.queueReqChan, metrics.GaugeAdd(1), m.logger)
}

// request dequeued
// - update number of queued requests
// - update queue time histogram
func (m *sglangMetricsAdapter) OnRequestDequeued(ev metrics.RequestDequeued) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.queueReqChan, metrics.GaugeAdd(-1), m.logger)
	common.WriteToChannel(m.queueTimeChan, metrics.Observation(ev.QueueTime), m.logger)
}

// request running
// - update number of running requests
func (m *sglangMetricsAdapter) OnRequestRunning(metrics.RequestRunning) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.runReqChan, metrics.GaugeAdd(1), m.logger)
}

func (m *sglangMetricsAdapter) OnPrefillStarted(metrics.PrefillStarted) {
	// State marker.
}

// prefill step ended
// - update TTFT histogram
func (m *sglangMetricsAdapter) OnPrefillEnded(ev metrics.PrefillEnded) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.ttftChan,
		histObserve(ev.PrefillDuration, ev.IsStreaming), m.logger)
}

func (m *sglangMetricsAdapter) OnDecodeStarted(metrics.DecodeStarted) {
	// State marker.
}

// token generated
// - update inter-token latency histogram
func (m *sglangMetricsAdapter) OnTokenGenerated(ev metrics.TokenGenerated) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.interTokenLatencyChan, metrics.Observation(ev.InterTokenLatency), m.logger)
}

// decode ended
// - update the per-request TPOT histogram
//
// A request that generated a single token has no time per output token to
// report, and one that ended for any other reason stopped mid-decode, so
// neither contributes, as neither does upstream.
func (m *sglangMetricsAdapter) OnDecodeEnded(ev metrics.DecodeEnded) {
	if m.fake != nil {
		return
	}
	if ev.GenerationTokens <= 0 || !tpotFinishReasons[ev.FinishReason] {
		return
	}
	common.WriteToChannel(m.reqTpotChan,
		histObserve(ev.DecodeDuration/float64(ev.GenerationTokens), ev.IsStreaming), m.logger)
}

// request processing finished successfully - update all relevant metrics
func (m *sglangMetricsAdapter) OnRequestSucceeded(ev metrics.RequestSucceeded) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.promptTokensTotalChan,
		counterAdd(float64(ev.PromptTokens), ev.IsStreaming), m.logger)
	common.WriteToChannel(m.generationTokensTotalChan,
		counterAdd(float64(ev.GenerationTokens), ev.IsStreaming), m.logger)
	common.WriteToChannel(m.e2eReqLatencyChan,
		histObserve(ev.E2ELatency, ev.IsStreaming), m.logger)
	common.WriteToChannel(m.requestsTotalChan, counterAdd(1, ev.IsStreaming), m.logger)

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
	if m.fake != nil {
		return
	}
	// Do not send update to e2eReqLatencyChan because sglang does not report a latency for failed requests
	m.finishRunning()
}

// change in kv cache utilization
// - update the token-usage gauges
//
// Not reached while ValidateConfig rejects an enabled KV cache, which leaves
// the gauges reporting the empty cache they are stamped with.
func (m *sglangMetricsAdapter) OnKVCacheUsageChanged(ev metrics.KVCacheUsageChanged) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.kvUsageChan, metrics.GaugeAdd(ev.KVCacheUsagePerc), m.logger)
}

// prefix cache lookup
// - update the cached-token counter and the cumulative hit rate
//
// Not reached while ValidateConfig rejects an enabled KV cache.
func (m *sglangMetricsAdapter) OnPrefixCacheQueried(ev metrics.PrefixCacheQueried) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.prefixCacheChan, prefixCacheUpdate{
		query: &prefixCacheQuery{cached: ev.CachedPromptTokens, queried: ev.QueriedTokens},
	}, m.logger)
}

func (m *sglangMetricsAdapter) OnLoRASetsChanged(metrics.LoRASetsChanged) {
	// This engine registers no LoRA flags, so no adapter can be requested.
}

func (m *sglangMetricsAdapter) finishRunning() {
	common.WriteToChannel(m.runReqChan, metrics.GaugeAdd(-1), m.logger)
}

func (m *sglangMetricsAdapter) createAndStartPrometheusChannels(ctx context.Context) {
	maxNumberOfRunningRequests, maxNumberOfWaitingRequests, _, maxNumberOfTokens := metrics.ChannelCapacities(m.config)

	m.runReqChan = common.NewChannel[metrics.GaugeUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.runReqChan, m.updateRunningRequests)

	m.queueReqChan = common.NewChannel[metrics.GaugeUpdate]("sglang", maxNumberOfWaitingRequests, ctx.Done())
	go common.Subscribe(ctx, m.queueReqChan, m.updateQueuedRequests)

	m.queueTimeChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfWaitingRequests, ctx.Done())
	go common.Subscribe(ctx, m.queueTimeChan, m.updateQueueTime)

	m.ttftChan = common.NewChannel[streamHistogramUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.ttftChan, m.updateTTFT)

	m.interTokenLatencyChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfTokens, ctx.Done())
	go common.Subscribe(ctx, m.interTokenLatencyChan, m.updateInterTokenLatency)

	m.reqTpotChan = common.NewChannel[streamHistogramUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqTpotChan, m.updateReqTpot)

	m.e2eReqLatencyChan = common.NewChannel[streamHistogramUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.e2eReqLatencyChan, m.updateE2EReqLatency)

	m.promptTokensHistChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.promptTokensHistChan, m.updatePromptTokensHist)

	m.uncachedPromptTokensHistChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.uncachedPromptTokensHistChan, m.updateUncachedPromptTokensHist)

	m.generationTokensHistChan = common.NewChannel[metrics.HistogramUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.generationTokensHistChan, m.updateGenerationTokensHist)

	m.promptTokensTotalChan = common.NewChannel[streamCounterUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.promptTokensTotalChan, m.updatePromptTokensTotal)

	m.generationTokensTotalChan = common.NewChannel[streamCounterUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.generationTokensTotalChan, m.updateGenerationTokensTotal)

	m.requestsTotalChan = common.NewChannel[streamCounterUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestsTotalChan, m.updateRequestsTotal)

	m.kvUsageChan = common.NewChannel[metrics.GaugeUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.kvUsageChan, m.updateKVCacheUsage)

	m.prefixCacheChan = common.NewChannel[prefixCacheUpdate]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.prefixCacheChan, m.updatePrefixCache)
}

// -- Per-metric updaters --------------------------------------------------

func (m *sglangMetricsAdapter) updateRunningRequests(upd metrics.GaugeUpdate) {
	switch {
	case upd.Add != nil && m.fake == nil:
		m.nRunningReqs += int64(*upd.Add)
	case upd.Reset != nil && m.fake != nil:
		m.nRunningReqs = int64(*upd.Reset)
	default:
		return
	}
	m.runningRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nRunningReqs))
}

func (m *sglangMetricsAdapter) updateQueuedRequests(upd metrics.GaugeUpdate) {
	switch {
	case upd.Add != nil && m.fake == nil:
		m.nQueuedReqs += int64(*upd.Add)
	case upd.Reset != nil && m.fake != nil:
		m.nQueuedReqs = int64(*upd.Reset)
	default:
		return
	}
	m.queuedRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nQueuedReqs))
}

func (m *sglangMetricsAdapter) updateQueueTime(upd metrics.HistogramUpdate) {
	m.updateHistogram(&m.queueTime, m.createAndRegisterQueueTimeHistogram, upd)
}

func (m *sglangMetricsAdapter) updateTTFT(upd streamHistogramUpdate) {
	m.updateStreamHistogram(&m.ttft, &m.fakeTTFT, m.createAndRegisterTTFTHistogram, upd)
}

func (m *sglangMetricsAdapter) updateInterTokenLatency(upd metrics.HistogramUpdate) {
	m.updateHistogram(&m.interTokenLatency, m.createAndRegisterInterTokenLatencyHistogram, upd)
}

func (m *sglangMetricsAdapter) updateReqTpot(upd streamHistogramUpdate) {
	m.updateStreamHistogram(&m.reqTpot, &m.fakeReqTpot, m.createAndRegisterReqTpotHistogram, upd)
}

func (m *sglangMetricsAdapter) updateE2EReqLatency(upd streamHistogramUpdate) {
	m.updateStreamHistogram(&m.e2eReqLatency, &m.fakeE2EReqLatency,
		m.createAndRegisterE2EReqLatencyHistogram, upd)
}

func (m *sglangMetricsAdapter) updatePromptTokensHist(upd metrics.HistogramUpdate) {
	m.updateHistogram(&m.promptTokensHist, m.createAndRegisterPromptTokensHistogram, upd)
}

func (m *sglangMetricsAdapter) updateUncachedPromptTokensHist(upd metrics.HistogramUpdate) {
	m.updateHistogram(&m.uncachedPromptTokensHist, m.createAndRegisterUncachedPromptTokensHistogram, upd)
}

func (m *sglangMetricsAdapter) updateGenerationTokensHist(upd metrics.HistogramUpdate) {
	m.updateHistogram(&m.generationTokensHist, m.createAndRegisterGenerationTokensHistogram, upd)
}

func (m *sglangMetricsAdapter) updatePromptTokensTotal(upd streamCounterUpdate) {
	m.updateStreamCounter(&m.promptTokensTotal, &m.fakePromptTokens,
		m.createAndRegisterPromptTokensTotalCounter, upd)
}

func (m *sglangMetricsAdapter) updateGenerationTokensTotal(upd streamCounterUpdate) {
	m.updateStreamCounter(&m.generationTokensTotal, &m.fakeGenerationTokens,
		m.createAndRegisterGenerationTokensTotalCounter, upd)
}

func (m *sglangMetricsAdapter) updateRequestsTotal(upd streamCounterUpdate) {
	m.updateStreamCounter(&m.requestsTotal, &m.fakeRequestsTotal,
		m.createAndRegisterRequestsTotalCounter, upd)
}

func (m *sglangMetricsAdapter) updateKVCacheUsage(upd metrics.GaugeUpdate) {
	switch {
	case upd.Add != nil && m.fake == nil:
		m.reportKVCacheUsage(*upd.Add)
	case upd.Reset != nil && m.fake != nil:
		m.reportKVCacheUsage(*upd.Reset)
	}
}

// updatePrefixCache accumulates a lookup's token counts, or on a reset replaces
// the tallies with absolute ones and recreates the cached-token counter. Either
// way the cumulative hit rate is republished, which sglang reports over the
// whole run rather than per query.
func (m *sglangMetricsAdapter) updatePrefixCache(upd prefixCacheUpdate) {
	model := m.config.DisplayModelName

	switch {
	case upd.query != nil && m.fake == nil:
		if upd.query.cached > 0 {
			m.cachedTokensTotal.WithLabelValues(model, cacheSourceTotal).Add(float64(upd.query.cached))
		}
		m.nCachedTokens += int64(upd.query.cached)
		m.nQueriedTokens += int64(upd.query.queried)
	case upd.reset != nil && m.fake != nil:
		// The tallies are the adapter's record of both, so an update naming one
		// of the pair leaves the other where it was.
		if upd.reset.cached != nil {
			m.nCachedTokens = *upd.reset.cached
		}
		if upd.reset.queried != nil {
			m.nQueriedTokens = *upd.reset.queried
		}
		m.resetCollector(m.cachedTokensTotal, m.createAndRegisterCachedTokensTotalCounter,
			"failed to recreate cached_tokens_total counter during fake-metrics reset", func() {
				m.cachedTokensTotal.WithLabelValues(model, cacheSourceTotal).Add(float64(m.nCachedTokens))
			})
	default:
		return
	}

	rate := float64(0)
	if m.nQueriedTokens > 0 {
		rate = float64(m.nCachedTokens) / float64(m.nQueriedTokens)
	}
	m.cacheHitRate.WithLabelValues(model).Set(rate)
}

// updateStreamCounter adds an event's value to the series matching the kind of
// request it came from, or on a reset folds the update into last and recreates
// the collector, stamping every series set so far. A series no configuration has
// ever named stays unstamped, so it reads as absent from a scrape rather than as
// an explicit zero.
func (m *sglangMetricsAdapter) updateStreamCounter(counterPP **prometheus.CounterVec,
	last *streamCounterReset, recreate func() error, upd streamCounterUpdate) {
	switch {
	case upd.add != nil:
		if m.fake != nil {
			return
		}
		(*counterPP).WithLabelValues(m.config.DisplayModelName,
			strconv.FormatBool(upd.add.isStreaming)).Add(upd.add.value)
	case upd.reset != nil:
		last.merge(*upd.reset)
		m.resetCollector(*counterPP, recreate, "failed to recreate counter during fake-metrics reset", func() {
			for isStreaming, value := range map[bool]*float64{
				true:  last.streaming,
				false: last.nonStreaming,
			} {
				if value != nil {
					(*counterPP).WithLabelValues(m.config.DisplayModelName,
						strconv.FormatBool(isStreaming)).Add(*value)
				}
			}
		})
	}
}

// resetCollector binds this adapter's registry and logger to
// metrics.ResetCollector.
func (m *sglangMetricsAdapter) resetCollector(current prometheus.Collector, recreate func() error,
	errMsg string, populate func()) {
	metrics.ResetCollector(m.registry, m.logger, current, recreate, errMsg, populate)
}

// reportKVCacheUsage mirrors the utilization fraction onto token_usage and
// derives the token counts beside it from the cache's capacity.
func (m *sglangMetricsAdapter) reportKVCacheUsage(usage float64) {
	used := usage * float64(m.kvCapacityTokens)

	model := m.config.DisplayModelName
	m.tokenUsage.WithLabelValues(model).Set(usage)
	m.usedTokens.WithLabelValues(model).Set(used)
	m.kvUsedTokens.WithLabelValues(model).Set(used)
	m.kvAvailableTokens.WithLabelValues(model).Set(float64(m.kvCapacityTokens) - used)
}

// updateHistogram records an observation on the single series of a histogram
// sglang does not split by request kind, or on a reset recreates the collector
// and replays the target bucket state.
func (m *sglangMetricsAdapter) updateHistogram(histPP **prometheus.HistogramVec,
	recreate func() error, upd metrics.HistogramUpdate) {
	switch {
	case upd.Observe != nil:
		if m.fake != nil {
			return
		}
		(*histPP).WithLabelValues(m.config.DisplayModelName).Observe(*upd.Observe)
	case upd.Reset != nil:
		m.resetCollector(*histPP, recreate, "failed to recreate histogram during fake-metrics reset", func() {
			metrics.InitFakeHistogram(*histPP, upd.Reset.Buckets, upd.Reset.Samples, m.config.DisplayModelName)
		})
	}
}

// updateStreamHistogram records an observation on the series matching the kind
// of request it came from, or on a reset folds the update into last and
// recreates the collector, replaying every series set so far. A series no
// configuration has ever named stays unstamped, so it reads as absent from a
// scrape rather than as an empty histogram.
func (m *sglangMetricsAdapter) updateStreamHistogram(histPP **prometheus.HistogramVec,
	last *streamHistogramReset, recreate func() error, upd streamHistogramUpdate) {
	switch {
	case upd.observe != nil:
		if m.fake != nil {
			return
		}
		(*histPP).WithLabelValues(m.config.DisplayModelName,
			strconv.FormatBool(upd.observe.isStreaming)).Observe(upd.observe.value)
	case upd.reset != nil:
		last.merge(*upd.reset)
		m.resetCollector(*histPP, recreate, "failed to recreate histogram during fake-metrics reset", func() {
			for isStreaming, samples := range map[bool][]int{
				true:  last.streaming,
				false: last.nonStreaming,
			} {
				if samples != nil {
					metrics.InitFakeHistogram(*histPP, last.buckets, samples,
						m.config.DisplayModelName, strconv.FormatBool(isStreaming))
				}
			}
		})
	}
}

// histObserve wraps a real event's observation for a stream-split histogram.
func histObserve(value float64, isStreaming bool) streamHistogramUpdate {
	return streamHistogramUpdate{observe: &streamObservation{value: value, isStreaming: isStreaming}}
}

// counterAdd wraps a real event's addition to one series of a stream-split
// counter.
func counterAdd(value float64, isStreaming bool) streamCounterUpdate {
	return streamCounterUpdate{add: &streamObservation{value: value, isStreaming: isStreaming}}
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
	if err := m.createAndRegisterTokenUsageGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterUsedTokensGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterKVUsedTokensGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterKVAvailableTokensGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterMaxTotalNumTokensGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterPageSizeGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterNumPagesGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterCacheHitRateGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterCachedTokensTotalCounter(); err != nil {
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

func (m *sglangMetricsAdapter) createAndRegisterTokenUsageGauge() error {
	m.tokenUsage = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangTokenUsageMetricName,
		Help: "The token usage.",
	}, modelLabel)
	return m.register(m.tokenUsage, "prometheus token usage gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterUsedTokensGauge() error {
	m.usedTokens = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangUsedTokensMetricName,
		Help: "The number of used tokens.",
	}, modelLabel)
	return m.register(m.usedTokens, "prometheus number of used tokens gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterKVUsedTokensGauge() error {
	m.kvUsedTokens = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangKVUsedTokensMetricName,
		Help: "Number of actively used token slots in the KV cache pool.",
	}, modelLabel)
	return m.register(m.kvUsedTokens, "prometheus kv used tokens gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterKVAvailableTokensGauge() error {
	m.kvAvailableTokens = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangKVAvailableTokensMetricName,
		Help: "Number of free token slots in the KV cache pool.",
	}, modelLabel)
	return m.register(m.kvAvailableTokens, "prometheus kv available tokens gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterMaxTotalNumTokensGauge() error {
	m.maxTotalNumTokens = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangMaxTotalNumTokensMetricName,
		Help: "Maximum total number of tokens in the KV cache pool.",
	}, modelLabel)
	return m.register(m.maxTotalNumTokens, "prometheus max total number of tokens gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterPageSizeGauge() error {
	m.pageSize = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangPageSizeMetricName,
		Help: "KV cache page size in tokens.",
	}, modelLabel)
	return m.register(m.pageSize, "prometheus page size gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterNumPagesGauge() error {
	m.numPages = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangNumPagesMetricName,
		Help: "Number of KV cache pages.",
	}, modelLabel)
	return m.register(m.numPages, "prometheus number of pages gauge register failed")
}

func (m *sglangMetricsAdapter) createAndRegisterCacheHitRateGauge() error {
	m.cacheHitRate = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangCacheHitRateMetricName,
		Help: "The prefix cache hit rate.",
	}, modelLabel)
	return m.register(m.cacheHitRate, "prometheus cache hit rate gauge register failed")
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

func (m *sglangMetricsAdapter) createAndRegisterCachedTokensTotalCounter() error {
	m.cachedTokensTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: SGLangCachedTokensTotalMetricName,
		Help: "Number of cached prompt tokens by source (device/host/storage).",
	}, []string{api.PromLabelModelName, promLabelCacheSource})
	return m.register(m.cachedTokensTotal, "prometheus cached_tokens_total counter register failed")
}

// setInitialValues zeroes the request and utilization gauges and stamps the
// ones describing the configuration, so the first scrape has a consistent
// baseline. The histograms and the per-request counters stay unstamped until a
// request is observed, since their series are per request kind.
//
// Under fake metrics the load gauges are left out: Start stamps those the
// configuration names, and one it does not name is absent from a scrape rather
// than reporting a zero no setting asked for. The gauges describing the
// configuration are stamped either way, since they report the geometry the
// simulator was given rather than anything a request changes.
func (m *sglangMetricsAdapter) setInitialValues() {
	model := m.config.DisplayModelName

	if m.fake == nil {
		m.runningRequests.WithLabelValues(model).Set(0)
		m.queuedRequests.WithLabelValues(model).Set(0)

		// An empty cache, which is also what these keep reporting while the
		// engine simulates no KV cache at all.
		m.tokenUsage.WithLabelValues(model).Set(0)
		m.usedTokens.WithLabelValues(model).Set(0)
		m.kvUsedTokens.WithLabelValues(model).Set(0)
		m.kvAvailableTokens.WithLabelValues(model).Set(float64(m.kvCapacityTokens))
		m.cacheHitRate.WithLabelValues(model).Set(0)
	}

	m.contextLen.WithLabelValues(model).Set(float64(m.config.MaxModelLen))
	m.maxTotalNumTokens.WithLabelValues(model).Set(float64(m.kvCapacityTokens))
	m.pageSize.WithLabelValues(model).Set(float64(m.config.KVCache.TokenBlockSize))
	m.numPages.WithLabelValues(model).Set(float64(m.config.KVCache.KVCacheSize))
}

// -------- Fake Metrics ------

// ApplyFakeMetricsUpdate narrows the engine-owned fake-metrics configuration to
// this engine's concrete type and applies it. A configuration belonging to
// another engine is logged and dropped.
func (m *sglangMetricsAdapter) ApplyFakeMetricsUpdate(update common.FakeMetrics) {
	sglangUpdate, ok := update.(*SGLangFakeMetrics)
	if !ok || sglangUpdate == nil {
		m.logger.Error(fmt.Errorf("unexpected fake-metrics configuration type %T", update),
			"ignoring fake-metrics update")
		return
	}
	if m.fake == nil {
		// There is no state to merge the partial onto, and applying it would
		// stamp fake values while the event handlers, which gate on fake being
		// set, keep reporting real ones; the two are never reported together.
		// Configuration.Update refuses a body carrying fake metrics in this
		// state, so this is reached only if that gate is bypassed.
		m.logger.Error(errors.New("the simulator is reporting real metrics"),
			"ignoring fake-metrics update")
		return
	}
	m.applyFakeMetrics(sglangUpdate)
}

// applyFakeMetrics enqueues the update on the per-metric channels and returns.
// The updater goroutines perform the collector unregister/recreate, logging and
// skipping any metric that fails to re-register.
//
// A metric the update does not name is left alone, and for the two that carry
// state across updates -- the stream-split counters and the prefix-cache
// tallies -- so is the half of them it does not name. Their updater goroutines
// hold that state, beside the collector it describes.
//
// Only the scalar metrics run inside Apply, since they are the ones that
// register or clear a generator. The rest only write to a channel.
func (m *sglangMetricsAdapter) applyFakeMetrics(update *SGLangFakeMetrics) {
	m.generators.Apply(func(gens *metrics.GeneratorSet) {
		if update.RunningRequests != nil {
			gens.Set(SGLangRunningReqsMetricName, update.RunningRequests, func(upd metrics.GaugeUpdate) {
				common.WriteToChannel(m.runReqChan, upd, m.logger)
			}, true)
		}
		if update.QueuedRequests != nil {
			gens.Set(SGLangQueueReqsMetricName, update.QueuedRequests, func(upd metrics.GaugeUpdate) {
				common.WriteToChannel(m.queueReqChan, upd, m.logger)
			}, true)
		}
		if update.TokenUsage != nil {
			gens.Set(SGLangTokenUsageMetricName, update.TokenUsage, func(upd metrics.GaugeUpdate) {
				common.WriteToChannel(m.kvUsageChan, upd, m.logger)
			}, false)
		}
	})

	if update.TotalPromptTokens != nil {
		common.WriteToChannel(m.promptTokensTotalChan,
			streamCounterUpdate{reset: update.TotalPromptTokens.reset()}, m.logger)
	}
	if update.TotalGenerationTokens != nil {
		common.WriteToChannel(m.generationTokensTotalChan,
			streamCounterUpdate{reset: update.TotalGenerationTokens.reset()}, m.logger)
	}
	if update.NumRequestsTotal != nil {
		common.WriteToChannel(m.requestsTotalChan,
			streamCounterUpdate{reset: update.NumRequestsTotal.reset()}, m.logger)
	}

	if update.CachedTokens != nil || update.QueriedTokens != nil {
		common.WriteToChannel(m.prefixCacheChan, prefixCacheUpdate{reset: &prefixCacheReset{
			cached:  update.CachedTokens,
			queried: update.QueriedTokens,
		}}, m.logger)
	}

	if update.TTFTBucketValues != nil {
		common.WriteToChannel(m.ttftChan,
			update.TTFTBucketValues.reset(ttftBucketsBoundaries), m.logger)
	}
	if update.ReqTPOTBucketValues != nil {
		common.WriteToChannel(m.reqTpotChan,
			update.ReqTPOTBucketValues.reset(interTokenLatencyBucketsBoundaries), m.logger)
	}
	if update.E2EReqLatencyBucketValues != nil {
		common.WriteToChannel(m.e2eReqLatencyChan,
			update.E2EReqLatencyBucketValues.reset(e2eReqLatencyBucketsBoundaries), m.logger)
	}

	for _, h := range []struct {
		samples []int
		buckets []float64
		channel common.Channel[metrics.HistogramUpdate]
	}{
		{update.InterTokenLatencyBucketValues, interTokenLatencyBucketsBoundaries, m.interTokenLatencyChan},
		{update.QueueTimeBucketValues, queueTimeBucketsBoundaries, m.queueTimeChan},
		{update.PromptTokensBucketValues, tokenBucketsBoundaries, m.promptTokensHistChan},
		{update.UncachedPromptTokensBucketValues, tokenBucketsBoundaries, m.uncachedPromptTokensHistChan},
		{update.GenerationTokensBucketValues, tokenBucketsBoundaries, m.generationTokensHistChan},
	} {
		if h.samples != nil {
			common.WriteToChannel(h.channel, metrics.HistogramUpdate{
				Reset: &metrics.HistogramReset{Buckets: h.buckets, Samples: h.samples},
			}, m.logger)
		}
	}
}

// reset turns s into the target bucket state of a stream-split histogram over
// boundaries, carrying only the series s names.
func (s *StreamSplitBuckets) reset(boundaries []float64) streamHistogramUpdate {
	return streamHistogramUpdate{reset: &streamHistogramReset{
		buckets:      boundaries,
		streaming:    s.Streaming,
		nonStreaming: s.NonStreaming,
	}}
}

// reset turns s into the target state of a stream-split counter, carrying only
// the series s names. A nil s names neither, which asks for no change.
func (s *StreamSplit) reset() *streamCounterReset {
	if s == nil {
		return &streamCounterReset{}
	}
	return &streamCounterReset{
		streaming:    int64ToFloat64(s.Streaming),
		nonStreaming: int64ToFloat64(s.NonStreaming),
	}
}

func int64ToFloat64(v *int64) *float64 {
	if v == nil {
		return nil
	}
	f := float64(*v)
	return &f
}
