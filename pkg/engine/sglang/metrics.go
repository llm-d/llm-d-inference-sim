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
	SGLangTokenUsageMetricName               = "sglang:token_usage"
	SGLangUsedTokensMetricName               = "sglang:num_used_tokens"
	SGLangKVUsedTokensMetricName             = "sglang:kv_used_tokens"
	SGLangKVAvailableTokensMetricName        = "sglang:kv_available_tokens"
	SGLangMaxTotalNumTokensMetricName        = "sglang:max_total_num_tokens"
	SGLangPageSizeMetricName                 = "sglang:page_size"
	SGLangNumPagesMetricName                 = "sglang:num_pages"
	SGLangCacheHitRateMetricName             = "sglang:cache_hit_rate"
	SGLangCachedTokensTotalMetricName        = "sglang:cached_tokens_total"
	SGLangCacheConfigMetricName              = "sglang:cache_config_info"
)

// promLabelCacheSource names the tier a cached prompt token came from.
const promLabelCacheSource = "cache_source"

// cacheSourceTotal is the cache_source value sglang falls back to when it has
// no per-tier breakdown, which is the only shape the simulator can report: its
// block cache has a single tier.
const cacheSourceTotal = "total"

// Labels of sglang:cache_config_info. The names are the ones llm-d-router
// reads out of this metric, since the router's SGLang configuration names it
// as the source of the block size and block count.
const (
	promLabelCachePageSize = "page_size"
	promLabelCacheNumPages = "num_pages"
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
	cacheConfig       *prometheus.GaugeVec

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
	runReqChan   common.Channel[gaugeDelta]
	queueReqChan common.Channel[gaugeDelta]

	queueTimeChan                common.Channel[observation]
	ttftChan                     common.Channel[streamObservation]
	interTokenLatencyChan        common.Channel[observation]
	reqTpotChan                  common.Channel[streamObservation]
	e2eReqLatencyChan            common.Channel[streamObservation]
	promptTokensHistChan         common.Channel[observation]
	uncachedPromptTokensHistChan common.Channel[observation]
	generationTokensHistChan     common.Channel[observation]

	promptTokensTotalChan     common.Channel[streamObservation]
	generationTokensTotalChan common.Channel[streamObservation]
	requestsTotalChan         common.Channel[streamObservation]

	kvUsageChan     common.Channel[observation]
	prefixCacheChan common.Channel[prefixCacheQuery]

	// nQueuedReqs / nRunningReqs are the adapter-local counters mirrored onto
	// the num_queue_reqs / num_running_reqs gauges.
	nQueuedReqs  int64
	nRunningReqs int64

	// Cumulative prefix-cache tallies behind cache_hit_rate, mutated only by
	// the prefix-cache updater goroutine.
	nCachedTokens  int64
	nQueriedTokens int64

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
		kvCapacityTokens: config.KVCache.KVCacheSize * config.KVCache.TokenBlockSize,
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
// - update queue time histogram
func (m *sglangMetricsAdapter) OnRequestDequeued(ev metrics.RequestDequeued) {
	common.WriteToChannel(m.queueReqChan, gaugeDelta(-1), m.logger)
	common.WriteToChannel(m.queueTimeChan, observation(ev.QueueTime), m.logger)
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

	common.WriteToChannel(m.promptTokensHistChan, observation(ev.PromptTokens), m.logger)
	common.WriteToChannel(m.uncachedPromptTokensHistChan,
		observation(ev.PromptTokens-ev.CachedPromptTokens), m.logger)
	common.WriteToChannel(m.generationTokensHistChan, observation(ev.GenerationTokens), m.logger)

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

// change in kv cache utilization
// - update the token-usage gauges
//
// Not reached while ValidateConfig rejects an enabled KV cache, which leaves
// the gauges reporting the empty cache they are stamped with.
func (m *sglangMetricsAdapter) OnKVCacheUsageChanged(ev metrics.KVCacheUsageChanged) {
	common.WriteToChannel(m.kvUsageChan, observation(ev.KVCacheUsagePerc), m.logger)
}

// prefix cache lookup
// - update the cached-token counter and the cumulative hit rate
//
// Not reached while ValidateConfig rejects an enabled KV cache.
func (m *sglangMetricsAdapter) OnPrefixCacheQueried(ev metrics.PrefixCacheQueried) {
	common.WriteToChannel(m.prefixCacheChan,
		prefixCacheQuery{cached: ev.CachedPromptTokens, queried: ev.QueriedTokens}, m.logger)
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

	m.queueTimeChan = common.NewChannel[observation]("sglang", maxNumberOfWaitingRequests, ctx.Done())
	go common.Subscribe(ctx, m.queueTimeChan, m.updateQueueTime)

	m.ttftChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.ttftChan, m.updateTTFT)

	m.interTokenLatencyChan = common.NewChannel[observation]("sglang", maxNumberOfTokens, ctx.Done())
	go common.Subscribe(ctx, m.interTokenLatencyChan, m.updateInterTokenLatency)

	m.reqTpotChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqTpotChan, m.updateReqTpot)

	m.e2eReqLatencyChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.e2eReqLatencyChan, m.updateE2EReqLatency)

	m.promptTokensHistChan = common.NewChannel[observation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.promptTokensHistChan, m.updatePromptTokensHist)

	m.uncachedPromptTokensHistChan = common.NewChannel[observation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.uncachedPromptTokensHistChan, m.updateUncachedPromptTokensHist)

	m.generationTokensHistChan = common.NewChannel[observation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.generationTokensHistChan, m.updateGenerationTokensHist)

	m.promptTokensTotalChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.promptTokensTotalChan, m.updatePromptTokensTotal)

	m.generationTokensTotalChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.generationTokensTotalChan, m.updateGenerationTokensTotal)

	m.requestsTotalChan = common.NewChannel[streamObservation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestsTotalChan, m.updateRequestsTotal)

	m.kvUsageChan = common.NewChannel[observation]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.kvUsageChan, m.updateKVCacheUsage)

	m.prefixCacheChan = common.NewChannel[prefixCacheQuery]("sglang", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.prefixCacheChan, m.updatePrefixCache)
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

func (m *sglangMetricsAdapter) updateQueueTime(upd observation) {
	m.observe(m.queueTime, upd)
}

func (m *sglangMetricsAdapter) updateTTFT(upd streamObservation) {
	m.observeByStream(m.ttft, upd)
}

func (m *sglangMetricsAdapter) updateInterTokenLatency(upd observation) {
	m.observe(m.interTokenLatency, upd)
}

func (m *sglangMetricsAdapter) updateReqTpot(upd streamObservation) {
	m.observeByStream(m.reqTpot, upd)
}

func (m *sglangMetricsAdapter) updateE2EReqLatency(upd streamObservation) {
	m.observeByStream(m.e2eReqLatency, upd)
}

func (m *sglangMetricsAdapter) updatePromptTokensHist(upd observation) {
	m.observe(m.promptTokensHist, upd)
}

func (m *sglangMetricsAdapter) updateUncachedPromptTokensHist(upd observation) {
	m.observe(m.uncachedPromptTokensHist, upd)
}

func (m *sglangMetricsAdapter) updateGenerationTokensHist(upd observation) {
	m.observe(m.generationTokensHist, upd)
}

func (m *sglangMetricsAdapter) updatePromptTokensTotal(upd streamObservation) {
	m.addByStream(m.promptTokensTotal, upd)
}

func (m *sglangMetricsAdapter) updateGenerationTokensTotal(upd streamObservation) {
	m.addByStream(m.generationTokensTotal, upd)
}

func (m *sglangMetricsAdapter) updateRequestsTotal(upd streamObservation) {
	m.addByStream(m.requestsTotal, upd)
}

// updateKVCacheUsage mirrors the utilization fraction onto token_usage and
// derives the token counts beside it from the cache's capacity.
func (m *sglangMetricsAdapter) updateKVCacheUsage(upd observation) {
	usage := float64(upd)
	used := usage * float64(m.kvCapacityTokens)

	model := m.config.DisplayModelName
	m.tokenUsage.WithLabelValues(model).Set(usage)
	m.usedTokens.WithLabelValues(model).Set(used)
	m.kvUsedTokens.WithLabelValues(model).Set(used)
	m.kvAvailableTokens.WithLabelValues(model).Set(float64(m.kvCapacityTokens) - used)
}

// updatePrefixCache accumulates the cached-token counter and republishes the
// cumulative hit rate, which sglang reports over the whole run rather than per
// query.
func (m *sglangMetricsAdapter) updatePrefixCache(upd prefixCacheQuery) {
	model := m.config.DisplayModelName
	if upd.cached > 0 {
		m.cachedTokensTotal.WithLabelValues(model, cacheSourceTotal).Add(float64(upd.cached))
	}

	m.nCachedTokens += int64(upd.cached)
	m.nQueriedTokens += int64(upd.queried)
	if m.nQueriedTokens > 0 {
		m.cacheHitRate.WithLabelValues(model).Set(float64(m.nCachedTokens) / float64(m.nQueriedTokens))
	}
}

// observe records upd on the single series of a histogram sglang does not
// split by request kind.
func (m *sglangMetricsAdapter) observe(hist *prometheus.HistogramVec, upd observation) {
	hist.WithLabelValues(m.config.DisplayModelName).Observe(float64(upd))
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
	if err := m.createAndRegisterCacheConfigGauge(); err != nil {
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

// createAndRegisterCacheConfigGauge registers sglang:cache_config_info, which
// carries the cache's block size and block count as labels on a constant 1.
//
// sglang itself has no such metric: it reports the two as the plain
// sglang:page_size and sglang:num_pages gauges registered above, which this
// engine reports too. This one exists because llm-d-router's SGLang
// configuration names it as where it reads them from, and a scrape missing it
// leaves the router with no block size or block count at all.
func (m *sglangMetricsAdapter) createAndRegisterCacheConfigGauge() error {
	m.cacheConfig = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: SGLangCacheConfigMetricName,
		Help: "Information of the KV cache configuration.",
	}, []string{promLabelCachePageSize, promLabelCacheNumPages})
	return m.register(m.cacheConfig, "prometheus cache config register failed")
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
func (m *sglangMetricsAdapter) setInitialValues() {
	model := m.config.DisplayModelName

	m.runningRequests.WithLabelValues(model).Set(0)
	m.queuedRequests.WithLabelValues(model).Set(0)
	m.contextLen.WithLabelValues(model).Set(float64(m.config.MaxModelLen))

	// An empty cache, which is also what these keep reporting while the engine
	// simulates no KV cache at all.
	m.tokenUsage.WithLabelValues(model).Set(0)
	m.usedTokens.WithLabelValues(model).Set(0)
	m.kvUsedTokens.WithLabelValues(model).Set(0)
	m.kvAvailableTokens.WithLabelValues(model).Set(float64(m.kvCapacityTokens))
	m.cacheHitRate.WithLabelValues(model).Set(0)

	m.maxTotalNumTokens.WithLabelValues(model).Set(float64(m.kvCapacityTokens))
	m.pageSize.WithLabelValues(model).Set(float64(m.config.KVCache.TokenBlockSize))
	m.numPages.WithLabelValues(model).Set(float64(m.config.KVCache.KVCacheSize))
	m.cacheConfig.WithLabelValues(
		strconv.Itoa(m.config.KVCache.TokenBlockSize),
		strconv.Itoa(m.config.KVCache.KVCacheSize),
	).Set(1)
}
