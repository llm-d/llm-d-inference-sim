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

// vLLM Prometheus implementation of metrics.EngineMetricsAdapter.

package vllm

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
)

const (
	VLLME2EReqLatencyMetricName           = "vllm:e2e_request_latency_seconds"
	VLLMReqQueueTimeMetricName            = "vllm:request_queue_time_seconds"
	VLLMReqInferenceTimeMetricName        = "vllm:request_inference_time_seconds"
	VLLMPrefillTimeMetricName             = "vllm:request_prefill_time_seconds"
	VLLMDecodeTimeMetricName              = "vllm:request_decode_time_seconds"
	VLLMTTFTMetricName                    = "vllm:time_to_first_token_seconds"
	VLLMTPOTMetricName                    = "vllm:time_per_output_token_seconds"
	VLLMReqTPOTMetricName                 = "vllm:request_time_per_output_token_seconds"
	VLLMInterTokenLatencyMetricName       = "vllm:inter_token_latency_seconds"
	VLLMMaxNumGenerationTokensMetricName  = "vllm:request_max_num_generation_tokens"
	VLLMGenerationTokensMetricName        = "vllm:request_generation_tokens"
	VLLMParamMaxTokensMetricName          = "vllm:request_params_max_tokens"
	VLLMPromptTokensMetricName            = "vllm:request_prompt_tokens"
	VLLMGenerationTokensTotalMetricName   = "vllm:generation_tokens_total"
	VLLMPromptTokensTotalMetricName       = "vllm:prompt_tokens_total"
	VLLMSuccessTotalMetricName            = "vllm:request_success_total"
	VLLMLoRARequestsMetricName            = "vllm:lora_requests_info"
	VLLMReqRunningMetricName              = "vllm:num_requests_running"
	VLLMReqWaitingMetricName              = "vllm:num_requests_waiting"
	VLLMKVCacheUsageMetricName            = "vllm:kv_cache_usage_perc"
	VLLMCacheConfigName                   = "vllm:cache_config_info"
	VLLMPrefixCacheHitsTotalMetricName    = "vllm:prefix_cache_hits_total"
	VLLMPrefixCacheQueriesTotalMetricName = "vllm:prefix_cache_queries_total"
)

var modelLabel = []string{api.PromLabelModelName}

// TokenBucketMantissas matches vLLM's build_1_2_5_buckets() in metrics.py.
//
// Reference: https://github.com/vllm-project/vllm/blob/main/vllm/engine/metrics.py#L175
var TokenBucketMantissas = []int{1, 2, 5}

// successTotalReset is the target state for request_success_total: one
// Add per (reason, count) pair after clearing the counter's series.
type successTotalReset struct {
	Reasons map[string]int64
}

// requestSuccessCounterUpdate is a requestSuccessTotalChan payload: an
// Increment with the finish-reason label, or a Reset to a per-reason target
// map. Exactly one field is non-nil.
type requestSuccessCounterUpdate struct {
	Increment *string
	Reset     *successTotalReset
}

// loraUpdate is the discriminated-union payload for lorasChan. Event-side
// producers set Snapshot with the current per-LoRA counts; the fake-metrics
// applier sets Reset.
type loraUpdate struct {
	Snapshot *metrics.LoRASetsChanged
	Reset    *loraReset
}

// loraReset carries the target state for lora_requests_info: clear the
// gauge's series, then stamp one series per entry (or a single zero-adapter
// row with the current timestamp when Entries is empty).
type loraReset struct {
	MaxLoRAs int
	Entries  []common.LorasMetrics
}

// vllmMetricsAdapter implements metrics.EngineMetricsAdapter and produces the
// vLLM Prometheus surface. The bus dispatches drained events to the On<Event>
// handlers, which fan out to per-metric channels, written to Prometheus by one
// updater goroutine per metric.
type vllmMetricsAdapter struct {
	logger logr.Logger
	config common.Configuration
	// fake is config.FakeMetrics narrowed to vLLM's concrete type, nil when
	// fake metrics are off or another engine owns the configuration.
	fake *VLLMFakeMetrics

	registry *prometheus.Registry

	// generators holds the fake-metric generators this adapter has registered
	// and the ticker that re-evaluates them.
	generators *metrics.FakeGenerators

	// gauges
	runningRequests        *prometheus.GaugeVec
	waitingRequests        *prometheus.GaugeVec
	kvCacheUsagePercentage *prometheus.GaugeVec
	loraInfo               *prometheus.GaugeVec
	cacheConfig            *prometheus.GaugeVec

	// histograms
	ttft                    *prometheus.HistogramVec
	tpot                    *prometheus.HistogramVec
	interTokenLatency       *prometheus.HistogramVec
	reqTpot                 *prometheus.HistogramVec
	e2eReqLatency           *prometheus.HistogramVec
	reqQueueTime            *prometheus.HistogramVec
	reqInferenceTime        *prometheus.HistogramVec
	reqPrefillTime          *prometheus.HistogramVec
	reqDecodeTime           *prometheus.HistogramVec
	requestPromptTokens     *prometheus.HistogramVec
	requestGenerationTokens *prometheus.HistogramVec
	maxNumGenerationTokens  *prometheus.HistogramVec
	requestParamsMaxTokens  *prometheus.HistogramVec

	// counters
	promptTokensTotal       *prometheus.CounterVec
	generationTokensTotal   *prometheus.CounterVec
	requestSuccessTotal     *prometheus.CounterVec
	prefixCacheHitsTotal    *prometheus.CounterVec
	prefixCacheQueriesTotal *prometheus.CounterVec

	// Channels: one per Prometheus metric family. Handlers push
	// here; the updater goroutines below drain each and perform the actual
	// Prometheus mutation. Created in Start once ctx.Done() is available.
	runReqChan       common.Channel[metrics.GaugeUpdate]
	waitingReqChan   common.Channel[metrics.GaugeUpdate]
	kvCacheUsageChan common.Channel[metrics.GaugeUpdate]

	ttftChan                    common.Channel[metrics.HistogramUpdate]
	perTokenLatencyChan         common.Channel[metrics.HistogramUpdate]
	e2eReqLatencyChan           common.Channel[metrics.HistogramUpdate]
	reqQueueTimeChan            common.Channel[metrics.HistogramUpdate]
	reqInferenceTimeChan        common.Channel[metrics.HistogramUpdate]
	reqPrefillTimeChan          common.Channel[metrics.HistogramUpdate]
	reqDecodeTimeChan           common.Channel[metrics.HistogramUpdate]
	reqTpotChan                 common.Channel[metrics.HistogramUpdate]
	requestPromptTokensChan     common.Channel[metrics.HistogramUpdate]
	requestGenerationTokensChan common.Channel[metrics.HistogramUpdate]
	maxNumGenerationTokensChan  common.Channel[metrics.HistogramUpdate]
	requestParamsMaxTokensChan  common.Channel[metrics.HistogramUpdate]

	promptTokensTotalChan       common.Channel[metrics.CounterUpdate]
	generationTokensTotalChan   common.Channel[metrics.CounterUpdate]
	prefixCacheHitsTotalChan    common.Channel[metrics.CounterUpdate]
	prefixCacheQueriesTotalChan common.Channel[metrics.CounterUpdate]

	lorasChan               common.Channel[loraUpdate]
	requestSuccessTotalChan common.Channel[requestSuccessCounterUpdate]

	// nWaitingReqs / nRunningReqs are the adapter-local counters mirrored
	// onto the num_requests_{waiting,running} gauges.
	nWaitingReqs int64
	nRunningReqs int64
}

// NewMetricsAdapter returns the vLLM metrics adapter.
func (Engine) NewMetricsAdapter(ctx context.Context, registry *prometheus.Registry,
	logger logr.Logger, config common.Configuration) (metrics.MetricsAdapter, error) {
	return newMetricsAdapter(ctx, registry, logger, config)
}

// newMetricsAdapter registers the Prometheus collectors on registry, stamps
// initial values, and spawns the per-metric updater goroutines. ctx must match
// the one passed to metrics.NewMetricsBus. Returns an error if any collector
// fails to register.
func newMetricsAdapter(ctx context.Context, registry *prometheus.Registry,
	logger logr.Logger, config common.Configuration) (*vllmMetricsAdapter, error) {
	m := &vllmMetricsAdapter{
		logger:     logger,
		registry:   registry,
		config:     config,
		generators: metrics.NewFakeGenerators(ctx, config.FakeMetricsRefreshInterval),
	}
	if fake, ok := config.FakeMetrics.(*VLLMFakeMetrics); ok {
		m.fake = fake
	}

	if err := m.buildMetrics(); err != nil {
		return nil, err
	}
	m.createAndStartPrometheusChannels(ctx)
	m.setInitialValues()

	return m, nil
}

func (m *vllmMetricsAdapter) Close() error {
	m.generators.Stop()

	return nil
}

// Start applies the initial fake-metrics configuration and starts the
// generator ticker. The bus has already subscribed the On<Event> handlers to
// its channels by the time this runs.
func (m *vllmMetricsAdapter) Start(_ context.Context) error {
	if m.fake != nil {
		fm := *m.fake
		m.applyFakeMetrics(&fm)
	}
	m.generators.Start()

	return nil
}

// -- Event handlers  -------------------

func (m *vllmMetricsAdapter) OnRequestReceived(_ metrics.RequestReceived) {
	// State marker; no exposed metric today.
}

func (m *vllmMetricsAdapter) OnRequestRejected(_ metrics.RequestRejected) {
	// State marker; no exposed metric today.
}

// request queued
// - update number of waiting requests
func (m *vllmMetricsAdapter) OnRequestQueued(ev metrics.RequestQueued) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.waitingReqChan, metrics.GaugeAdd(1), m.logger)
}

// request dequeued
// - update number of waiting requests
// - update queue time histogram
func (m *vllmMetricsAdapter) OnRequestDequeued(ev metrics.RequestDequeued) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.waitingReqChan, metrics.GaugeAdd(-1), m.logger)
	common.WriteToChannel(m.reqQueueTimeChan, metrics.Observation(ev.QueueTime), m.logger)
}

// request running
// - update number of running requests
func (m *vllmMetricsAdapter) OnRequestRunning(ev metrics.RequestRunning) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.runReqChan, metrics.GaugeAdd(1), m.logger)
}

// prefill started
func (m *vllmMetricsAdapter) OnPrefillStarted(_ metrics.PrefillStarted) {
	// State marker.
}

// prefill step ended
// - update prefill time histogram
// - update TTFT histogram
func (m *vllmMetricsAdapter) OnPrefillEnded(ev metrics.PrefillEnded) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.reqPrefillTimeChan, metrics.Observation(ev.PrefillDuration), m.logger)
	common.WriteToChannel(m.ttftChan, metrics.Observation(ev.PrefillDuration), m.logger)
}

func (m *vllmMetricsAdapter) OnDecodeStarted(_ metrics.DecodeStarted) {
	// State marker.
}

// token generated
// - update tpot and itl histograms
func (m *vllmMetricsAdapter) OnTokenGenerated(ev metrics.TokenGenerated) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.perTokenLatencyChan, metrics.Observation(ev.InterTokenLatency), m.logger)
}

// decode ended
// - update decode time histogram
// - update requests tpot histogram
func (m *vllmMetricsAdapter) OnDecodeEnded(ev metrics.DecodeEnded) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.reqDecodeTimeChan, metrics.Observation(ev.DecodeDuration), m.logger)

	if ev.GenerationTokens > 0 {
		common.WriteToChannel(m.reqTpotChan, metrics.Observation(ev.DecodeDuration/float64(ev.GenerationTokens)), m.logger)
	}
}

// request processing finished successfully - update all relevant metrics
func (m *vllmMetricsAdapter) OnRequestSucceeded(ev metrics.RequestSucceeded) {
	if m.fake != nil {
		return
	}

	// update number of successful requests per finish reason
	common.WriteToChannel(m.requestSuccessTotalChan, requestSuccessCounterUpdate{Increment: &ev.FinishReason}, m.logger)

	// request finished successfully, update number of prompt and generated tokens
	// both total and histogram metrics
	common.WriteToChannel(m.requestPromptTokensChan, metrics.Observation(float64(ev.PromptTokens)), m.logger)
	common.WriteToChannel(m.requestGenerationTokensChan, metrics.Observation(float64(ev.GenerationTokens)), m.logger)
	promptTokens := float64(ev.PromptTokens)
	generationTokens := float64(ev.GenerationTokens)
	common.WriteToChannel(m.promptTokensTotalChan, metrics.CounterUpdate{Add: &promptTokens}, m.logger)
	common.WriteToChannel(m.generationTokensTotalChan, metrics.CounterUpdate{Add: &generationTokens}, m.logger)

	// if max_tokens is set, update the request_params_max_tokens histogram
	if ev.MaxTokens != nil {
		common.WriteToChannel(m.requestParamsMaxTokensChan, metrics.Observation(float64(*ev.MaxTokens)), m.logger)
	}
	if maxGenTokens, err := common.MaxIntSlice(ev.GenTokensPerChoice); err == nil {
		common.WriteToChannel(m.maxNumGenerationTokensChan, metrics.Observation(float64(maxGenTokens)), m.logger)
	}

	common.WriteToChannel(m.e2eReqLatencyChan, metrics.Observation(ev.E2ELatency), m.logger)
	common.WriteToChannel(m.reqInferenceTimeChan, metrics.Observation(ev.InferenceTime), m.logger)

	m.finishRunning()
}

// request processing failed
// - update all relevant metrics
func (m *vllmMetricsAdapter) OnRequestFailed(ev metrics.RequestFailed) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.e2eReqLatencyChan, metrics.Observation(ev.E2ELatency), m.logger)
	common.WriteToChannel(m.reqInferenceTimeChan, metrics.Observation(ev.InferenceTime), m.logger)

	m.finishRunning()
}

// change in kv cache utilization
// - update kv cache usage gauge
func (m *vllmMetricsAdapter) OnKVCacheUsageChanged(ev metrics.KVCacheUsageChanged) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.kvCacheUsageChan, metrics.GaugeAdd(ev.KVCacheUsagePerc), m.logger)
}

// change in prefix cache utilization
// - update prefix cache hits and queries counters
func (m *vllmMetricsAdapter) OnPrefixCacheQueried(ev metrics.PrefixCacheQueried) {
	if m.fake != nil {
		return
	}
	hit := float64(ev.CachedPromptTokens)
	queried := float64(ev.QueriedTokens)
	common.WriteToChannel(m.prefixCacheHitsTotalChan, metrics.CounterUpdate{Add: &hit}, m.logger)
	common.WriteToChannel(m.prefixCacheQueriesTotalChan, metrics.CounterUpdate{Add: &queried}, m.logger)
}

// OnLoRASetsChanged receives the per-LoRA waiting/running snapshot produced
// by the bus after each LoRAChanged event and forwards it to the LoRA
// updater goroutine.
func (m *vllmMetricsAdapter) OnLoRASetsChanged(ev metrics.LoRASetsChanged) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.lorasChan, loraUpdate{Snapshot: &ev}, m.logger)
}

// finishRunning decrements the running-request counter for a terminal
// request. LoRA state transitions are handled separately via LoRAChanged.
func (m *vllmMetricsAdapter) finishRunning() {
	common.WriteToChannel(m.runReqChan, metrics.GaugeAdd(-1), m.logger)
}

func (m *vllmMetricsAdapter) createAndStartPrometheusChannels(ctx context.Context) {
	maxNumberOfRunningRequests, maxNumberOfWaitingRequests, maxNumberOfRequests, maxNumberOfTokens := metrics.ChannelCapacities(m.config)

	m.runReqChan = common.NewChannel[metrics.GaugeUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.runReqChan, m.updateRunningRequests)

	m.waitingReqChan = common.NewChannel[metrics.GaugeUpdate]("vllm", maxNumberOfWaitingRequests, ctx.Done())
	go common.Subscribe(ctx, m.waitingReqChan, m.updateWaitingRequests)

	m.kvCacheUsageChan = common.NewChannel[metrics.GaugeUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.kvCacheUsageChan, m.updateKVCacheUsage)

	m.ttftChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.ttftChan, m.updateTTFT)

	m.perTokenLatencyChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfTokens, ctx.Done())
	go common.Subscribe(ctx, m.perTokenLatencyChan, m.updatePerTokenLatency)

	m.e2eReqLatencyChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.e2eReqLatencyChan, m.updateE2EReqLatency)

	m.reqQueueTimeChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfWaitingRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqQueueTimeChan, m.updateReqQueueTime)

	m.reqInferenceTimeChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqInferenceTimeChan, m.updateReqInferenceTime)

	m.reqPrefillTimeChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqPrefillTimeChan, m.updateReqPrefillTime)

	m.reqDecodeTimeChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqDecodeTimeChan, m.updateReqDecodeTime)

	m.reqTpotChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqTpotChan, m.updateReqTpot)

	m.lorasChan = common.NewChannel[loraUpdate]("vllm", maxNumberOfRequests, ctx.Done())
	go common.Subscribe(ctx, m.lorasChan, m.updateLoRAs)

	m.requestPromptTokensChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestPromptTokensChan, m.updateRequestPromptTokens)

	m.requestGenerationTokensChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestGenerationTokensChan, m.updateRequestGenerationTokens)

	m.maxNumGenerationTokensChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.maxNumGenerationTokensChan, m.updateMaxNumGenerationTokens)

	m.requestParamsMaxTokensChan = common.NewChannel[metrics.HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestParamsMaxTokensChan, m.updateRequestParamsMaxTokens)

	m.promptTokensTotalChan = common.NewChannel[metrics.CounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.promptTokensTotalChan, m.updatePromptTokensTotal)

	m.generationTokensTotalChan = common.NewChannel[metrics.CounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.generationTokensTotalChan, m.updateGenerationTokensTotal)

	m.requestSuccessTotalChan = common.NewChannel[requestSuccessCounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestSuccessTotalChan, m.updateRequestSuccessTotal)

	m.prefixCacheHitsTotalChan = common.NewChannel[metrics.CounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.prefixCacheHitsTotalChan, m.updatePrefixCacheHitsTotal)

	m.prefixCacheQueriesTotalChan = common.NewChannel[metrics.CounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.prefixCacheQueriesTotalChan, m.updatePrefixCacheQueriesTotal)
}

// -- Per-metric updaters --------------------------------------------------

func (m *vllmMetricsAdapter) updateRequestPromptTokens(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.requestPromptTokens, upd)
}

func (m *vllmMetricsAdapter) updateRequestGenerationTokens(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.requestGenerationTokens, upd)
}

func (m *vllmMetricsAdapter) updateMaxNumGenerationTokens(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.maxNumGenerationTokens, upd)
}

func (m *vllmMetricsAdapter) updateRequestParamsMaxTokens(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.requestParamsMaxTokens, upd)
}

func (m *vllmMetricsAdapter) updatePromptTokensTotal(upd metrics.CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.promptTokensTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.resetCounter(m.promptTokensTotal, upd.Reset.Value)
	}
}

func (m *vllmMetricsAdapter) updateGenerationTokensTotal(upd metrics.CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.generationTokensTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.resetCounter(m.generationTokensTotal, upd.Reset.Value)
	}
}

func (m *vllmMetricsAdapter) updateRequestSuccessTotal(upd requestSuccessCounterUpdate) {
	switch {
	case upd.Increment != nil:
		if m.fake != nil {
			return
		}
		m.requestSuccessTotal.WithLabelValues(m.config.DisplayModelName, *upd.Increment).Inc()
	case upd.Reset != nil:
		m.resetSuccessTotal(upd.Reset.Reasons)
	}
}

func (m *vllmMetricsAdapter) updatePrefixCacheHitsTotal(upd metrics.CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.prefixCacheHitsTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.resetCounter(m.prefixCacheHitsTotal, upd.Reset.Value)
	}
}

func (m *vllmMetricsAdapter) updatePrefixCacheQueriesTotal(upd metrics.CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.prefixCacheQueriesTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.resetCounter(m.prefixCacheQueriesTotal, upd.Reset.Value)
	}
}

// updateHistogram records an Observe, or on Reset clears the histogram's
// series and replays the target bucket state.
func (m *vllmMetricsAdapter) updateHistogram(hist *prometheus.HistogramVec, upd metrics.HistogramUpdate) {
	switch {
	case upd.Observe != nil:
		if m.fake != nil {
			return
		}
		hist.WithLabelValues(m.config.DisplayModelName).Observe(*upd.Observe)
	case upd.Reset != nil:
		hist.Reset()
		metrics.InitFakeHistogram(hist, upd.Reset.Buckets, upd.Reset.Samples, m.config.DisplayModelName)
	}
}

func (m *vllmMetricsAdapter) updateWaitingRequests(upd metrics.GaugeUpdate) {
	switch {
	case upd.Add != nil && m.fake == nil:
		m.nWaitingReqs += int64(*upd.Add)
	case upd.Reset != nil && m.fake != nil:
		m.nWaitingReqs = int64(*upd.Reset)
	default:
		return
	}
	m.reportWaitingRequests()
}

func (m *vllmMetricsAdapter) updateRunningRequests(upd metrics.GaugeUpdate) {
	switch {
	case upd.Add != nil && m.fake == nil:
		m.nRunningReqs += int64(*upd.Add)
	case upd.Reset != nil && m.fake != nil:
		m.nRunningReqs = int64(*upd.Reset)
	default:
		return
	}
	m.reportRunningRequests()
}

func (m *vllmMetricsAdapter) updateKVCacheUsage(upd metrics.GaugeUpdate) {
	switch {
	case upd.Add != nil && m.fake == nil:
		m.reportKVCacheUsage(*upd.Add)
	case upd.Reset != nil && m.fake != nil:
		m.reportKVCacheUsage(*upd.Reset)
	}
}

func (m *vllmMetricsAdapter) updateTTFT(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.ttft, upd)
}

func (m *vllmMetricsAdapter) updatePerTokenLatency(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.tpot, upd)
	m.updateHistogram(m.interTokenLatency, upd)
}

func (m *vllmMetricsAdapter) updateE2EReqLatency(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.e2eReqLatency, upd)
}

func (m *vllmMetricsAdapter) updateReqQueueTime(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.reqQueueTime, upd)
}

func (m *vllmMetricsAdapter) updateReqInferenceTime(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.reqInferenceTime, upd)
}

func (m *vllmMetricsAdapter) updateReqPrefillTime(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.reqPrefillTime, upd)
}

func (m *vllmMetricsAdapter) updateReqDecodeTime(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.reqDecodeTime, upd)
}

func (m *vllmMetricsAdapter) updateReqTpot(upd metrics.HistogramUpdate) {
	m.updateHistogram(m.reqTpot, upd)
}

// lorasUpdater republishes lora_requests_info from Snapshot events, or on
// Reset clears the gauge's series and stamps the supplied entries.
func (m *vllmMetricsAdapter) updateLoRAs(upd loraUpdate) {
	switch {
	case upd.Snapshot != nil:
		m.reportLoras(*upd.Snapshot)
	case upd.Reset != nil:
		m.resetLoRA(upd.Reset)
	}
}

// resetCounter clears counter's series and records the target value via a
// single Add. A nil value leaves the counter with no series stamped, so it
// reads as absent from /metrics.
func (m *vllmMetricsAdapter) resetCounter(counter *prometheus.CounterVec, value *float64) {
	counter.Reset()
	if value != nil {
		counter.WithLabelValues(m.config.DisplayModelName).Add(*value)
	}
}

// resetSuccessTotal clears requestSuccessTotal's series, then Adds each
// (reason, count) pair. Nil or empty reasons leaves the counter with no
// series stamped.
func (m *vllmMetricsAdapter) resetSuccessTotal(reasons map[string]int64) {
	m.requestSuccessTotal.Reset()
	for reason, count := range reasons {
		m.requestSuccessTotal.WithLabelValues(m.config.DisplayModelName, reason).Add(float64(count))
	}
}

// resetLoRA clears loraInfo's series, then stamps one series per entry.
// Empty entries emits a single zero-adapter row with the current timestamp.
func (m *vllmMetricsAdapter) resetLoRA(reset *loraReset) {
	// new value is empty - ensure /metrics does not report it
	if reset == nil || len(reset.Entries) == 0 {
		if m.loraInfo != nil {
			m.registry.Unregister(m.loraInfo)
			m.loraInfo = nil
		}
	} else {
		// new value is non-empty - reset or create the gauge and stamp the entries
		if m.loraInfo == nil {
			if err := m.createAndRegisterLoraInfoGauge(); err != nil {
				return
			}
		} else {
			m.loraInfo.Reset()
		}
		for _, entry := range reset.Entries {
			m.loraInfo.WithLabelValues(
				strconv.Itoa(reset.MaxLoRAs),
				entry.RunningLoras,
				entry.WaitingLoras,
			).Set(entry.Timestamp)
		}
	}
}

// -- Report helpers (Prometheus writes) -------------------------------------

func (m *vllmMetricsAdapter) reportRunningRequests() {
	if m.runningRequests != nil {
		m.runningRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nRunningReqs))
	}
}

func (m *vllmMetricsAdapter) reportWaitingRequests() {
	if m.waitingRequests != nil {
		m.waitingRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nWaitingReqs))
	}
}

func (m *vllmMetricsAdapter) reportKVCacheUsage(value float64) {
	if m.kvCacheUsagePercentage != nil {
		m.kvCacheUsagePercentage.WithLabelValues(m.config.DisplayModelName).Set(value)
	}
}

func (m *vllmMetricsAdapter) reportLoras(snap metrics.LoRASetsChanged) {
	if m.fake != nil {
		return
	}
	if m.loraInfo == nil {
		return
	}

	runningLoras := strings.Join(slices.Collect(maps.Keys(snap.Running)), ",")
	waitingLoras := strings.Join(slices.Collect(maps.Keys(snap.Waiting)), ",")

	m.loraInfo.WithLabelValues(
		strconv.Itoa(m.config.Lora.MaxLoras),
		runningLoras,
		waitingLoras,
	).SetToCurrentTime()
}

// -- Prometheus wiring ------------------------------------------------------

// buildMetrics constructs and registers all Prometheus collectors. It is called once during adapter construction.
func (m *vllmMetricsAdapter) buildMetrics() error {
	// vLLM's Prometheus client exports the standard process metrics, and
	// consumers read process_start_time_seconds to tell engine restarts apart.
	if err := m.register(collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
		"prometheus process collector register failed"); err != nil {
		return err
	}
	tokenBuckets := metrics.BuildBuckets(m.config.MaxModelLen, TokenBucketMantissas)

	if err := m.createAndRegisterRunningRequestsGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterWaitingRequestsGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterKVCacheUsageGauge(); err != nil {
		return err
	}
	// vLLM creates this gauge only when LoRA is enabled.
	if len(m.config.Lora.LoraModules) > 0 {
		if err := m.createAndRegisterLoraInfoGauge(); err != nil {
			return err
		}
	}
	if err := m.createAndRegisterCacheConfigGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterTTFTHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterTPOTHistogram(); err != nil {
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
	if err := m.createAndRegisterReqQueueTimeHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqInferenceTimeHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqPrefillTimeHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqDecodeTimeHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqPromptTokensHistogram(tokenBuckets); err != nil {
		return err
	}
	if err := m.createAndRegisterReqGenerationTokensHistogram(tokenBuckets); err != nil {
		return err
	}
	if err := m.createAndRegisterMaxNumGenerationTokensHistogram(tokenBuckets); err != nil {
		return err
	}
	if err := m.createAndRegisterReqParamsMaxTokensHistogram(tokenBuckets); err != nil {
		return err
	}
	if err := m.createAndRegisterPromptTokensTotalCounter(); err != nil {
		return err
	}
	if err := m.createAndRegisterGenerationTokensTotalCounter(); err != nil {
		return err
	}
	if err := m.createAndRegisterRequestSuccessTotalCounter(); err != nil {
		return err
	}
	if err := m.createAndRegisterPrefixCacheHitsTotalCounter(); err != nil {
		return err
	}
	if err := m.createAndRegisterPrefixCacheQueriesTotalCounter(); err != nil {
		return err
	}
	return nil
}

func (m *vllmMetricsAdapter) createAndRegisterRunningRequestsGauge() error {
	m.runningRequests = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMReqRunningMetricName,
		Help: "Number of requests currently running on GPU.",
	}, modelLabel)
	return m.register(m.runningRequests, "prometheus number of running requests gauge register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterWaitingRequestsGauge() error {
	m.waitingRequests = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMReqWaitingMetricName,
		Help: "Prometheus metric for the number of queued requests.",
	}, modelLabel)
	return m.register(m.waitingRequests, "prometheus number of requests in queue gauge register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterKVCacheUsageGauge() error {
	m.kvCacheUsagePercentage = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMKVCacheUsageMetricName,
		Help: "Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).",
	}, modelLabel)
	return m.register(m.kvCacheUsagePercentage, "prometheus kv cache usage percentage gauge register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterLoraInfoGauge() error {
	m.loraInfo = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMLoRARequestsMetricName,
		Help: "Running stats on lora requests.",
	}, []string{api.PromLabelMaxLora, api.PromLabelRunningLoraAdapters, api.PromLabelWaitingLoraAdapters})
	return m.register(m.loraInfo, "prometheus lora info gauge register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterCacheConfigGauge() error {
	m.cacheConfig = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMCacheConfigName,
		Help: "Information of the LLMEngine CacheConfig.",
	}, []string{
		api.PromLabelCacheBlockSize,
		api.PromLabelCacheDType,
		api.PromLabelCacheNumCPUBlocks,
		api.PromLabelCacheNumGPUBlocks,
	})
	return m.register(m.cacheConfig, "prometheus cache config register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterTTFTHistogram() error {
	m.ttft = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMTTFTMetricName,
		Help:    "Histogram of time to first token in seconds.",
		Buckets: common.TTFTBucketsBoundaries,
	}, modelLabel)
	return m.register(m.ttft, "prometheus time to first token histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterTPOTHistogram() error {
	m.tpot = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMTPOTMetricName,
		Help:    "Histogram of time per output token in seconds.",
		Buckets: common.TPOTBucketsBoundaries,
	}, modelLabel)
	return m.register(m.tpot, "prometheus time per output token histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterInterTokenLatencyHistogram() error {
	m.interTokenLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMInterTokenLatencyMetricName,
		Help:    "Histogram of inter-token latency in seconds.",
		Buckets: common.TPOTBucketsBoundaries,
	}, modelLabel)
	return m.register(m.interTokenLatency, "prometheus inter-token latency histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterReqTpotHistogram() error {
	m.reqTpot = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMReqTPOTMetricName,
		Help:    "Histogram of time_per_output_token_seconds per request.",
		Buckets: common.TPOTBucketsBoundaries,
	}, modelLabel)
	return m.register(m.reqTpot, "prometheus time_per_output_token_seconds per request histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterE2EReqLatencyHistogram() error {
	m.e2eReqLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLME2EReqLatencyMetricName,
		Help:    "Histogram of end to end request latency in seconds.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.e2eReqLatency, "prometheus e2e request latency histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterReqQueueTimeHistogram() error {
	m.reqQueueTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMReqQueueTimeMetricName,
		Help:    "Histogram of time spent in WAITING phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.reqQueueTime, "prometheus request queue time histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterReqInferenceTimeHistogram() error {
	m.reqInferenceTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMReqInferenceTimeMetricName,
		Help:    "Histogram of time spent in RUNNING phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.reqInferenceTime, "prometheus request inference time histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterReqPrefillTimeHistogram() error {
	m.reqPrefillTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMPrefillTimeMetricName,
		Help:    "Histogram of time spent in PREFILL phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.reqPrefillTime, "prometheus request prefill time histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterReqDecodeTimeHistogram() error {
	m.reqDecodeTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMDecodeTimeMetricName,
		Help:    "Histogram of time spent in DECODE phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)

	return m.register(m.reqDecodeTime, "prometheus request decode time histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterReqPromptTokensHistogram(tokenBuckets []float64) error {
	m.requestPromptTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMPromptTokensMetricName,
		Help:    "Number of prefill tokens processed.",
		Buckets: tokenBuckets,
	}, modelLabel)
	return m.register(m.requestPromptTokens, "prometheus request_prompt_tokens histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterReqGenerationTokensHistogram(tokenBuckets []float64) error {
	m.requestGenerationTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMGenerationTokensMetricName,
		Help:    "Number of generation tokens processed.",
		Buckets: tokenBuckets,
	}, modelLabel)
	return m.register(m.requestGenerationTokens, "prometheus request_generation_tokens histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterMaxNumGenerationTokensHistogram(tokenBuckets []float64) error {
	m.maxNumGenerationTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMMaxNumGenerationTokensMetricName,
		Help:    "Histogram of maximum number of requested generation tokens.",
		Buckets: tokenBuckets,
	}, modelLabel)
	return m.register(m.maxNumGenerationTokens, "prometheus max_num_generation_tokens histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterReqParamsMaxTokensHistogram(tokenBuckets []float64) error {
	m.requestParamsMaxTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMParamMaxTokensMetricName,
		Help:    "Histogram of the max_tokens request parameter.",
		Buckets: tokenBuckets,
	}, modelLabel)
	return m.register(m.requestParamsMaxTokens, "prometheus request_params_max_tokens histogram register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterPromptTokensTotalCounter() error {
	m.promptTokensTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMPromptTokensTotalMetricName,
		Help: "Total number of prompt tokens processed.",
	}, modelLabel)
	return m.register(m.promptTokensTotal, "prometheus prompt_tokens_total counter register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterGenerationTokensTotalCounter() error {
	m.generationTokensTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMGenerationTokensTotalMetricName,
		Help: "Total number of generated tokens.",
	}, modelLabel)
	return m.register(m.generationTokensTotal, "prometheus generation_tokens_total counter register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterRequestSuccessTotalCounter() error {
	m.requestSuccessTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMSuccessTotalMetricName,
		Help: "Count of successfully processed requests.",
	}, []string{api.PromLabelModelName, api.PromLabelFinishReason})
	return m.register(m.requestSuccessTotal, "prometheus request_success_total counter register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterPrefixCacheHitsTotalCounter() error {
	m.prefixCacheHitsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMPrefixCacheHitsTotalMetricName,
		Help: "Prefix cache hits, in terms of number of cached tokens.",
	}, modelLabel)
	return m.register(m.prefixCacheHitsTotal, "prometheus prefix_cache_hits_total counter register failed")
}

func (m *vllmMetricsAdapter) createAndRegisterPrefixCacheQueriesTotalCounter() error {
	m.prefixCacheQueriesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMPrefixCacheQueriesTotalMetricName,
		Help: "Prefix cache queries, in terms of number of queried tokens.",
	}, modelLabel)
	return m.register(m.prefixCacheQueriesTotal, "prometheus prefix_cache_queries_total counter register failed")
}

// register registers c with the bus's Prometheus registry, logging errMsg
// on failure.
func (m *vllmMetricsAdapter) register(c prometheus.Collector, errMsg string) error {
	if err := m.registry.Register(c); err != nil {
		m.logger.Error(err, errMsg)
		return err
	}
	return nil
}

// setInitialValues zeroes the request/kv-cache gauges and stamps the
// one-shot cache_config_info series so the first scrape has a consistent
// baseline. lora_requests_info has no baseline: like vLLM, it first appears
// with the first LoRA event.
func (m *vllmMetricsAdapter) setInitialValues() {
	if m.fake == nil {
		m.runningRequests.WithLabelValues(m.config.DisplayModelName).Set(0)
		m.waitingRequests.WithLabelValues(m.config.DisplayModelName).Set(0)
		m.kvCacheUsagePercentage.WithLabelValues(m.config.DisplayModelName).Set(0)
	}
	m.cacheConfig.WithLabelValues(
		strconv.Itoa(m.config.KVCache.TokenBlockSize),
		m.config.KVCache.KVCacheDType,
		"0",
		strconv.Itoa(m.config.KVCache.KVCacheSize),
	).Set(1)
}

// -------- Fake Metrics ------

// resolveTokenTotal returns the target absolute value for a token counter
// paired with a histogram: explicit wins when set, else the sum of the
// histogram Samples.
func resolveTokenTotal(buckets []float64, samples []int, explicit *float64) *float64 {
	if explicit != nil {
		return explicit
	}

	return metrics.InitFakeHistogram(nil, buckets, samples)
}

// ApplyFakeMetricsUpdate narrows the engine-owned fake-metrics configuration
// to vLLM's concrete type and applies it. A configuration belonging to another
// engine, or one arriving while this adapter reports real metrics, is logged
// and dropped.
func (m *vllmMetricsAdapter) ApplyFakeMetricsUpdate(update common.FakeMetrics) {
	vllmUpdate, ok := update.(*VLLMFakeMetrics)
	if !ok || vllmUpdate == nil {
		m.logger.Error(fmt.Errorf("unexpected fake-metrics configuration type %T", update),
			"ignoring fake-metrics update")
		return
	}
	if m.fake == nil {
		// Applying it would stamp fake values while the event handlers, which
		// gate on fake being set, keep reporting real ones, and the two are
		// never reported together. Configuration.Update refuses a body carrying
		// fake metrics in this state, so this is reached only if that gate is
		// bypassed.
		m.logger.Error(errors.New("the simulator is reporting real metrics"),
			"ignoring fake-metrics update")
		return
	}
	m.applyFakeMetrics(vllmUpdate)
}

// applyFakeMetrics enqueues the update on the per-metric channels and returns.
// The updater goroutines clear and repopulate each collector's series.
func (m *vllmMetricsAdapter) applyFakeMetrics(update *VLLMFakeMetrics) {
	m.generators.Apply(func(gens *metrics.GeneratorSet) {
		if update.RunningRequests != nil {
			gens.Set(VLLMReqRunningMetricName, update.RunningRequests, func(upd metrics.GaugeUpdate) {
				common.WriteToChannel(m.runReqChan, upd, m.logger)
			}, true)
		}
		if update.WaitingRequests != nil {
			gens.Set(VLLMReqWaitingMetricName, update.WaitingRequests, func(upd metrics.GaugeUpdate) {
				common.WriteToChannel(m.waitingReqChan, upd, m.logger)
			}, true)
		}
		if update.KVCacheUsagePercentage != nil {
			gens.Set(VLLMKVCacheUsageMetricName, update.KVCacheUsagePercentage, func(upd metrics.GaugeUpdate) {
				common.WriteToChannel(m.kvCacheUsageChan, upd, m.logger)
			}, false)
		}
	})

	if update.TTFTBucketValues != nil {
		common.WriteToChannel(m.ttftChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: common.TTFTBucketsBoundaries, Samples: update.TTFTBucketValues}}, m.logger)
	}
	if update.TPOTBucketValues != nil {
		common.WriteToChannel(m.perTokenLatencyChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: common.TPOTBucketsBoundaries, Samples: update.TPOTBucketValues}}, m.logger)
	}
	if update.E2ERequestLatencyBucketValues != nil {
		common.WriteToChannel(m.e2eReqLatencyChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.E2ERequestLatencyBucketValues}}, m.logger)
	}
	if update.ReqQueueTimeBucketValues != nil {
		common.WriteToChannel(m.reqQueueTimeChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqQueueTimeBucketValues}}, m.logger)
	}
	if update.ReqInfTimeBucketValues != nil {
		common.WriteToChannel(m.reqInferenceTimeChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqInfTimeBucketValues}}, m.logger)
	}
	if update.ReqPrefillTimeBucketValues != nil {
		common.WriteToChannel(m.reqPrefillTimeChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqPrefillTimeBucketValues}}, m.logger)
	}
	if update.ReqDecodeTimeBucketValues != nil {
		common.WriteToChannel(m.reqDecodeTimeChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqDecodeTimeBucketValues}}, m.logger)
	}
	if update.ReqTPOTBucketValues != nil {
		common.WriteToChannel(m.reqTpotChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: common.TPOTBucketsBoundaries, Samples: update.ReqTPOTBucketValues}}, m.logger)
	}

	tokenBuckets := metrics.BuildBuckets(m.config.MaxModelLen, TokenBucketMantissas)

	if update.RequestParamsMaxTokens != nil {
		common.WriteToChannel(m.requestParamsMaxTokensChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: tokenBuckets, Samples: update.RequestParamsMaxTokens}}, m.logger)
	}
	if update.RequestMaxGenerationTokens != nil {
		common.WriteToChannel(m.maxNumGenerationTokensChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: tokenBuckets, Samples: update.RequestMaxGenerationTokens}}, m.logger)
	}

	// update histogram of the prompt tokens
	if update.RequestPromptTokens != nil {
		common.WriteToChannel(m.requestPromptTokensChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: tokenBuckets, Samples: update.RequestPromptTokens}}, m.logger)
	}
	// update the total prompt tokens counter according the histogram (if the total is not provided) or
	// according to the explicit total (if provided)
	if update.RequestPromptTokens != nil || update.TotalPromptTokens != nil {
		total := resolveTokenTotal(tokenBuckets, update.RequestPromptTokens, metrics.Int64ToFloat64(update.TotalPromptTokens))
		common.WriteToChannel(m.promptTokensTotalChan, metrics.CounterUpdate{Reset: &metrics.CounterReset{Value: total}}, m.logger)
	}

	if update.RequestGenerationTokens != nil {
		common.WriteToChannel(m.requestGenerationTokensChan, metrics.HistogramUpdate{Reset: &metrics.HistogramReset{Buckets: tokenBuckets, Samples: update.RequestGenerationTokens}}, m.logger)
	}
	if update.RequestGenerationTokens != nil || update.TotalGenerationTokens != nil {
		total := resolveTokenTotal(tokenBuckets, update.RequestGenerationTokens, metrics.Int64ToFloat64(update.TotalGenerationTokens))
		common.WriteToChannel(m.generationTokensTotalChan, metrics.CounterUpdate{Reset: &metrics.CounterReset{Value: total}}, m.logger)
	}

	if update.PrefixCacheQueries != nil {
		common.WriteToChannel(m.prefixCacheQueriesTotalChan, metrics.CounterUpdate{Reset: &metrics.CounterReset{Value: metrics.Int64ToFloat64(update.PrefixCacheQueries)}}, m.logger)
	}
	if update.PrefixCacheHits != nil {
		common.WriteToChannel(m.prefixCacheHitsTotalChan, metrics.CounterUpdate{Reset: &metrics.CounterReset{Value: metrics.Int64ToFloat64(update.PrefixCacheHits)}}, m.logger)
	}

	if update.RequestSuccessTotal != nil {
		common.WriteToChannel(m.requestSuccessTotalChan, requestSuccessCounterUpdate{Reset: &successTotalReset{Reasons: update.RequestSuccessTotal}}, m.logger)
	}

	if update.LoraMetrics != nil {
		common.WriteToChannel(m.lorasChan, loraUpdate{Reset: &loraReset{MaxLoRAs: m.config.Lora.MaxLoras, Entries: update.LoraMetrics}}, m.logger)
	}

}
