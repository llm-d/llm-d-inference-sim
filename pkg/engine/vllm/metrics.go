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
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"

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
	VLLMMaxNumGenerationTokensMetricName  = "vllm:max_num_generation_tokens"
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

// GaugeUpdate is a gauge-channel payload: either an Add delta from a real
// event or a Reset to an absolute fake-metrics value. Exactly one variant
// must be non-nil.
type GaugeUpdate struct {
	Add   *float64
	Reset *float64
}

// HistogramUpdate is a histogram-channel payload: either a single Observe
// or a Reset to a target bucket state. Exactly one variant must be non-nil.
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

// SuccessTotalReset is the target state for request_success_total: one
// Add per (reason, count) pair after recreate.
type SuccessTotalReset struct {
	Reasons map[string]int64
}

// RequestSuccessCounterUpdate is a requestSuccessTotalChan payload: an
// Increment with the finish-reason label, or a Reset to a per-reason target
// map. Exactly one field is non-nil.
type RequestSuccessCounterUpdate struct {
	Increment *string
	Reset     *SuccessTotalReset
}

// LoRAUpdate is the discriminated-union payload for lorasChan. Event-side
// producers set Snapshot with the current per-LoRA counts; the fake-metrics
// applier sets Reset.
type LoRAUpdate struct {
	Snapshot *metrics.LoRASetsChanged
	Reset    *LoRAReset
}

// LoRAReset carries the target state for lora_requests_info: unregister,
// recreate, then stamp one series per entry (or a single zero-adapter row
// with the current timestamp when Entries is empty).
type LoRAReset struct {
	MaxLoRAs int
	Entries  []common.LorasMetrics
}

// activeGenerator is one fake-metric generator bound to the gauge channel it
// feeds. The ticker evaluates fn on every refresh and pushes the result.
type activeGenerator struct {
	fn         metrics.Generator
	params     *common.FunctionInfo
	roundToInt bool
	updateFunc func(upd GaugeUpdate)
}

// VLLMMetricsAdapter implements metrics.EngineMetricsAdapter and produces the
// vLLM Prometheus surface. The bus dispatches drained events to the On<Event>
// handlers, which fan out to per-metric channels, written to Prometheus by one
// updater goroutine per metric.
type VLLMMetricsAdapter struct {
	logger logr.Logger
	config common.Configuration
	// fake is config.FakeMetrics narrowed to vLLM's concrete type, nil when
	// fake metrics are off or another engine owns the configuration.
	fake *VLLMFakeMetrics
	ctx  context.Context

	registry *prometheus.Registry

	// genMu guards the fake-metrics generator set and ticker lifecycle:
	// generators, started, tickerRunning, tickerCancel, tickerStart. Held
	// briefly by ApplyFakeMetricsUpdate, Start, Close, and tick (snapshot
	// only) so the ticker goroutine and admin-driven updates cannot race on
	// the map.
	genMu         sync.Mutex
	generators    map[string]activeGenerator
	started       bool
	tickerRunning bool
	tickerCancel  context.CancelFunc
	tickerStart   time.Time

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
	runReqChan       common.Channel[GaugeUpdate]
	waitingReqChan   common.Channel[GaugeUpdate]
	kvCacheUsageChan common.Channel[GaugeUpdate]

	ttftChan                    common.Channel[HistogramUpdate]
	perTokenLatencyChan         common.Channel[HistogramUpdate]
	e2eReqLatencyChan           common.Channel[HistogramUpdate]
	reqQueueTimeChan            common.Channel[HistogramUpdate]
	reqInferenceTimeChan        common.Channel[HistogramUpdate]
	reqPrefillTimeChan          common.Channel[HistogramUpdate]
	reqDecodeTimeChan           common.Channel[HistogramUpdate]
	reqTpotChan                 common.Channel[HistogramUpdate]
	requestPromptTokensChan     common.Channel[HistogramUpdate]
	requestGenerationTokensChan common.Channel[HistogramUpdate]
	maxNumGenerationTokensChan  common.Channel[HistogramUpdate]
	requestParamsMaxTokensChan  common.Channel[HistogramUpdate]

	promptTokensTotalChan       common.Channel[CounterUpdate]
	generationTokensTotalChan   common.Channel[CounterUpdate]
	prefixCacheHitsTotalChan    common.Channel[CounterUpdate]
	prefixCacheQueriesTotalChan common.Channel[CounterUpdate]

	lorasChan               common.Channel[LoRAUpdate]
	requestSuccessTotalChan common.Channel[RequestSuccessCounterUpdate]

	// nWaitingReqs / nRunningReqs are the adapter-local counters mirrored
	// onto the num_requests_{waiting,running} gauges.
	nWaitingReqs int64
	nRunningReqs int64
}

// NewMetricsAdapter returns the vLLM metrics adapter.
func (Engine) NewMetricsAdapter(ctx context.Context, registry *prometheus.Registry,
	logger logr.Logger, config common.Configuration) (metrics.MetricsAdapter, error) {
	m, err := newMetricsAdapter(ctx, registry, logger, config)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// newMetricsAdapter registers the Prometheus collectors on registry, stamps
// initial values, and spawns the per-metric updater goroutines. ctx must match
// the one passed to metrics.NewMetricsBus. Returns an error if any collector
// fails to register.
func newMetricsAdapter(ctx context.Context, registry *prometheus.Registry,
	logger logr.Logger, config common.Configuration) (*VLLMMetricsAdapter, error) {
	m := &VLLMMetricsAdapter{
		logger:     logger,
		registry:   registry,
		config:     config,
		generators: make(map[string]activeGenerator),
		ctx:        ctx,
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

func (m *VLLMMetricsAdapter) Close() error {
	m.genMu.Lock()
	defer m.genMu.Unlock()
	m.stopTickerLocked()
	m.started = false

	return nil
}

// Start applies the initial fake-metrics configuration and starts the
// generator ticker. The bus has already subscribed the On<Event> handlers to
// its channels by the time this runs.
func (m *VLLMMetricsAdapter) Start(_ context.Context) error {
	if m.fake != nil {
		fm := *m.fake
		if fm.LoraMetrics == nil {
			fm.LoraMetrics = []common.LorasMetrics{}
		}
		m.applyFakeMetrics(&fm)
	}

	m.genMu.Lock()
	defer m.genMu.Unlock()
	m.started = true
	if len(m.generators) > 0 && !m.tickerRunning {
		m.startTickerLocked()
	}

	return nil
}

// -- Event handlers  -------------------

func (m *VLLMMetricsAdapter) OnRequestReceived(_ metrics.RequestReceived) {
	// State marker; no exposed metric today.
}

func (m *VLLMMetricsAdapter) OnRequestRejected(_ metrics.RequestRejected) {
	// State marker; no exposed metric today.
}

// request queued
// - update number of waiting requests
func (m *VLLMMetricsAdapter) OnRequestQueued(ev metrics.RequestQueued) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.waitingReqChan, gaugeAdd(1), m.logger)
}

// request dequeued
// - update number of waiting requests
// - update queue time histogram
func (m *VLLMMetricsAdapter) OnRequestDequeued(ev metrics.RequestDequeued) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.waitingReqChan, gaugeAdd(-1), m.logger)
	common.WriteToChannel(m.reqQueueTimeChan, observation(ev.QueueTime), m.logger)
}

// request running
// - update number of running requests
func (m *VLLMMetricsAdapter) OnRequestRunning(ev metrics.RequestRunning) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.runReqChan, gaugeAdd(1), m.logger)
}

// prefill started
func (m *VLLMMetricsAdapter) OnPrefillStarted(_ metrics.PrefillStarted) {
	// State marker.
}

// prefill step ended
// - update prefill time histogram
// - update TTFT histogram
func (m *VLLMMetricsAdapter) OnPrefillEnded(ev metrics.PrefillEnded) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.reqPrefillTimeChan, observation(ev.PrefillDuration), m.logger)
	common.WriteToChannel(m.ttftChan, observation(ev.PrefillDuration), m.logger)
}

func (m *VLLMMetricsAdapter) OnDecodeStarted(_ metrics.DecodeStarted) {
	// State marker.
}

// token generated
// - update tpot and itl histograms
func (m *VLLMMetricsAdapter) OnTokenGenerated(ev metrics.TokenGenerated) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.perTokenLatencyChan, observation(ev.InterTokenLatency), m.logger)
}

// decode ended
// - update decode time histogram
// - update requests tpot histogram
func (m *VLLMMetricsAdapter) OnDecodeEnded(ev metrics.DecodeEnded) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.reqDecodeTimeChan, observation(ev.DecodeDuration), m.logger)

	if ev.GenerationTokens > 0 {
		common.WriteToChannel(m.reqTpotChan, observation(ev.DecodeDuration/float64(ev.GenerationTokens)), m.logger)
	}
}

// request processing finished successfully - update all relevant metrics
func (m *VLLMMetricsAdapter) OnRequestSucceeded(ev metrics.RequestSucceeded) {
	if m.fake != nil {
		return
	}

	// update number of successful requests per finish reason
	common.WriteToChannel(m.requestSuccessTotalChan, RequestSuccessCounterUpdate{Increment: &ev.FinishReason}, m.logger)

	// request finished successfully, update number of prompt and generated tokens
	// both total and histogram metrics
	common.WriteToChannel(m.requestPromptTokensChan, observation(float64(ev.PromptTokens)), m.logger)
	common.WriteToChannel(m.requestGenerationTokensChan, observation(float64(ev.GenerationTokens)), m.logger)
	promptTokens := float64(ev.PromptTokens)
	generationTokens := float64(ev.GenerationTokens)
	common.WriteToChannel(m.promptTokensTotalChan, CounterUpdate{Add: &promptTokens}, m.logger)
	common.WriteToChannel(m.generationTokensTotalChan, CounterUpdate{Add: &generationTokens}, m.logger)

	// if max_tokens is set, update the request_params_max_tokens histogram
	if ev.MaxTokens != nil {
		common.WriteToChannel(m.requestParamsMaxTokensChan, observation(float64(*ev.MaxTokens)), m.logger)
	}
	if maxGenTokens, err := common.MaxIntSlice(ev.GenTokensPerChoice); err == nil {
		common.WriteToChannel(m.maxNumGenerationTokensChan, observation(float64(maxGenTokens)), m.logger)
	}

	common.WriteToChannel(m.e2eReqLatencyChan, observation(ev.E2ELatency), m.logger)
	common.WriteToChannel(m.reqInferenceTimeChan, observation(ev.InferenceTime), m.logger)

	m.finishRunning()
}

// request processing failed
// - update all relevant metrics
func (m *VLLMMetricsAdapter) OnRequestFailed(ev metrics.RequestFailed) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.e2eReqLatencyChan, observation(ev.E2ELatency), m.logger)
	common.WriteToChannel(m.reqInferenceTimeChan, observation(ev.InferenceTime), m.logger)

	m.finishRunning()
}

// change in kv cache utilization
// - update kv cache usage gauge
func (m *VLLMMetricsAdapter) OnKVCacheUsageChanged(ev metrics.KVCacheUsageChanged) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.kvCacheUsageChan, gaugeAdd(ev.KVCacheUsagePerc), m.logger)
}

// change in prefix cache utilization
// - update prefix cache hits and queries counters
func (m *VLLMMetricsAdapter) OnPrefixCacheQueried(ev metrics.PrefixCacheQueried) {
	if m.fake != nil {
		return
	}
	hit := float64(ev.CachedPromptTokens)
	queried := float64(ev.QueriedTokens)
	common.WriteToChannel(m.prefixCacheHitsTotalChan, CounterUpdate{Add: &hit}, m.logger)
	common.WriteToChannel(m.prefixCacheQueriesTotalChan, CounterUpdate{Add: &queried}, m.logger)
}

// OnLoRASetsChanged receives the per-LoRA waiting/running snapshot produced
// by the bus after each LoRAChanged event and forwards it to the LoRA
// updater goroutine.
func (m *VLLMMetricsAdapter) OnLoRASetsChanged(ev metrics.LoRASetsChanged) {
	if m.fake != nil {
		return
	}
	common.WriteToChannel(m.lorasChan, LoRAUpdate{Snapshot: &ev}, m.logger)
}

// finishRunning decrements the running-request counter for a terminal
// request. LoRA state transitions are handled separately via LoRAChanged.
func (m *VLLMMetricsAdapter) finishRunning() {
	common.WriteToChannel(m.runReqChan, gaugeAdd(-1), m.logger)
}

func (m *VLLMMetricsAdapter) createAndStartPrometheusChannels(ctx context.Context) {
	maxNumberOfRunningRequests, maxNumberOfWaitingRequests, maxNumberOfRequests, maxNumberOfTokens := metrics.ChannelCapacities(m.config)

	m.runReqChan = common.NewChannel[GaugeUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.runReqChan, m.updateRunningRequests)

	m.waitingReqChan = common.NewChannel[GaugeUpdate]("vllm", maxNumberOfWaitingRequests, ctx.Done())
	go common.Subscribe(ctx, m.waitingReqChan, m.updateWaitingRequests)

	m.kvCacheUsageChan = common.NewChannel[GaugeUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.kvCacheUsageChan, m.updateKVCacheUsage)

	m.ttftChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.ttftChan, m.updateTTFT)

	m.perTokenLatencyChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfTokens, ctx.Done())
	go common.Subscribe(ctx, m.perTokenLatencyChan, m.updatePerTokenLatency)

	m.e2eReqLatencyChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.e2eReqLatencyChan, m.updateE2EReqLatency)

	m.reqQueueTimeChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfWaitingRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqQueueTimeChan, m.updateReqQueueTime)

	m.reqInferenceTimeChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqInferenceTimeChan, m.updateReqInferenceTime)

	m.reqPrefillTimeChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqPrefillTimeChan, m.updateReqPrefillTime)

	m.reqDecodeTimeChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqDecodeTimeChan, m.updateReqDecodeTime)

	m.reqTpotChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.reqTpotChan, m.updateReqTpot)

	m.lorasChan = common.NewChannel[LoRAUpdate]("vllm", maxNumberOfRequests, ctx.Done())
	go common.Subscribe(ctx, m.lorasChan, m.updateLoRAs)

	m.requestPromptTokensChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestPromptTokensChan, m.updateRequestPromptTokens)

	m.requestGenerationTokensChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestGenerationTokensChan, m.updateRequestGenerationTokens)

	m.maxNumGenerationTokensChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.maxNumGenerationTokensChan, m.updateMaxNumGenerationTokens)

	m.requestParamsMaxTokensChan = common.NewChannel[HistogramUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestParamsMaxTokensChan, m.updateRequestParamsMaxTokens)

	m.promptTokensTotalChan = common.NewChannel[CounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.promptTokensTotalChan, m.updatePromptTokensTotal)

	m.generationTokensTotalChan = common.NewChannel[CounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.generationTokensTotalChan, m.updateGenerationTokensTotal)

	m.requestSuccessTotalChan = common.NewChannel[RequestSuccessCounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.requestSuccessTotalChan, m.updateRequestSuccessTotal)

	m.prefixCacheHitsTotalChan = common.NewChannel[CounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.prefixCacheHitsTotalChan, m.updatePrefixCacheHitsTotal)

	m.prefixCacheQueriesTotalChan = common.NewChannel[CounterUpdate]("vllm", maxNumberOfRunningRequests, ctx.Done())
	go common.Subscribe(ctx, m.prefixCacheQueriesTotalChan, m.updatePrefixCacheQueriesTotal)
}

// -- Per-metric updaters --------------------------------------------------

func (m *VLLMMetricsAdapter) updateRequestPromptTokens(upd HistogramUpdate) {
	m.updateHistogram(&m.requestPromptTokens, m.createAndRegisterReqPromptTokensHistogram, upd)
}

func (m *VLLMMetricsAdapter) updateRequestGenerationTokens(upd HistogramUpdate) {
	m.updateHistogram(&m.requestGenerationTokens, m.createAndRegisterReqGenerationTokensHistogram, upd)
}

func (m *VLLMMetricsAdapter) updateMaxNumGenerationTokens(upd HistogramUpdate) {
	m.updateHistogram(&m.maxNumGenerationTokens, m.createAndRegisterMaxNumGenerationTokensHistogram, upd)
}

func (m *VLLMMetricsAdapter) updateRequestParamsMaxTokens(upd HistogramUpdate) {
	m.updateHistogram(&m.requestParamsMaxTokens, m.createAndRegisterReqParamsMaxTokensHistogram, upd)
}

func (m *VLLMMetricsAdapter) updatePromptTokensTotal(upd CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.promptTokensTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.resetCounter(&m.promptTokensTotal, m.createAndRegisterPromptTokensTotalCounter,
			m.config.DisplayModelName, upd.Reset.Value)
	}
}

func (m *VLLMMetricsAdapter) updateGenerationTokensTotal(upd CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.generationTokensTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.resetCounter(&m.generationTokensTotal, m.createAndRegisterGenerationTokensTotalCounter,
			m.config.DisplayModelName, upd.Reset.Value)
	}
}

func (m *VLLMMetricsAdapter) updateRequestSuccessTotal(upd RequestSuccessCounterUpdate) {
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

func (m *VLLMMetricsAdapter) updatePrefixCacheHitsTotal(upd CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.prefixCacheHitsTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.resetCounter(&m.prefixCacheHitsTotal, m.createAndRegisterPrefixCacheHitsTotalCounter,
			m.config.DisplayModelName, upd.Reset.Value)
	}
}

func (m *VLLMMetricsAdapter) updatePrefixCacheQueriesTotal(upd CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.prefixCacheQueriesTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.resetCounter(&m.prefixCacheQueriesTotal, m.createAndRegisterPrefixCacheQueriesTotalCounter,
			m.config.DisplayModelName, upd.Reset.Value)
	}
}

// observation wraps a single Observe value for a histogram-family channel.
func observation(v float64) HistogramUpdate {
	return HistogramUpdate{Observe: &v}
}

// gaugeAdd wraps a real-event delta for a gauge channel.
func gaugeAdd(v float64) GaugeUpdate {
	return GaugeUpdate{Add: &v}
}

// gaugeReset wraps an absolute fake-metrics value for a gauge channel.
func gaugeReset(v float64) GaugeUpdate {
	return GaugeUpdate{Reset: &v}
}

// updateHistogram records an Observe, or on Reset unregisters the
// collector, recreates it, and replays the target bucket state.
func (m *VLLMMetricsAdapter) updateHistogram(histPP **prometheus.HistogramVec, recreate func() error, upd HistogramUpdate) {
	switch {
	case upd.Observe != nil:
		if m.fake != nil {
			return
		}
		if *histPP != nil {
			(*histPP).WithLabelValues(m.config.DisplayModelName).Observe(*upd.Observe)
		}
	case upd.Reset != nil:
		m.registry.Unregister(*histPP)
		if err := recreate(); err != nil {
			m.logger.Error(err, "failed to recreate histogram during fake-metrics reset")
			return
		}
		metrics.InitFakeHistogram(*histPP, m.config.DisplayModelName, upd.Reset.Buckets, upd.Reset.Samples)
	}
}

func (m *VLLMMetricsAdapter) updateWaitingRequests(upd GaugeUpdate) {
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

func (m *VLLMMetricsAdapter) updateRunningRequests(upd GaugeUpdate) {
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

func (m *VLLMMetricsAdapter) updateKVCacheUsage(upd GaugeUpdate) {
	switch {
	case upd.Add != nil && m.fake == nil:
		m.reportKVCacheUsage(*upd.Add)
	case upd.Reset != nil && m.fake != nil:
		m.reportKVCacheUsage(*upd.Reset)
	}
}

func (m *VLLMMetricsAdapter) updateTTFT(upd HistogramUpdate) {
	m.updateHistogram(&m.ttft, m.createAndRegisterTTFTHistogram, upd)
}

func (m *VLLMMetricsAdapter) updatePerTokenLatency(upd HistogramUpdate) {
	m.updateHistogram(&m.tpot, m.createAndRegisterTPOTHistogram, upd)
	m.updateHistogram(&m.interTokenLatency, m.createAndRegisterInterTokenLatencyHistogram, upd)
}

func (m *VLLMMetricsAdapter) updateE2EReqLatency(upd HistogramUpdate) {
	m.updateHistogram(&m.e2eReqLatency, m.createAndRegisterE2EReqLatencyHistogram, upd)
}

func (m *VLLMMetricsAdapter) updateReqQueueTime(upd HistogramUpdate) {
	m.updateHistogram(&m.reqQueueTime, m.createAndRegisterReqQueueTimeHistogram, upd)
}

func (m *VLLMMetricsAdapter) updateReqInferenceTime(upd HistogramUpdate) {
	m.updateHistogram(&m.reqInferenceTime, m.createAndRegisterReqInferenceTimeHistogram, upd)
}

func (m *VLLMMetricsAdapter) updateReqPrefillTime(upd HistogramUpdate) {
	m.updateHistogram(&m.reqPrefillTime, m.createAndRegisterReqPrefillTimeHistogram, upd)
}

func (m *VLLMMetricsAdapter) updateReqDecodeTime(upd HistogramUpdate) {
	m.updateHistogram(&m.reqDecodeTime, m.createAndRegisterReqDecodeTimeHistogram, upd)
}

func (m *VLLMMetricsAdapter) updateReqTpot(upd HistogramUpdate) {
	m.updateHistogram(&m.reqTpot, m.createAndRegisterReqTpotHistogram, upd)
}

// lorasUpdater republishes lora_requests_info from Snapshot events, or on
// Reset recreates the collector and stamps the supplied entries.
func (m *VLLMMetricsAdapter) updateLoRAs(upd LoRAUpdate) {
	switch {
	case upd.Snapshot != nil:
		m.reportLoras(*upd.Snapshot)
	case upd.Reset != nil:
		m.resetLoRA(upd.Reset)
	}
}

// resetCollector unregisters current, recreates it via recreate, then calls
// populate to stamp the recreated collector's series. Called only from
// updater goroutines.
func (m *VLLMMetricsAdapter) resetCollector(current prometheus.Collector, recreate func() error, errMsg string, populate func()) {
	m.registry.Unregister(current)
	if err := recreate(); err != nil {
		m.logger.Error(err, errMsg)
		return
	}
	populate()
}

// resetCounter resets *counterPP and records the target value via a
// single Add. A nil value leaves the recreated counter with no series
// stamped, so it reads as absent from /metrics.
func (m *VLLMMetricsAdapter) resetCounter(counterPP **prometheus.CounterVec, recreate func() error, modelName string, value *float64) {
	m.resetCollector(*counterPP, recreate, "failed to recreate counter during fake-metrics reset", func() {
		if value != nil {
			(*counterPP).WithLabelValues(modelName).Add(*value)
		}
	})
}

// resetSuccessTotal resets requestSuccessTotal, then Adds each
// (reason, count) pair. Nil or empty reasons leaves the recreated counter
// with no series stamped.
func (m *VLLMMetricsAdapter) resetSuccessTotal(reasons map[string]int64) {
	m.resetCollector(m.requestSuccessTotal, m.createAndRegisterRequestSuccessTotalCounter, "failed to recreate request_success_total counter during fake-metrics reset", func() {
		for reason, count := range reasons {
			m.requestSuccessTotal.WithLabelValues(m.config.DisplayModelName, reason).Add(float64(count))
		}
	})
}

// resetLoRA resets loraInfo, then stamps one series per entry. Empty
// entries emits a single zero-adapter row with the current timestamp
// (matching the fake-metrics default).
func (m *VLLMMetricsAdapter) resetLoRA(reset *LoRAReset) {
	m.resetCollector(m.loraInfo, m.createAndRegisterLoraInfoGauge, "failed to recreate lora_requests_info gauge during fake-metrics reset", func() {
		if len(reset.Entries) == 0 {
			m.loraInfo.WithLabelValues(
				strconv.Itoa(reset.MaxLoRAs),
				"",
				"",
			).Set(float64(time.Now().Unix()))
			return
		}
		for _, entry := range reset.Entries {
			m.loraInfo.WithLabelValues(
				strconv.Itoa(reset.MaxLoRAs),
				entry.RunningLoras,
				entry.WaitingLoras,
			).Set(entry.Timestamp)
		}
	})
}

// -- Report helpers (Prometheus writes) -------------------------------------

func (m *VLLMMetricsAdapter) reportRunningRequests() {
	if m.runningRequests != nil {
		m.runningRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nRunningReqs))
	}
}

func (m *VLLMMetricsAdapter) reportWaitingRequests() {
	if m.waitingRequests != nil {
		m.waitingRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nWaitingReqs))
	}
}

func (m *VLLMMetricsAdapter) reportKVCacheUsage(value float64) {
	if m.kvCacheUsagePercentage != nil {
		m.kvCacheUsagePercentage.WithLabelValues(m.config.DisplayModelName).Set(value)
	}
}

func (m *VLLMMetricsAdapter) reportLoras(snap metrics.LoRASetsChanged) {
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
	).Set(float64(time.Now().Unix()))
}

// -- Prometheus wiring ------------------------------------------------------

// buildMetrics constructs and registers all Prometheus collectors. It is called once during adapter construction.
func (m *VLLMMetricsAdapter) buildMetrics() error {
	if err := m.createAndRegisterRunningRequestsGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterWaitingRequestsGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterKVCacheUsageGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterLoraInfoGauge(); err != nil {
		return err
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
	if err := m.createAndRegisterReqPromptTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqGenerationTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterMaxNumGenerationTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqParamsMaxTokensHistogram(); err != nil {
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

// register registers c with the bus's Prometheus registry, logging errMsg
// on failure.
func (m *VLLMMetricsAdapter) register(c prometheus.Collector, errMsg string) error {
	if err := m.registry.Register(c); err != nil {
		m.logger.Error(err, errMsg)
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterRunningRequestsGauge() error {
	m.runningRequests = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMReqRunningMetricName,
		Help: "Number of requests currently running on GPU.",
	}, modelLabel)
	return m.register(m.runningRequests, "prometheus number of running requests gauge register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterWaitingRequestsGauge() error {
	m.waitingRequests = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMReqWaitingMetricName,
		Help: "Prometheus metric for the number of queued requests.",
	}, modelLabel)
	return m.register(m.waitingRequests, "prometheus number of requests in queue gauge register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterKVCacheUsageGauge() error {
	m.kvCacheUsagePercentage = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMKVCacheUsageMetricName,
		Help: "Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).",
	}, modelLabel)
	return m.register(m.kvCacheUsagePercentage, "prometheus kv cache usage percentage gauge register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterLoraInfoGauge() error {
	m.loraInfo = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMLoRARequestsMetricName,
		Help: "Running stats on lora requests.",
	}, []string{api.PromLabelMaxLora, api.PromLabelRunningLoraAdapters, api.PromLabelWaitingLoraAdapters})
	return m.register(m.loraInfo, "prometheus lora info gauge register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterCacheConfigGauge() error {
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

func (m *VLLMMetricsAdapter) createAndRegisterTTFTHistogram() error {
	m.ttft = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMTTFTMetricName,
		Help:    "Histogram of time to first token in seconds.",
		Buckets: common.TTFTBucketsBoundaries,
	}, modelLabel)
	return m.register(m.ttft, "prometheus time to first token histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterTPOTHistogram() error {
	m.tpot = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMTPOTMetricName,
		Help:    "Histogram of time per output token in seconds.",
		Buckets: common.TPOTBucketsBoundaries,
	}, modelLabel)
	return m.register(m.tpot, "prometheus time per output token histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterInterTokenLatencyHistogram() error {
	m.interTokenLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMInterTokenLatencyMetricName,
		Help:    "Histogram of inter-token latency in seconds.",
		Buckets: common.TPOTBucketsBoundaries,
	}, modelLabel)
	return m.register(m.interTokenLatency, "prometheus inter-token latency histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqTpotHistogram() error {
	m.reqTpot = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMReqTPOTMetricName,
		Help:    "Histogram of time_per_output_token_seconds per request.",
		Buckets: common.TPOTBucketsBoundaries,
	}, modelLabel)
	return m.register(m.reqTpot, "prometheus time_per_output_token_seconds per request histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterE2EReqLatencyHistogram() error {
	m.e2eReqLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLME2EReqLatencyMetricName,
		Help:    "Histogram of end to end request latency in seconds.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.e2eReqLatency, "prometheus e2e request latency histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqQueueTimeHistogram() error {
	m.reqQueueTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMReqQueueTimeMetricName,
		Help:    "Histogram of time spent in WAITING phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.reqQueueTime, "prometheus request queue time histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqInferenceTimeHistogram() error {
	m.reqInferenceTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMReqInferenceTimeMetricName,
		Help:    "Histogram of time spent in RUNNING phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.reqInferenceTime, "prometheus request inference time histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqPrefillTimeHistogram() error {
	m.reqPrefillTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMPrefillTimeMetricName,
		Help:    "Histogram of time spent in PREFILL phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.reqPrefillTime, "prometheus request prefill time histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqDecodeTimeHistogram() error {
	m.reqDecodeTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMDecodeTimeMetricName,
		Help:    "Histogram of time spent in DECODE phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	return m.register(m.reqDecodeTime, "prometheus request decode time histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqPromptTokensHistogram() error {
	m.requestPromptTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMPromptTokensMetricName,
		Help:    "Number of prefill tokens processed.",
		Buckets: metrics.BuildBuckets(m.config.MaxModelLen, TokenBucketMantissas),
	}, modelLabel)
	return m.register(m.requestPromptTokens, "prometheus request_prompt_tokens histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqGenerationTokensHistogram() error {
	m.requestGenerationTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMGenerationTokensMetricName,
		Help:    "Number of generation tokens processed.",
		Buckets: metrics.BuildBuckets(m.config.MaxModelLen, TokenBucketMantissas),
	}, modelLabel)
	return m.register(m.requestGenerationTokens, "prometheus request_generation_tokens histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterMaxNumGenerationTokensHistogram() error {
	m.maxNumGenerationTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMMaxNumGenerationTokensMetricName,
		Help:    "Histogram of maximum number of requested generation tokens.",
		Buckets: metrics.BuildBuckets(m.config.MaxModelLen, TokenBucketMantissas),
	}, modelLabel)
	return m.register(m.maxNumGenerationTokens, "prometheus max_num_generation_tokens histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqParamsMaxTokensHistogram() error {
	m.requestParamsMaxTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMParamMaxTokensMetricName,
		Help:    "Histogram of the max_tokens request parameter.",
		Buckets: metrics.BuildBuckets(m.config.MaxModelLen, TokenBucketMantissas),
	}, modelLabel)
	return m.register(m.requestParamsMaxTokens, "prometheus request_params_max_tokens histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterPromptTokensTotalCounter() error {
	m.promptTokensTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMPromptTokensTotalMetricName,
		Help: "Total number of prompt tokens processed.",
	}, modelLabel)
	return m.register(m.promptTokensTotal, "prometheus prompt_tokens_total counter register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterGenerationTokensTotalCounter() error {
	m.generationTokensTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMGenerationTokensTotalMetricName,
		Help: "Total number of generated tokens.",
	}, modelLabel)
	return m.register(m.generationTokensTotal, "prometheus generation_tokens_total counter register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterRequestSuccessTotalCounter() error {
	m.requestSuccessTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMSuccessTotalMetricName,
		Help: "Count of successfully processed requests.",
	}, []string{api.PromLabelModelName, api.PromLabelFinishReason})
	return m.register(m.requestSuccessTotal, "prometheus request_success_total counter register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterPrefixCacheHitsTotalCounter() error {
	m.prefixCacheHitsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMPrefixCacheHitsTotalMetricName,
		Help: "Prefix cache hits, in terms of number of cached tokens.",
	}, modelLabel)
	return m.register(m.prefixCacheHitsTotal, "prometheus prefix_cache_hits_total counter register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterPrefixCacheQueriesTotalCounter() error {
	m.prefixCacheQueriesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMPrefixCacheQueriesTotalMetricName,
		Help: "Prefix cache queries, in terms of number of queried tokens.",
	}, modelLabel)
	return m.register(m.prefixCacheQueriesTotal, "prometheus prefix_cache_queries_total counter register failed")
}

// setInitialValues zeroes the request/kv-cache gauges and stamps the
// one-shot cache_config_info and lora_requests_info series so the first
// scrape has a consistent baseline.
func (m *VLLMMetricsAdapter) setInitialValues() {
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
	m.loraInfo.WithLabelValues(
		strconv.Itoa(m.config.Lora.MaxLoras),
		"",
		"",
	).Set(float64(time.Now().Unix()))
}

// -------- Fake Metrics ------
func (m *VLLMMetricsAdapter) updateScalarLocked(key string, fm *common.FakeMetricWithFunction, updateFunc func(upd GaugeUpdate), roundToInt bool) {
	if fm.IsFunction {
		gen := activeGenerator{
			fn:         metrics.Dispatch(fm.Function.Name),
			params:     fm.Function,
			roundToInt: roundToInt,
			updateFunc: updateFunc,
		}
		m.generators[key] = gen
		value := gen.fn(gen.params, 0)
		if roundToInt {
			value = float64(int64(value))
		}
		updateFunc(gaugeReset(value))
		return
	}
	delete(m.generators, key)
	updateFunc(gaugeReset(fm.FixedValue))
}

func float64Ptr(v *int64) *float64 {
	if v == nil {
		return nil
	}
	f := float64(*v)
	return &f
}

// resolveTokenTotal returns the target absolute value for a token counter
// paired with a histogram: explicit wins when set, else the sum of the
// histogram Samples.
func resolveTokenTotal(buckets []float64, samples []int, explicit *float64) *float64 {
	if explicit != nil {
		return explicit
	}

	return metrics.InitFakeHistogram(nil, "", buckets, samples)
}

// ApplyFakeMetricsUpdate narrows the engine-owned fake-metrics configuration
// to vLLM's concrete type and applies it. A configuration belonging to another
// engine is logged and dropped.
func (m *VLLMMetricsAdapter) ApplyFakeMetricsUpdate(update common.FakeMetrics) {
	vllmUpdate, ok := update.(*VLLMFakeMetrics)
	if !ok || vllmUpdate == nil {
		m.logger.Error(fmt.Errorf("unexpected fake-metrics configuration type %T", update),
			"ignoring fake-metrics update")
		return
	}
	m.applyFakeMetrics(vllmUpdate)
}

// applyFakeMetrics enqueues the update on the per-metric channels and returns.
// The updater goroutines perform the collector unregister/recreate, logging and
// skipping any metric that fails to re-register.
func (m *VLLMMetricsAdapter) applyFakeMetrics(update *VLLMFakeMetrics) {
	m.genMu.Lock()
	defer m.genMu.Unlock()
	generatorsWereEmpty := len(m.generators) == 0

	if update.RunningRequests != nil {
		m.updateScalarLocked(VLLMReqRunningMetricName, update.RunningRequests, func(upd GaugeUpdate) {
			common.WriteToChannel(m.runReqChan, upd, m.logger)
		}, true)
	}
	if update.WaitingRequests != nil {
		m.updateScalarLocked(VLLMReqWaitingMetricName, update.WaitingRequests, func(upd GaugeUpdate) {
			common.WriteToChannel(m.waitingReqChan, upd, m.logger)
		}, true)
	}
	if update.KVCacheUsagePercentage != nil {
		m.updateScalarLocked(VLLMKVCacheUsageMetricName, update.KVCacheUsagePercentage, func(upd GaugeUpdate) {
			common.WriteToChannel(m.kvCacheUsageChan, upd, m.logger)
		}, false)
	}

	if update.TTFTBucketValues != nil {
		common.WriteToChannel(m.ttftChan, HistogramUpdate{Reset: &HistogramReset{Buckets: common.TTFTBucketsBoundaries, Samples: update.TTFTBucketValues}}, m.logger)
	}
	if update.TPOTBucketValues != nil {
		common.WriteToChannel(m.perTokenLatencyChan, HistogramUpdate{Reset: &HistogramReset{Buckets: common.TPOTBucketsBoundaries, Samples: update.TPOTBucketValues}}, m.logger)
	}
	if update.E2ERequestLatencyBucketValues != nil {
		common.WriteToChannel(m.e2eReqLatencyChan, HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.E2ERequestLatencyBucketValues}}, m.logger)
	}
	if update.ReqQueueTimeBucketValues != nil {
		common.WriteToChannel(m.reqQueueTimeChan, HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqQueueTimeBucketValues}}, m.logger)
	}
	if update.ReqInfTimeBucketValues != nil {
		common.WriteToChannel(m.reqInferenceTimeChan, HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqInfTimeBucketValues}}, m.logger)
	}
	if update.ReqPrefillTimeBucketValues != nil {
		common.WriteToChannel(m.reqPrefillTimeChan, HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqPrefillTimeBucketValues}}, m.logger)
	}
	if update.ReqDecodeTimeBucketValues != nil {
		common.WriteToChannel(m.reqDecodeTimeChan, HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqDecodeTimeBucketValues}}, m.logger)
	}
	if update.ReqTPOTBucketValues != nil {
		common.WriteToChannel(m.reqTpotChan, HistogramUpdate{Reset: &HistogramReset{Buckets: common.TPOTBucketsBoundaries, Samples: update.ReqTPOTBucketValues}}, m.logger)
	}

	tokenBuckets := metrics.BuildBuckets(m.config.MaxModelLen, TokenBucketMantissas)

	if update.RequestParamsMaxTokens != nil {
		common.WriteToChannel(m.requestParamsMaxTokensChan, HistogramUpdate{Reset: &HistogramReset{Buckets: tokenBuckets, Samples: update.RequestParamsMaxTokens}}, m.logger)
	}
	if update.RequestMaxGenerationTokens != nil {
		common.WriteToChannel(m.maxNumGenerationTokensChan, HistogramUpdate{Reset: &HistogramReset{Buckets: tokenBuckets, Samples: update.RequestMaxGenerationTokens}}, m.logger)
	}

	// update histogram of the propmpt tokens
	if update.RequestPromptTokens != nil {
		common.WriteToChannel(m.requestPromptTokensChan, HistogramUpdate{Reset: &HistogramReset{Buckets: tokenBuckets, Samples: update.RequestPromptTokens}}, m.logger)
	}
	// update the total prompt tokens counter according the histogram (if the total is not provided) or
	// according to the explicit total (if provided)
	if update.RequestPromptTokens != nil || update.TotalPromptTokens != nil {
		total := resolveTokenTotal(tokenBuckets, update.RequestPromptTokens, float64Ptr(update.TotalPromptTokens))
		common.WriteToChannel(m.promptTokensTotalChan, CounterUpdate{Reset: &CounterReset{Value: total}}, m.logger)
	}

	if update.RequestGenerationTokens != nil {
		common.WriteToChannel(m.requestGenerationTokensChan, HistogramUpdate{Reset: &HistogramReset{Buckets: tokenBuckets, Samples: update.RequestGenerationTokens}}, m.logger)
	}
	if update.RequestGenerationTokens != nil || update.TotalGenerationTokens != nil {
		total := resolveTokenTotal(tokenBuckets, update.RequestGenerationTokens, float64Ptr(update.TotalGenerationTokens))
		common.WriteToChannel(m.generationTokensTotalChan, CounterUpdate{Reset: &CounterReset{Value: total}}, m.logger)
	}

	if update.PrefixCacheQueries != nil {
		common.WriteToChannel(m.prefixCacheQueriesTotalChan, CounterUpdate{Reset: &CounterReset{Value: float64Ptr(update.PrefixCacheQueries)}}, m.logger)
	}
	if update.PrefixCacheHits != nil {
		common.WriteToChannel(m.prefixCacheHitsTotalChan, CounterUpdate{Reset: &CounterReset{Value: float64Ptr(update.PrefixCacheHits)}}, m.logger)
	}

	if update.RequestSuccessTotal != nil {
		common.WriteToChannel(m.requestSuccessTotalChan, RequestSuccessCounterUpdate{Reset: &SuccessTotalReset{Reasons: update.RequestSuccessTotal}}, m.logger)
	}

	if update.LoraMetrics != nil {
		common.WriteToChannel(m.lorasChan, LoRAUpdate{Reset: &LoRAReset{MaxLoRAs: m.config.Lora.MaxLoras, Entries: update.LoraMetrics}}, m.logger)
	}

	generatorsAreEmpty := len(m.generators) == 0

	if m.started {
		switch {
		case generatorsWereEmpty && !generatorsAreEmpty:
			m.startTickerLocked()
		case !generatorsWereEmpty && generatorsAreEmpty:
			m.stopTickerLocked()
		}
	}
}

func (m *VLLMMetricsAdapter) startTickerLocked() {
	tickerCtx, cancel := context.WithCancel(m.ctx)
	m.tickerCancel = cancel
	m.tickerRunning = true
	m.tickerStart = time.Now()
	interval := m.config.FakeMetricsRefreshInterval
	go m.runTicker(tickerCtx, interval, m.tickerStart)
}

func (m *VLLMMetricsAdapter) stopTickerLocked() {
	if !m.tickerRunning {
		return
	}
	m.tickerCancel()
	m.tickerCancel = nil
	m.tickerRunning = false
}

func (m *VLLMMetricsAdapter) runTicker(ctx context.Context, interval time.Duration, start time.Time) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.tick(time.Since(start))
		}
	}
}

func (m *VLLMMetricsAdapter) tick(t time.Duration) {
	m.genMu.Lock()
	snapshot := make([]activeGenerator, 0, len(m.generators))
	for _, gen := range m.generators {
		snapshot = append(snapshot, gen)
	}
	m.genMu.Unlock()

	for _, gen := range snapshot {
		value := gen.fn(gen.params, t)
		if gen.roundToInt {
			value = float64(int64(value))
		}
		gen.updateFunc(gaugeReset(value))
	}
}
