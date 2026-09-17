/*
Copyright 2025 The llm-d-inference-simference-sim Authors.

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
	"github.com/llm-d/llm-d-inference-sim/pkg/engine/vllm/fakemetrics"
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
	fake *fakemetrics.Config
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
	logger logr.Logger, config common.Configuration) (metrics.EngineMetricsAdapter, error) {
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
	if fake, ok := config.FakeMetrics.(*fakemetrics.Config); ok {
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

func (m *VLLMMetricsAdapter) createAndStartPrometheusChannels(ctx context.Context) {
	maxNumberOfRunningRequests, maxNumberOfWaitingRequests, maxNumberOfRequests, maxNumberOfTokens := metrics.ChannelCapacities(m.config)

	m.runReqChan = common.Channel[GaugeUpdate]{
		Channel: make(chan GaugeUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.runReqChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.runReqChan, m.runningRequestsUpdater)

	m.waitingReqChan = common.Channel[GaugeUpdate]{
		Channel: make(chan GaugeUpdate, maxNumberOfWaitingRequests),
		Name:    "vllm.waitingReqChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.waitingReqChan, m.waitingRequestsUpdater)

	m.kvCacheUsageChan = common.Channel[GaugeUpdate]{
		Channel: make(chan GaugeUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.kvCacheUsageChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.kvCacheUsageChan, m.kvCacheUsageUpdater)

	m.ttftChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.ttftChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.ttftChan, m.ttftUpdater)

	m.perTokenLatencyChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfTokens),
		Name:    "vllm.perTokenLatencyChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.perTokenLatencyChan, m.perTokenLatencyUpdater)

	m.e2eReqLatencyChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.e2eReqLatencyChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.e2eReqLatencyChan, m.e2eReqLatencyUpdater)

	m.reqQueueTimeChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfWaitingRequests),
		Name:    "vllm.reqQueueTimeChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.reqQueueTimeChan, m.reqQueueTimeUpdater)

	m.reqInferenceTimeChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.reqInferenceTimeChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.reqInferenceTimeChan, m.reqInferenceTimeUpdater)

	m.reqPrefillTimeChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.reqPrefillTimeChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.reqPrefillTimeChan, m.reqPrefillTimeUpdater)

	m.reqDecodeTimeChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.reqDecodeTimeChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.reqDecodeTimeChan, m.reqDecodeTimeUpdater)

	m.reqTpotChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.reqTpotChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.reqTpotChan, m.reqTpotUpdater)

	m.lorasChan = common.Channel[LoRAUpdate]{
		Channel: make(chan LoRAUpdate, maxNumberOfRequests),
		Name:    "vllm.lorasChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.lorasChan, m.lorasUpdater)

	m.requestPromptTokensChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.requestPromptTokensChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.requestPromptTokensChan, m.requestPromptTokensUpdater)

	m.requestGenerationTokensChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.requestGenerationTokensChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.requestGenerationTokensChan, m.requestGenerationTokensUpdater)

	m.maxNumGenerationTokensChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.maxNumGenerationTokensChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.maxNumGenerationTokensChan, m.maxNumGenerationTokensUpdater)

	m.requestParamsMaxTokensChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.requestParamsMaxTokensChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.requestParamsMaxTokensChan, m.requestParamsMaxTokensUpdater)

	m.promptTokensTotalChan = common.Channel[CounterUpdate]{
		Channel: make(chan CounterUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.promptTokensTotalChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.promptTokensTotalChan, m.promptTokensTotalUpdater)

	m.generationTokensTotalChan = common.Channel[CounterUpdate]{
		Channel: make(chan CounterUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.generationTokensTotalChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.generationTokensTotalChan, m.generationTokensTotalUpdater)

	m.requestSuccessTotalChan = common.Channel[RequestSuccessCounterUpdate]{
		Channel: make(chan RequestSuccessCounterUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.requestSuccessTotalChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.requestSuccessTotalChan, m.requestSuccessTotalUpdater)

	m.prefixCacheHitsTotalChan = common.Channel[CounterUpdate]{
		Channel: make(chan CounterUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.prefixCacheHitsTotalChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.prefixCacheHitsTotalChan, m.prefixCacheHitsTotalUpdater)

	m.prefixCacheQueriesTotalChan = common.Channel[CounterUpdate]{
		Channel: make(chan CounterUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.prefixCacheQueriesTotalChan",
		Done:    ctx.Done(),
	}
	go metrics.Subscribe(ctx, m.prefixCacheQueriesTotalChan, m.prefixCacheQueriesTotalUpdater)
}

// -- Per-metric write helpers ---------------------------------------------

func (m *VLLMMetricsAdapter) writeToRequestPromptTokens(upd HistogramUpdate) {
	common.WriteToChannel(m.requestPromptTokensChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToRequestGenerationTokens(upd HistogramUpdate) {
	common.WriteToChannel(m.requestGenerationTokensChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToMaxNumGenerationTokens(upd HistogramUpdate) {
	common.WriteToChannel(m.maxNumGenerationTokensChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToRequestParamsMaxTokens(upd HistogramUpdate) {
	common.WriteToChannel(m.requestParamsMaxTokensChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToPromptTokensTotal(upd CounterUpdate) {
	common.WriteToChannel(m.promptTokensTotalChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToGenerationTokensTotal(upd CounterUpdate) {
	common.WriteToChannel(m.generationTokensTotalChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToRequestSuccessTotal(upd RequestSuccessCounterUpdate) {
	common.WriteToChannel(m.requestSuccessTotalChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToPrefixCacheHitsTotal(upd CounterUpdate) {
	common.WriteToChannel(m.prefixCacheHitsTotalChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToPrefixCacheQueriesTotal(upd CounterUpdate) {
	common.WriteToChannel(m.prefixCacheQueriesTotalChan, upd, m.logger)
}

// -- Per-metric updaters --------------------------------------------------

func (m *VLLMMetricsAdapter) requestPromptTokensUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.requestPromptTokens, m.createAndRegisterReqPromptTokensHistogram, upd)
}

func (m *VLLMMetricsAdapter) requestGenerationTokensUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.requestGenerationTokens, m.createAndRegisterReqGenerationTokensHistogram, upd)
}

func (m *VLLMMetricsAdapter) maxNumGenerationTokensUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.maxNumGenerationTokens, m.createAndRegisterMaxNumGenerationTokensHistogram, upd)
}

func (m *VLLMMetricsAdapter) requestParamsMaxTokensUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.requestParamsMaxTokens, m.createAndRegisterReqParamsMaxTokensHistogram, upd)
}

func (m *VLLMMetricsAdapter) promptTokensTotalUpdater(upd CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.promptTokensTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.applyCounterReset(&m.promptTokensTotal, m.createAndRegisterPromptTokensTotalCounter,
			m.config.DisplayModelName, upd.Reset.Value)
	}
}

func (m *VLLMMetricsAdapter) generationTokensTotalUpdater(upd CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.generationTokensTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.applyCounterReset(&m.generationTokensTotal, m.createAndRegisterGenerationTokensTotalCounter,
			m.config.DisplayModelName, upd.Reset.Value)
	}
}

func (m *VLLMMetricsAdapter) requestSuccessTotalUpdater(upd RequestSuccessCounterUpdate) {
	switch {
	case upd.Increment != nil:
		if m.fake != nil {
			return
		}
		m.requestSuccessTotal.WithLabelValues(m.config.DisplayModelName, *upd.Increment).Inc()
	case upd.Reset != nil:
		m.applySuccessTotalReset(upd.Reset.Reasons)
	}
}

func (m *VLLMMetricsAdapter) prefixCacheHitsTotalUpdater(upd CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.prefixCacheHitsTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.applyCounterReset(&m.prefixCacheHitsTotal, m.createAndRegisterPrefixCacheHitsTotalCounter,
			m.config.DisplayModelName, upd.Reset.Value)
	}
}

func (m *VLLMMetricsAdapter) prefixCacheQueriesTotalUpdater(upd CounterUpdate) {
	switch {
	case upd.Add != nil:
		if m.fake != nil {
			return
		}
		m.prefixCacheQueriesTotal.WithLabelValues(m.config.DisplayModelName).Add(*upd.Add)
	case upd.Reset != nil:
		m.applyCounterReset(&m.prefixCacheQueriesTotal, m.createAndRegisterPrefixCacheQueriesTotalCounter,
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

// applyHistogramUpdate records an Observe, or on Reset unregisters the
// collector, recreates it, and replays the target bucket state.
func (m *VLLMMetricsAdapter) applyHistogramUpdate(histPP **prometheus.HistogramVec, recreate func() error, upd HistogramUpdate) {
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

// -- Channel write helpers --------------------------------------------------
//
// One helper per per-metric channel. Producers - event handlers and the
// fake-metrics applier alike - go through these wrappers so a channel is
// named in one place.

func (m *VLLMMetricsAdapter) writeToRunReq(upd GaugeUpdate) {
	common.WriteToChannel(m.runReqChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToWaitingReq(upd GaugeUpdate) {
	common.WriteToChannel(m.waitingReqChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToKVCacheUsage(upd GaugeUpdate) {
	common.WriteToChannel(m.kvCacheUsageChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToTTFT(upd HistogramUpdate) {
	common.WriteToChannel(m.ttftChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToPerTokenLatency(upd HistogramUpdate) {
	common.WriteToChannel(m.perTokenLatencyChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToE2EReqLatency(upd HistogramUpdate) {
	common.WriteToChannel(m.e2eReqLatencyChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToReqQueueTime(upd HistogramUpdate) {
	common.WriteToChannel(m.reqQueueTimeChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToReqInferenceTime(upd HistogramUpdate) {
	common.WriteToChannel(m.reqInferenceTimeChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToReqPrefillTime(upd HistogramUpdate) {
	common.WriteToChannel(m.reqPrefillTimeChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToReqDecodeTime(upd HistogramUpdate) {
	common.WriteToChannel(m.reqDecodeTimeChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToReqTpot(upd HistogramUpdate) {
	common.WriteToChannel(m.reqTpotChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) writeToLoRAs(upd LoRAUpdate) {
	common.WriteToChannel(m.lorasChan, upd, m.logger)
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
// - update LoRA state if applicable
func (m *VLLMMetricsAdapter) OnRequestQueued(ev metrics.RequestQueued) {
	if m.fake != nil {
		return
	}
	m.writeToWaitingReq(gaugeAdd(1))
}

// request dequeued
// - update number of waiting requests
// - update queue time histogram
// lora will be marked as runnning in OnRequestRunning
func (m *VLLMMetricsAdapter) OnRequestDequeued(ev metrics.RequestDequeued) {
	if m.fake != nil {
		return
	}
	m.writeToWaitingReq(gaugeAdd(-1))

	m.writeToReqQueueTime(observation(ev.QueueTime))
}

// request running
// - update number of running requests
// - update LoRA state if applicable
func (m *VLLMMetricsAdapter) OnRequestRunning(ev metrics.RequestRunning) {
	if m.fake != nil {
		return
	}
	m.writeToRunReq(gaugeAdd(1))
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
	m.writeToReqPrefillTime(observation(ev.PrefillDuration))
	m.writeToTTFT(observation(ev.PrefillDuration))
}

func (m *VLLMMetricsAdapter) OnDecodeStarted(_ metrics.DecodeStarted) {
	// State marker.
}

// token generated
// - update tpot and itl latency histograms
func (m *VLLMMetricsAdapter) OnTokenGenerated(ev metrics.TokenGenerated) {
	if m.fake != nil {
		return
	}
	m.writeToPerTokenLatency(observation(ev.InterTokenLatency))
}

// decode ended
// - update decode time histogram
// - update requests tpot histogram
func (m *VLLMMetricsAdapter) OnDecodeEnded(ev metrics.DecodeEnded) {
	if m.fake != nil {
		return
	}
	m.writeToReqDecodeTime(observation(ev.DecodeDuration))

	if ev.GenerationTokens > 0 {
		m.writeToReqTpot(observation(ev.DecodeDuration / float64(ev.GenerationTokens)))
	}
}

// request processing finished successfully - update all relevant metrics
func (m *VLLMMetricsAdapter) OnRequestSucceeded(ev metrics.RequestSucceeded) {
	if m.fake != nil {
		return
	}

	// update number of successful requests per finish reason
	m.writeToRequestSuccessTotal(RequestSuccessCounterUpdate{Increment: &ev.FinishReason})

	// request finished successfully, update number of prompt and generated tokens
	// both total and histogram metrics
	m.writeToRequestPromptTokens(observation(float64(ev.PromptTokens)))
	m.writeToRequestGenerationTokens(observation(float64(ev.GenerationTokens)))
	promptTokens := float64(ev.PromptTokens)
	generationTokens := float64(ev.GenerationTokens)
	m.writeToPromptTokensTotal(CounterUpdate{Add: &promptTokens})
	m.writeToGenerationTokensTotal(CounterUpdate{Add: &generationTokens})

	// if max_tokens is set, update the request_params_max_tokens histogram
	if ev.MaxTokens != nil {
		m.writeToRequestParamsMaxTokens(observation(float64(*ev.MaxTokens)))
	}
	if maxGenTokens, err := common.MaxIntSlice(ev.GenTokensPerChoice); err == nil {
		m.writeToMaxNumGenerationTokens(observation(float64(maxGenTokens)))
	}

	m.writeToE2EReqLatency(observation(ev.E2ELatency))
	m.writeToReqInferenceTime(observation(ev.InferenceTime))

	m.finishRunning()
}

// request processing failed
// - update all relevant metrics
func (m *VLLMMetricsAdapter) OnRequestFailed(ev metrics.RequestFailed) {
	if m.fake != nil {
		return
	}
	m.writeToE2EReqLatency(observation(ev.E2ELatency))
	m.writeToReqInferenceTime(observation(ev.InferenceTime))

	m.finishRunning()
}

// change in kv cache utilization
// - update kv cache usage gauge
func (m *VLLMMetricsAdapter) OnKVCacheUsageChanged(ev metrics.KVCacheUsageChanged) {
	if m.fake != nil {
		return
	}
	m.writeToKVCacheUsage(gaugeAdd(ev.KVCacheUsagePerc))
}

// change in prefix cache utilization
// - update prefix cache hits and queries counters
func (m *VLLMMetricsAdapter) OnPrefixCacheQueried(ev metrics.PrefixCacheQueried) {
	if m.fake != nil {
		return
	}
	hit := float64(ev.CachedPromptTokens)
	queried := float64(ev.QueriedTokens)
	m.writeToPrefixCacheHitsTotal(CounterUpdate{Add: &hit})
	m.writeToPrefixCacheQueriesTotal(CounterUpdate{Add: &queried})

}

// OnLoRASetsChanged receives the per-LoRA waiting/running snapshot produced
// by the bus after each LoRAChanged event and forwards it to the LoRA
// updater goroutine.
func (m *VLLMMetricsAdapter) OnLoRASetsChanged(ev metrics.LoRASetsChanged) {
	if m.fake != nil {
		return
	}
	m.writeToLoRAs(LoRAUpdate{Snapshot: &ev})
}

// finishRunning decrements the running-request counter for a terminal
// request. LoRA state transitions are handled separately via LoRAChanged.
func (m *VLLMMetricsAdapter) finishRunning() {
	m.writeToRunReq(gaugeAdd(-1))
}

// -- Channel updates  -------------------

// -- Updaters (per-metric channels -> Prometheus) ------------------

func (m *VLLMMetricsAdapter) waitingRequestsUpdater(upd GaugeUpdate) {
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

func (m *VLLMMetricsAdapter) runningRequestsUpdater(upd GaugeUpdate) {
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

func (m *VLLMMetricsAdapter) kvCacheUsageUpdater(upd GaugeUpdate) {
	switch {
	case upd.Add != nil && m.fake == nil:
		m.reportKVCacheUsage(*upd.Add)
	case upd.Reset != nil && m.fake != nil:
		m.reportKVCacheUsage(*upd.Reset)
	}
}

func (m *VLLMMetricsAdapter) ttftUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.ttft, m.createAndRegisterTTFTHistogram, upd)
}

func (m *VLLMMetricsAdapter) perTokenLatencyUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.tpot, m.createAndRegisterTPOTHistogram, upd)
	m.applyHistogramUpdate(&m.interTokenLatency, m.createAndRegisterInterTokenLatencyHistogram, upd)
}

func (m *VLLMMetricsAdapter) e2eReqLatencyUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.e2eReqLatency, m.createAndRegisterE2EReqLatencyHistogram, upd)
}

func (m *VLLMMetricsAdapter) reqQueueTimeUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.reqQueueTime, m.createAndRegisterReqQueueTimeHistogram, upd)
}

func (m *VLLMMetricsAdapter) reqInferenceTimeUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.reqInferenceTime, m.createAndRegisterReqInferenceTimeHistogram, upd)
}

func (m *VLLMMetricsAdapter) reqPrefillTimeUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.reqPrefillTime, m.createAndRegisterReqPrefillTimeHistogram, upd)
}

func (m *VLLMMetricsAdapter) reqDecodeTimeUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.reqDecodeTime, m.createAndRegisterReqDecodeTimeHistogram, upd)
}

func (m *VLLMMetricsAdapter) reqTpotUpdater(upd HistogramUpdate) {
	m.applyHistogramUpdate(&m.reqTpot, m.createAndRegisterReqTpotHistogram, upd)
}

// lorasUpdater republishes lora_requests_info from Snapshot events, or on
// Reset recreates the collector and stamps the supplied entries.
func (m *VLLMMetricsAdapter) lorasUpdater(upd LoRAUpdate) {
	switch {
	case upd.Snapshot != nil:
		m.reportLoras(*upd.Snapshot)
	case upd.Reset != nil:
		m.applyLoRAReset(upd.Reset)
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

// applyCounterReset resets *counterPP and records the target value via a
// single Add. A nil value leaves the recreated counter with no series
// stamped, so it reads as absent from /metrics.
func (m *VLLMMetricsAdapter) applyCounterReset(counterPP **prometheus.CounterVec, recreate func() error, modelName string, value *float64) {
	m.resetCollector(*counterPP, recreate, "failed to recreate counter during fake-metrics reset", func() {
		if value != nil {
			(*counterPP).WithLabelValues(modelName).Add(*value)
		}
	})
}

// applySuccessTotalReset resets requestSuccessTotal, then Adds each
// (reason, count) pair. Nil or empty reasons leaves the recreated counter
// with no series stamped.
func (m *VLLMMetricsAdapter) applySuccessTotalReset(reasons map[string]int64) {
	m.resetCollector(m.requestSuccessTotal, m.createAndRegisterRequestSuccessTotalCounter, "failed to recreate request_success_total counter during fake-metrics reset", func() {
		for reason, count := range reasons {
			m.requestSuccessTotal.WithLabelValues(m.config.DisplayModelName, reason).Add(float64(count))
		}
	})
}

// applyLoRAReset resets loraInfo, then stamps one series per entry. Empty
// entries emits a single zero-adapter row with the current timestamp
// (matching the fake-metrics default).
func (m *VLLMMetricsAdapter) applyLoRAReset(reset *LoRAReset) {
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
		Buckets: metrics.Build125Buckets(m.config.MaxModelLen),
	}, modelLabel)
	return m.register(m.requestPromptTokens, "prometheus request_prompt_tokens histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqGenerationTokensHistogram() error {
	m.requestGenerationTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMGenerationTokensMetricName,
		Help:    "Number of generation tokens processed.",
		Buckets: metrics.Build125Buckets(m.config.MaxModelLen),
	}, modelLabel)
	return m.register(m.requestGenerationTokens, "prometheus request_generation_tokens histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterMaxNumGenerationTokensHistogram() error {
	m.maxNumGenerationTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMMaxNumGenerationTokensMetricName,
		Help:    "Histogram of maximum number of requested generation tokens.",
		Buckets: metrics.Build125Buckets(m.config.MaxModelLen),
	}, modelLabel)
	return m.register(m.maxNumGenerationTokens, "prometheus max_num_generation_tokens histogram register failed")
}

func (m *VLLMMetricsAdapter) createAndRegisterReqParamsMaxTokensHistogram() error {
	m.requestParamsMaxTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMParamMaxTokensMetricName,
		Help:    "Histogram of the max_tokens request parameter.",
		Buckets: metrics.Build125Buckets(m.config.MaxModelLen),
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
	vllmUpdate, ok := update.(*fakemetrics.Config)
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
func (m *VLLMMetricsAdapter) applyFakeMetrics(update *fakemetrics.Config) {
	m.genMu.Lock()
	defer m.genMu.Unlock()
	generatorsWereEmpty := len(m.generators) == 0

	if update.RunningRequests != nil {
		m.updateScalarLocked(VLLMReqRunningMetricName, update.RunningRequests, m.writeToRunReq, true)
	}
	if update.WaitingRequests != nil {
		m.updateScalarLocked(VLLMReqWaitingMetricName, update.WaitingRequests, m.writeToWaitingReq, true)
	}
	if update.KVCacheUsagePercentage != nil {
		m.updateScalarLocked(VLLMKVCacheUsageMetricName, update.KVCacheUsagePercentage, m.writeToKVCacheUsage, false)
	}

	if update.TTFTBucketValues != nil {
		m.writeToTTFT(HistogramUpdate{Reset: &HistogramReset{Buckets: common.TTFTBucketsBoundaries, Samples: update.TTFTBucketValues}})
	}
	if update.TPOTBucketValues != nil {
		m.writeToPerTokenLatency(HistogramUpdate{Reset: &HistogramReset{Buckets: common.TPOTBucketsBoundaries, Samples: update.TPOTBucketValues}})
	}
	if update.E2ERequestLatencyBucketValues != nil {
		m.writeToE2EReqLatency(HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.E2ERequestLatencyBucketValues}})
	}
	if update.ReqQueueTimeBucketValues != nil {
		m.writeToReqQueueTime(HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqQueueTimeBucketValues}})
	}
	if update.ReqInfTimeBucketValues != nil {
		m.writeToReqInferenceTime(HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqInfTimeBucketValues}})
	}
	if update.ReqPrefillTimeBucketValues != nil {
		m.writeToReqPrefillTime(HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqPrefillTimeBucketValues}})
	}
	if update.ReqDecodeTimeBucketValues != nil {
		m.writeToReqDecodeTime(HistogramUpdate{Reset: &HistogramReset{Buckets: common.RequestLatencyBucketsBoundaries, Samples: update.ReqDecodeTimeBucketValues}})
	}
	if update.ReqTPOTBucketValues != nil {
		m.writeToReqTpot(HistogramUpdate{Reset: &HistogramReset{Buckets: common.TPOTBucketsBoundaries, Samples: update.ReqTPOTBucketValues}})
	}

	tokenBuckets := metrics.Build125Buckets(m.config.MaxModelLen)

	if update.RequestParamsMaxTokens != nil {
		m.writeToRequestParamsMaxTokens(HistogramUpdate{Reset: &HistogramReset{Buckets: tokenBuckets, Samples: update.RequestParamsMaxTokens}})
	}
	if update.RequestMaxGenerationTokens != nil {
		m.writeToMaxNumGenerationTokens(HistogramUpdate{Reset: &HistogramReset{Buckets: tokenBuckets, Samples: update.RequestMaxGenerationTokens}})
	}

	// update histogram of the propmpt tokens
	if update.RequestPromptTokens != nil {
		m.writeToRequestPromptTokens(HistogramUpdate{Reset: &HistogramReset{Buckets: tokenBuckets, Samples: update.RequestPromptTokens}})
	}
	// update the total prompt tokens counter according the histogram (if the total is not provided) or
	// according to the explicit total (if provided)
	if update.RequestPromptTokens != nil || update.TotalPromptTokens != nil {
		total := resolveTokenTotal(tokenBuckets, update.RequestPromptTokens, float64Ptr(update.TotalPromptTokens))
		m.writeToPromptTokensTotal(CounterUpdate{Reset: &CounterReset{Value: total}})
	}

	if update.RequestGenerationTokens != nil {
		m.writeToRequestGenerationTokens(HistogramUpdate{Reset: &HistogramReset{Buckets: tokenBuckets, Samples: update.RequestGenerationTokens}})
	}
	if update.RequestGenerationTokens != nil || update.TotalGenerationTokens != nil {
		total := resolveTokenTotal(tokenBuckets, update.RequestGenerationTokens, float64Ptr(update.TotalGenerationTokens))
		m.writeToGenerationTokensTotal(CounterUpdate{Reset: &CounterReset{Value: total}})
	}

	if update.PrefixCacheQueries != nil {
		m.writeToPrefixCacheQueriesTotal(CounterUpdate{Reset: &CounterReset{Value: float64Ptr(update.PrefixCacheQueries)}})
	}
	if update.PrefixCacheHits != nil {
		m.writeToPrefixCacheHitsTotal(CounterUpdate{Reset: &CounterReset{Value: float64Ptr(update.PrefixCacheHits)}})
	}

	if update.RequestSuccessTotal != nil {
		m.writeToRequestSuccessTotal(RequestSuccessCounterUpdate{Reset: &SuccessTotalReset{Reasons: update.RequestSuccessTotal}})
	}

	if update.LoraMetrics != nil {
		m.writeToLoRAs(LoRAUpdate{Reset: &LoRAReset{MaxLoRAs: m.config.Lora.MaxLoras, Entries: update.LoraMetrics}})
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
