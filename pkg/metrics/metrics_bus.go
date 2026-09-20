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
	"errors"
	"sync"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/prometheus/client_golang/prometheus"
)

// MetricsAdapter turns MetricsBus events into engine-specific metric
// observations. The bus spawns one drainer goroutine per channel, each
// dispatching to the matching On<Event> handler. Handlers run serially per
// channel, so a slow handler cannot block other event kinds. Implementations
// live with their engine (vLLM's is in pkg/engine/vllm), hence the exported
// handler names.
type MetricsAdapter interface {
	Start(ctx context.Context) error
	Close() error

	OnRequestReceived(ev RequestReceived)
	OnRequestQueued(ev RequestQueued)
	OnRequestDequeued(ev RequestDequeued)
	OnRequestRunning(ev RequestRunning)
	OnPrefillStarted(ev PrefillStarted)
	OnPrefillEnded(ev PrefillEnded)
	OnDecodeStarted(ev DecodeStarted)
	OnTokenGenerated(ev TokenGenerated)
	OnDecodeEnded(ev DecodeEnded)
	OnRequestSucceeded(ev RequestSucceeded)
	OnRequestFailed(ev RequestFailed)
	OnRequestRejected(ev RequestRejected)
	OnKVCacheUsageChanged(ev KVCacheUsageChanged)
	OnPrefixCacheQueried(ev PrefixCacheQueried)
	OnLoRASetsChanged(ev LoRASetsChanged)

	// ApplyFakeMetricsUpdate enqueues a partial fake-metrics update on the
	// adapter's per-metric channels. The update carries the engine-owned
	// FakeMetrics interface; each adapter narrows it to its own concrete type
	// and ignores a configuration belonging to another engine. Application is
	// asynchronous: the updater goroutines log and skip a metric whose
	// collector cannot be recreated, so a caller cannot observe such a failure.
	ApplyFakeMetricsUpdate(update common.FakeMetrics)
}

// AdapterFactory builds the metrics adapter for one engine backend. The active
// engine supplies it, so pkg/metrics never names a concrete backend.
type AdapterFactory func(ctx context.Context, registry *prometheus.Registry,
	logger logr.Logger, config common.Configuration) (MetricsAdapter, error)

// MetricsBus carries state-change events from producers to the engine
// metrics adapter. Producers push via common.WriteToChannel; the adapter
// runs one drainer goroutine per channel.
type MetricsBus struct {
	adapter  MetricsAdapter
	logger   logr.Logger
	registry *prometheus.Registry

	// Per-LoRA waiting/running request counts, mutated only by the LoRAChanged subscriber
	runningLoras sync.Map
	waitingLoras sync.Map

	RequestQueued    common.Channel[RequestQueued]
	RequestDequeued  common.Channel[RequestDequeued]
	RequestRunning   common.Channel[RequestRunning]
	PrefillStarted   common.Channel[PrefillStarted]
	PrefillEnded     common.Channel[PrefillEnded]
	DecodeStarted    common.Channel[DecodeStarted]
	TokenGenerated   common.Channel[TokenGenerated]
	DecodeEnded      common.Channel[DecodeEnded]
	RequestSucceeded common.Channel[RequestSucceeded]
	RequestFailed    common.Channel[RequestFailed]
	KVCacheUsage     common.Channel[KVCacheUsageChanged]
	PrefixCacheQuery common.Channel[PrefixCacheQueried]
	// LoRAChanged carries all LoRA request transitions on one channel to preserve per-request ordering.
	LoRAChanged common.Channel[LoRAChanged]
	// LoRASetsChanged is emitted by the bus after each LoRAChanged event with the per-LoRA waiting/running counts.
	LoRASetsChanged common.Channel[LoRASetsChanged]
}

func (b *MetricsBus) Start(ctx context.Context) error {
	// start LoRA counter loop before starting the adapter,
	// so that the first LoRAChanged event is processed before the first LoRASetsChanged event.
	go b.loraCounterLoop(ctx)

	b.subscribeAdapter(ctx)

	return b.adapter.Start(ctx)
}

// Close releases the adapter's resources. Callers must cancel the context
// passed to Start first, so the drainer goroutines are gone before the adapter
// tears itself down.
func (b *MetricsBus) Close() error {
	return b.adapter.Close()
}

// subscribeAdapter spawns one drainer goroutine per event channel, each
// dispatching to the adapter's matching On<Event> handler.
func (b *MetricsBus) subscribeAdapter(ctx context.Context) {
	go common.Subscribe(ctx, b.RequestQueued, b.adapter.OnRequestQueued)
	go common.Subscribe(ctx, b.RequestDequeued, b.adapter.OnRequestDequeued)
	go common.Subscribe(ctx, b.RequestRunning, b.adapter.OnRequestRunning)
	go common.Subscribe(ctx, b.PrefillStarted, b.adapter.OnPrefillStarted)
	go common.Subscribe(ctx, b.PrefillEnded, b.adapter.OnPrefillEnded)
	go common.Subscribe(ctx, b.DecodeStarted, b.adapter.OnDecodeStarted)
	go common.Subscribe(ctx, b.TokenGenerated, b.adapter.OnTokenGenerated)
	go common.Subscribe(ctx, b.DecodeEnded, b.adapter.OnDecodeEnded)
	go common.Subscribe(ctx, b.RequestSucceeded, b.adapter.OnRequestSucceeded)
	go common.Subscribe(ctx, b.RequestFailed, b.adapter.OnRequestFailed)
	go common.Subscribe(ctx, b.KVCacheUsage, b.adapter.OnKVCacheUsageChanged)
	go common.Subscribe(ctx, b.PrefixCacheQuery, b.adapter.OnPrefixCacheQueried)
	go common.Subscribe(ctx, b.LoRASetsChanged, b.adapter.OnLoRASetsChanged)
}

// loraCounterLoop subscribes to LoRAChanged, mutates the per-LoRA waiting/running
// counters, and forwards a snapshot on LoRASetsChanged.
func (b *MetricsBus) loraCounterLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case ev := <-b.LoRAChanged.Channel:
			switch ev.State {
			case LoRAWaiting:
				b.incrementLoraRefCount(ev.Model, &b.waitingLoras)
			case LoRARunning:
				b.decrementLoraRefCount(ev.Model, &b.waitingLoras)
				b.incrementLoraRefCount(ev.Model, &b.runningLoras)
			case LoRADone:
				b.decrementLoraRefCount(ev.Model, &b.runningLoras)
			default:
				// invalid event
				continue
			}

			common.WriteToChannel(b.LoRASetsChanged, b.snapshotLoraSets(), b.logger)
		}
	}
}

func (b *MetricsBus) snapshotLoraSets() LoRASetsChanged {
	running := make(map[string]int)
	b.runningLoras.Range(func(k, v any) bool {
		running[k.(string)] = v.(int)
		return true
	})
	waiting := make(map[string]int)
	b.waitingLoras.Range(func(k, v any) bool {
		waiting[k.(string)] = v.(int)
		return true
	})
	return LoRASetsChanged{
		Running: running,
		Waiting: waiting,
	}
}

func (b *MetricsBus) incrementLoraRefCount(lora string, theMap *sync.Map) {
	count := 0
	if value, ok := theMap.Load(lora); ok {
		count = value.(int)
	}
	theMap.Store(lora, count+1)
}

func (b *MetricsBus) decrementLoraRefCount(lora string, theMap *sync.Map) {
	if value, ok := theMap.Load(lora); ok {
		count := value.(int)
		if count > 1 {
			theMap.Store(lora, count-1)
		} else {
			theMap.Delete(lora)
		}
	}
}

// ApplyFakeMetricsUpdate forwards a partial fake-metrics update to the
// adapter's fake controller when the adapter supports it. Adapters not in
// fake mode ignore it; the caller is expected to have already gated on
// config.FakeMetrics != nil.
func (b *MetricsBus) ApplyFakeMetricsUpdate(update common.FakeMetrics) {
	if b == nil || update == nil {
		return
	}
	b.adapter.ApplyFakeMetricsUpdate(update)
}

// -- Events -----------------------------------------------------------------
//

// RequestReceived fires when a request enters HandleRequest, before queue admission.
// State marker: not consumed by any exposed metric today.
type RequestReceived struct {
}

// RequestRejected fires when a request is refused before it can be queued
// (queue full, invalid model, injected failure).
// State marker: not consumed by any exposed metric today.
type RequestRejected struct {
	Err error
}

// RequestQueued fires when a request is admitted to the waiting queue.
// Drives num_requests_waiting (+1) and the LoRA waiting-set add.
type RequestQueued struct {
}

// RequestDequeued fires when a request is pulled from the waiting queue.
// Drives num_requests_waiting (-1) and request_queue_time_seconds.
type RequestDequeued struct {
	QueueTime float64 // seconds
}

// RequestRunning fires when a worker begins processing a request, before
// prefill. Drives num_requests_running (+1) and the LoRA waiting->running move.
type RequestRunning struct {
}

// PrefillStarted fires at the start of simulated prefill. State marker.
type PrefillStarted struct {
}

// PrefillEnded fires after the simulated prefill delay. Drives
// request_prefill_time_seconds and time_to_first_token_seconds (same
// value observed on both histograms).
type PrefillEnded struct {
	PrefillDuration float64 // seconds
}

// DecodeStarted fires before the per-token generation loop. State marker.
type DecodeStarted struct {
}

// TokenGenerated fires once per generated token (from the second token
// onward). InterTokenLatency is the elapsed time since the previous
// token, computed by the producer.
type TokenGenerated struct {
	InterTokenLatency float64 // seconds
}

// DecodeEnded fires after the last token. Drives request_decode_time_seconds
// (Observe DecodeDuration) and request_time_per_output_token_seconds
// (Observe DecodeDuration/GenerationTokens when GenerationTokens > 0).
type DecodeEnded struct {
	GenerationTokens int
	DecodeDuration   float64 // seconds
}

// RequestSucceeded is the terminal event for a request that produced a
// response. Drives all token, success, and both request-level latency
// histograms, plus num_requests_running (-1) and LoRA running-set removal.
type RequestSucceeded struct {
	PromptTokens       int
	GenerationTokens   int
	GenTokensPerChoice []int
	MaxTokens          *int64
	FinishReason       string
	E2ELatency         float64 // seconds
	InferenceTime      float64 // seconds
}

// RequestFailed is the terminal event for a request that errored out. Same
// running-counter, LoRA, and latency bookkeeping as RequestSucceeded, but
// no token or success counter increments.
type RequestFailed struct {
	E2ELatency    float64 // seconds
	InferenceTime float64 // seconds
}

// KVCacheUsageChanged fires when block-cache utilization changes.
// Cache-wide (not per-request); Model is empty.
type KVCacheUsageChanged struct {
	KVCacheUsagePerc float64
}

// PrefixCacheQueried fires on prefix-cache lookup at request start.
// Cache-wide; Model is empty.
type PrefixCacheQueried struct {
	QueriedTokens      int
	CachedPromptTokens int
}

// LoRAState identifies the transition a LoRA-carrying request has made.
type LoRAState int

const (
	// LoRAWaiting: request has just been enqueued for a LoRA.
	LoRAWaiting LoRAState = iota
	// LoRARunning: request for a LoRA has been picked up by a worker.
	LoRARunning
	// LoRADone: request for a LoRA has completed (success or failure).
	LoRADone
)

// LoRAChanged signals a waiting/running/done transition for a LoRA request.
type LoRAChanged struct {
	Model string
	State LoRAState
}

// LoRASetsChanged is emitted after the bus applies a LoRAChanged event; it
// carries the current per-LoRA waiting and running request counts
type LoRASetsChanged struct {
	Running map[string]int
	Waiting map[string]int
}

// ChannelCapacities returns the buffered-channel sizes derived from config,
// shared by the bus and every EngineMetricsAdapter so buffer sizing stays
// consistent across both.
// The per-token stream is buffered on the bus and again on the adapter channel
// behind it; both must use tokens, or the narrower one decides when
// observations are dropped and the wider one is reserved for nothing.
func ChannelCapacities(config common.Configuration) (running, waiting, requests, tokens int) {
	running = config.MaxNumSeqs * 2
	waiting = config.MaxWaitingQueueLength * 2
	requests = (config.MaxNumSeqs + config.MaxWaitingQueueLength) * 2
	tokens = config.MaxNumSeqs * config.MaxModelLen
	return
}

// --------------------------------
func NewMetricsBus(ctx context.Context, config common.Configuration, registry *prometheus.Registry,
	logger logr.Logger, newAdapter AdapterFactory) (*MetricsBus, error) {
	if newAdapter == nil {
		return nil, errors.New("metrics adapter factory is required")
	}

	mBus := &MetricsBus{
		registry: registry,
		logger:   logger,
	}

	// create channels with capacity based on config
	done := ctx.Done()

	maxNumberOfRunningRequests, maxNumberOfWaitingRequests, maxNumberOfRequests, maxNumberOfTokens := ChannelCapacities(config)

	mBus.RequestQueued = common.NewChannel[RequestQueued]("bus", maxNumberOfWaitingRequests, done)
	mBus.RequestDequeued = common.NewChannel[RequestDequeued]("bus", maxNumberOfWaitingRequests, done)
	mBus.RequestRunning = common.NewChannel[RequestRunning]("bus", maxNumberOfRunningRequests, done)
	mBus.PrefillStarted = common.NewChannel[PrefillStarted]("bus", maxNumberOfRunningRequests, done)
	mBus.PrefillEnded = common.NewChannel[PrefillEnded]("bus", maxNumberOfRunningRequests, done)
	mBus.DecodeStarted = common.NewChannel[DecodeStarted]("bus", maxNumberOfRunningRequests, done)
	mBus.TokenGenerated = common.NewChannel[TokenGenerated]("bus", maxNumberOfTokens, done)
	mBus.DecodeEnded = common.NewChannel[DecodeEnded]("bus", maxNumberOfRunningRequests, done)
	mBus.RequestSucceeded = common.NewChannel[RequestSucceeded]("bus", maxNumberOfRunningRequests, done)
	mBus.RequestFailed = common.NewChannel[RequestFailed]("bus", maxNumberOfRunningRequests, done)
	mBus.KVCacheUsage = common.NewChannel[KVCacheUsageChanged]("bus", maxNumberOfRunningRequests, done)
	mBus.PrefixCacheQuery = common.NewChannel[PrefixCacheQueried]("bus", maxNumberOfRunningRequests, done)
	mBus.LoRAChanged = common.NewChannel[LoRAChanged]("bus", maxNumberOfRequests, done)
	mBus.LoRASetsChanged = common.NewChannel[LoRASetsChanged]("bus", maxNumberOfRequests, done)

	adapter, err := newAdapter(ctx, registry, logger, config)
	if err != nil {
		return nil, err
	}
	mBus.adapter = adapter

	return mBus, nil
}
