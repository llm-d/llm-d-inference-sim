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

// Contains functions related to prometheus metrics

package llmdinferencesim

import (
	"context"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

// createAndRegisterPrometheus creates and registers prometheus metrics used by vLLM simulator
// Metrics reported:
// - lora_requests_info
func (s *VllmSimulator) createAndRegisterPrometheus() error {
	s.metrics.registry = prometheus.NewRegistry()

	s.metrics.loraInfo = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      "vllm:lora_requests_info",
			Help:      "Running stats on lora requests.",
		},
		[]string{vllmapi.PromLabelMaxLora, vllmapi.PromLabelRunningLoraAdapters, vllmapi.PromLabelWaitingLoraAdapters},
	)

	if err := s.metrics.registry.Register(s.metrics.loraInfo); err != nil {
		s.logger.Error(err, "Prometheus lora info gauge register failed")
		return err
	}

	s.metrics.runningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      "vllm:num_requests_running",
			Help:      "Number of requests currently running on GPU.",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.runningRequests); err != nil {
		s.logger.Error(err, "Prometheus number of running requests gauge register failed")
		return err
	}

	// not supported for now, reports constant value
	s.metrics.waitingRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      "vllm:num_requests_waiting",
			Help:      "Prometheus metric for the number of queued requests.",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.waitingRequests); err != nil {
		s.logger.Error(err, "Prometheus number of requests in queue gauge register failed")
		return err
	}

	s.metrics.ttft = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:time_to_first_token_seconds",
			Help:      "Histogram of time to first token in seconds.",
			Buckets:   common.TTFTBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.ttft); err != nil {
		s.logger.Error(err, "Prometheus time to first token histogram register failed")
		return err
	}

	s.metrics.tpot = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:time_per_output_token_seconds",
			Help:      "Histogram of time per output token in seconds.",
			Buckets:   common.TPOTBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.tpot); err != nil {
		s.logger.Error(err, "Prometheus time per output token histogram register failed")
		return err
	}

	s.metrics.e2eReqLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:e2e_request_latency_seconds",
			Help:      "Histogram of end to end request latency in seconds.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.e2eReqLatency); err != nil {
		s.logger.Error(err, "Prometheus end to end request latency histogram register failed")
		return err
	}

	s.metrics.reqQueueTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:request_queue_time_seconds",
			Help:      "Histogram of time spent in WAITING phase for request.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.reqQueueTime); err != nil {
		s.logger.Error(err, "Prometheus request queue time histogram register failed")
		return err
	}

	s.metrics.reqInferenceTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:request_inference_time_seconds",
			Help:      "Histogram of time spent in RUNNING phase for request.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.reqInferenceTime); err != nil {
		s.logger.Error(err, "Prometheus request inerence time histogram register failed")
		return err
	}

	s.metrics.reqPrefillTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:request_prefill_time_seconds",
			Help:      "Histogram of time spent in PREFILL phase for request.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.reqPrefillTime); err != nil {
		s.logger.Error(err, "Prometheus request prefill time histogram register failed")
		return err
	}

	s.metrics.reqDecodeTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:request_queue_time_seconds",
			Help:      "Histogram of time spent in DECODE phase for request.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.reqDecodeTime); err != nil {
		s.logger.Error(err, "Prometheus request decode time histogram register failed")
		return err
	}

	s.metrics.kvCacheUsagePercentage = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      "vllm:gpu_cache_usage_perc",
			Help:      "Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.kvCacheUsagePercentage); err != nil {
		s.logger.Error(err, "Prometheus kv cache usage percentage gauge register failed")
		return err
	}

	s.metrics.requestPromptTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:request_prompt_tokens",
			Help:      "Number of prefill tokens processed.",
			Buckets:   build125Buckets(s.config.MaxModelLen),
		},
		[]string{vllmapi.PromLabelModelName},
	)
	if err := s.metrics.registry.Register(s.metrics.requestPromptTokens); err != nil {
		s.logger.Error(err, "Prometheus request_prompt_tokens histogram register failed")
		return err
	}

	s.metrics.requestGenerationTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:request_generation_tokens",
			Help:      "Number of generation tokens processed.",
			Buckets:   build125Buckets(s.config.MaxModelLen),
		},
		[]string{vllmapi.PromLabelModelName},
	)
	if err := s.metrics.registry.Register(s.metrics.requestGenerationTokens); err != nil {
		s.logger.Error(err, "Prometheus request_generation_tokens histogram register failed")
		return err
	}

	s.metrics.requestParamsMaxTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      "vllm:request_params_max_tokens",
			Help:      "Histogram of the max_tokens request parameter.",
			Buckets:   build125Buckets(s.config.MaxModelLen),
		},
		[]string{vllmapi.PromLabelModelName},
	)
	if err := s.metrics.registry.Register(s.metrics.requestParamsMaxTokens); err != nil {
		s.logger.Error(err, "Prometheus request_params_max_tokens histogram register failed")
		return err
	}

	s.metrics.requestSuccessTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "",
			Name:      "vllm:request_success_total",
			Help:      "Count of successfully processed requests.",
		},
		[]string{vllmapi.PromLabelModelName, vllmapi.PromLabelFinishReason},
	)
	if err := s.metrics.registry.Register(s.metrics.requestSuccessTotal); err != nil {
		s.logger.Error(err, "Prometheus request_success_total counter register failed")
		return err
	}

	s.setInitialPrometheusMetrics()

	return nil
}

// setInitialPrometheusMetrics sends the default values to prometheus or
// the fake metrics if set
func (s *VllmSimulator) setInitialPrometheusMetrics() {
	var nRunningReqs, nWaitingReqs, kvCacheUsage float64
	modelName := s.getDisplayedModelName(s.config.Model)
	if s.config.FakeMetrics != nil {
		nRunningReqs = float64(s.config.FakeMetrics.RunningRequests)
		nWaitingReqs = float64(s.config.FakeMetrics.WaitingRequests)
		kvCacheUsage = float64(s.config.FakeMetrics.KVCacheUsagePercentage)
		if s.config.FakeMetrics.TTFTBucketValues != nil {
			s.initFakeHistogram(s.metrics.ttft, common.TTFTBucketsBoundaries, s.config.FakeMetrics.TTFTBucketValues)
		}

		if s.config.FakeMetrics.TPOTBucketValues != nil {
			s.initFakeHistogram(s.metrics.tpot, common.TPOTBucketsBoundaries, s.config.FakeMetrics.TPOTBucketValues)
		}
		buckets := build125Buckets(s.config.MaxModelLen)
		if s.config.FakeMetrics.RequestPromptTokens != nil {
			s.initFakeHistogram(s.metrics.requestPromptTokens, buckets, s.config.FakeMetrics.RequestPromptTokens)
		}
		if s.config.FakeMetrics.RequestGenerationTokens != nil {
			s.initFakeHistogram(s.metrics.requestParamsMaxTokens, buckets, s.config.FakeMetrics.RequestGenerationTokens)
		}
		if s.config.FakeMetrics.RequestParamsMaxTokens != nil {
			s.initFakeHistogram(s.metrics.requestGenerationTokens, buckets, s.config.FakeMetrics.RequestParamsMaxTokens)
		}

		for reason, requestSuccessTotal := range s.config.FakeMetrics.RequestSuccessTotal {
			s.metrics.requestSuccessTotal.WithLabelValues(modelName, reason).Add(float64(requestSuccessTotal))
		}

		if s.config.FakeMetrics.E2ERequestLatencyBucketValues != nil {
			s.initFakeHistogram(s.metrics.tpot, common.RequestLatencyBucketsBoundaries, s.config.FakeMetrics.E2ERequestLatencyBucketValues)
		}
	}

	s.metrics.runningRequests.WithLabelValues(modelName).Set(nRunningReqs)
	s.metrics.waitingRequests.WithLabelValues(modelName).Set(nWaitingReqs)
	s.metrics.kvCacheUsagePercentage.WithLabelValues(modelName).Set(kvCacheUsage)

	if s.config.FakeMetrics != nil && len(s.config.FakeMetrics.LoraMetrics) != 0 {
		for _, metrics := range s.config.FakeMetrics.LoraMetrics {
			s.metrics.loraInfo.WithLabelValues(
				strconv.Itoa(s.config.MaxLoras),
				metrics.RunningLoras,
				metrics.WaitingLoras).Set(metrics.Timestamp)
		}
	} else {
		s.metrics.loraInfo.WithLabelValues(
			strconv.Itoa(s.config.MaxLoras),
			"",
			"").Set(float64(time.Now().Unix()))
	}
}

// initFakeHistogram initializes the given histogram values based on the input
// bucketsBoundaries - upper boudaries of all buckets except the last one. Actual number of buckets is len(bucketsBoundaries)+1.
// This includes the last bucket (last_boundary, +Inf].
// bucketsSamplesCount - array containing number of samples per bucket, starting from the first bucket.
// Trailing empty buckets are not included in this array, so its length can be <= len(bucketsBoundaries)+1
func (s *VllmSimulator) initFakeHistogram(hist *prometheus.HistogramVec, bucketsBoundaries []float64, bucketsSamplesCount []int) {
	var valueToObserve float64
	numOfBoundaries := len(bucketsBoundaries)
	modelName := s.getDisplayedModelName(s.config.Model)

	for i, bucketSamplesCount := range bucketsSamplesCount {
		// for each bucket calculate value to use for Observe function
		// for all buckets except the last one it will be the upper boundary (which is included in the bucket)
		// for the last bucket it will be top boundary of the previous bucket + 1
		if i < numOfBoundaries {
			valueToObserve = bucketsBoundaries[i]
		} else {
			// this is last bucket - use number larger than the upper bound of the previous bucket
			valueToObserve = bucketsBoundaries[numOfBoundaries-1] + 1
		}

		for range bucketSamplesCount {
			// create required number of observations for the calculated sample
			hist.WithLabelValues(modelName).Observe(valueToObserve)
		}
	}
}

// reportLoras sets information about loaded LoRA adapters
func (s *VllmSimulator) reportLoras() {
	if s.config.FakeMetrics != nil {
		return
	}
	if s.metrics.loraInfo == nil {
		// Happens in the tests
		return
	}

	var runningLoras []string
	s.metrics.runningLoras.Range(func(key any, _ any) bool {
		if lora, ok := key.(string); ok {
			runningLoras = append(runningLoras, lora)
		}
		return true
	})
	var waitingLoras []string
	s.metrics.waitingLoras.Range(func(key any, _ any) bool {
		if lora, ok := key.(string); ok {
			waitingLoras = append(waitingLoras, lora)
		}
		return true
	})

	s.metrics.loraInfo.WithLabelValues(
		strconv.Itoa(s.config.MaxLoras),
		strings.Join(runningLoras, ","),
		strings.Join(waitingLoras, ",")).Set(float64(time.Now().Unix()))
}

// reportRunningRequests sets information about running completion requests
func (s *VllmSimulator) reportRunningRequests() {
	if s.config.FakeMetrics != nil {
		return
	}
	if s.metrics.runningRequests != nil {
		s.metrics.runningRequests.WithLabelValues(
			s.getDisplayedModelName(s.config.Model)).Set(float64(s.metrics.nRunningReqs))
	}
}

// reportWaitingRequests sets information about waiting completion requests
func (s *VllmSimulator) reportWaitingRequests() {
	if s.config.FakeMetrics != nil {
		return
	}
	if s.metrics.waitingRequests != nil {
		s.metrics.waitingRequests.WithLabelValues(
			s.getDisplayedModelName(s.config.Model)).Set(float64(s.metrics.nWaitingReqs))
	}
}

// reportHistogramValue sets the given value in the given histogram
func (s *VllmSimulator) reportHistogramValue(hist *prometheus.HistogramVec, val float64) {
	if s.config.FakeMetrics != nil {
		return
	}
	if hist != nil {
		hist.WithLabelValues(
			s.getDisplayedModelName(s.config.Model)).Observe(val)
	}
}

// reportKVCacheUsage sets information about kv cache usage
func (s *VllmSimulator) reportKVCacheUsage(value float64) {
	if s.config.FakeMetrics != nil {
		return
	}
	if s.metrics.kvCacheUsagePercentage != nil {
		s.metrics.kvCacheUsagePercentage.WithLabelValues(
			s.getDisplayedModelName(s.config.Model)).Set(value)
	}
}

// startMetricsUpdaters starts the various metrics updaters
func (s *VllmSimulator) startMetricsUpdaters(ctx context.Context) {
	go s.waitingRequestsUpdater(ctx)
	go s.runningRequestsUpdater(ctx)
	go s.lorasUpdater(ctx)
	go s.kvCacheUsageUpdater(ctx)
	go s.ttftUpdater(ctx)
	go s.tpotUpdater(ctx)
	go s.recordRequestUpdater(ctx)
	go s.e2eReqLatencyUpdater(ctx)
	go s.reqQueueTimeUpdater(ctx)
	go s.reqInferenceTimeUpdater(ctx)
	go s.reqPrefillTimeUpdater(ctx)
	go s.reqDecodeTimeUpdater(ctx)
}

// waitingRequestsUpdater updates the waiting requests metric by listening on the relevant channel
func (s *VllmSimulator) waitingRequestsUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case inc := <-s.metrics.waitingReqChan:
			s.metrics.nWaitingReqs += inc
			s.reportWaitingRequests()
		}
	}
}

// runningRequestsUpdater updates the running requests metric by listening on the relevant channel
func (s *VllmSimulator) runningRequestsUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case inc := <-s.metrics.runReqChan:
			s.metrics.nRunningReqs += inc
			s.reportRunningRequests()
		}
	}
}

// kvCacheUsageUpdater updates the kv cache usage  metric by listening on the relevant channel
func (s *VllmSimulator) kvCacheUsageUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.kvCacheUsageChan:
			s.reportKVCacheUsage(value)
		}
	}
}

// ttftUpdater updates the time to first token metric by listening on the relevant channel
func (s *VllmSimulator) ttftUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.ttftChan:
			s.reportHistogramValue(s.metrics.ttft, value)
		}
	}
}

// tpotUpdater updates the time per output token metric by listening on the relevant channel
func (s *VllmSimulator) tpotUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.tpotChan:
			s.reportHistogramValue(s.metrics.tpot, value)
		}
	}
}

// e2eReqLatencyUpdater updates the e2e request latency metric by listening on the relevant channel
func (s *VllmSimulator) e2eReqLatencyUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.e2eReqLatencyChan:
			s.reportHistogramValue(s.metrics.e2eReqLatency, value)
		}
	}
}

// reqQueueTimeUpdater updates the request queue time metric by listening on the relevant channel
func (s *VllmSimulator) reqQueueTimeUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.reqQueueTimeChan:
			s.reportHistogramValue(s.metrics.reqQueueTime, value)
		}
	}
}

// reqInferenceTimeUpdater updates the request inference time metric by listening on the relevant channel
func (s *VllmSimulator) reqInferenceTimeUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.reqInferenceTimeChan:
			s.reportHistogramValue(s.metrics.reqInferenceTime, value)
		}
	}
}

// reqPrefillTimeUpdater updates the request prefill time metric by listening on the relevant channel
func (s *VllmSimulator) reqPrefillTimeUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.reqPrefillTimeChan:
			s.reportHistogramValue(s.metrics.reqPrefillTime, value)
		}
	}
}

// reqDecodeTimeUpdater updates the request decode time metric by listening on the relevant channel
func (s *VllmSimulator) reqDecodeTimeUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.reqDecodeTimeChan:
			s.reportHistogramValue(s.metrics.reqDecodeTime, value)
		}
	}
}

// lorasUpdater updates the running loras metric by listening on the relevant channel
// one function updates both waiting and running loras since they a part of the same prometheus gauge
func (s *VllmSimulator) lorasUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case loraUpdate := <-s.metrics.lorasChan:
			switch loraUpdate.state {
			case waitingUsageState:
				s.incrementLoraRefCount(loraUpdate.name, &s.metrics.waitingLoras)
			case runningUsageState:
				s.decrementLoraRefCount(loraUpdate.name, &s.metrics.waitingLoras)
				s.incrementLoraRefCount(loraUpdate.name, &s.metrics.runningLoras)
			case doneUsageState:
				s.decrementLoraRefCount(loraUpdate.name, &s.metrics.runningLoras)
			}
			s.reportLoras()
		}
	}
}

func (s *VllmSimulator) incrementLoraRefCount(lora string, theMap *sync.Map) {
	count := 0
	if value, ok := theMap.Load(lora); ok {
		// if lora is already in the map - increment its counter
		count = value.(int)
	}
	theMap.Store(lora, count+1)
}

func (s *VllmSimulator) decrementLoraRefCount(lora string, theMap *sync.Map) {
	if value, ok := theMap.Load(lora); ok {
		count := value.(int)
		if count > 1 {
			theMap.Store(lora, count-1)
		} else {
			// last lora instance stopped its execution - remove from the map
			theMap.Delete(lora)
		}
	}
}

// recordRequestUpdater listens on requestSuccessChan and drives the Prometheus metric
// for successfully completed requests.
func (s *VllmSimulator) recordRequestUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case event := <-s.metrics.requestSuccessChan:
			s.recordRequestMetricsOnSuccess(
				event.promptTokens,
				event.generationTokens,
				event.maxTokens,
				event.finishReason,
			)
		}
	}
}

// requestSuccessEvent represents the data associated with a successfully completed request,
// which is sent through the requestSuccessChan for asynchronous metrics recording.
type requestSuccessEvent struct {
	// promptTokens is the number of input (prompt) tokens in the request
	promptTokens int
	// generationTokens is the number of generated (output) tokens in the response
	generationTokens int
	// maxTokens is the maximum number of tokens allowed for generation (if specified in the request)
	maxTokens *int64
	// finishReason indicates why the generation stopped (e.g., "stop", "length", "tool_calls")
	finishReason string
}

// recordRequestMetricsOnSuccess records metrics for a successfully completed request
func (s *VllmSimulator) recordRequestMetricsOnSuccess(promptTokens,
	generationTokens int, maxTokens *int64, finishReason string) {
	modelName := s.getDisplayedModelName(s.config.Model)
	s.metrics.requestPromptTokens.WithLabelValues(modelName).Observe(float64(promptTokens))
	s.metrics.requestGenerationTokens.WithLabelValues(modelName).Observe(float64(generationTokens))
	if maxTokens != nil {
		s.metrics.requestParamsMaxTokens.WithLabelValues(modelName).Observe(float64(*maxTokens))
	}
	s.metrics.requestSuccessTotal.WithLabelValues(modelName, finishReason).Inc()
}

// build125Buckets generates histogram buckets in powers of 10 scaled by [1,2,5].
// This matches vLLM's build_1_2_5_buckets() in metrics.py.
//
// Reference: https://github.com/vllm-project/vllm/blob/main/vllm/engine/metrics.py#L175
func build125Buckets(maxValue int) []float64 {
	if maxValue <= 0 {
		return []float64{}
	}
	var buckets []float64
	exponent := 0
	mantissa := []int{1, 2, 5}

	for {
		complete := true
		for _, m := range mantissa {
			value := m * int(math.Pow10(exponent))
			if value <= maxValue {
				buckets = append(buckets, float64(value))
				complete = false
			}
		}
		if complete {
			break
		}
		exponent++
	}
	return buckets
}
