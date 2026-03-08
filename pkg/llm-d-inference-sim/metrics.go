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
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

const (
	E2EReqLatencyMetricName          = "vllm:e2e_request_latency_seconds"
	ReqQueueTimeMetricName           = "vllm:request_queue_time_seconds"
	ReqInferenceTimeMetricName       = "vllm:request_inference_time_seconds"
	PrefillTimeMetricName            = "vllm:request_prefill_time_seconds"
	DecodeTimeMetricName             = "vllm:request_decode_time_seconds"
	TTFTMetricName                   = "vllm:time_to_first_token_seconds"
	TPOTMetricName                   = "vllm:time_per_output_token_seconds"
	InterTokenLatencyMetricName      = "vllm:inter_token_latency_seconds"
	MaxNumGenerationTokensMetricName = "vllm:max_num_generation_tokens"
	GenerationTokensMetricName       = "vllm:request_generation_tokens"
	ParamMaxTokensMetricName         = "vllm:request_params_max_tokens"
	PromptTokensMetricName           = "vllm:request_prompt_tokens"
	GenerationTokensTotalMetricName  = "vllm:generation_tokens_total"
	PromptTokensTotalMetricName      = "vllm:prompt_tokens_total"
	SuccessTotalMetricName           = "vllm:request_success_total"
	LoRARequestsMetricName           = "vllm:lora_requests_info"
	ReqRunningMetricName             = "vllm:num_requests_running"
	ReqWaitingMetricName             = "vllm:num_requests_waiting"
	KVCacheUsageMetricName           = "vllm:kv_cache_usage_perc"
	CacheConfigName                  = "vllm:cache_config_info"
	PrefixCacheHitsMetricName        = "vllm:prefix_cache_hits"
	PrefixCacheQueriesMetricName     = "vllm:prefix_cache_queries"
)

func (s *SimContext) MetricsRegistry() *prometheus.Registry {
	return s.metrics.registry
}

// createAndRegisterPrometheus creates and registers prometheus metrics used by vLLM simulator
func (s *SimContext) createAndRegisterPrometheus(ctx context.Context) error {
	maxNumberOfRequests := s.Config.MaxNumSeqs + s.Config.MaxWaitingQueueLength

	s.metrics.registry = prometheus.NewRegistry()

	s.metrics.loraInfo = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      LoRARequestsMetricName,
			Help:      "Running stats on lora requests.",
		},
		[]string{vllmapi.PromLabelMaxLora, vllmapi.PromLabelRunningLoraAdapters, vllmapi.PromLabelWaitingLoraAdapters},
	)

	if err := s.metrics.registry.Register(s.metrics.loraInfo); err != nil {
		s.logger.Error(err, "prometheus lora info gauge register failed")
		return err
	}

	s.metrics.lorasChan = make(chan loraUsage, maxNumberOfRequests)
	go s.lorasUpdater(ctx)

	s.metrics.runningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      ReqRunningMetricName,
			Help:      "Number of requests currently running on GPU.",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.runningRequests); err != nil {
		s.logger.Error(err, "prometheus number of running requests gauge register failed")
		return err
	}

	s.metrics.runReqChan = make(chan int64, maxNumberOfRequests)
	go s.runningRequestsUpdater(ctx)

	s.metrics.waitingRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      ReqWaitingMetricName,
			Help:      "Prometheus metric for the number of queued requests.",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.waitingRequests); err != nil {
		s.logger.Error(err, "prometheus number of requests in queue gauge register failed")
		return err
	}

	s.metrics.waitingReqChan = make(chan int64, maxNumberOfRequests)
	go s.waitingRequestsUpdater(ctx)

	s.metrics.ttft = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      TTFTMetricName,
			Help:      "Histogram of time to first token in seconds.",
			Buckets:   common.TTFTBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.ttft); err != nil {
		s.logger.Error(err, "prometheus time to first token histogram register failed")
		return err
	}

	s.metrics.ttftChan = make(chan float64, maxNumberOfRequests)
	go s.ttftUpdater(ctx)

	s.metrics.tpot = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      TPOTMetricName,
			Help:      "Histogram of time per output token in seconds.",
			Buckets:   common.TPOTBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.tpot); err != nil {
		s.logger.Error(err, "prometheus time per output token histogram register failed")
		return err
	}

	s.metrics.tpotChan = make(chan float64, maxNumberOfRequests*s.Config.MaxModelLen)
	go s.tpotUpdater(ctx)

	// Register inter_token_latency_seconds (new standard since vLLM 0.11)
	s.metrics.interTokenLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      InterTokenLatencyMetricName,
			Help:      "Histogram of inter-token latency in seconds.",
			Buckets:   common.TPOTBucketsBoundaries, // Reuse same buckets as TPOT
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.interTokenLatency); err != nil {
		s.logger.Error(err, "prometheus inter-token latency histogram register failed")
		return err
	}

	s.metrics.e2eReqLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      E2EReqLatencyMetricName,
			Help:      "Histogram of end to end request latency in seconds.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.e2eReqLatency); err != nil {
		s.logger.Error(err, "Prometheus end to end request latency histogram register failed")
		return err
	}

	s.metrics.e2eReqLatencyChan = make(chan float64, maxNumberOfRequests)
	go s.e2eReqLatencyUpdater(ctx)

	s.metrics.reqQueueTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      ReqQueueTimeMetricName,
			Help:      "Histogram of time spent in WAITING phase for request.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.reqQueueTime); err != nil {
		s.logger.Error(err, "Prometheus request queue time histogram register failed")
		return err
	}

	s.metrics.reqQueueTimeChan = make(chan float64, maxNumberOfRequests)
	go s.reqQueueTimeUpdater(ctx)

	s.metrics.reqInferenceTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      ReqInferenceTimeMetricName,
			Help:      "Histogram of time spent in RUNNING phase for request.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.reqInferenceTime); err != nil {
		s.logger.Error(err, "Prometheus request inference time histogram register failed")
		return err
	}

	s.metrics.reqInferenceTimeChan = make(chan float64, maxNumberOfRequests)
	go s.reqInferenceTimeUpdater(ctx)

	s.metrics.reqPrefillTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      PrefillTimeMetricName,
			Help:      "Histogram of time spent in PREFILL phase for request.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.reqPrefillTime); err != nil {
		s.logger.Error(err, "Prometheus request prefill time histogram register failed")
		return err
	}

	s.metrics.reqPrefillTimeChan = make(chan float64, maxNumberOfRequests)
	go s.reqPrefillTimeUpdater(ctx)

	s.metrics.reqDecodeTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      DecodeTimeMetricName,
			Help:      "Histogram of time spent in DECODE phase for request.",
			Buckets:   common.RequestLatencyBucketsBoundaries,
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.reqDecodeTime); err != nil {
		s.logger.Error(err, "Prometheus request decode time histogram register failed")
		return err
	}

	s.metrics.reqDecodeTimeChan = make(chan float64, maxNumberOfRequests)
	go s.reqDecodeTimeUpdater(ctx)

	s.metrics.kvCacheUsagePercentage = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      KVCacheUsageMetricName,
			Help:      "Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.kvCacheUsagePercentage); err != nil {
		s.logger.Error(err, "prometheus kv cache usage percentage gauge register failed")
		return err
	}

	s.metrics.kvCacheUsageChan = make(chan float64, maxNumberOfRequests)
	go s.kvCacheUsageUpdater(ctx)

	s.metrics.prefixCacheHits = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "",
			Name:      PrefixCacheHitsMetricName,
			Help:      "Prefix cache hits, in terms of number of cached tokens.",
		},
		[]string{vllmapi.PromLabelModelName},
	)
	if err := s.metrics.registry.Register(s.metrics.prefixCacheHits); err != nil {
		s.logger.Error(err, "prometheus prefix_cache_hits counter register failed")
		return err
	}

	s.metrics.prefixCacheQueries = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "",
			Name:      PrefixCacheQueriesMetricName,
			Help:      "Prefix cache queries, in terms of number of queried tokens.",
		},
		[]string{vllmapi.PromLabelModelName},
	)
	if err := s.metrics.registry.Register(s.metrics.prefixCacheQueries); err != nil {
		s.logger.Error(err, "prometheus prefix_cache_queries counter register failed")
		return err
	}

	s.metrics.prefixCacheStatsChan = make(chan kvcache.PrefixCacheStats, maxNumberOfRequests)
	go s.prefixCacheStatsUpdater(ctx)

	s.metrics.requestPromptTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      PromptTokensMetricName,
			Help:      "Number of prefill tokens processed.",
			Buckets:   Build125Buckets(s.Config.MaxModelLen),
		},
		[]string{vllmapi.PromLabelModelName},
	)
	if err := s.metrics.registry.Register(s.metrics.requestPromptTokens); err != nil {
		s.logger.Error(err, "prometheus request_prompt_tokens histogram register failed")
		return err
	}

	s.metrics.maxNumGenerationTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      MaxNumGenerationTokensMetricName,
			Help:      "Histogram of maximum number of requested generation tokens.",
			Buckets:   Build125Buckets(s.Config.MaxModelLen),
		},
		[]string{vllmapi.PromLabelModelName},
	)
	if err := s.metrics.registry.Register(s.metrics.maxNumGenerationTokens); err != nil {
		s.logger.Error(err, "prometheus max_num_generation_tokens histogram register failed")
		return err
	}

	s.metrics.requestGenerationTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      GenerationTokensMetricName,
			Help:      "Number of generation tokens processed.",
			Buckets:   Build125Buckets(s.Config.MaxModelLen),
		},
		[]string{vllmapi.PromLabelModelName},
	)
	if err := s.metrics.registry.Register(s.metrics.requestGenerationTokens); err != nil {
		s.logger.Error(err, "prometheus request_generation_tokens histogram register failed")
		return err
	}

	s.metrics.requestParamsMaxTokens = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: "",
			Name:      ParamMaxTokensMetricName,
			Help:      "Histogram of the max_tokens request parameter.",
			Buckets:   Build125Buckets(s.Config.MaxModelLen),
		},
		[]string{vllmapi.PromLabelModelName},
	)
	if err := s.metrics.registry.Register(s.metrics.requestParamsMaxTokens); err != nil {
		s.logger.Error(err, "prometheus request_params_max_tokens histogram register failed")
		return err
	}

	s.metrics.promptTokensTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "",
			Name:      PromptTokensTotalMetricName,
			Help:      "Total number of prompt tokens processed.",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.promptTokensTotal); err != nil {
		s.logger.Error(err, "prometheus prompt_tokens_total counter register failed")
		return err
	}

	s.metrics.generationTokensTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "",
			Name:      GenerationTokensTotalMetricName,
			Help:      "Total number of generated tokens.",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.metrics.registry.Register(s.metrics.generationTokensTotal); err != nil {
		s.logger.Error(err, "prometheus generation_tokens_total counter register failed")
		return err
	}

	s.metrics.requestSuccessTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "",
			Name:      SuccessTotalMetricName,
			Help:      "Count of successfully processed requests.",
		},
		[]string{vllmapi.PromLabelModelName, vllmapi.PromLabelFinishReason},
	)
	if err := s.metrics.registry.Register(s.metrics.requestSuccessTotal); err != nil {
		s.logger.Error(err, "prometheus request_success_total counter register failed")
		return err
	}

	s.metrics.requestSuccessChan = make(chan requestSuccessEvent, maxNumberOfRequests)
	go s.recordRequestUpdater(ctx)

	cacheConfig := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      CacheConfigName,
			Help:      "Information of the LLMEngine CacheConfig.",
		},
		[]string{vllmapi.PromLabelCacheBlockSize, vllmapi.PromLabelCacheNumGPUBlocks},
	)
	if err := s.metrics.registry.Register(cacheConfig); err != nil {
		s.logger.Error(err, "prometheus cache config register failed")
		return err
	}

	s.setInitialPrometheusMetrics(cacheConfig)

	return nil
}

// setInitialPrometheusMetrics sends the default values to prometheus or
// the fake metrics if set
func (s *SimContext) setInitialPrometheusMetrics(cacheConfig *prometheus.GaugeVec) {
	var nRunningReqs, nWaitingReqs, kvCacheUsage float64
	modelName := s.getDisplayedModelName(s.Config.Model)
	if s.Config.FakeMetrics != nil {
		nRunningReqs = float64(s.Config.FakeMetrics.RunningRequests)
		nWaitingReqs = float64(s.Config.FakeMetrics.WaitingRequests)
		kvCacheUsage = float64(s.Config.FakeMetrics.KVCacheUsagePercentage)
		if s.Config.FakeMetrics.TTFTBucketValues != nil {
			s.initFakeHistogram(s.metrics.ttft, common.TTFTBucketsBoundaries, s.Config.FakeMetrics.TTFTBucketValues)
		}

		if s.Config.FakeMetrics.TPOTBucketValues != nil {
			s.initFakeHistogram(s.metrics.tpot, common.TPOTBucketsBoundaries, s.Config.FakeMetrics.TPOTBucketValues)
			s.initFakeHistogram(s.metrics.interTokenLatency, common.TPOTBucketsBoundaries, s.Config.FakeMetrics.TPOTBucketValues)
		}
		buckets := Build125Buckets(s.Config.MaxModelLen)
		if s.Config.FakeMetrics.RequestPromptTokens != nil {
			s.initFakeHistogram(s.metrics.requestPromptTokens, buckets, s.Config.FakeMetrics.RequestPromptTokens)
			var promptTotal int64
			if s.Config.FakeMetrics.TotalPromptTokens != nil {
				promptTotal = *s.Config.FakeMetrics.TotalPromptTokens
			} else {
				promptTotal = EstimateTokenTotal(s.Config.FakeMetrics.RequestPromptTokens, buckets)
			}
			s.metrics.promptTokensTotal.WithLabelValues(modelName).Add(float64(promptTotal))
		}
		if s.Config.FakeMetrics.RequestGenerationTokens != nil {
			s.initFakeHistogram(s.metrics.requestParamsMaxTokens, buckets, s.Config.FakeMetrics.RequestGenerationTokens)
			var genTotal int64
			if s.Config.FakeMetrics.TotalGenerationTokens != nil {
				genTotal = *s.Config.FakeMetrics.TotalGenerationTokens
			} else {
				genTotal = EstimateTokenTotal(s.Config.FakeMetrics.RequestGenerationTokens, buckets)
			}
			s.metrics.generationTokensTotal.WithLabelValues(modelName).Add(float64(genTotal))
		}
		if s.Config.FakeMetrics.RequestParamsMaxTokens != nil {
			s.initFakeHistogram(s.metrics.requestGenerationTokens, buckets, s.Config.FakeMetrics.RequestParamsMaxTokens)
		}
		if s.Config.FakeMetrics.RequestMaxGenerationTokens != nil {
			s.initFakeHistogram(s.metrics.maxNumGenerationTokens, buckets, s.Config.FakeMetrics.RequestMaxGenerationTokens)
		}

		for reason, requestSuccessTotal := range s.Config.FakeMetrics.RequestSuccessTotal {
			s.metrics.requestSuccessTotal.WithLabelValues(modelName, reason).Add(float64(requestSuccessTotal))
		}

		if s.Config.FakeMetrics.E2ERequestLatencyBucketValues != nil {
			s.initFakeHistogram(s.metrics.e2eReqLatency, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.E2ERequestLatencyBucketValues)
		}

		if s.Config.FakeMetrics.ReqQueueTimeBucketValues != nil {
			s.initFakeHistogram(s.metrics.reqQueueTime, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.ReqQueueTimeBucketValues)
		}

		if s.Config.FakeMetrics.ReqInfTimeBucketValues != nil {
			s.initFakeHistogram(s.metrics.reqInferenceTime, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.ReqInfTimeBucketValues)
		}

		if s.Config.FakeMetrics.ReqPrefillTimeBucketValues != nil {
			s.initFakeHistogram(s.metrics.reqPrefillTime, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.ReqPrefillTimeBucketValues)
		}

		if s.Config.FakeMetrics.ReqDecodeTimeBucketValues != nil {
			s.initFakeHistogram(s.metrics.reqDecodeTime, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.ReqDecodeTimeBucketValues)
		}
		if s.Config.FakeMetrics.PrefixCacheQueries != nil {
			s.metrics.prefixCacheQueries.WithLabelValues(modelName).Add(float64(*s.Config.FakeMetrics.PrefixCacheQueries))
		}
		if s.Config.FakeMetrics.PrefixCacheHits != nil {
			s.metrics.prefixCacheHits.WithLabelValues(modelName).Add(float64(*s.Config.FakeMetrics.PrefixCacheHits))
		}
	}

	s.metrics.runningRequests.WithLabelValues(modelName).Set(nRunningReqs)
	s.metrics.waitingRequests.WithLabelValues(modelName).Set(nWaitingReqs)
	s.metrics.kvCacheUsagePercentage.WithLabelValues(modelName).Set(kvCacheUsage)

	cacheConfig.WithLabelValues(strconv.Itoa(s.Config.TokenBlockSize), strconv.Itoa(s.Config.KVCacheSize)).Set(1)

	if s.Config.FakeMetrics != nil && len(s.Config.FakeMetrics.LoraMetrics) != 0 {
		for _, metrics := range s.Config.FakeMetrics.LoraMetrics {
			s.metrics.loraInfo.WithLabelValues(
				strconv.Itoa(s.Config.MaxLoras),
				metrics.RunningLoras,
				metrics.WaitingLoras).Set(metrics.Timestamp)
		}
	} else {
		s.metrics.loraInfo.WithLabelValues(
			strconv.Itoa(s.Config.MaxLoras),
			"",
			"").Set(float64(time.Now().Unix()))
	}
}

// initFakeHistogram initializes the given histogram values based on the input
// bucketsBoundaries - upper boudaries of all buckets except the last one. Actual number of buckets is len(bucketsBoundaries)+1.
// This includes the last bucket (last_boundary, +Inf].
// bucketsSamplesCount - array containing number of samples per bucket, starting from the first bucket.
// Trailing empty buckets are not included in this array, so its length can be <= len(bucketsBoundaries)+1
func (s *SimContext) initFakeHistogram(hist *prometheus.HistogramVec, bucketsBoundaries []float64, bucketsSamplesCount []int) {
	var valueToObserve float64
	numOfBoundaries := len(bucketsBoundaries)
	modelName := s.getDisplayedModelName(s.Config.Model)

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
func (s *SimContext) reportLoras() {
	if s.Config.FakeMetrics != nil {
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
		strconv.Itoa(s.Config.MaxLoras),
		strings.Join(runningLoras, ","),
		strings.Join(waitingLoras, ",")).Set(float64(time.Now().Unix()))
}

// reportRunningRequests sets information about running completion requests
func (s *SimContext) reportRunningRequests() {
	if s.Config.FakeMetrics != nil {
		return
	}
	if s.metrics.runningRequests != nil {
		s.metrics.runningRequests.WithLabelValues(
			s.getDisplayedModelName(s.Config.Model)).Set(float64(s.metrics.nRunningReqs))
	}
}

// reportWaitingRequests sets information about waiting completion requests
func (s *SimContext) reportWaitingRequests() {
	if s.Config.FakeMetrics != nil {
		return
	}
	if s.metrics.waitingRequests != nil {
		s.metrics.waitingRequests.WithLabelValues(
			s.getDisplayedModelName(s.Config.Model)).Set(float64(s.metrics.nWaitingReqs))
	}
}

// reportHistogramValue sets the given value in the given histogram
func (s *SimContext) reportHistogramValue(hist *prometheus.HistogramVec, val float64) {
	if s.Config.FakeMetrics != nil {
		return
	}
	if hist != nil {
		hist.WithLabelValues(
			s.getDisplayedModelName(s.Config.Model)).Observe(val)
	}
}

// reportKVCacheUsage sets information about kv cache usage
func (s *SimContext) reportKVCacheUsage(value float64) {
	if s.Config.FakeMetrics != nil {
		return
	}
	if s.metrics.kvCacheUsagePercentage != nil {
		s.metrics.kvCacheUsagePercentage.WithLabelValues(
			s.getDisplayedModelName(s.Config.Model)).Set(value)
	}
}

// waitingRequestsUpdater updates the waiting requests metric by listening on the relevant channel
func (s *SimContext) waitingRequestsUpdater(ctx context.Context) {
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
func (s *SimContext) runningRequestsUpdater(ctx context.Context) {
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
func (s *SimContext) kvCacheUsageUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.kvCacheUsageChan:
			s.reportKVCacheUsage(value)
		}
	}
}

// prefixCacheStatsUpdater increments prefix cache hit/query counters by listening on the relevant channel
func (s *SimContext) prefixCacheStatsUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case stats := <-s.metrics.prefixCacheStatsChan:
			s.reportPrefixCacheStats(stats)
		}
	}
}

// reportPrefixCacheStats increments the prefix cache counters
func (s *SimContext) reportPrefixCacheStats(stats kvcache.PrefixCacheStats) {
	if s.Config.FakeMetrics != nil {
		return
	}
	modelName := s.getDisplayedModelName(s.Config.Model)
	if s.metrics.prefixCacheQueries != nil {
		s.metrics.prefixCacheQueries.WithLabelValues(modelName).Add(float64(stats.QueriedTokens))
	}
	if s.metrics.prefixCacheHits != nil {
		s.metrics.prefixCacheHits.WithLabelValues(modelName).Add(float64(stats.CachedTokens))
	}
}

// ttftUpdater updates the time to first token metric by listening on the relevant channel
func (s *SimContext) ttftUpdater(ctx context.Context) {
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
func (s *SimContext) tpotUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.metrics.tpotChan:
			s.reportHistogramValue(s.metrics.tpot, value)
			s.reportHistogramValue(s.metrics.interTokenLatency, value)
		}
	}
}

// e2eReqLatencyUpdater updates the e2e request latency metric by listening on the relevant channel
func (s *SimContext) e2eReqLatencyUpdater(ctx context.Context) {
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
func (s *SimContext) reqQueueTimeUpdater(ctx context.Context) {
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
func (s *SimContext) reqInferenceTimeUpdater(ctx context.Context) {
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
func (s *SimContext) reqPrefillTimeUpdater(ctx context.Context) {
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
func (s *SimContext) reqDecodeTimeUpdater(ctx context.Context) {
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
func (s *SimContext) lorasUpdater(ctx context.Context) {
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

func (s *SimContext) incrementLoraRefCount(lora string, theMap *sync.Map) {
	count := 0
	if value, ok := theMap.Load(lora); ok {
		// if lora is already in the map - increment its counter
		count = value.(int)
	}
	theMap.Store(lora, count+1)
}

func (s *SimContext) decrementLoraRefCount(lora string, theMap *sync.Map) {
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
func (s *SimContext) recordRequestUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case event := <-s.metrics.requestSuccessChan:
			s.recordRequestMetricsOnSuccess(
				event.promptTokens,
				event.generationTokens,
				event.genTokensPerChoice,
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
	// generationTokens is the number of generated (output) tokens in the response,
	// in case of response with multiple choices contains sum of lengths of all choices
	generationTokens int
	// genTokensPerChoice array of generated tokens count per choice,
	// sum of all elements in this array should be equal to generationTokens
	genTokensPerChoice []int
	// maxTokens is the maximum number of tokens allowed for generation (if specified in the request)
	maxTokens *int64
	// finishReason indicates why the generation stopped (e.g., "stop", "length", "tool_calls")
	finishReason string
}

// recordRequestMetricsOnSuccess records metrics for a successfully completed request
func (s *SimContext) recordRequestMetricsOnSuccess(promptTokens,
	generationTokens int, genTokensPerChoice []int, maxTokens *int64, finishReason string) {
	modelName := s.getDisplayedModelName(s.Config.Model)
	s.metrics.requestPromptTokens.WithLabelValues(modelName).Observe(float64(promptTokens))
	s.metrics.requestGenerationTokens.WithLabelValues(modelName).Observe(float64(generationTokens))
	s.metrics.promptTokensTotal.WithLabelValues(modelName).Add(float64(promptTokens))
	s.metrics.generationTokensTotal.WithLabelValues(modelName).Add(float64(generationTokens))
	if maxTokens != nil {
		s.metrics.requestParamsMaxTokens.WithLabelValues(modelName).Observe(float64(*maxTokens))
	}
	s.metrics.requestSuccessTotal.WithLabelValues(modelName, finishReason).Inc()
	if maxGenTokens, err := common.MaxIntSlice(genTokensPerChoice); err == nil {
		s.metrics.maxNumGenerationTokens.WithLabelValues(modelName).Observe(float64(maxGenTokens))
	}
}

// Build125Buckets generates histogram buckets in powers of 10 scaled by [1,2,5].
// This matches vLLM's build_1_2_5_buckets() in metrics.py.
//
// Reference: https://github.com/vllm-project/vllm/blob/main/vllm/engine/metrics.py#L175
func Build125Buckets(maxValue int) []float64 {
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

// EstimateTokenTotal estimates the total number of tokens based on histogram bucket boundaries
// and the number of requests in each bucket. It assumes that requests in a bucket have token
// lengths uniformly distributed between the bucket's lower and upper bounds, and uses the
// midpoint as a representative value for estimation.
//
// The last bucket is treated as [buckets[len(buckets)-1], +Inf), so its upper bound is approximated
// as twice the lower bound for midpoint calculation.
func EstimateTokenTotal(counts []int, buckets []float64) int64 {
	if len(counts) == 0 || len(buckets) == 0 {
		return 0
	}

	nCounts := len(counts)
	nBuckets := len(buckets)

	var total int64
	lower := 0.0

	for i := 0; i < nCounts; i++ {
		count := counts[i]
		if count == 0 {
			// Advance lower bound even if count is zero, to stay aligned with buckets
			if i < nBuckets {
				lower = buckets[i]
			}
			continue
		}

		var upper float64
		if i < nBuckets {
			// Bucket i corresponds to (lower, buckets[i]]
			upper = buckets[i]
		} else {
			// Last bucket: (buckets[nBuckets-1], +Inf) → approximate upper = 2 * lower
			upper = lower * 2.0
		}

		mid := (lower + upper) / 2.0
		total += int64(float64(count) * mid)

		// Update lower for next iteration
		lower = upper
	}

	return total
}
