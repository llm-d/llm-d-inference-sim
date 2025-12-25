/*
Copyright 2025 The llm-d-inference-sim Authors.

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

package llmdinferencesim

import (
	"sync"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
)

type loraUsageState int

type loraUsage struct {
	// the lora adapter name
	name string
	// state of the lora usage - waiting/running/done
	state loraUsageState
}

// Prometheus metrics
type metricsData struct {
	// runningLoras is a collection of running loras,
	// the key is lora's name, the value is the number of running requests using this lora
	runningLoras sync.Map
	// waitingLoras is a collection of waiting loras,
	// the key is lora's name, the value is the number of waiting requests using this lora
	waitingLoras sync.Map
	// lorasChan is a channel to update waitingLoras and runningLoras
	lorasChan chan loraUsage
	// nRunningReqs is the number of inference requests that are currently being processed
	nRunningReqs int64
	// runReqChan is a channel to update nRunningReqs
	runReqChan chan int64
	// requestSuccessChan is a channel to update requestSuccessReqs
	requestSuccessChan chan requestSuccessEvent
	// nWaitingReqs is the number of inference requests that are waiting to be processed
	nWaitingReqs int64
	// waitingReqChan is a channel to update nWaitingReqs
	waitingReqChan chan int64
	// ttftChan is a channel to update time to first token
	ttftChan chan float64
	// tpotChan is a channel to update time per output token
	tpotChan chan float64
	// e2eReqLatencyChan is a channel to update request e2e latency
	e2eReqLatencyChan chan float64
	// reqQueueTimeChan is a channel to update request queue time
	reqQueueTimeChan chan float64
	// reqInferenceTimeChan is a channel to update request inference time
	reqInferenceTimeChan chan float64
	// reqPrefillTimeChan is a channel to update request prefill time
	reqPrefillTimeChan chan float64
	// reqDecodeTimeChan is a channel to update request decode time
	reqDecodeTimeChan chan float64
	// kvCacheUsageChan is a channel to update kvCacheUsagePercentage
	kvCacheUsageChan chan float64
	// registry is a Prometheus registry
	registry *prometheus.Registry
	// loraInfo is prometheus gauge
	loraInfo *prometheus.GaugeVec
	// runningRequests is prometheus gauge
	runningRequests *prometheus.GaugeVec
	// waitingRequests is prometheus gauge for number of queued requests
	waitingRequests *prometheus.GaugeVec
	// ttft is prometheus histogram for time to first token in seconds
	ttft *prometheus.HistogramVec
	// tpot is prometheus histogram for time per output token in seconds (deprecated since vLLM 0.11
	tpot *prometheus.HistogramVec
	// interTokenLatency is prometheus histogram for inter-token latency in seconds (replaces tpot since vLLM 0.11)
	interTokenLatency *prometheus.HistogramVec
	// e2eReqLatency is prometheus histogram of end to end request latency in seconds
	e2eReqLatency *prometheus.HistogramVec
	// reqQueueTime is prometheus histogram of request queue time in seconds
	reqQueueTime *prometheus.HistogramVec
	// reqInferenceTime is prometheus histogram of request inference time in seconds
	reqInferenceTime *prometheus.HistogramVec
	// reqPrefillTime is prometheus histogram of request prefill time in seconds
	reqPrefillTime *prometheus.HistogramVec
	// reqDecodeTime is prometheus histogram of request decode time in seconds
	reqDecodeTime *prometheus.HistogramVec
	// kvCacheUsagePercentage is prometheus gauge
	kvCacheUsagePercentage *prometheus.GaugeVec
	// requestPromptTokens is prometheus histogram for number of input (prompt) tokens in request
	requestPromptTokens *prometheus.HistogramVec
	// requestGenerationTokens is prometheus histogram for number of generated tokens in request
	requestGenerationTokens *prometheus.HistogramVec
	// promptTokensTotal is prometheus counter for total number of input (prompt) tokens
	promptTokensTotal *prometheus.CounterVec
	// generationTokensTotal is prometheus counter for total number of generated tokens
	generationTokensTotal *prometheus.CounterVec
	// maxNumGenerationTokens is prometheus histogram for maximum number of generated tokens in request
	maxNumGenerationTokens *prometheus.HistogramVec
	// requestParamsMaxTokens is prometheus histogram for 'max_tokens' parameter in request
	requestParamsMaxTokens *prometheus.HistogramVec
	// requestSuccessTotal is prometheus counter for total number of successful requests
	requestSuccessTotal *prometheus.CounterVec
}

// LoRAs usage info for requests execution
type lorasUsageInfo struct {
	mux sync.RWMutex
	// lora adapter name -> reference count (number of currently running requests)
	loadedLoras map[string]int
	// channel for "there is a LoRA that can be removed" event
	loraRemovable chan int
	// maximum number of LoRAs that can be used simultaneously
	maxLoras int
}

type simContext struct {
	// logger is used for information and errors logging
	logger logr.Logger
	// metrics contains all Prometheus metrics related data
	metrics metricsData
	// config is the simulator's configuration
	config *common.Configuration
	// loraAdaptors contains list of LoRA available adaptors
	loraAdaptors sync.Map
	// loras contains information about which LoRAs are in use
	loras *lorasUsageInfo
	// rand with a configurable seed to generate reproducible random responses
	random *common.Random
	// kv cache functionality
	kvcacheHelper *kvcache.KVCacheHelper
	// dataset is used for token generation in responses
	dataset dataset.Dataset
	// latencyCalculator calculates the delays in simulator's responses
	latencyCalculator LatencyCalculator
}

// isLora returns true if the given model name is one of loaded LoRAs
func (s *simContext) isLora(model string) bool {
	for _, lora := range s.getLoras() {
		if model == lora {
			return true
		}
	}

	return false
}

// getDisplayedModelName returns the model name that must appear in API
// responses.  LoRA adapters keep their explicit name, while all base-model
// requests are surfaced as the first alias from --served-model-name.
func (s *simContext) getDisplayedModelName(reqModel string) string {
	if s.isLora(reqModel) {
		return reqModel
	}
	return s.config.ServedModelNames[0]
}
