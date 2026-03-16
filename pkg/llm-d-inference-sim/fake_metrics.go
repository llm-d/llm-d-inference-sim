/*
Copyright 2026 The llm-d-inference-simference-sim Authors.

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

// Contains functions related to fake prometheus metrics

package llmdinferencesim

import (
	"math"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

type generator func(params *common.FunctionInfo, t time.Duration) float64

type generatedFakeMetrics struct {
	updateChan     chan *common.MetricInfo
	updateChanName string
	genFun         generator
	params         *common.FunctionInfo
	roundToInt     bool
}

func (s *SimContext) setInitialFakeMetrics() {
	modelName := s.getDisplayedModelName(s.Config.Model)

	var nRunningReqs, nWaitingReqs, kvCacheUsage float64
	if s.Config.FakeMetrics.RunningRequests.IsFunction {
		genFakeMetric := generatedFakeMetrics{
			updateChan:     s.metrics.runReqChan,
			updateChanName: "metrics.runReqChan",
			genFun:         mapFun(s.Config.FakeMetrics.RunningRequests.Function.Name),
			params:         s.Config.FakeMetrics.RunningRequests.Function,
			roundToInt:     true,
		}
		s.metrics.generatedFakeMetrics = append(s.metrics.generatedFakeMetrics, genFakeMetric)
		nRunningReqs = genFakeMetric.genFun(genFakeMetric.params, 0)
	} else {
		nRunningReqs = s.Config.FakeMetrics.RunningRequests.FixedValue
	}
	s.metrics.runningRequests.WithLabelValues(modelName).Set(nRunningReqs)

	if s.Config.FakeMetrics.WaitingRequests.IsFunction {
		genFakeMetric := generatedFakeMetrics{
			updateChan:     s.metrics.waitingReqChan,
			updateChanName: "metrics.waitingReqChan",
			genFun:         mapFun(s.Config.FakeMetrics.WaitingRequests.Function.Name),
			params:         s.Config.FakeMetrics.WaitingRequests.Function,
			roundToInt:     true,
		}
		s.metrics.generatedFakeMetrics = append(s.metrics.generatedFakeMetrics, genFakeMetric)
		nWaitingReqs = genFakeMetric.genFun(genFakeMetric.params, 0)
	} else {
		nWaitingReqs = s.Config.FakeMetrics.WaitingRequests.FixedValue
	}
	s.metrics.waitingRequests.WithLabelValues(modelName).Set(nWaitingReqs)

	if s.Config.FakeMetrics.KVCacheUsagePercentage.IsFunction {
		genFakeMetric := generatedFakeMetrics{
			updateChan:     s.metrics.kvCacheUsageChan,
			updateChanName: "metrics.kvCacheUsageChan",
			genFun:         mapFun(s.Config.FakeMetrics.KVCacheUsagePercentage.Function.Name),
			params:         s.Config.FakeMetrics.KVCacheUsagePercentage.Function,
		}
		s.metrics.generatedFakeMetrics = append(s.metrics.generatedFakeMetrics, genFakeMetric)
		kvCacheUsage = genFakeMetric.genFun(genFakeMetric.params, 0)
	} else {
		kvCacheUsage = s.Config.FakeMetrics.KVCacheUsagePercentage.FixedValue
	}
	s.metrics.kvCacheUsagePercentage.WithLabelValues(modelName).Set(kvCacheUsage)

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
		s.initFakeHistogram(s.metrics.requestGenerationTokens, buckets, s.Config.FakeMetrics.RequestGenerationTokens)
		var genTotal int64
		if s.Config.FakeMetrics.TotalGenerationTokens != nil {
			genTotal = *s.Config.FakeMetrics.TotalGenerationTokens
		} else {
			genTotal = EstimateTokenTotal(s.Config.FakeMetrics.RequestGenerationTokens, buckets)
		}
		s.metrics.generationTokensTotal.WithLabelValues(modelName).Add(float64(genTotal))
	}
	if s.Config.FakeMetrics.RequestParamsMaxTokens != nil {
		s.initFakeHistogram(s.metrics.requestParamsMaxTokens, buckets, s.Config.FakeMetrics.RequestParamsMaxTokens)
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

	if len(s.Config.FakeMetrics.LoraMetrics) != 0 {
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

	go s.updateFakeMetrics()
}

func (s *SimContext) updateFakeMetrics() {
	start := time.Now()
	ticker := time.NewTicker(s.Config.FakeMetricsRefreshInterval)
	defer ticker.Stop()
	for range ticker.C {
		t := time.Since(start)
		for _, metric := range s.metrics.generatedFakeMetrics {
			value := metric.genFun(metric.params, t)
			if metric.roundToInt {
				rounded := int64(value)
				value = float64(rounded)
			}
			update := common.MetricInfo{
				Value:  value,
				IsFake: true,
			}
			common.WriteToChannel(metric.updateChan, &update, s.logger, metric.updateChanName)
		}
	}
}

func mapFun(name string) generator {
	switch name {
	case common.OscillateFuncName:
		return oscillate
	case common.RampFuncName:
		return ramp
	case common.RampWithResetFuncName:
		return rampWithReset
	case common.SquarewaveFuncName:
		return squarewave
	}
	return nil
}

// oscillate: generates a smooth sine-wave between min and max over each period
func oscillate(params *common.FunctionInfo, t time.Duration) float64 {
	phase := (2 * math.Pi * t.Seconds()) / params.Period.Seconds()
	amp := (params.End - params.Start) / 2
	mid := (params.Start + params.End) / 2
	return mid + amp*math.Sin(phase)
}

// ramp returns a value that ramps from min to max over period, then stays at max
func ramp(params *common.FunctionInfo, t time.Duration) float64 {
	frac := t.Seconds() / params.Period.Seconds() // 0..∞
	if frac >= 1 {
		return params.End
	}
	return params.Start + frac*(params.End-params.Start)
}

// rampWithReset returns a value in [min,max] that ramps linearly and resets every period
func rampWithReset(params *common.FunctionInfo, t time.Duration) float64 {
	// elapsed within current period in seconds (0..period)
	elapsedSec := (t % params.Period).Seconds()
	periodSec := params.Period.Seconds()
	frac := elapsedSec / periodSec // in [0,1]
	if frac > 1 {
		frac = 1
	}
	return params.Start + frac*(params.End-params.Start)
}

// squarewave alternates between min and max, staying at each level for half of the period
func squarewave(params *common.FunctionInfo, t time.Duration) float64 {
	// Time within current period
	within := t % params.Period
	half := params.Period / 2
	if within < half {
		return params.Start
	}
	return params.End
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
