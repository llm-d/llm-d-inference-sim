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
	"encoding/json"
	"math"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

type generator func(params *common.FunctionInfo, t time.Duration) float64

type generatedFakeMetrics struct {
	updateChan common.Channel[common.MetricInfo]
	genFun     generator
	params     *common.FunctionInfo
	roundToInt bool
}

func (s *SimContext) setInitialFakeMetrics() error {
	s.metrics.generatedFakeMetrics = make(map[string]generatedFakeMetrics)

	// Build a map of all configured JSON keys so that UpdateFakeMetrics
	// processes every field. Fields with omitempty that are nil/zero are
	// naturally excluded by json.Marshal.
	data, err := json.Marshal(s.Config.FakeMetrics)
	if err != nil {
		return err
	}
	var allKeys map[string]any
	if err := json.Unmarshal(data, &allKeys); err != nil {
		return err
	}

	// Remove keys with null values — they represent unconfigured metrics
	// (nil slices/maps) and should not be processed. Without this,
	// updateTokenMetrics would create empty counter time series (e.g.
	// prompt_tokens_total=0) for metrics the user never configured.
	for k, v := range allKeys {
		if v == nil {
			delete(allKeys, k)
		}
	}
	// Loras always need processing: when unconfigured, the default
	// entry (empty adapters, current timestamp) must be set.
	allKeys["loras"] = true

	// No previous values on initial setup.
	return s.UpdateFakeMetrics(allKeys, &common.FakeMetrics{})
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
			common.WriteToChannel(metric.updateChan, update, s.logger)
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
func (s *SimContext) initFakeHistogram(hist *prometheus.HistogramVec, bucketsBoundaries []float64, bucketsSamplesCount []int) *int64 {
	var valueToObserve float64
	var total int64
	numOfBoundaries := len(bucketsBoundaries)
	modelName := s.getDisplayedModelName(s.Config.Model)

	if len(bucketsSamplesCount) == 0 || len(bucketsBoundaries) == 0 {
		return nil
	}

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

		total += int64(bucketSamplesCount) * int64(valueToObserve)
	}

	return &total
}

// UpdateFakeMetrics applies a partial update to the simulator's Prometheus metrics
// based on the keys present in fakeMetricsMap. Only metrics whose JSON key appears
// in the map are touched — absent keys are left unchanged.
//
// For histogram and counter metrics, if oldFakeMetrics indicates the metric had
// previous fake values, the old Prometheus collector is unregistered and re-created
// before applying the new observations. This ensures updated values replace (not
// accumulate on top of) the old ones. When oldFakeMetrics has nil/zero values for a
// metric, the metric is assumed clean and observations are added directly.
//
// This function is called both during initial setup (via setInitialFakeMetrics with
// all configured keys and an empty oldFakeMetrics) and at runtime via the POST
// /fake_metrics HTTP endpoint (with only the keys the caller supplied).
func (s *SimContext) UpdateFakeMetrics(fakeMetricsMap map[string]any, oldFakeMetrics *common.FakeMetrics) error {
	modelName := s.getDisplayedModelName(s.Config.Model)

	var generatedFakeMetricsWasEmpty bool
	if len(s.metrics.generatedFakeMetrics) == 0 {
		generatedFakeMetricsWasEmpty = true
	}
	if _, ok := fakeMetricsMap["running-requests"]; ok {
		s.setFakeMetricWithFunction(modelName, &s.Config.FakeMetrics.RunningRequests, s.metrics.runningRequests,
			s.metrics.runReqChan, true)
	}
	if _, ok := fakeMetricsMap["waiting-requests"]; ok {
		s.setFakeMetricWithFunction(modelName, &s.Config.FakeMetrics.WaitingRequests, s.metrics.waitingRequests,
			s.metrics.waitingReqChan, true)
	}
	if _, ok := fakeMetricsMap["kv-cache-usage"]; ok {
		s.setFakeMetricWithFunction(modelName, &s.Config.FakeMetrics.KVCacheUsagePercentage, s.metrics.kvCacheUsagePercentage,
			s.metrics.kvCacheUsageChan, false)
	}

	if _, ok := fakeMetricsMap["ttft-buckets-values"]; ok {
		if oldFakeMetrics.TTFTBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.ttft)
			if err := s.createAndRegisterTTFTMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.ttft, common.TTFTBucketsBoundaries, s.Config.FakeMetrics.TTFTBucketValues)
	}

	if _, ok := fakeMetricsMap["tpot-buckets-values"]; ok {
		if oldFakeMetrics.TPOTBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.tpot)
			s.metrics.registry.Unregister(s.metrics.interTokenLatency)
			if err := s.createAndRegisterTPOTAndInterTokenMetrics(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.tpot, common.TPOTBucketsBoundaries, s.Config.FakeMetrics.TPOTBucketValues)
		s.initFakeHistogram(s.metrics.interTokenLatency, common.TPOTBucketsBoundaries, s.Config.FakeMetrics.TPOTBucketValues)
	}

	if _, ok := fakeMetricsMap["e2erl-buckets-values"]; ok {
		if oldFakeMetrics.E2ERequestLatencyBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.e2eReqLatency)
			if err := s.createAndRegisterE2EReqLatencyMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.e2eReqLatency, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.E2ERequestLatencyBucketValues)
	}

	if _, ok := fakeMetricsMap["queue-time-buckets-values"]; ok {
		if oldFakeMetrics.ReqQueueTimeBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.reqQueueTime)
			if err := s.createAndRegisterReqQueueTimeMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.reqQueueTime, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.ReqQueueTimeBucketValues)
	}

	if _, ok := fakeMetricsMap["inf-time-buckets-values"]; ok {
		if oldFakeMetrics.ReqInfTimeBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.reqInferenceTime)
			if err := s.createAndRegisterReqInferenceTimeMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.reqInferenceTime, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.ReqInfTimeBucketValues)
	}

	if _, ok := fakeMetricsMap["prefill-time-buckets-values"]; ok {
		if oldFakeMetrics.ReqPrefillTimeBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.reqPrefillTime)
			if err := s.createAndRegisterReqPrefillTimeMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.reqPrefillTime, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.ReqPrefillTimeBucketValues)
	}

	if _, ok := fakeMetricsMap["decode-time-buckets-values"]; ok {
		if oldFakeMetrics.ReqDecodeTimeBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.reqDecodeTime)
			if err := s.createAndRegisterReqDecodeTimeMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.reqDecodeTime, common.RequestLatencyBucketsBoundaries, s.Config.FakeMetrics.ReqDecodeTimeBucketValues)
	}

	buckets := Build125Buckets(s.Config.MaxModelLen)

	if _, ok := fakeMetricsMap["request-params-max-tokens"]; ok {
		if oldFakeMetrics.RequestParamsMaxTokens != nil {
			s.metrics.registry.Unregister(s.metrics.requestParamsMaxTokens)
			if err := s.createAndRegisterReqParamsMaxTokensMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.requestParamsMaxTokens, buckets, s.Config.FakeMetrics.RequestParamsMaxTokens)
	}

	if _, ok := fakeMetricsMap["request-max-generation-tokens"]; ok {
		if oldFakeMetrics.RequestMaxGenerationTokens != nil {
			s.metrics.registry.Unregister(s.metrics.maxNumGenerationTokens)
			if err := s.createAndRegisterMaxNumGenerationTokensMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.maxNumGenerationTokens, buckets, s.Config.FakeMetrics.RequestMaxGenerationTokens)
	}

	if err := s.updateTokenMetrics(
		modelName, buckets, fakeMetricsMap,
		"request-prompt-tokens", "total-prompt-tokens",
		s.Config.FakeMetrics.RequestPromptTokens, oldFakeMetrics.RequestPromptTokens,
		s.Config.FakeMetrics.TotalPromptTokens, oldFakeMetrics.TotalPromptTokens,
		&s.metrics.requestPromptTokens, &s.metrics.promptTokensTotal,
		s.createAndRegisterReqPromptTokensMetrics, s.createAndRegisterPromptTokensTotalMetrics,
		func() { s.Config.FakeMetrics.TotalPromptTokens = nil },
	); err != nil {
		return err
	}

	if err := s.updateTokenMetrics(
		modelName, buckets, fakeMetricsMap,
		"request-generation-tokens", "total-generation-tokens",
		s.Config.FakeMetrics.RequestGenerationTokens, oldFakeMetrics.RequestGenerationTokens,
		s.Config.FakeMetrics.TotalGenerationTokens, oldFakeMetrics.TotalGenerationTokens,
		&s.metrics.requestGenerationTokens, &s.metrics.generationTokensTotal,
		s.createAndRegisterReqGenerationTokensMetrics, s.createAndRegisterGenerationTokensTotalMetrics,
		func() { s.Config.FakeMetrics.TotalGenerationTokens = nil },
	); err != nil {
		return err
	}

	if _, ok := fakeMetricsMap["prefix-cache-queries"]; ok {
		if oldFakeMetrics.PrefixCacheQueries != nil {
			s.metrics.registry.Unregister(s.metrics.prefixCacheQueries)
			if err := s.createAndRegisterPrefixCacheQueriesMetric(); err != nil {
				return err
			}
		}
		if s.Config.FakeMetrics.PrefixCacheQueries != nil {
			s.metrics.prefixCacheQueries.WithLabelValues(modelName).Add(float64(*s.Config.FakeMetrics.PrefixCacheQueries))
		}
	}

	if _, ok := fakeMetricsMap["prefix-cache-hits"]; ok {
		if oldFakeMetrics.PrefixCacheHits != nil {
			s.metrics.registry.Unregister(s.metrics.prefixCacheHits)
			if err := s.createAndRegisterPrefixCacheHitsMetric(); err != nil {
				return err
			}
		}
		if s.Config.FakeMetrics.PrefixCacheHits != nil {
			s.metrics.prefixCacheHits.WithLabelValues(modelName).Add(float64(*s.Config.FakeMetrics.PrefixCacheHits))
		}
	}

	if _, ok := fakeMetricsMap["request-success-total"]; ok {
		if oldFakeMetrics.RequestSuccessTotal != nil {
			s.metrics.registry.Unregister(s.metrics.requestSuccessTotal)
			if err := s.createAndRegisterRequestSuccessTotalMetric(); err != nil {
				return err
			}
		}
		for reason, requestSuccessTotal := range s.Config.FakeMetrics.RequestSuccessTotal {
			s.metrics.requestSuccessTotal.WithLabelValues(modelName, reason).Add(float64(requestSuccessTotal))
		}
	}

	if _, ok := fakeMetricsMap["loras"]; ok {
		s.metrics.registry.Unregister(s.metrics.loraInfo)
		if err := s.createAndRegisterLoraInfoMetric(); err != nil {
			return err
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
	}

	if generatedFakeMetricsWasEmpty && len(s.metrics.generatedFakeMetrics) > 0 {
		go s.updateFakeMetrics()
	}

	return nil
}

// updateTokenMetrics handles the update logic for a histogram+counter token metric pair.
// It updates the histogram if new values are provided, then conditionally resets and updates
// the associated total counter based on what changed between old and new configurations.
func (s *SimContext) updateTokenMetrics(
	modelName string,
	buckets []float64,
	fakeMetricsMap map[string]any,
	histKey string,
	totalKey string,
	newHistValues []int,
	oldHistValues []int,
	newExplicitTotal *int64,
	oldExplicitTotal *int64,
	hist **prometheus.HistogramVec,
	counter **prometheus.CounterVec,
	recreateHist func() error,
	recreateCounter func() error,
	clearExplicit func(),
) error {
	_, newHasHist := fakeMetricsMap[histKey]
	_, newHasExplicit := fakeMetricsMap[totalKey]

	// Update histogram if new values are provided.
	var histTotal *int64
	if newHasHist {
		if oldHistValues != nil {
			s.metrics.registry.Unregister(*hist)
			if err := recreateHist(); err != nil {
				return err
			}
		}
		histTotal = s.initFakeHistogram(*hist, buckets, newHistValues)
	}

	// The counter can be set from two sources: an explicit total value,
	// or derived from the request histogram.
	needsUpdate := (newHasExplicit && newExplicitTotal != nil) || newHasHist
	if !needsUpdate {
		return nil
	}

	// Reset (unregister + re-register) if the counter already had a value.
	if oldExplicitTotal != nil || oldHistValues != nil {
		s.metrics.registry.Unregister(*counter)
		if err := recreateCounter(); err != nil {
			return err
		}
	}

	// Use the explicit total if provided, otherwise use the total derived from the histogram.
	tokenTotal := histTotal
	if newHasExplicit && newExplicitTotal != nil {
		tokenTotal = newExplicitTotal
	}
	if tokenTotal != nil {
		(*counter).WithLabelValues(modelName).Add(float64(*tokenTotal))
	}

	// Clear the stale explicit total when a histogram is newly introduced
	// without an accompanying explicit total — the counter will now be
	// derived from the histogram.
	if !newHasExplicit && newHasHist && oldHistValues == nil {
		clearExplicit()
	}

	return nil
}

func (s *SimContext) setFakeMetricWithFunction(modelName string, fm *common.FakeMetricWithFunction, metric *prometheus.GaugeVec,
	channel common.Channel[common.MetricInfo], roundToInt bool) {
	var value float64
	if fm.IsFunction {
		genFakeMetric := generatedFakeMetrics{
			updateChan: channel,
			genFun:     mapFun(fm.Function.Name),
			params:     fm.Function,
			roundToInt: roundToInt,
		}
		s.metrics.generatedFakeMetrics[channel.Name] = genFakeMetric
		value = genFakeMetric.genFun(genFakeMetric.params, 0)
	} else {
		delete(s.metrics.generatedFakeMetrics, channel.Name)
		value = fm.FixedValue
	}
	metric.WithLabelValues(modelName).Set(value)
}
