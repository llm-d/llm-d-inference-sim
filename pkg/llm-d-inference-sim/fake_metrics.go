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

// UpdateFakeMetricsFromBody applies a partial fake-metrics update parsed from
// the body of the deprecated /fake_metrics endpoint. The body is the partial
// itself (not wrapped in {"fake-metrics": ...}); we wrap it and dispatch
// through ApplyConfigUpdate so it goes through the same validation, atomic
// config swap, and Prometheus side-effect path as /admin/config.
// Will be removed in v0.12.
func (s *SimContext) UpdateFakeMetricsFromBody(body []byte) error {
	wrapped := append(append([]byte(`{"fake-metrics":`), body...), '}')
	return s.ApplyConfigUpdate(wrapped)
}

type generator func(params *common.FunctionInfo, t time.Duration) float64

type generatedFakeMetrics struct {
	updateChan common.Channel[common.MetricInfo]
	genFun     generator
	params     *common.FunctionInfo
	roundToInt bool
}

func (s *SimContext) setInitialFakeMetrics() error {
	s.metrics.generatedFakeMetrics = make(map[string]generatedFakeMetrics)

	initial := s.Config().FakeMetrics

	// Loras always need processing on initial setup so the default empty
	// entry (no adapters, current timestamp) gets registered. Parser
	// initializes LoraMetrics to a non-nil (possibly empty) slice for the
	// configured case; force non-nil here to cover any path that didn't.
	if initial.LoraMetrics == nil {
		initial.LoraMetrics = []common.LorasMetrics{}
	}
	return s.updateFakeMetrics(initial, nil)
}

func (s *SimContext) updateGeneratedFakeMetrics() {
	start := time.Now()
	ticker := time.NewTicker(s.Config().FakeMetricsRefreshInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.metrics.stopFakeMetricsTicker:
			return
		case <-ticker.C:
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
			hist.WithLabelValues(s.Config().DisplayModelName).Observe(valueToObserve)
		}

		total += int64(bucketSamplesCount) * int64(valueToObserve)
	}

	return &total
}

// updateFakeMetrics applies a partial update to the simulator's Prometheus
// metrics. update carries the fields to apply (nil fields are skipped); old
// is the previous FakeMetrics state used to decide whether a collector
// already exists and must be unregistered+recreated to drop accumulated
// observations. old is nil at startup (setInitialFakeMetrics) where there is
// no prior state.
//
// This function does not mutate any shared state — the merged FakeMetrics is
// produced by Configuration.Update and swapped in by the caller via
// SetConfig.
func (s *SimContext) updateFakeMetrics(update *common.FakeMetrics, old *common.FakeMetrics) error {
	hadOld := func(check func(o *common.FakeMetrics) bool) bool {
		return old != nil && check(old)
	}

	var generatedFakeMetricsWasEmpty bool
	if len(s.metrics.generatedFakeMetrics) == 0 {
		generatedFakeMetricsWasEmpty = true
	}

	if update.RunningRequests != nil {
		s.setFakeMetricWithFunction(s.Config().DisplayModelName, update.RunningRequests, s.metrics.runningRequests,
			s.metrics.runReqChan, true)
	}
	if update.WaitingRequests != nil {
		s.setFakeMetricWithFunction(s.Config().DisplayModelName, update.WaitingRequests, s.metrics.waitingRequests,
			s.metrics.waitingReqChan, true)
	}
	if update.KVCacheUsagePercentage != nil {
		s.setFakeMetricWithFunction(s.Config().DisplayModelName, update.KVCacheUsagePercentage, s.metrics.kvCacheUsagePercentage,
			s.metrics.kvCacheUsageChan, false)
	}

	if update.TTFTBucketValues != nil {
		if hadOld(func(o *common.FakeMetrics) bool { return o.TTFTBucketValues != nil }) {
			s.metrics.registry.Unregister(s.metrics.ttft)
			if err := s.createAndRegisterTTFTMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.ttft, common.TTFTBucketsBoundaries, update.TTFTBucketValues)
	}

	if update.TPOTBucketValues != nil {
		if hadOld(func(o *common.FakeMetrics) bool { return o.TPOTBucketValues != nil }) {
			s.metrics.registry.Unregister(s.metrics.tpot)
			s.metrics.registry.Unregister(s.metrics.interTokenLatency)
			if err := s.createAndRegisterTPOTAndInterTokenMetrics(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.tpot, common.TPOTBucketsBoundaries, update.TPOTBucketValues)
		s.initFakeHistogram(s.metrics.interTokenLatency, common.TPOTBucketsBoundaries, update.TPOTBucketValues)
	}

	if update.E2ERequestLatencyBucketValues != nil {
		if hadOld(func(o *common.FakeMetrics) bool { return o.E2ERequestLatencyBucketValues != nil }) {
			s.metrics.registry.Unregister(s.metrics.e2eReqLatency)
			if err := s.createAndRegisterE2EReqLatencyMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.e2eReqLatency, common.RequestLatencyBucketsBoundaries, update.E2ERequestLatencyBucketValues)
	}

	if update.ReqQueueTimeBucketValues != nil {
		if hadOld(func(o *common.FakeMetrics) bool { return o.ReqQueueTimeBucketValues != nil }) {
			s.metrics.registry.Unregister(s.metrics.reqQueueTime)
			if err := s.createAndRegisterReqQueueTimeMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.reqQueueTime, common.RequestLatencyBucketsBoundaries, update.ReqQueueTimeBucketValues)
	}

	if update.ReqInfTimeBucketValues != nil {
		if hadOld(func(o *common.FakeMetrics) bool { return o.ReqInfTimeBucketValues != nil }) {
			s.metrics.registry.Unregister(s.metrics.reqInferenceTime)
			if err := s.createAndRegisterReqInferenceTimeMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.reqInferenceTime, common.RequestLatencyBucketsBoundaries, update.ReqInfTimeBucketValues)
	}

	if update.ReqPrefillTimeBucketValues != nil {
		if hadOld(func(o *common.FakeMetrics) bool { return o.ReqPrefillTimeBucketValues != nil }) {
			s.metrics.registry.Unregister(s.metrics.reqPrefillTime)
			if err := s.createAndRegisterReqPrefillTimeMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.reqPrefillTime, common.RequestLatencyBucketsBoundaries, update.ReqPrefillTimeBucketValues)
	}

	if update.ReqDecodeTimeBucketValues != nil {
		if hadOld(func(o *common.FakeMetrics) bool { return o.ReqDecodeTimeBucketValues != nil }) {
			s.metrics.registry.Unregister(s.metrics.reqDecodeTime)
			if err := s.createAndRegisterReqDecodeTimeMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.reqDecodeTime, common.RequestLatencyBucketsBoundaries, update.ReqDecodeTimeBucketValues)
	}

	buckets := Build125Buckets(s.Config().MaxModelLen)

	if update.RequestParamsMaxTokens != nil {
		if hadOld(func(o *common.FakeMetrics) bool { return o.RequestParamsMaxTokens != nil }) {
			s.metrics.registry.Unregister(s.metrics.requestParamsMaxTokens)
			if err := s.createAndRegisterReqParamsMaxTokensMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.requestParamsMaxTokens, buckets, update.RequestParamsMaxTokens)
	}

	if update.RequestMaxGenerationTokens != nil {
		if hadOld(func(o *common.FakeMetrics) bool { return o.RequestMaxGenerationTokens != nil }) {
			s.metrics.registry.Unregister(s.metrics.maxNumGenerationTokens)
			if err := s.createAndRegisterMaxNumGenerationTokensMetric(); err != nil {
				return err
			}
		}
		s.initFakeHistogram(s.metrics.maxNumGenerationTokens, buckets, update.RequestMaxGenerationTokens)
	}

	var oldRequestPromptTokens, oldRequestGenerationTokens []int
	var oldTotalPromptTokens, oldTotalGenerationTokens *int64
	var oldPrefixCacheQueries, oldPrefixCacheHits *int64
	var oldRequestSuccessTotal map[string]int64
	if old != nil {
		oldRequestPromptTokens = old.RequestPromptTokens
		oldRequestGenerationTokens = old.RequestGenerationTokens
		oldTotalPromptTokens = old.TotalPromptTokens
		oldTotalGenerationTokens = old.TotalGenerationTokens
		oldPrefixCacheQueries = old.PrefixCacheQueries
		oldPrefixCacheHits = old.PrefixCacheHits
		oldRequestSuccessTotal = old.RequestSuccessTotal
	}

	if err := s.updateTokenMetrics(
		s.Config().DisplayModelName, buckets,
		update.RequestPromptTokens, oldRequestPromptTokens,
		update.TotalPromptTokens, oldTotalPromptTokens,
		&s.metrics.requestPromptTokens, &s.metrics.promptTokensTotal,
		s.createAndRegisterReqPromptTokensMetrics, s.createAndRegisterPromptTokensTotalMetrics,
	); err != nil {
		return err
	}

	if err := s.updateTokenMetrics(
		s.Config().DisplayModelName, buckets,
		update.RequestGenerationTokens, oldRequestGenerationTokens,
		update.TotalGenerationTokens, oldTotalGenerationTokens,
		&s.metrics.requestGenerationTokens, &s.metrics.generationTokensTotal,
		s.createAndRegisterReqGenerationTokensMetrics, s.createAndRegisterGenerationTokensTotalMetrics,
	); err != nil {
		return err
	}

	if update.PrefixCacheQueries != nil {
		if oldPrefixCacheQueries != nil {
			s.metrics.registry.Unregister(s.metrics.prefixCacheQueries)
			if err := s.createAndRegisterPrefixCacheQueriesMetric(); err != nil {
				return err
			}
		}
		s.metrics.prefixCacheQueries.WithLabelValues(s.Config().DisplayModelName).Add(float64(*update.PrefixCacheQueries))
	}

	if update.PrefixCacheHits != nil {
		if oldPrefixCacheHits != nil {
			s.metrics.registry.Unregister(s.metrics.prefixCacheHits)
			if err := s.createAndRegisterPrefixCacheHitsMetric(); err != nil {
				return err
			}
		}
		s.metrics.prefixCacheHits.WithLabelValues(s.Config().DisplayModelName).Add(float64(*update.PrefixCacheHits))
	}

	if update.RequestSuccessTotal != nil {
		if oldRequestSuccessTotal != nil {
			s.metrics.registry.Unregister(s.metrics.requestSuccessTotal)
			if err := s.createAndRegisterRequestSuccessTotalMetric(); err != nil {
				return err
			}
		}
		for reason, requestSuccessTotal := range update.RequestSuccessTotal {
			s.metrics.requestSuccessTotal.WithLabelValues(s.Config().DisplayModelName, reason).Add(float64(requestSuccessTotal))
		}
	}

	if update.LoraMetrics != nil {
		s.metrics.registry.Unregister(s.metrics.loraInfo)
		if err := s.createAndRegisterLoraInfoMetric(); err != nil {
			return err
		}
		if len(update.LoraMetrics) != 0 {
			for _, metrics := range update.LoraMetrics {
				s.metrics.loraInfo.WithLabelValues(
					strconv.Itoa(s.Config().MaxLoras),
					metrics.RunningLoras,
					metrics.WaitingLoras).Set(metrics.Timestamp)
			}
		} else {
			s.metrics.loraInfo.WithLabelValues(
				strconv.Itoa(s.Config().MaxLoras),
				"",
				"").Set(float64(time.Now().Unix()))
		}
	}

	if generatedFakeMetricsWasEmpty && len(s.metrics.generatedFakeMetrics) > 0 {
		s.metrics.stopFakeMetricsTicker = make(chan struct{})
		go s.updateGeneratedFakeMetrics()
	} else if !generatedFakeMetricsWasEmpty && len(s.metrics.generatedFakeMetrics) == 0 {
		close(s.metrics.stopFakeMetricsTicker)
	}

	return nil
}

// updateTokenMetrics handles the update logic for a histogram+counter token
// metric pair. It updates the histogram if new values are provided, then
// conditionally resets and updates the associated total counter based on what
// changed between old and new configurations. No state is mutated; the merged
// FakeMetrics is produced separately by Configuration.Update.
func (s *SimContext) updateTokenMetrics(
	modelName string,
	buckets []float64,
	newHistValues []int,
	oldHistValues []int,
	newExplicitTotal *int64,
	oldExplicitTotal *int64,
	hist **prometheus.HistogramVec,
	counter **prometheus.CounterVec,
	recreateHist func() error,
	recreateCounter func() error,
) error {
	newHasHist := newHistValues != nil
	newHasExplicit := newExplicitTotal != nil

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
	if !newHasHist && !newHasExplicit {
		return nil
	}

	// Reset (unregister + re-register) if the counter already had a value.
	if oldExplicitTotal != nil || oldHistValues != nil {
		s.metrics.registry.Unregister(*counter)
		if err := recreateCounter(); err != nil {
			return err
		}
	}

	// Use the explicit total if provided, otherwise use the total derived from
	// the histogram.
	tokenTotal := histTotal
	if newHasExplicit {
		tokenTotal = newExplicitTotal
	}
	if tokenTotal != nil {
		(*counter).WithLabelValues(modelName).Add(float64(*tokenTotal))
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
