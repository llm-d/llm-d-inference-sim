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
package metrics

import (
	"math"

	"github.com/prometheus/client_golang/prometheus"
)

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

// InitFakeHistogram initializes the given histogram values based on the input
// bucketsBoundaries - upper boundaries of all buckets except the last one. Actual number of buckets is len(bucketsBoundaries)+1.
// This includes the last bucket (last_boundary, +Inf].
// bucketsSamplesCount - array containing number of samples per bucket, starting from the first bucket.
// Trailing empty buckets are not included in this array, so its length can be <= len(bucketsBoundaries)+1
func InitFakeHistogram(hist *prometheus.HistogramVec, modelName string, bucketsBoundaries []float64, bucketsSamplesCount []int) *float64 {
	var valueToObserve float64
	var total float64
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

		// hist could be nil for case when only total calculaion is needed, without updating the histogram
		// to keep in-sync values of the histogram, it sum and the total counter
		if hist != nil {
			for range bucketSamplesCount {
				// create required number of observations for the calculated sample
				hist.WithLabelValues(modelName).Observe(valueToObserve)
			}
		}

		total += float64(bucketSamplesCount) * valueToObserve
	}

	return &total
}
