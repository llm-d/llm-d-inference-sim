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

package vllm

import (
	"errors"
	"fmt"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
)

// VLLMFakeMetrics implements common.VLLMFakeMetrics for the vLLM backend.
type VLLMFakeMetrics struct {
	// LoraMetrics
	LoraMetrics []common.LorasMetrics `json:"loras"`
	LorasString []string              `yaml:"loras" json:"-"`

	// All scalar fields are pointers so that absence (nil) can be
	// distinguished from a zero value during partial updates via
	// /admin/config — only non-nil fields are applied.

	// RunningRequests is the number of inference requests that are currently being processed
	RunningRequests *common.FakeMetricWithFunction `yaml:"running-requests" json:"running-requests,omitempty"`
	// WaitingRequests is the number of inference requests that are waiting to be processed
	WaitingRequests *common.FakeMetricWithFunction `yaml:"waiting-requests" json:"waiting-requests,omitempty"`
	// KVCacheUsagePercentage  is the fraction of KV-cache blocks currently in use (from 0 to 1)
	KVCacheUsagePercentage *common.FakeMetricWithFunction `yaml:"kv-cache-usage" json:"kv-cache-usage,omitempty"`

	// Histogram metrics - defined by array of values.
	// Each value in this array is a value for the corresponding bucket.
	// Array may contain less values than number of buckets, all trailing missing values assumed as 0.

	// TTFTBuckets is an array of values for time-to-first-token buckets.
	// Buckets upper boundaries in seconds are:
	// 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
	// 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0, +Inf
	TTFTBucketValues []int `yaml:"ttft-buckets-values" json:"ttft-buckets-values"`
	// TPOTBuckets is an array of values for time-per-output-token buckets.
	// Buckets upper boundaries in seconds are:
	// 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
	// 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, +Inf
	TPOTBucketValues []int `yaml:"tpot-buckets-values" json:"tpot-buckets-values"`
	// RequestPromptTokens RequestGenerationTokens RequestParamsMaxTokens Histogram fake-observation arrays for init.
	// Each value in these arrays is passed to Observe() exactly once at startup.
	// By default:
	//   - The sum of RequestPromptTokens initializes the metric vllm:prompt_tokens_total.
	//   - The sum of RequestGenerationTokens initializes the metric vllm:generation_tokens_total.
	//
	// If TotalPromptTokens or TotalGenerationTokens are explicitly provided,
	// they override the above sums and are used directly as the initial total token counts.
	RequestPromptTokens        []int `yaml:"request-prompt-tokens" json:"request-prompt-tokens"`                 // prompt-length samples
	RequestGenerationTokens    []int `yaml:"request-generation-tokens" json:"request-generation-tokens"`         // generation-length samples
	RequestParamsMaxTokens     []int `yaml:"request-params-max-tokens" json:"request-params-max-tokens"`         // max_tokens parameter samples
	RequestMaxGenerationTokens []int `yaml:"request-max-generation-tokens" json:"request-max-generation-tokens"` // request_max_num_generation_tokens samples
	// RequestSuccessTotal is the number of successful requests, key: finish-reason (stop, length, etc.).
	RequestSuccessTotal map[string]int64 `yaml:"request-success-total" json:"request-success-total"`

	// TotalPromptTokens is the total number of prompt tokens processed
	TotalPromptTokens *int64 `yaml:"total-prompt-tokens" json:"total-prompt-tokens,omitempty"`
	// TotalGenerationTokens is the total number of generated tokens
	TotalGenerationTokens *int64 `yaml:"total-generation-tokens" json:"total-generation-tokens,omitempty"`

	// Latency histograms - have same buckets upper boundaries in seconds are:
	// 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0,
	// 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0, +Inf

	// E2ERequestLatencyBucketValues is an array of values for e2e request latency buckets.
	E2ERequestLatencyBucketValues []int `yaml:"e2erl-buckets-values" json:"e2erl-buckets-values"`
	// ReqQueueTimeBucketValues is an array of values for request queue time buckets.
	ReqQueueTimeBucketValues []int `yaml:"queue-time-buckets-values" json:"queue-time-buckets-values"`
	// ReqInfTimeBucketValues is an array of values for request inference time buckets.
	ReqInfTimeBucketValues []int `yaml:"inf-time-buckets-values" json:"inf-time-buckets-values"`
	// ReqPrefillTimeBucketValues is an array of values for request prefill time buckets.
	ReqPrefillTimeBucketValues []int `yaml:"prefill-time-buckets-values" json:"prefill-time-buckets-values"`
	// ReqDecodeTimeBucketValues is an array of values for request decode time buckets.
	ReqDecodeTimeBucketValues []int `yaml:"decode-time-buckets-values" json:"decode-time-buckets-values"`
	// ReqTpotBucketValues is an array of values for request time-per-output-token buckets.
	ReqTPOTBucketValues []int `yaml:"request-tpot-buckets-values" json:"request-tpot-buckets-values"`

	// PrefixCacheHits is the initial value for the prefix cache hits counter (in tokens)
	PrefixCacheHits *int64 `yaml:"prefix-cache-hits" json:"prefix-cache-hits,omitempty"`
	// PrefixCacheQueries is the initial value for the prefix cache queries counter (in tokens)
	PrefixCacheQueries *int64 `yaml:"prefix-cache-queries" json:"prefix-cache-queries,omitempty"`
}

// New returns a fresh, zero-valued *Config.
func (f *VLLMFakeMetrics) New() common.FakeMetrics {
	return &VLLMFakeMetrics{}
}

// Validate checks the fake-metrics configuration. Called by the vLLM engine's
// ValidateConfig (fake metrics are an engine-specific feature).
func (f *VLLMFakeMetrics) Validate() error {
	if err := common.ValidateFakeCount("running-requests", f.RunningRequests); err != nil {
		return err
	}
	if err := common.ValidateFakeCount("waiting-requests", f.WaitingRequests); err != nil {
		return err
	}
	if err := common.ValidateFakeFraction("kv-cache-usage", f.KVCacheUsagePercentage); err != nil {
		return err
	}

	// The token-length histograms are missing here: their boundaries scale with
	// the model's context window, which this method does not have. The engine's
	// ValidateConfig checks them through ValidateTokenBuckets.
	for _, h := range []struct {
		name    string
		buckets []float64
		samples []int
	}{
		{"ttft-buckets-values", common.TTFTBucketsBoundaries, f.TTFTBucketValues},
		{"tpot-buckets-values", common.TPOTBucketsBoundaries, f.TPOTBucketValues},
		{"request-tpot-buckets-values", common.TPOTBucketsBoundaries, f.ReqTPOTBucketValues},
		{"e2erl-buckets-values", common.RequestLatencyBucketsBoundaries, f.E2ERequestLatencyBucketValues},
		{"queue-time-buckets-values", common.RequestLatencyBucketsBoundaries, f.ReqQueueTimeBucketValues},
		{"inf-time-buckets-values", common.RequestLatencyBucketsBoundaries, f.ReqInfTimeBucketValues},
		{"prefill-time-buckets-values", common.RequestLatencyBucketsBoundaries, f.ReqPrefillTimeBucketValues},
		{"decode-time-buckets-values", common.RequestLatencyBucketsBoundaries, f.ReqDecodeTimeBucketValues},
	} {
		if err := common.ValidateFakeBuckets(h.name, h.buckets, h.samples); err != nil {
			return err
		}
	}

	if f.RequestSuccessTotal != nil {
		for reason, count := range f.RequestSuccessTotal {
			if count < 0 {
				return fmt.Errorf("fake metrics request-success-total.%s "+
					"cannot be negative, got %d", reason, count)
			}
			if _, ok := common.ValidFinishReasons[reason]; !ok {
				return fmt.Errorf("invalid finish reason in request-success-total: "+
					"%s (valid reasons: %v)", reason, common.RequiredFinishReasons)
			}
		}
		for _, reason := range common.RequiredFinishReasons {
			if _, exists := f.RequestSuccessTotal[reason]; !exists {
				f.RequestSuccessTotal[reason] = 0
			}
		}
	}

	if f.PrefixCacheHits != nil && *f.PrefixCacheHits < 0 {
		return errors.New("fake metrics prefix-cache-hits cannot be negative")
	}
	if f.PrefixCacheQueries != nil && *f.PrefixCacheQueries < 0 {
		return errors.New("fake metrics prefix-cache-queries cannot be negative")
	}
	if (f.PrefixCacheHits == nil) != (f.PrefixCacheQueries == nil) {
		return errors.New("fake metrics prefix-cache-hits and prefix-cache-queries must be specified together")
	}
	if f.PrefixCacheHits != nil && f.PrefixCacheQueries != nil &&
		*f.PrefixCacheHits > *f.PrefixCacheQueries {
		return errors.New("fake metrics prefix-cache-hits cannot exceed prefix-cache-queries")
	}

	return nil
}

// ValidateTokenBuckets checks the token-length histogram arrays against the
// buckets maxModelLen builds, which is why it is not part of Validate: the
// boundaries are not a property of the fake-metrics configuration alone. Called
// by this engine's ValidateConfig, which has the configuration.
func (f *VLLMFakeMetrics) ValidateTokenBuckets(maxModelLen int) error {
	buckets := metrics.BuildBuckets(maxModelLen, TokenBucketMantissas)

	for _, h := range []struct {
		name    string
		samples []int
	}{
		{"request-prompt-tokens", f.RequestPromptTokens},
		{"request-generation-tokens", f.RequestGenerationTokens},
		{"request-params-max-tokens", f.RequestParamsMaxTokens},
		{"request-max-generation-tokens", f.RequestMaxGenerationTokens},
	} {
		if err := common.ValidateFakeBuckets(h.name, buckets, h.samples); err != nil {
			return err
		}
	}
	return nil
}
