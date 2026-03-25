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

package common

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	OscillateFuncName     = "oscillate"
	RampFuncName          = "ramp"
	RampWithResetFuncName = "rampreset"
	SquarewaveFuncName    = "squarewave"
)

type FakeMetrics struct {
	// LoraMetrics
	LoraMetrics []LorasMetrics `json:"loras"`
	LorasString []string       `yaml:"loras"`

	// The fake metrics of type FakeMetricWithFunction can be either a fixed number or a generator
	// function that produces fake metric values over time, using the parameters start, end, and period.
	// Supported functions are:
	//  - oscillate: Generates a smooth sine-wave between start and end over each period.
	//  - ramp: Interpolates linearly from start to end over one period and then stays at end.
	//  - rampreset: Interpolates linearly from start to end over each period, then jumps back to start and repeats.
	//  - squarewave: Alternates between start and end, staying at each level for half of the period.
	// The configuration format is: fun:start:end:period, for example: ramp:10:0:5s or oscillate:0:10:5s.

	// RunningRequests is the number of inference requests that are currently being processed
	RunningRequests FakeMetricWithFunction `yaml:"running-requests" json:"running-requests"`
	// WaitingRequests is the number of inference requests that are waiting to be processed
	WaitingRequests FakeMetricWithFunction `yaml:"waiting-requests" json:"waiting-requests"`
	// KVCacheUsagePercentage  is the fraction of KV-cache blocks currently in use (from 0 to 1)
	KVCacheUsagePercentage FakeMetricWithFunction `yaml:"kv-cache-usage" json:"kv-cache-usage"`

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
	TotalPromptTokens *int64 `json:"total-prompt-tokens,omitempty"`
	// TotalGenerationTokens is the total number of generated tokens
	TotalGenerationTokens *int64 `json:"total-generation-tokens,omitempty"`

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

	// PrefixCacheHits is the initial value for the prefix cache hits counter (in tokens)
	PrefixCacheHits *int64 `yaml:"prefix-cache-hits" json:"prefix-cache-hits,omitempty"`
	// PrefixCacheQueries is the initial value for the prefix cache queries counter (in tokens)
	PrefixCacheQueries *int64 `yaml:"prefix-cache-queries" json:"prefix-cache-queries,omitempty"`
}

type FakeMetricWithFunction struct {
	FixedValue float64
	Function   *FunctionInfo
	IsFunction bool
}

type FunctionInfo struct {
	Name   string
	Start  float64
	End    float64
	Period time.Duration
}

func parseFunc(parts []string) (*FunctionInfo, error) {
	if len(parts) != 4 {
		return nil, errors.New("need func:start:end:period in fake metric generation function")
	}
	start, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		return nil, err
	}
	end, err := strconv.ParseFloat(parts[2], 64)
	if err != nil {
		return nil, err
	}
	period, err := time.ParseDuration(parts[3])
	if err != nil {
		return nil, err
	}
	return &FunctionInfo{Name: parts[0], Start: start, End: end, Period: period}, nil
}

func (f *FakeMetricWithFunction) parseFunction(s string) error {
	parts := strings.Split(s, ":")
	if config, err := parseFunc(parts); err != nil {
		return fmt.Errorf("unknown format in fake metric generation function: %s", err.Error())
	} else {
		f.Function = config
		f.IsFunction = true
		return nil
	}
}

func (f *FakeMetricWithFunction) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind == yaml.ScalarNode {
		// Try number first
		if n, err := strconv.ParseFloat(value.Value, 64); err == nil {
			f.FixedValue = n
			return nil
		}
	}

	return f.parseFunction(value.Value)
}

func (f *FakeMetricWithFunction) UnmarshalJSON(data []byte) error {
	// Try number first
	var n float64
	if err := json.Unmarshal(data, &n); err == nil {
		f.FixedValue = n
		return nil
	}

	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}

	return f.parseFunction(s)
}

type LorasMetrics struct {
	// RunningLoras is a comma separated list of running LoRAs
	RunningLoras string `json:"running"`
	// WaitingLoras is a comma separated list of waiting LoRAs
	WaitingLoras string `json:"waiting"`
	// Timestamp is the timestamp of the metric
	Timestamp float64 `json:"timestamp"`
}

func (c *Configuration) unmarshalFakeMetrics(fakeMetricsString string) error {
	var metrics *FakeMetrics
	if err := json.Unmarshal([]byte(fakeMetricsString), &metrics); err != nil {
		return err
	}
	c.FakeMetrics = metrics
	return nil
}

func (c *Configuration) unmarshalLoraFakeMetrics() error {
	if c.FakeMetrics != nil {
		c.FakeMetrics.LoraMetrics = make([]LorasMetrics, 0)
		for _, jsonStr := range c.FakeMetrics.LorasString {
			var lora LorasMetrics
			if err := json.Unmarshal([]byte(jsonStr), &lora); err != nil {
				return err
			}
			c.FakeMetrics.LoraMetrics = append(c.FakeMetrics.LoraMetrics, lora)
		}
	}
	return nil
}

type fakeMetricsAlias FakeMetrics

// UnmarshalUpdateJSON applies a partial JSON update to the receiver.
// Unlike standard json.Unmarshal, it only overwrites fields whose JSON keys
// are explicitly present in data — unmentioned fields are left unchanged.
// This allows callers to PATCH individual metrics without resetting the rest.
//
// It uses a type alias (fakeMetricsAlias) to decode data without triggering
// custom UnmarshalJSON methods, then copies only the present fields into f
// via reflection.
//
// Returns:
//   - before: a snapshot of f's state before any mutation, used by the caller
//     to detect what changed (e.g. whether to unregister old Prometheus metrics).
//   - updatedKeys: the raw JSON key map (from json.Unmarshal into map[string]any),
//     so the caller knows exactly which keys were supplied. Keys with explicit
//     null values ARE included — this lets callers distinguish "field set to null"
//     (clear the metric) from "field absent" (leave it alone).
//   - err: any JSON decoding error.
func (f *FakeMetrics) UnmarshalUpdateJSON(data []byte) (before *FakeMetrics, updatedKeys map[string]any, err error) {
	// First decode into a raw map to see which keys are present.
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, nil, err
	}

	// Decode into an alias so zero values are set, then selectively copy.
	var aux fakeMetricsAlias
	if err := json.Unmarshal(data, &aux); err != nil {
		return nil, nil, err
	}

	// Snapshot the current state before mutation.
	old := *f
	before = &old

	// Use reflection to copy only present fields from aux into f.
	v := reflect.ValueOf(f).Elem()
	t := v.Type()

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		jsonTag := field.Tag.Get("json")
		// remove ,omitempty
		parts := strings.SplitN(jsonTag, ",", 2)
		jsonTag = strings.TrimSpace(parts[0])

		if _, ok := raw[jsonTag]; ok {
			auxVal := reflect.ValueOf(&aux).Elem().Field(i)
			dest := v.Field(i)

			// Direct copy: auxVal already has correct type/value for JSON fields
			if auxVal.Type() == dest.Type() && auxVal.CanInterface() {
				dest.Set(auxVal)
			}
		}
	}

	return before, raw, nil
}

func (f *FakeMetrics) validate() error {
	if f.RunningRequests.FixedValue < 0 || f.WaitingRequests.FixedValue < 0 {
		return errors.New("fake metrics request counters cannot be negative")
	}
	if f.KVCacheUsagePercentage.FixedValue < 0 || f.KVCacheUsagePercentage.FixedValue > 1 {
		return errors.New("fake metrics KV cache usage must be between 0 and 1")
	}
	if err := f.RunningRequests.Function.validate(); err != nil {
		return err
	}
	if err := f.WaitingRequests.Function.validate(); err != nil {
		return err
	}
	if err := f.KVCacheUsagePercentage.Function.validate(); err != nil {
		return err
	}
	if f.KVCacheUsagePercentage.IsFunction {
		if f.KVCacheUsagePercentage.Function.Start < 0 || f.KVCacheUsagePercentage.Function.Start > 1 ||
			f.KVCacheUsagePercentage.Function.End < 0 || f.KVCacheUsagePercentage.Function.End > 1 {
			return errors.New("fake metrics KV cache usage start and end must be between 0 and 1")
		}
	}

	if f.TTFTBucketValues != nil {
		if len(f.TTFTBucketValues) > len(TTFTBucketsBoundaries)+1 {
			return errors.New("fake time-to-first-token array is too long")
		}
		for _, v := range f.TTFTBucketValues {
			if v < 0 {
				return errors.New("time-to-first-token fake metrics should contain only non-negative values")
			}
		}
	}
	if f.TPOTBucketValues != nil {
		if len(f.TPOTBucketValues) > len(TPOTBucketsBoundaries)+1 {
			return errors.New("fake time-per-output-token array is too long")
		}
		for _, v := range f.TPOTBucketValues {
			if v < 0 {
				return errors.New("time-per-output-token fake metrics should contain only non-negative values")
			}
		}
	}
	if f.RequestSuccessTotal != nil {
		for reason, count := range f.RequestSuccessTotal {
			if count < 0 {
				return fmt.Errorf("fake metrics request-success-total.%s "+
					"cannot be negative, got %d", reason, count)
			}
			if _, ok := validFinishReasons[reason]; !ok {
				return fmt.Errorf("invalid finish reason in request-success-total: "+
					"%s (valid reasons: %v)", reason, requiredFinishReasons)
			}
		}
		for _, reason := range requiredFinishReasons {
			if _, exists := f.RequestSuccessTotal[reason]; !exists {
				f.RequestSuccessTotal[reason] = 0
			}
		}
	}
	for _, v := range f.RequestPromptTokens {
		if v < 0 {
			return errors.New("fake metrics request-prompt-tokens cannot contain negative values")
		}
	}
	for _, v := range f.RequestGenerationTokens {
		if v < 0 {
			return errors.New("fake metrics request-generation-tokens cannot contain negative values")
		}
	}
	for _, v := range f.RequestParamsMaxTokens {
		if v < 0 {
			return errors.New("fake metrics request-params-max-tokens cannot contain negative values")
		}
	}
	for _, v := range f.RequestMaxGenerationTokens {
		if v < 0 {
			return errors.New("fake metrics request-max-generation-tokens cannot contain negative values")
		}
	}

	for _, v := range f.E2ERequestLatencyBucketValues {
		if v < 0 {
			return errors.New("fake metrics e2erl-buckets-values cannot contain negative values")
		}
	}
	for _, v := range f.ReqQueueTimeBucketValues {
		if v < 0 {
			return errors.New("fake metrics queue-time-buckets-values cannot contain negative values")
		}
	}
	for _, v := range f.ReqInfTimeBucketValues {
		if v < 0 {
			return errors.New("fake metrics inf-time-buckets-values cannot contain negative values")
		}
	}
	for _, v := range f.ReqPrefillTimeBucketValues {
		if v < 0 {
			return errors.New("fake metrics prefill-time-buckets-values cannot contain negative values")
		}
	}
	for _, v := range f.ReqDecodeTimeBucketValues {
		if v < 0 {
			return errors.New("fake metrics decode-time-buckets-values cannot contain negative values")
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

func (g *FunctionInfo) validate() error {
	if g == nil {
		return nil
	}
	if g.Name != OscillateFuncName && g.Name != RampFuncName && g.Name != RampWithResetFuncName && g.Name != SquarewaveFuncName {
		return fmt.Errorf("invalid fake metrics generation function %s, must be one of the following: %s, %s, %s, %s",
			g.Name, OscillateFuncName, RampFuncName, RampWithResetFuncName, SquarewaveFuncName)
	}
	if g.End < 0 || g.Start < 0 || g.Period < 0 {
		return errors.New("invalid fake metrics generation parameter: start and end must not be negative")
	}
	if g.Period <= 0 {
		return errors.New("invalid fake metrics generation parameter: period must be positive")
	}
	return nil
}
