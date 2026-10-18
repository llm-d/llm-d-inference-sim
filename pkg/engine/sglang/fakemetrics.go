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

package sglang

import (
	"errors"
	"fmt"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// SGLangFakeMetrics implements common.FakeMetrics for the sglang backend.
//
// The fakeable surface is this engine's load metrics. The gauges describing the
// cache geometry and the context window keep the values stamped at startup,
// since they report the configuration the simulator was given rather than
// anything a request changes.
//
// A metric this names nothing for reports nothing: real and fake metrics are
// never served together, so a scrape under fake metrics carries what the
// configuration sets and the startup gauges, and no other series at all.
//
// No total is derived from a histogram the way the vLLM engine derives its token
// counters: sglang splits prompt_tokens_total by is_streaming but reports
// prompt_tokens_histogram without the label, so one carries no per-series sum
// for the other. The totals are set on their own.
type SGLangFakeMetrics struct {
	// All fields are pointers so that absence (nil) can be distinguished from a
	// zero value during partial updates via /admin/config -- only non-nil fields
	// are applied.

	// RunningRequests is the number of inference requests currently being processed
	RunningRequests *common.FakeMetricWithFunction `yaml:"running-requests" json:"running-requests,omitempty"`
	// QueuedRequests is the number of inference requests waiting to be processed
	QueuedRequests *common.FakeMetricWithFunction `yaml:"queue-requests" json:"queue-requests,omitempty"`
	// TokenUsage is the fraction of the KV cache's token capacity in use (from 0
	// to 1). The absolute token counts reported beside it are derived from it and
	// the configured capacity, as they are for a real cache.
	TokenUsage *common.FakeMetricWithFunction `yaml:"token-usage" json:"token-usage,omitempty"`

	// TotalPromptTokens is the number of prefill tokens processed
	TotalPromptTokens *StreamSplit `yaml:"total-prompt-tokens" json:"total-prompt-tokens,omitempty"`
	// TotalGenerationTokens is the number of generation tokens processed
	TotalGenerationTokens *StreamSplit `yaml:"total-generation-tokens" json:"total-generation-tokens,omitempty"`
	// NumRequestsTotal is the number of requests processed
	NumRequestsTotal *StreamSplit `yaml:"num-requests-total" json:"num-requests-total,omitempty"`

	// CachedTokens and QueriedTokens are the prefix cache's cumulative tallies,
	// in tokens. The first is reported on its own and the two together as the
	// hit rate, so neither is meaningful without the other.
	CachedTokens  *int64 `yaml:"cached-tokens" json:"cached-tokens,omitempty"`
	QueriedTokens *int64 `yaml:"queried-tokens" json:"queried-tokens,omitempty"`

	// Histogram metrics are defined by an array of bucket sample counts, in
	// bucket order. An array may be shorter than the number of buckets, which
	// leaves the trailing ones empty. Every sample is observed at its bucket's
	// upper boundary, so a histogram's _sum follows from which buckets are
	// filled rather than being configurable on its own.
	//
	// The three sglang labels by is_streaming take one array per series; the
	// rest take a single array.

	// TTFTBucketValues fills sglang:time_to_first_token_seconds, over the
	// boundaries of sglang's --bucket-time-to-first-token.
	TTFTBucketValues *StreamSplitBuckets `yaml:"ttft-buckets-values" json:"ttft-buckets-values,omitempty"`
	// ReqTPOTBucketValues fills sglang:request_time_per_output_token_seconds,
	// over the boundaries of sglang's --bucket-inter-token-latency.
	ReqTPOTBucketValues *StreamSplitBuckets `yaml:"request-tpot-buckets-values" json:"request-tpot-buckets-values,omitempty"`
	// E2EReqLatencyBucketValues fills sglang:e2e_request_latency_seconds, over
	// the boundaries of sglang's --bucket-e2e-request-latency.
	E2EReqLatencyBucketValues *StreamSplitBuckets `yaml:"e2erl-buckets-values" json:"e2erl-buckets-values,omitempty"`

	// InterTokenLatencyBucketValues fills sglang:inter_token_latency_seconds,
	// over the same boundaries as the per-request TPOT histogram.
	InterTokenLatencyBucketValues []int `yaml:"itl-buckets-values" json:"itl-buckets-values,omitempty"`
	// QueueTimeBucketValues fills sglang:queue_time_seconds.
	QueueTimeBucketValues []int `yaml:"queue-time-buckets-values" json:"queue-time-buckets-values,omitempty"`
	// PromptTokensBucketValues fills sglang:prompt_tokens_histogram.
	PromptTokensBucketValues []int `yaml:"prompt-tokens-buckets-values" json:"prompt-tokens-buckets-values,omitempty"`
	// UncachedPromptTokensBucketValues fills sglang:uncached_prompt_tokens_histogram.
	UncachedPromptTokensBucketValues []int `yaml:"uncached-prompt-tokens-buckets-values" json:"uncached-prompt-tokens-buckets-values,omitempty"`
	// GenerationTokensBucketValues fills sglang:generation_tokens_histogram.
	GenerationTokensBucketValues []int `yaml:"generation-tokens-buckets-values" json:"generation-tokens-buckets-values,omitempty"`
}

// StreamSplitBuckets holds the bucket samples of both series of a histogram
// sglang labels by is_streaming. A nil series is one the configuration does not
// name, which leaves that series absent from a scrape.
type StreamSplitBuckets struct {
	Streaming    []int `yaml:"streaming" json:"streaming,omitempty"`
	NonStreaming []int `yaml:"non-streaming" json:"non-streaming,omitempty"`
}

// StreamSplit holds both series of a metric sglang labels by is_streaming. Each
// series is a pointer, so a series the configuration does not name is absent
// from a scrape rather than reporting an explicit zero.
type StreamSplit struct {
	Streaming    *int64 `yaml:"streaming" json:"streaming,omitempty"`
	NonStreaming *int64 `yaml:"non-streaming" json:"non-streaming,omitempty"`
}

// New returns a fresh, zero-valued *SGLangFakeMetrics.
func (f *SGLangFakeMetrics) New() common.FakeMetrics {
	return &SGLangFakeMetrics{}
}

// Validate checks the fake-metrics configuration. Called by the sglang engine's
// ValidateConfig (fake metrics are an engine-specific feature).
func (f *SGLangFakeMetrics) Validate() error {
	if err := common.ValidateFakeCount("running-requests", f.RunningRequests); err != nil {
		return err
	}
	if err := common.ValidateFakeCount("queue-requests", f.QueuedRequests); err != nil {
		return err
	}
	if err := common.ValidateFakeFraction("token-usage", f.TokenUsage); err != nil {
		return err
	}

	if err := f.TotalPromptTokens.validate("total-prompt-tokens"); err != nil {
		return err
	}
	if err := f.TotalGenerationTokens.validate("total-generation-tokens"); err != nil {
		return err
	}
	if err := f.NumRequestsTotal.validate("num-requests-total"); err != nil {
		return err
	}

	for _, h := range []struct {
		name    string
		buckets []float64
		split   *StreamSplitBuckets
		unsplit []int
	}{
		{name: "ttft-buckets-values", buckets: ttftBucketsBoundaries, split: f.TTFTBucketValues},
		{name: "request-tpot-buckets-values", buckets: interTokenLatencyBucketsBoundaries, split: f.ReqTPOTBucketValues},
		{name: "e2erl-buckets-values", buckets: e2eReqLatencyBucketsBoundaries, split: f.E2EReqLatencyBucketValues},
		{name: "itl-buckets-values", buckets: interTokenLatencyBucketsBoundaries, unsplit: f.InterTokenLatencyBucketValues},
		{name: "queue-time-buckets-values", buckets: queueTimeBucketsBoundaries, unsplit: f.QueueTimeBucketValues},
		{name: "prompt-tokens-buckets-values", buckets: tokenBucketsBoundaries, unsplit: f.PromptTokensBucketValues},
		{name: "uncached-prompt-tokens-buckets-values", buckets: tokenBucketsBoundaries, unsplit: f.UncachedPromptTokensBucketValues},
		{name: "generation-tokens-buckets-values", buckets: tokenBucketsBoundaries, unsplit: f.GenerationTokensBucketValues},
	} {
		if h.split != nil {
			if err := common.ValidateFakeBuckets(h.name, h.buckets, h.split.Streaming); err != nil {
				return err
			}
			if err := common.ValidateFakeBuckets(h.name, h.buckets, h.split.NonStreaming); err != nil {
				return err
			}
			continue
		}
		if err := common.ValidateFakeBuckets(h.name, h.buckets, h.unsplit); err != nil {
			return err
		}
	}

	if (f.CachedTokens == nil) != (f.QueriedTokens == nil) {
		return errors.New("fake metrics cached-tokens and queried-tokens must be specified together")
	}
	if f.CachedTokens != nil {
		if *f.CachedTokens < 0 || *f.QueriedTokens < 0 {
			return errors.New("fake metrics cached-tokens and queried-tokens cannot be negative")
		}
		if *f.CachedTokens > *f.QueriedTokens {
			return errors.New("fake metrics cached-tokens cannot exceed queried-tokens")
		}
	}

	return nil
}

// validate checks both of s's series. name identifies the metric in an error.
func (s *StreamSplit) validate(name string) error {
	if s == nil {
		return nil
	}
	if (s.Streaming != nil && *s.Streaming < 0) || (s.NonStreaming != nil && *s.NonStreaming < 0) {
		return fmt.Errorf("fake metrics %s cannot be negative", name)
	}
	return nil
}
