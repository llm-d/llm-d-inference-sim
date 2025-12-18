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

// Package vllmsim implements the vLLM simulator.
package llmdinferencesim

import (
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

type TTFTParams struct {
	PromptTokens       int
	CachedPromptTokens int
	DoRemotePrefill    bool
	RunningReqs        int64
}

type InterTokenParams struct {
	RunningReqs int64
}

type LatencyCalculator interface {
	// GetTimeToFirstToken returns time to first token. The simulator will wait
	// this amount of time before generating the first token.
	GetTimeToFirstToken(params *TTFTParams) time.Duration
	// GetInterTokenLatency returns inter-token latency. The simulator will wait
	// this amount of time before generating each response token (except the first one).
	GetInterTokenLatency(params *InterTokenParams) time.Duration
}

type baseCalculator struct {
	interTokenLatency       int
	interTokenLatencyStdDev int
	timeFactorUnderLoad     float64
	maxNumSeqs              int
	random                  *common.Random
}

// returns inter token latency
func (b *baseCalculator) GetInterTokenLatency(params *InterTokenParams) time.Duration {
	latency := int(float64(b.interTokenLatency) * b.getCurrLoadFactor(params.RunningReqs))
	return b.randomNorm(latency, b.interTokenLatencyStdDev)
}

func (b *baseCalculator) getCurrLoadFactor(nRunningReqs int64) float64 {
	if b.maxNumSeqs <= 1 {
		return 1.0
	}
	return 1 + (b.timeFactorUnderLoad-1)*float64(nRunningReqs-1)/float64(b.maxNumSeqs-1)
}

func (b *baseCalculator) randomNorm(mean int, stdDev int) time.Duration {
	return time.Duration(b.random.RandomNormTruncated(mean, stdDev)) * time.Millisecond
}

type defaultCalculator struct {
	baseCalculator
	timeToFirstToken             int
	timeToFirstTokenStdDev       int
	kVCacheTransferLatency       int
	kVCacheTransferLatencyStdDev int
	kVCacheTransferTimePerToken  int
	kVCacheTransferTimeStdDev    int
	prefillOverhead              int
	prefillTimePerToken          int
	prefillTimeStdDev            int
}

func newDefaultCalculator(config *common.Configuration, random *common.Random) *defaultCalculator {
	return &defaultCalculator{
		baseCalculator: baseCalculator{
			interTokenLatency:       config.InterTokenLatency,
			interTokenLatencyStdDev: config.InterTokenLatencyStdDev,
			timeFactorUnderLoad:     config.TimeFactorUnderLoad,
			maxNumSeqs:              config.MaxNumSeqs,
			random:                  random,
		},
		timeToFirstToken:             config.TimeToFirstToken,
		timeToFirstTokenStdDev:       config.TimeToFirstTokenStdDev,
		kVCacheTransferLatency:       config.KVCacheTransferLatency,
		kVCacheTransferLatencyStdDev: config.KVCacheTransferLatencyStdDev,
		kVCacheTransferTimePerToken:  config.KVCacheTransferTimePerToken,
		kVCacheTransferTimeStdDev:    config.KVCacheTransferTimeStdDev,
		prefillOverhead:              config.PrefillOverhead,
		prefillTimePerToken:          config.PrefillTimePerToken,
		prefillTimeStdDev:            config.PrefillTimeStdDev,
	}
}

// returns time to first token
func (d *defaultCalculator) GetTimeToFirstToken(params *TTFTParams) time.Duration {
	if params.DoRemotePrefill {
		if d.kVCacheTransferLatency == 0 && d.kVCacheTransferLatencyStdDev == 0 {
			// is disaggregated PD and ttft is calculated using number of prompt tokens
			kvCacheTransT := d.kVCacheTransferTimePerToken * params.PromptTokens
			return d.randomNorm(kvCacheTransT, d.kVCacheTransferTimeStdDev)
		}
		// is disaggregated PD and *not* using number of prompt tokens
		return d.randomNorm(d.kVCacheTransferLatency, d.kVCacheTransferLatencyStdDev)
	}
	if d.timeToFirstToken == 0 && d.timeToFirstTokenStdDev == 0 {
		// is aggregated PD and ttft is calculated using number of prompt tokens that are not in kv cache
		prefillOverhead := int(float64(d.prefillOverhead) * d.getCurrLoadFactor(params.RunningReqs))
		prefillTimePerToken := int(float64(d.prefillTimePerToken) * d.getCurrLoadFactor(params.RunningReqs))
		prefillTime := prefillOverhead + (params.PromptTokens-params.CachedPromptTokens)*prefillTimePerToken
		return d.randomNorm(prefillTime, d.prefillTimeStdDev)
	}
	// is aggregated PD and *not* using number of prompt tokens
	ttft := int(float64(d.timeToFirstToken) * d.getCurrLoadFactor(params.RunningReqs))
	return d.randomNorm(ttft, d.timeToFirstTokenStdDev)
}
