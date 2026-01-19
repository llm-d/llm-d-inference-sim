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
	interTokenLatency       time.Duration
	interTokenLatencyStdDev time.Duration
	timeFactorUnderLoad     float64
	maxNumSeqs              int
	random                  *common.Random
}

// returns inter token latency
func (b *baseCalculator) GetInterTokenLatency(params *InterTokenParams) time.Duration {
	latency := time.Duration(float64(b.interTokenLatency) * b.getCurrLoadFactor(params.RunningReqs))
	return b.random.RandomNormDuration(latency, b.interTokenLatencyStdDev)
}

func (b *baseCalculator) getCurrLoadFactor(nRunningReqs int64) float64 {
	if b.maxNumSeqs <= 1 {
		return 1.0
	}
	return 1 + (b.timeFactorUnderLoad-1)*float64(nRunningReqs-1)/float64(b.maxNumSeqs-1)
}

// Default latency calculator. Decides whether to use per token or
// constant latency calculations based on the values of time-to-first-token
// and kv-cache-transfer-latency.
type defaultCalculator struct {
	baseCalculator
	timeToFirstToken             time.Duration
	timeToFirstTokenStdDev       time.Duration
	kVCacheTransferLatency       time.Duration
	kVCacheTransferLatencyStdDev time.Duration
	kVCacheTransferTimePerToken  time.Duration
	kVCacheTransferTimeStdDev    time.Duration
	prefillOverhead              time.Duration
	prefillTimePerToken          time.Duration
	prefillTimeStdDev            time.Duration
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
			kvCacheTransT := d.kVCacheTransferTimePerToken * time.Duration(params.PromptTokens)
			return d.random.RandomNormDuration(kvCacheTransT, d.kVCacheTransferTimeStdDev)
		}
		// is disaggregated PD and *not* using number of prompt tokens
		return d.random.RandomNormDuration(d.kVCacheTransferLatency, d.kVCacheTransferLatencyStdDev)
	}
	if d.timeToFirstToken == 0 && d.timeToFirstTokenStdDev == 0 {
		// is aggregated PD and ttft is calculated using number of prompt tokens that are not in kv cache
		prefillOverhead := time.Duration(float64(d.prefillOverhead) * d.getCurrLoadFactor(params.RunningReqs))
		prefillTimePerToken := time.Duration(float64(d.prefillTimePerToken) * d.getCurrLoadFactor(params.RunningReqs))
		prefillTime := prefillOverhead + time.Duration(params.PromptTokens-params.CachedPromptTokens)*prefillTimePerToken
		return d.random.RandomNormDuration(prefillTime, d.prefillTimeStdDev)
	}

	// is aggregated PD and *not* using number of prompt tokens
	ttft := time.Duration(float64(d.timeToFirstToken) * d.getCurrLoadFactor(params.RunningReqs))
	return d.random.RandomNormDuration(ttft, d.timeToFirstTokenStdDev)
}

// Constant latency calculator doesn't take the prompt size into account, uses
// time-to-first-token and kv-cache-transfer-latency and their std devs.
type constantCalculator struct {
	baseCalculator
	timeToFirstToken             time.Duration
	timeToFirstTokenStdDev       time.Duration
	kVCacheTransferLatency       time.Duration
	kVCacheTransferLatencyStdDev time.Duration
}

func newConstantCalculator(config *common.Configuration, random *common.Random) *constantCalculator {
	return &constantCalculator{
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
	}
}

// returns time to first token
func (c *constantCalculator) GetTimeToFirstToken(params *TTFTParams) time.Duration {
	if params.DoRemotePrefill {
		// is disaggregated PD and *not* using number of prompt tokens
		return c.random.RandomNormDuration(c.kVCacheTransferLatency, c.kVCacheTransferLatencyStdDev)
	}
	// is aggregated PD and *not* using number of prompt tokens
	ttft := time.Duration(float64(c.timeToFirstToken) * c.getCurrLoadFactor(params.RunningReqs))
	return c.random.RandomNormDuration(ttft, c.timeToFirstTokenStdDev)
}

// Per token calculator takes the prompt size into account
type perTokenCalculator struct {
	baseCalculator
	kVCacheTransferTimePerToken time.Duration
	kVCacheTransferTimeStdDev   time.Duration
	prefillOverhead             time.Duration
	prefillTimePerToken         time.Duration
	prefillTimeStdDev           time.Duration
}

func newPerTokenCalculator(config *common.Configuration, random *common.Random) *perTokenCalculator {
	return &perTokenCalculator{
		baseCalculator: baseCalculator{
			interTokenLatency:       config.InterTokenLatency,
			interTokenLatencyStdDev: config.InterTokenLatencyStdDev,
			timeFactorUnderLoad:     config.TimeFactorUnderLoad,
			maxNumSeqs:              config.MaxNumSeqs,
			random:                  random,
		},
		kVCacheTransferTimePerToken: config.KVCacheTransferTimePerToken,
		kVCacheTransferTimeStdDev:   config.KVCacheTransferTimeStdDev,
		prefillOverhead:             config.PrefillOverhead,
		prefillTimePerToken:         config.PrefillTimePerToken,
		prefillTimeStdDev:           config.PrefillTimeStdDev,
	}
}

// returns time to first token
func (p *perTokenCalculator) GetTimeToFirstToken(params *TTFTParams) time.Duration {
	if params.DoRemotePrefill {
		// is disaggregated PD and ttft is calculated using number of prompt tokens
		kvCacheTransT := p.kVCacheTransferTimePerToken * time.Duration(params.PromptTokens)
		return p.random.RandomNormDuration(kvCacheTransT, p.kVCacheTransferTimeStdDev)
	}
	// is aggregated PD and ttft is calculated using number of prompt tokens that are not in kv cache
	prefillOverhead := time.Duration(float64(p.prefillOverhead) * p.getCurrLoadFactor(params.RunningReqs))
	prefillTimePerToken := time.Duration(float64(p.prefillTimePerToken) * p.getCurrLoadFactor(params.RunningReqs))
	prefillTime := prefillOverhead + time.Duration(params.PromptTokens-params.CachedPromptTokens)*prefillTimePerToken
	return p.random.RandomNormDuration(prefillTime, p.prefillTimeStdDev)
}
