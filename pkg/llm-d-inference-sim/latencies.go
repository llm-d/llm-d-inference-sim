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
)

func (s *VllmSimulator) getCurrLoadFactor() float64 {
	if s.config.MaxNumSeqs <= 1 {
		return 1.0
	}
	return 1 + (s.config.TimeFactorUnderLoad-1)*float64(s.metrics.nRunningReqs-1)/float64(s.config.MaxNumSeqs-1)
}

func (s *VllmSimulator) getTimeToFirstToken() time.Duration {
	return time.Duration(float64(s.config.TimeToFirstToken) * s.getCurrLoadFactor())
}

func (s *VllmSimulator) getPrefillOverhead() time.Duration {
	return time.Duration(float64(s.config.PrefillOverhead) * s.getCurrLoadFactor())
}

func (s *VllmSimulator) getPrefillTimePerToken() time.Duration {
	return time.Duration(float64(s.config.PrefillTimePerToken) * s.getCurrLoadFactor())
}

// returns time to first token based on the current request's doRemotePrefill
func (s *VllmSimulator) getWaitTimeToFirstToken(nPromptTokens int, nCachedPromptTokens int, doRemotePrefill bool) time.Duration {
	if doRemotePrefill {
		if s.config.KVCacheTransferLatency == 0 && s.config.KVCacheTransferLatencyStdDev == 0 {
			// is disaggregated PD and ttft is calculated using number of prompt tokens
			kvCacheTransT := s.config.KVCacheTransferTimePerToken.ToDuration() * time.Duration(nPromptTokens)
			return s.random.RandomNormDuration(kvCacheTransT, s.config.KVCacheTransferTimeStdDev.ToDuration())
		}
		// is disaggregated PD and *not* using number of prompt tokens
		return s.random.RandomNormDuration(s.config.KVCacheTransferLatency.ToDuration(), s.config.KVCacheTransferLatencyStdDev.ToDuration())
	}
	if s.config.TimeToFirstToken == 0 && s.config.TimeToFirstTokenStdDev == 0 {
		// is aggregated PD and ttft is calculated using number of prompt tokens that are not in kv cache
		prefillTime := s.getPrefillOverhead() + time.Duration(nPromptTokens-nCachedPromptTokens)*s.getPrefillTimePerToken()
		return s.random.RandomNormDuration(prefillTime, s.config.PrefillTimeStdDev.ToDuration())
	}
	// is aggregated PD and *not* using number of prompt tokens
	return s.random.RandomNormDuration(s.getTimeToFirstToken(), s.config.TimeToFirstTokenStdDev.ToDuration())
}

// returns inter token latency
func (s *VllmSimulator) getInterTokenLatency() time.Duration {
	latency := time.Duration(float64(s.config.InterTokenLatency) * s.getCurrLoadFactor())
	return s.random.RandomNormDuration(latency, s.config.InterTokenLatencyStdDev.ToDuration())
}
