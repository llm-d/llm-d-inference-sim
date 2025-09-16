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

import "github.com/llm-d/llm-d-inference-sim/pkg/common"

func (s *VllmSimulator) getCurrLoadFactor() float64 {
	if s.config.MaxNumSeqs <= 1 {
		return 1.0
	}
	return 1 + (s.config.TimeFactorUnderLoad-1)*float64(s.nRunningReqs-1)/float64(s.config.MaxNumSeqs-1)
}

func (s *VllmSimulator) getTimeToFirstToken() int {
	return int(float64(s.config.TimeToFirstToken) * s.getCurrLoadFactor())
}

func (s *VllmSimulator) getPrefillOverhead() int {
	return int(float64(s.config.PrefillOverhead) * s.getCurrLoadFactor())
}

func (s *VllmSimulator) getPrefillTimePerToken() int {
	return int(float64(s.config.PrefillTimePerToken) * s.getCurrLoadFactor())
}

// returns time to first token based on the current request's doRemotePrefill
func (s *VllmSimulator) getWaitTimeToFirstToken(nPromptTokens int, nCachedPromptTokens int, doRemotePrefill bool) int {
	if doRemotePrefill {
		if s.config.KVCacheTransferLatency == 0 && s.config.KVCacheTransferLatencyStdDev == 0 {
			// is disaggregated PD and ttft is calculated using number of prompt tokens
			kvCacheTransT := s.config.KVCacheTransferTimePerToken * nPromptTokens
			return common.RandomNorm(kvCacheTransT, s.config.KVCacheTransferTimeStdDev)
		}
		// is disaggregated PD and *not* using number of prompt tokens
		return common.RandomNorm(s.config.KVCacheTransferLatency, s.config.KVCacheTransferLatencyStdDev)
	}
	if s.config.TimeToFirstToken == 0 && s.config.TimeToFirstTokenStdDev == 0 {
		// is aggregated PD and ttft is calculated using number of prompt tokens that are not in kv cache
		prefillTime := s.getPrefillOverhead() + (nPromptTokens-nCachedPromptTokens)*s.getPrefillTimePerToken()
		return common.RandomNorm(prefillTime, s.config.PrefillTimeStdDev)
	}
	// is aggregated PD and *not* using number of prompt tokens
	return common.RandomNorm(s.getTimeToFirstToken(), s.config.TimeToFirstTokenStdDev)
}

// returns inter token latency
func (s *VllmSimulator) getInterTokenLatency() int {
	latency := int(float64(s.config.InterTokenLatency) * s.getCurrLoadFactor())
	return common.RandomNorm(latency, s.config.InterTokenLatencyStdDev)
}
