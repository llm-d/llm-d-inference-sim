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

package endpoint

import (
	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

// Runtime is the seam through which request processing reaches the
// simulator engine it runs on. It is satisfied implicitly by
// *simulator.SimContext, which keeps this package free of any dependency on
// the simulator package.
type Runtime interface {
	// Config returns the simulator's current configuration.
	Config() *common.Configuration
	// GetRandom returns the simulator's configured random source.
	GetRandom() *common.Random
	// GetTokenizer returns the simulator's tokenizer.
	GetTokenizer() tokenizer.Tokenizer
	// Logger returns the simulator's logger.
	Logger() logr.Logger
	// RequestStarted records that req has begun processing: increments the
	// running-request metric and, if req targets a LoRA, stamps its LoRA ID
	// and marks the LoRA as running.
	RequestStarted(req api.Request)
	// GetResponseTokens generates response tokens for req from the
	// simulator's configured dataset.
	GetResponseTokens(req api.Request) (*api.Tokenized, string, error)
	// KVCacheOnRequestStart records req's arrival in the KV cache, if enabled.
	KVCacheOnRequestStart(req api.Request) (kvcache.PrefixCacheStats, *api.Error)
	// KVCacheOnRequestEnd records the request's completion in the KV cache, if enabled.
	KVCacheOnRequestEnd(requestID string)
}
