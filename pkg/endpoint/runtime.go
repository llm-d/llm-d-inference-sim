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
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

// Runtime is the seam through which request-processing code and the
// transport layer (pkg/communication) reach the simulator engine. It is
// satisfied implicitly by *simulator.SimContext, which keeps this package
// free of any dependency on the simulator package.
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
	// Sleep transitions the simulator into sleep mode. Returns whether it
	// actually slept.
	Sleep() bool
	// WakeUp wakes the simulator, activating the KV cache when
	// activateKVCache is true and KV cache support is enabled.
	WakeUp(activateKVCache bool)
	// IsSleeping reports whether the simulator is currently sleeping.
	IsSleeping() bool
	// ValidateBaseModel checks that model is a known base model, rejecting
	// LoRA adapters.
	ValidateBaseModel(model string) *api.Error
	// ShouldSendImage decides whether an Omni response should include an
	// image. headerOverride, when true, forces an image regardless of the
	// emission rate.
	ShouldSendImage(headerOverride bool) bool
	// MooncakeEngineMap returns the dp_rank -> {engine_id} map served by
	// /query, stable for the simulator's lifetime.
	MooncakeEngineMap() map[string]map[string]string
	// CreateEmbeddings computes embedding vectors for req.
	CreateEmbeddings(req *api.EmbeddingRequest) (*api.EmbeddingResponse, *api.Error)
	// MetricsRegistry returns the simulator's Prometheus registry.
	MetricsRegistry() *prometheus.Registry
	// LoadLoraAdaptor loads a LoRA adapter described by body.
	LoadLoraAdaptor(body []byte) error
	// UnloadLoraAdaptor unloads a LoRA adapter described by body.
	UnloadLoraAdaptor(body []byte) error
	// CreateModelsResponse lists the base model and any loaded LoRA adapters.
	CreateModelsResponse() *api.ModelsResponse
	// ApplyConfigUpdate validates and applies a partial admin-config update.
	ApplyConfigUpdate(body []byte) error
	// UpdateFakeMetricsFromBody applies a partial fake-metrics update.
	UpdateFakeMetricsFromBody(body []byte) error
}
