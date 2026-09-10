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
)

// ValidateConfig checks the vLLM-specific fields of cfg. Called after
// cfg's common fields have already been validated.
func (Engine) ValidateConfig(cfg *common.Configuration) error {
	if cfg.Latencies.KVCacheTransferTimePerToken < 0 {
		return errors.New("kv-cache transfer time per token cannot be negative")
	}
	if cfg.Latencies.KVCacheTransferTimeStdDev < 0 {
		return errors.New("kv-cache transfer time standard deviation cannot be negative")
	}
	// No upper-bound check on KVCacheTransferTimeStdDev for the same reason as
	// PrefillTimeStdDev: it is applied to the total transfer time
	// (n × kv-cache-transfer-time-per-token), which depends on the prompt
	// length n and is unknown at config time. Runtime clamping in
	// RandomNormDuration handles oversized std-devs.

	if cfg.Latencies.KVCacheTransferLatency < 0 {
		return errors.New("kv-cache transfer time cannot be negative")
	}
	if cfg.Latencies.KVCacheTransferLatencyStdDev < 0 {
		return errors.New("kv-cache transfer time standard deviation cannot be negative")
	}
	if float32(cfg.Latencies.KVCacheTransferLatencyStdDev) > 0.3*float32(cfg.Latencies.KVCacheTransferLatency) {
		return errors.New("kv-cache transfer standard deviation cannot be more than 30% of kv-cache transfer")
	}

	if cfg.MaxLoras < 1 {
		return errors.New("max LoRAs cannot be less than 1")
	}
	if cfg.MaxCPULoras == 0 {
		// max CPU LoRAs by default is same as max LoRAs
		cfg.MaxCPULoras = cfg.MaxLoras
	}
	if cfg.MaxCPULoras < cfg.MaxLoras {
		return errors.New("max CPU LoRAs cannot be less than max LoRAs")
	}

	for _, lora := range cfg.LoraModules {
		if lora.Name == "" {
			return errors.New("empty LoRA name")
		}
		if lora.BaseModelName != "" && lora.BaseModelName != cfg.Model {
			return fmt.Errorf("unknown base model '%s' for LoRA '%s'", lora.BaseModelName, lora.Name)
		}
	}

	if cfg.KVCache.EnableKVCache {
		if cfg.KVCache.TokenBlockSize != 8 && cfg.KVCache.TokenBlockSize != 16 && cfg.KVCache.TokenBlockSize != 32 &&
			cfg.KVCache.TokenBlockSize != 64 && cfg.KVCache.TokenBlockSize != 128 {
			return errors.New("token block size should be one of the following: 8, 16, 32, 64, 128")
		}

		if cfg.KVCache.KVCacheSize < 0 {
			return errors.New("KV cache size cannot be negative")
		}
		if cfg.KVCache.EventBatchSize < 1 {
			return errors.New("event batch size cannot less than 1")
		}

		if cfg.KVCache.KVEventsReplayEndpoint != "" && cfg.KVCache.KVEventsReplayQueueSize < 1 {
			return errors.New("kv-events-replay-queue-size cannot be less than 1")
		}

		if err := validateEndpointPortsDontCollide(cfg); err != nil {
			return err
		}
	} else {
		// KV cache is disabled: its sizing/hashing/eventing settings are unused,
		// so report them as all-zero rather than leaking whatever defaults or
		// unused flags/YAML values happened to be set.
		cfg.KVCache = common.KVCacheConfig{}
	}

	if cfg.FakeMetrics != nil {
		if err := cfg.FakeMetrics.Validate(); err != nil {
			return err
		}
		if cfg.FakeMetricsRefreshInterval <= 0 {
			return errors.New("fake metrics refresh interval must be positive")
		}
	}

	if cfg.GlobalCacheHitThreshold < 0 || cfg.GlobalCacheHitThreshold > 1 {
		return errors.New("global cache hit threshold must be between in range [0, 1]")
	}

	return nil
}

// validateEndpointPortsDontCollide ensures the ZMQ publish endpoint and the
// KV-events-replay endpoint don't end up bound to the same port once each
// rank's offset is applied.
//
// This holds even when data-parallel-rank is set to a single fixed value for
// this process: the other ranks of the same cluster are still out there,
// each running with their own fixed rank in 0..data-parallel-size-1 and the
// same base endpoints, so this rank's ZMQ port can still collide with some
// other rank's replay port (or vice versa). The check therefore always
// spans the full 0..DPSize-1 range rather than narrowing to this process's
// own rank.
func validateEndpointPortsDontCollide(cfg *common.Configuration) error {
	if cfg.KVCache.ZMQEndpoint == "" || cfg.KVCache.KVEventsReplayEndpoint == "" {
		return nil
	}

	_, zmqPort, ok := common.ParseEndpointPort(cfg.KVCache.ZMQEndpoint)
	if !ok {
		return nil
	}
	_, replayPort, ok := common.ParseEndpointPort(cfg.KVCache.KVEventsReplayEndpoint)
	if !ok {
		return nil
	}

	// Ports occupied across ranks 0..DPSize-1: [port, port+DPSize-1].
	maxRank := cfg.DPSize - 1
	if zmqPort <= replayPort+maxRank && replayPort <= zmqPort+maxRank {
		return fmt.Errorf("zmq-endpoint (%s) and kv-events-replay-endpoint (%s) ports collide"+
			" once offset by data-parallel rank", cfg.KVCache.ZMQEndpoint, cfg.KVCache.KVEventsReplayEndpoint)
	}
	return nil
}
