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

import "github.com/llm-d/llm-d-inference-sim/pkg/common"

// ApplyDefaults fills in the default values of the configuration groups this
// engine owns. Called before a config file is loaded and before BindFlags, so
// that a YAML value overrides a default and a flag overrides both.
func (Engine) ApplyDefaults(cfg *common.Configuration) {
	// Metrics are opt-in, as they are in sglang: its own enable_metrics defaults
	// to false and nothing is served at /metrics until --enable-metrics is
	// given.
	cfg.EnableMetrics = false

	// One LoRA slot, so the simulator's adapter bookkeeping has a table to
	// index. No adapter can occupy it: this engine registers no LoRA flags and
	// claims no LoRA configuration keys.
	cfg.Lora = common.LoraConfig{MaxLoras: 1}

	// The cache's geometry, which the metrics describing it report whether or
	// not it is simulated, so that a scrape names a real block size and block
	// count rather than two zeroes. Simulation stays off: EnableKVCache is left
	// false and ValidateConfig rejects a configuration turning it on. The rest
	// of the group, which only a running cache reads, is left zero-valued.
	cfg.KVCache = common.KVCacheConfig{
		KVCacheSize:    1024,
		TokenBlockSize: 16,
	}
}
