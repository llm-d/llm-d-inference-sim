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

import "github.com/llm-d/llm-d-inference-sim/pkg/common"

// ApplyDefaults fills in the default values of the configuration groups this
// engine owns. Called before a config file is loaded and before BindFlags, so
// that a YAML value overrides a default and a flag overrides both.
func (Engine) ApplyDefaults(cfg *common.Configuration) {
	cfg.Lora = common.LoraConfig{MaxLoras: 1}
	cfg.KVCache = common.KVCacheConfig{
		KVCacheSize:             1024,
		KVCacheDType:            "auto",
		TokenBlockSize:          16,
		ZMQEndpoint:             "tcp://127.0.0.1:5557",
		KVEventsReplayQueueSize: 1024,
		EventBatchSize:          16,
	}
}
