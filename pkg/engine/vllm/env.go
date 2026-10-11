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
	"os"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const (
	// devModeEnv enables vLLM's development-only endpoints, which is what
	// gates sleep mode.
	devModeEnv = "VLLM_SERVER_DEV_MODE"
	// pythonHashSeedEnv is read when the --hash-seed flag is not passed; see
	// configuration precedence in the docs.
	pythonHashSeedEnv = "PYTHONHASHSEED"
)

// ApplyEnv applies this engine's own environment-variable settings to cfg.
// Called after the flags have been parsed and before validation.
func (Engine) ApplyEnv(cfg *common.Configuration, changed func(flag string) bool) {
	cfg.DevMode = os.Getenv(devModeEnv) == "1"

	// hash-seed is registered by BindFlags, so a command-line value wins over
	// the env var.
	if !changed("hash-seed") {
		if v := os.Getenv(pythonHashSeedEnv); v != "" {
			cfg.KVCache.HashSeed = v
		}
	}
}
