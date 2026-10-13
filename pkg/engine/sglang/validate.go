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

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// loraTargetAll is the lora-target-modules value that targets every module.
const loraTargetAll = "all"

// loraTargetModules are the modules sglang can apply LoRA to, from
// SUPPORTED_LORA_TARGET_MODULES in sglang's utils/common.py.
var loraTargetModules = []string{
	"q_proj", "k_proj", "v_proj", "o_proj", "q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj",
	"wq_b", "wk", "weights_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj", "gate_up_proj",
	"embed_tokens", "lm_head", "qkvr", "wo_ud",
}

// ValidateConfig checks the sglang-specific fields of cfg. Called after cfg's
// common fields have already been validated, and again on every admin-config
// update.
//
// The LoRA rules are sglang's own, from check_lora_server_args, which also
// resolves an unset enable-lora: LoRA paths imply it, and an explicit
// enable-lora=false drops them. The other two are the features BindFlags
// accepts a toggle for but this engine cannot honor yet. The wording says so
// rather than calling the setting invalid, because the setting is one the
// engine is expected to grow.
func (Engine) ValidateConfig(cfg *common.Configuration) error {
	if err := validateLora(&cfg.Lora); err != nil {
		return err
	}

	if cfg.KVCache.EnableKVCache {
		return errors.New("KV-cache simulation is not implemented by the sglang engine yet")
	}
	if cfg.FakeMetrics != nil {
		return errors.New("fake metrics are not implemented by the sglang engine yet")
	}
	return nil
}

func validateLora(lora *common.LoraConfig) error {
	if lora.MaxLoras < 1 {
		return errors.New("max_loras_per_batch must be positive")
	}

	// sglang's argparse checks these choices whether LoRA is enabled or not.
	for _, module := range lora.TargetModules {
		if module != loraTargetAll && !slices.Contains(loraTargetModules, module) {
			return fmt.Errorf("invalid choice for lora-target-modules: '%s' (choose from %s, %s)",
				module, strings.Join(loraTargetModules, ", "), loraTargetAll)
		}
	}

	if len(lora.LoraModules) > 0 {
		if lora.EnableLora == nil {
			enabled := true
			lora.EnableLora = &enabled
		} else if !*lora.EnableLora {
			lora.LoraModules = nil
		}
	}
	if lora.EnableLora == nil {
		enabled := false
		lora.EnableLora = &enabled
	}
	if !*lora.EnableLora {
		return nil
	}

	if slices.Contains(lora.TargetModules, loraTargetAll) && len(lora.TargetModules) != 1 {
		return errors.New("if 'all' is specified in --lora-target-modules, it should be the only module specified")
	}
	if len(lora.LoraModules) == 0 && (lora.MaxLoraRank == 0 || len(lora.TargetModules) == 0) {
		return errors.New("when no initial --lora-paths is provided, you need to specify both --max-lora-rank " +
			"and --lora-target-modules for LoRA initialization")
	}

	if lora.MaxCPULoras != 0 {
		if lora.MaxCPULoras < lora.MaxLoras {
			return fmt.Errorf("max_loaded_loras should be greater than or equal to max_loras_per_batch. "+
				"max_loaded_loras=%d, max_loras_per_batch=%d", lora.MaxCPULoras, lora.MaxLoras)
		}
		if len(lora.LoraModules) > lora.MaxCPULoras {
			return fmt.Errorf("the number of LoRA paths should not exceed max_loaded_loras. "+
				"max_loaded_loras=%d, lora_paths=%d", lora.MaxCPULoras, len(lora.LoraModules))
		}
	}

	// sglang loads the paths one by one at startup, failing on the first one
	// its registry would refuse.
	var names []string
	pinned := 0
	for _, module := range lora.LoraModules {
		if slices.Contains(names, module.Name) {
			return fmt.Errorf("failed to load LoRA adapter %s: Failed to load LoRA adapter %s because it is already loaded",
				module.Name, module.Name)
		}
		if module.Pinned && pinned >= lora.MaxLoras-1 {
			return fmt.Errorf("failed to load LoRA adapter %s: Failed to load LoRA adapter %s as a pinned adapter. "+
				"It is not allowed to pin all slots in the LoRA memory pool to avoid starvation for unpinned adapters "+
				"and base models. Please increase your `--max-loras-per-batch` or load it as unpinned LoRA adapters",
				module.Name, module.Name)
		}
		names = append(names, module.Name)
		if module.Pinned {
			pinned++
		}
	}
	return nil
}
