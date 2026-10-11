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

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// ValidateConfig checks the sglang-specific fields of cfg. Called after cfg's
// common fields have already been validated, and again on every admin-config
// update.
//
// These are the features BindFlags accepts a toggle for but this engine cannot
// honor yet. The wording says so rather than calling the setting invalid,
// because the setting is one the engine is expected to grow.
func (Engine) ValidateConfig(cfg *common.Configuration) error {
	if cfg.KVCache.EnableKVCache {
		return errors.New("KV-cache simulation is not implemented by the sglang engine yet")
	}
	if cfg.FakeMetrics != nil {
		return errors.New("fake metrics are not implemented by the sglang engine yet")
	}
	return nil
}
