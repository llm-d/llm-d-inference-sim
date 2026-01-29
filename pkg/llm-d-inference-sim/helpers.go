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
	"encoding/json"
	"fmt"

	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

// isValidModel checks if the given model is the base model or one of "loaded" LoRAs
func (s *VllmSimulator) isValidModel(model string) bool {
	for _, name := range s.context.config.ServedModelNames {
		if model == name {
			return true
		}
	}
	for _, lora := range s.context.getLoras() {
		if model == lora {
			return true
		}
	}

	return false
}

func (s *VllmSimulator) showConfig(dp bool) error {
	cfgJSON, err := json.Marshal(s.context.config)
	if err != nil {
		return fmt.Errorf("failed to marshal configuration to JSON: %w", err)
	}

	var m map[string]interface{}
	err = json.Unmarshal(cfgJSON, &m)
	if err != nil {
		return fmt.Errorf("failed to unmarshal JSON to map: %w", err)
	}
	if dp {
		// remove the port
		delete(m, "port")
	}
	// clean LoraModulesString field
	m["lora-modules"] = m["LoraModules"]
	delete(m, "LoraModules")
	delete(m, "LoraModulesString")

	// clean fake-metrics field
	if field, ok := m["fake-metrics"].(map[string]interface{}); ok {
		delete(field, "LorasString")
	}

	// show in JSON
	cfgJSON, err = json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal configuration to JSON: %w", err)
	}
	s.context.logger.V(logging.INFO).Info("Configuration:", "", string(cfgJSON))
	return nil
}

func getNumberOfPromptTokens(req openaiserverapi.Request) int {
	return len(req.TokenizedPrompt().Strings) // TODO: Change to Tokens
}

func validateRequest(req openaiserverapi.Request) (string, int) {
	if req.GetMaxCompletionTokens() != nil && *req.GetMaxCompletionTokens() <= 0 {
		return "Max completion tokens and max tokens should be positive", fasthttp.StatusBadRequest
	}

	if req.IsDoRemoteDecode() && req.IsStream() {
		return "Prefill does not support streaming", fasthttp.StatusBadRequest
	}

	if req.GetIgnoreEOS() && req.GetMaxCompletionTokens() == nil {
		return "Ignore_eos is true but max_completion_tokens (or max_tokens) is not set", fasthttp.StatusBadRequest
	}

	return "", fasthttp.StatusOK
}
