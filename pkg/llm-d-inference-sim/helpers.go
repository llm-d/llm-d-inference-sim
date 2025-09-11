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
)

// isValidModel checks if the given model is the base model or one of "loaded" LoRAs
func (s *VllmSimulator) isValidModel(model string) bool {
	for _, name := range s.config.ServedModelNames {
		if model == name {
			return true
		}
	}
	for _, lora := range s.getLoras() {
		if model == lora {
			return true
		}
	}

	return false
}

// isLora returns true if the given model name is one of loaded LoRAs
func (s *VllmSimulator) isLora(model string) bool {
	for _, lora := range s.getLoras() {
		if model == lora {
			return true
		}
	}

	return false
}

// getDisplayedModelName returns the model name that must appear in API
// responses.  LoRA adapters keep their explicit name, while all base-model
// requests are surfaced as the first alias from --served-model-name.
func (s *VllmSimulator) getDisplayedModelName(reqModel string) string {
	if s.isLora(reqModel) {
		return reqModel
	}
	return s.config.ServedModelNames[0]
}

func (s *VllmSimulator) showConfig(dp bool) error {
	cfgJSON, err := json.Marshal(s.config)
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
	s.logger.Info("Configuration:", "", string(cfgJSON))
	return nil
}
