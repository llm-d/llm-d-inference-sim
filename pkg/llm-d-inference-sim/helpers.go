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
	"fmt"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

// isValidModel checks if the given model is the base model or one of "loaded" LoRAs
func (s *VllmSimulator) isValidModel(model string) bool {
	for _, name := range s.Context.Config.ServedModelNames {
		if model == name {
			return true
		}
	}
	for _, lora := range s.Context.getLoras() {
		if model == lora {
			return true
		}
	}

	return false
}

// ValidateBaseModel checks that model is a known base model. LoRA adapters
// are rejected because the render endpoints tokenize against the base model
// and don't go through the LoRA loading path.
func (s *VllmSimulator) ValidateBaseModel(model string) (string, int) {
	if !s.isValidModel(model) {
		return fmt.Sprintf("The model `%s` does not exist.", model), fasthttp.StatusNotFound
	}
	if s.Context.isLora(model) {
		return fmt.Sprintf("The model `%s` is a LoRA adapter and is not supported by the render endpoints.",
			model), fasthttp.StatusBadRequest
	}
	return "", 0
}

func getNumberOfPromptTokens(req openaiserverapi.Request) int {
	return req.TokenizedPrompt().Length()
}

func validateRequest(req openaiserverapi.Request) (string, int) {
	if req.GetMaxCompletionTokens() != nil && *req.GetMaxCompletionTokens() <= 0 {
		return common.InvalidMaxTokensErrMsg, fasthttp.StatusBadRequest
	}

	if req.IsDoRemoteDecode() && req.IsStream() {
		return "Prefill does not support streaming", fasthttp.StatusBadRequest
	}

	if req.GetIgnoreEOS() && req.GetMaxCompletionTokens() == nil {
		return "Ignore_eos is true but max_completion_tokens (or max_tokens) is not set", fasthttp.StatusBadRequest
	}

	return "", fasthttp.StatusOK
}
