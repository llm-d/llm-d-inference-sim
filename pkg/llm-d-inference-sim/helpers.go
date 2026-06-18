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
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/valyala/fasthttp"
)

// isValidModel checks if the given model is the base model or one of "loaded" LoRAs
func (s *VllmSimulator) isValidModel(model string) bool {
	for _, name := range s.Context.Config().ServedModelNames {
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
func (s *VllmSimulator) ValidateBaseModel(model string) *api.Error {
	if !s.isValidModel(model) {
		serverErr := api.NewError(fmt.Sprintf("The model `%s` does not exist.", model),
			fasthttp.StatusNotFound, nil)
		return &serverErr
	}
	if s.Context.isLora(model) {
		serverErr := api.NewError(fmt.Sprintf("The model `%s` is a LoRA adapter and is not supported by the render endpoints.",
			model), fasthttp.StatusBadRequest, nil)
		return &serverErr
	}
	return nil
}

func getNumberOfPromptTokens(req api.Request) int {
	return req.TokenizedPrompt().Length()
}

func validateRequest(req api.Request) *api.Error {
	if n := req.GetRawN(); n != nil && *n <= 0 {
		err := api.NewError("n must be at least 1", fasthttp.StatusBadRequest, nil)
		return &err
	}

	if req.GetMaxCompletionTokens() != nil && *req.GetMaxCompletionTokens() <= 0 {
		err := api.NewError(common.InvalidMaxTokensErrMsg, fasthttp.StatusBadRequest, nil)
		return &err
	}

	if req.IsDoRemoteDecode() && req.IsStream() {
		err := api.NewError("Prefill does not support streaming", fasthttp.StatusBadRequest, nil)
		return &err
	}

	if req.GetIgnoreEOS() && req.GetMaxCompletionTokens() == nil {
		err := api.NewError("Ignore_eos is true but max_completion_tokens (or max_tokens) is not set",
			fasthttp.StatusBadRequest, nil)
		return &err
	}

	return nil
}

// buildECTransferParams creates simulated ECTransferParams for each MM hash.
func buildECTransferParams(mmHashes map[string][]string) map[string]api.ECTransferParams {
	params := make(map[string]api.ECTransferParams)

	for _, hashes := range mmHashes {
		for _, hash := range hashes {
			params[hash] = api.ECTransferParams{
				PeerHost:      "DUMMY",
				PeerPort:      1234,
				SizeBytes:     2359296,
				NixlAgentData: []byte("NIXL_METADATA_PLACEHOLDER"),
			}
		}
	}
	return params
}
