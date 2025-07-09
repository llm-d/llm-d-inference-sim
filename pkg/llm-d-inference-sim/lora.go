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

// LoRA related structures and functions
package llmdinferencesim

import (
	"encoding/json"

	"github.com/valyala/fasthttp"
)

type loadLoraRequest struct {
	LoraName string `json:"lora_name"`
	LoraPath string `json:"lora_path"`
}

type unloadLoraRequest struct {
	LoraName string `json:"lora_name"`
}

func (s *VllmSimulator) getLoras() []string {
	loras := make([]string, 0)

	s.loraAdaptors.Range(func(key, _ any) bool {
		if lora, ok := key.(string); ok {
			loras = append(loras, lora)
		} else {
			s.logger.Info("Stored LoRA is not a string", "value", key)
		}
		return true
	})

	return loras
}

func (s *VllmSimulator) loadLora(ctx *fasthttp.RequestCtx) {
	var req loadLoraRequest
	err := json.Unmarshal(ctx.Request.Body(), &req)
	if err != nil {
		s.logger.Error(err, "failed to read and parse load lora request body")
		ctx.Error("failed to read and parse load lora request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	s.loraAdaptors.Store(req.LoraName, "")
}

func (s *VllmSimulator) unloadLora(ctx *fasthttp.RequestCtx) {
	var req unloadLoraRequest
	err := json.Unmarshal(ctx.Request.Body(), &req)
	if err != nil {
		s.logger.Error(err, "failed to read and parse unload lora request body")
		ctx.Error("failed to read and parse unload lora request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	s.loraAdaptors.Delete(req.LoraName)
}
