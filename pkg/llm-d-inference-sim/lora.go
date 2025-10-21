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

func (s *VllmSimulator) loadLoraAdaptor(ctx *fasthttp.RequestCtx) {
	var req loadLoraRequest
	err := json.Unmarshal(ctx.Request.Body(), &req)
	if err != nil {
		s.logger.Error(err, "failed to read and parse load lora request body")
		ctx.Error("failed to read and parse load lora request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	s.loraAdaptors.Store(req.LoraName, "")
}

func (s *VllmSimulator) unloadLoraAdaptor(ctx *fasthttp.RequestCtx) {
	var req unloadLoraRequest
	err := json.Unmarshal(ctx.Request.Body(), &req)
	if err != nil {
		s.logger.Error(err, "failed to read and parse unload lora request body")
		ctx.Error("failed to read and parse unload lora request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}

	s.loraAdaptors.Delete(req.LoraName)
}

// Checks if the LoRA adaptor is loaded
func (s *VllmSimulator) loraIsLoaded(model string) bool {
	if !s.isLora(model) {
		return true
	}

	s.loras.mux.RLock()
	defer s.loras.mux.RUnlock()

	_, ok := s.loras.loadedLoras[model]
	return ok
}

// Load the LoRA adaptor if possible. Return false if not.
func (s *VllmSimulator) loadLora(model string) bool {
	if !s.isLora(model) {
		return true
	}

	s.loras.mux.Lock()
	defer s.loras.mux.Unlock()

	// check if this LoRA is already loaded or within maxLoras slots
	_, ok := s.loras.loadedLoras[model]
	ok = ok || len(s.loras.loadedLoras) < s.loras.maxLoras
	if !ok {
		// if this LoRA is not loaded, and the number of loaded LoRAs reached
		// maxLoras, try to find a LoRA that is not in use, and unload it
		for lora, count := range s.loras.loadedLoras {
			if count == 0 {
				delete(s.loras.loadedLoras, lora)
				ok = true
				break
			}
		}
	}
	if ok {
		s.loras.loadedLoras[model]++
	}
	return ok
}

// incrementLora increments the count of running requests using the model
// (if the model is a LoRA). Can be called only for loaded LoRAs (that are
// already in loras.loadedLoras)
func (s *VllmSimulator) incrementLora(model string) {
	if !s.isLora(model) {
		return
	}

	s.loras.mux.Lock()
	defer s.loras.mux.Unlock()
	s.loras.loadedLoras[model]++
}

// decrementLora decrements the count of running requests using the model
// (if the model is a LoRA)
func (s *VllmSimulator) decrementLora(model string) {
	if model == "" || !s.isLora(model) {
		return
	}

	s.loras.mux.Lock()
	defer s.loras.mux.Unlock()

	s.loras.loadedLoras[model]--
	if s.loras.loadedLoras[model] <= 0 {
		// last usage of this LoRA
		s.loras.loraRemovable <- 1
	}
}
