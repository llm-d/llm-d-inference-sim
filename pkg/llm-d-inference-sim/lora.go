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

// Checks if a request with this model can run under maxLoras limit
func (s *VllmSimulator) loraIsLoaded(model string) bool {
	if !s.isLora(model) {
		return true
	}

	s.loras.mux.RLock()
	defer s.loras.mux.RUnlock()

	_, ok := s.loras.usedLoras[model]
	s.logger.Info("is loaded", "lora", model, "ok", ok)

	return ok
}

// Checks if a request with this model can run under maxLoras limit
func (s *VllmSimulator) loadLora(model string) bool {
	if !s.isLora(model) {
		return true
	}

	s.loras.mux.Lock()
	defer s.loras.mux.Unlock()

	// check if this lora is already loaded or within maxLoras slots
	_, ok := s.loras.usedLoras[model]
	ok = ok || len(s.loras.usedLoras) < s.loras.maxLoras
	s.logger.Info("load", "lora", model, "count", s.loras.usedLoras[model], "size", len(s.loras.usedLoras), "ok", ok)
	if !ok {
		for lora, count := range s.loras.usedLoras {
			s.logger.Info("loop", "lora", lora, "count", count)
			if count == 0 {
				s.logger.Info("loop delete", "lora", lora)
				delete(s.loras.usedLoras, lora)
				ok = true
				break
			}
		}
	}
	if ok {
		s.loras.usedLoras[model]++
	}
	s.logger.Info("load", "ok", ok, "lora", model, "count", s.loras.usedLoras[model], "size", len(s.loras.usedLoras))

	return ok
}

func (s *VllmSimulator) incrementLora(model string) {
	if !s.isLora(model) {
		return
	}

	s.loras.mux.Lock()
	defer s.loras.mux.Unlock()
	s.loras.usedLoras[model]++
}

func (s *VllmSimulator) decrementLora(model string) {
	if !s.isLora(model) {
		return
	}

	s.loras.mux.Lock()
	defer s.loras.mux.Unlock()

	s.loras.usedLoras[model] -= 1
	s.logger.Info("decrement", "lora", model)
	if s.loras.usedLoras[model] <= 0 {
		// last usage of this lora - remove it from the used loras list
		// delete(s.loras.usedLoras, model)
		s.loras.loraRemovable <- 1
	}
}
