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
	"encoding/json"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/lora"
)

type loadLoraRequest struct {
	LoraName string `json:"lora_name"`
	LoraPath string `json:"lora_path"`
}

type unloadLoraRequest struct {
	LoraName string `json:"lora_name"`
}

// loraRegistry is this engine's lora.Registry. A request runs on an adapter by
// naming it in its model field, and the load and unload endpoints record and
// drop adapters without checking them.
type loraRegistry struct {
	// adapters maps a loaded adapter's name to its path
	adapters         sync.Map
	servedModelNames []string
	maxModelLen      int
	ownedBy          string
}

// NewLoraRegistry builds this engine's LoRA registry for one rank, seeded with
// the adapters config preloads.
func (e Engine) NewLoraRegistry(config common.Configuration) lora.Registry {
	r := &loraRegistry{
		servedModelNames: config.ServedModelNames,
		maxModelLen:      config.MaxModelLen,
		ownedBy:          e.Name(),
	}
	for _, module := range config.Lora.LoraModules {
		r.adapters.Store(module.Name, module.Path)
	}
	return r
}

// ResolveModel accepts a served model name or a loaded adapter. LoRA adapters
// keep their explicit name, while all base-model requests are surfaced as the
// first alias from --served-model-name. An unset model is checked as "".
func (r *loraRegistry) ResolveModel(_ api.ErrorRoute, modelField *string) (string, *api.Error) {
	model := ""
	if modelField != nil {
		model = *modelField
	}
	if r.IsAdapter(model) {
		return model, nil
	}
	if !slices.Contains(r.servedModelNames, model) {
		err := api.NewError(fmt.Sprintf("The model `%s` does not exist.", model), fasthttp.StatusNotFound, nil)
		return "", &err
	}
	return r.servedModelNames[0], nil
}

// Acquire runs every prompt on the adapter the model field names, if it names
// one. lora_path is not a vLLM field, so it is ignored.
func (r *loraRegistry) Acquire(_ api.ErrorRoute, model string, loraPaths []*api.LoraPath) ([]string, *api.Error) {
	adapter := ""
	if r.IsAdapter(model) {
		adapter = model
	}
	adapters := make([]string, len(loraPaths))
	for i := range adapters {
		adapters[i] = adapter
	}
	return adapters, nil
}

func (r *loraRegistry) Release(string) {}

func (r *loraRegistry) IsAdapter(name string) bool {
	_, ok := r.adapters.Load(name)
	return ok
}

// ModelCards roots each adapter at its path, or at its name when it was
// loaded without one.
func (r *loraRegistry) ModelCards(parent string) []api.ModelsResponseModelInfo {
	var cards []api.ModelsResponseModelInfo
	r.adapters.Range(func(key, value any) bool {
		name, _ := key.(string)
		root, _ := value.(string)
		if root == "" {
			root = name
		}
		maxModelLen := r.maxModelLen
		cards = append(cards, api.ModelsResponseModelInfo{
			ID:          name,
			Object:      api.ObjectModel,
			Created:     time.Now().Unix(),
			OwnedBy:     r.ownedBy,
			Root:        root,
			Parent:      &parent,
			MaxModelLen: &maxModelLen,
		})
		return true
	})
	return cards
}

func (r *loraRegistry) Load(body []byte) (int, any, *api.Error) {
	var req loadLoraRequest
	if err := json.Unmarshal(body, &req); err != nil {
		apiErr := api.NewError("failed to read and parse load lora request body, "+err.Error(),
			fasthttp.StatusBadRequest, nil)
		return 0, nil, &apiErr
	}
	r.adapters.Store(req.LoraName, req.LoraPath)
	return fasthttp.StatusOK, nil, nil
}

func (r *loraRegistry) Unload(body []byte) (int, any, *api.Error) {
	var req unloadLoraRequest
	if err := json.Unmarshal(body, &req); err != nil {
		apiErr := api.NewError("failed to read and parse unload lora request body, "+err.Error(),
			fasthttp.StatusBadRequest, nil)
		return 0, nil, &apiErr
	}
	r.adapters.Delete(req.LoraName)
	return fasthttp.StatusOK, nil, nil
}
