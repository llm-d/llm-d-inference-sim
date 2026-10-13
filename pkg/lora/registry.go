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

// Package lora defines the seam between the simulator core and an engine's
// management of LoRA adapters.
package lora

import (
	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

// Registry is the set of LoRA adapters a server has loaded, managed the way one
// engine manages it: how a request names its model and the adapter it runs on,
// what the load and unload endpoints accept and answer, and when an adapter
// stops being available. The batch slots adapters occupy while requests run
// are the simulator core's, not the registry's.
//
// A Registry is safe for concurrent use.
type Registry interface {
	// ResolveModel checks a request's model field, as received on route (nil
	// when the request left it unset), and returns the model name its response
	// reports.
	ResolveModel(route api.ErrorRoute, model *string) (displayedModel string, err *api.Error)
	// Acquire returns the adapter each prompt of a request received on route
	// runs on, or "" for the base model, as named by the request's model field
	// and the prompt's lora_path field (nil for none; a prompt runs on its first
	// entry). All of a request's prompts are acquired together, since making one
	// prompt's adapter available may evict another's. Each non-empty adapter
	// must be released with Release once its prompt completes; on error none is
	// acquired.
	Acquire(route api.ErrorRoute, model string, loraPaths []*api.LoraPath) (adapters []string, err *api.Error)
	// Release ends one acquisition of adapter made by Acquire.
	Release(adapter string)
	// IsAdapter reports whether name is a loaded adapter.
	IsAdapter(name string) bool
	// ModelCards lists the loaded adapters as /v1/models entries of the base
	// model parent.
	ModelCards(parent string) []api.ModelsResponseModelInfo
	// Load handles the body of a load request, returning the response status
	// and body (nil for none), or the error to answer with instead.
	Load(body []byte) (status int, response any, err *api.Error)
	// Unload is Load's counterpart for an unload request.
	Unload(body []byte) (status int, response any, err *api.Error)
}

// StubRegistry is a Registry with no adapters, which accepts every model name
// and every load and unload request. For tests that need a Registry but no
// engine's LoRA behavior.
type StubRegistry struct{}

func (StubRegistry) ResolveModel(_ api.ErrorRoute, model *string) (string, *api.Error) {
	if model == nil {
		return "", nil
	}
	return *model, nil
}
func (StubRegistry) Acquire(_ api.ErrorRoute, _ string, loraPaths []*api.LoraPath) ([]string, *api.Error) {
	return make([]string, len(loraPaths)), nil
}
func (StubRegistry) Release(string)                                  {}
func (StubRegistry) IsAdapter(string) bool                           { return false }
func (StubRegistry) ModelCards(string) []api.ModelsResponseModelInfo { return nil }
func (StubRegistry) Load([]byte) (int, any, *api.Error)              { return fasthttp.StatusOK, nil, nil }
func (StubRegistry) Unload([]byte) (int, any, *api.Error)            { return fasthttp.StatusOK, nil, nil }
