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
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/lora"
)

// loraRef is one adapter, as sglang's LoRARef records it.
type loraRef struct {
	name   string
	path   string
	pinned bool
}

// loraRegistry is this engine's lora.Registry, following sglang's tokenizer
// manager and LoRARegistry:
//
//   - A request names its adapter as "base:adapter" in its model field, or in
//     its lora_path field; a bare adapter name in model runs the base model.
//     The model field is checked only on the Responses API.
//   - Loading a name already loaded, or unloading one that is not, fails, and
//     unloading waits for the requests holding the adapter to finish.
//   - Every adapter ever loaded stays known, so a request naming one that was
//     since unloaded or evicted loads it again.
//   - Past max-loaded-loras, each load evicts the least recently used unpinned
//     adapter, where every request naming an adapter counts as a use.
type loraRegistry struct {
	enabled          bool
	maxLorasPerBatch int
	// maxLoadedLoras is 0 for no limit
	maxLoadedLoras   int
	servedModelNames []string
	ownedBy          string

	// updateMu serializes loads and unloads, explicit and implicit, as sglang's
	// lora_update_lock does.
	updateMu sync.Mutex

	mu sync.Mutex
	// released is signalled whenever a request releases an adapter
	released *sync.Cond
	// registered are the loaded adapters, least recently used first
	registered []loraRef
	// known holds every adapter ever loaded, in the order first loaded
	known []loraRef
	// inFlight counts the requests holding each adapter
	inFlight  map[string]int
	numPinned int
}

// NewLoraRegistry builds this engine's LoRA registry for one rank, seeded with
// the adapters config preloads.
func (e Engine) NewLoraRegistry(config common.Configuration) lora.Registry {
	r := &loraRegistry{
		enabled:          config.Lora.EnableLora != nil && *config.Lora.EnableLora,
		maxLorasPerBatch: config.Lora.MaxLoras,
		maxLoadedLoras:   config.Lora.MaxCPULoras,
		servedModelNames: config.ServedModelNames,
		ownedBy:          e.Name(),
		inFlight:         make(map[string]int),
	}
	r.released = sync.NewCond(&r.mu)
	for _, module := range config.Lora.LoraModules {
		ref := loraRef{name: module.Name, path: module.Path, pinned: module.Pinned}
		r.registered = append(r.registered, ref)
		r.remember(ref)
		if ref.pinned {
			r.numPinned++
		}
	}
	return r
}

// parseModel splits sglang's "base:adapter" model syntax on its first colon.
// adapter is "" when model names none.
func parseModel(model string) (base, adapter string) {
	base, adapter, found := strings.Cut(model, ":")
	if !found {
		return model, ""
	}
	return strings.TrimSpace(base), strings.TrimSpace(adapter)
}

// ResolveModel reports the model as the request names it. Only the Responses
// API checks it, unless unset, against the served model names and the loaded
// adapters, by the whole name or its base part.
func (r *loraRegistry) ResolveModel(route api.ErrorRoute, modelField *string) (string, *api.Error) {
	if modelField == nil {
		return "", nil
	}
	model := *modelField
	if route != api.ErrorRouteResponses {
		return model, nil
	}
	base, _ := parseModel(model)
	if r.isKnownModel(model) || r.isKnownModel(base) {
		return model, nil
	}
	err := api.NewError(fmt.Sprintf("The model `%s` does not exist.", model), fasthttp.StatusNotFound, nil)
	return "", &err
}

func (r *loraRegistry) isKnownModel(name string) bool {
	return slices.Contains(r.servedModelNames, name) || r.IsAdapter(name)
}

// Acquire loads again every adapter the request names that was unloaded or
// evicted, and only then acquires them all, so that no reload can wait on an
// adapter the request itself holds. The Responses API runs every request on
// the base model.
func (r *loraRegistry) Acquire(route api.ErrorRoute, model string, loraPaths []*api.LoraPath) ([]string, *api.Error) {
	for _, loraPath := range loraPaths {
		if raw := loraPath.Invalid(); raw != nil {
			return nil, invalidLoraPath(raw)
		}
	}
	adapters := make([]string, len(loraPaths))
	if route == api.ErrorRouteResponses {
		return adapters, nil
	}
	_, fromModel := parseModel(model)
	for i, loraPath := range loraPaths {
		switch named := loraPath.ForPrompt(0); {
		case fromModel != "":
			adapters[i] = fromModel
		case named != nil:
			adapters[i] = *named
		}
	}

	var requested []string
	for _, adapter := range adapters {
		if adapter != "" && !slices.Contains(requested, adapter) {
			requested = append(requested, adapter)
		}
	}
	if len(requested) == 0 {
		return adapters, nil
	}
	if !r.enabled {
		return nil, badRequest(fmt.Sprintf("LoRA adapter '%s' was requested, but LoRA is not enabled. "+
			"Please launch the server with --enable-lora flag and preload adapters "+
			"using --lora-paths or /load_lora_adapter endpoint.", requested[0]))
	}
	// A prompt on the base model among prompts on adapters comes from a null
	// lora_path entry, which sglang counts as one more unique adapter.
	unique := len(requested)
	if slices.Contains(adapters, "") {
		unique++
	}
	if r.maxLoadedLoras > 0 && unique > r.maxLoadedLoras {
		return nil, badRequest(fmt.Sprintf("Received request with %d unique loras requested but max loaded loras is %d",
			unique, r.maxLoadedLoras))
	}

	for _, adapter := range requested {
		if r.IsAdapter(adapter) {
			continue
		}
		ref, ok := r.knownRef(adapter)
		if !ok {
			return nil, badRequest(fmt.Sprintf("Got LoRA adapter that has never been loaded: %s\nAll loaded adapters: %s.",
				adapter, pyKeys("dict_keys", r.knownNames())))
		}
		if result := r.load(ref); !result.Success && !strings.Contains(*result.ErrorMessage, "already loaded") {
			return nil, badRequest(fmt.Sprintf("Failed to implicitly load LoRA adapter %s: %s",
				adapter, *result.ErrorMessage))
		}
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	for _, adapter := range requested {
		if r.indexOf(adapter) < 0 {
			return nil, badRequest(fmt.Sprintf("The following requested LoRA adapters are not loaded: %s\nLoaded adapters: %s.",
				adapter, pyKeys("odict_keys", r.registeredNames())))
		}
	}
	for _, adapter := range adapters {
		if adapter == "" {
			continue
		}
		r.inFlight[adapter]++
		r.touch(adapter)
	}
	return adapters, nil
}

func (r *loraRegistry) Release(adapter string) {
	if adapter == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.inFlight[adapter]--
	if r.inFlight[adapter] <= 0 {
		delete(r.inFlight, adapter)
	}
	r.released.Broadcast()
}

func (r *loraRegistry) IsAdapter(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.indexOf(name) >= 0
}

// ModelCards lists the adapters least recently used first, with no context
// window of their own, and none at all when LoRA is disabled.
func (r *loraRegistry) ModelCards(parent string) []api.ModelsResponseModelInfo {
	if !r.enabled {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	cards := make([]api.ModelsResponseModelInfo, 0, len(r.registered))
	for _, ref := range r.registered {
		cards = append(cards, api.ModelsResponseModelInfo{
			ID:      ref.name,
			Object:  api.ObjectModel,
			Created: time.Now().Unix(),
			OwnedBy: r.ownedBy,
			Root:    ref.path,
			Parent:  &parent,
		})
	}
	return cards
}

func (r *loraRegistry) Load(body []byte) (int, any, *api.Error) {
	fields, invalid := decodeLoraRequest(body, loadLoraFields)
	if invalid != nil {
		return 0, nil, invalid
	}
	name, _ := fields.get("lora_name")
	path, _ := fields.get("lora_path")
	pinned, _ := fields.get("pinned")
	isPinned, _ := pinned.(bool)
	return loraUpdateStatus(r.load(loraRef{name: name.(string), path: path.(string), pinned: isPinned}))
}

func (r *loraRegistry) Unload(body []byte) (int, any, *api.Error) {
	fields, invalid := decodeLoraRequest(body, unloadLoraFields)
	if invalid != nil {
		return 0, nil, invalid
	}
	if !r.enabled {
		return loraUpdateStatus(failedUpdate(loraNotEnabled))
	}
	name, _ := fields.get("lora_name")
	r.updateMu.Lock()
	defer r.updateMu.Unlock()
	return loraUpdateStatus(r.unloadLocked(name.(string)))
}

const loraNotEnabled = "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."

func loraUpdateStatus(result loraUpdateOutput) (int, any, *api.Error) {
	if result.Success {
		return fasthttp.StatusOK, result, nil
	}
	return fasthttp.StatusBadRequest, result, nil
}

// load registers ref and then evicts the least recently used unpinned adapters
// until max-loaded-loras is met, waiting for each to be released.
func (r *loraRegistry) load(ref loraRef) loraUpdateOutput {
	if !r.enabled {
		return failedUpdate(loraNotEnabled)
	}
	r.updateMu.Lock()
	defer r.updateMu.Unlock()

	result := r.register(ref)
	if r.maxLoadedLoras == 0 {
		return result
	}
	for {
		r.mu.Lock()
		over := len(r.registered) > r.maxLoadedLoras
		lru := ""
		for _, candidate := range r.registered {
			if !candidate.pinned {
				lru = candidate.name
				break
			}
		}
		names := r.registeredNames()
		r.mu.Unlock()
		if !over {
			return result
		}
		if lru == "" {
			return failedUpdate(fmt.Sprintf("Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "+
				"LoRA registry is: %s", pyKeys("odict_keys", names)))
		}
		if evicted := r.unloadLocked(lru); !evicted.Success {
			return failedUpdate(fmt.Sprintf("Error while unloading LRU LoRA adapter '%s': %s",
				lru, *evicted.ErrorMessage))
		}
		delete(result.LoadedAdapters, lru)
	}
}

// register adds ref as the most recently used adapter. Caller must hold updateMu.
func (r *loraRegistry) register(ref loraRef) loraUpdateOutput {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.indexOf(ref.name) >= 0 {
		return r.failedUpdateLocked(fmt.Sprintf("Failed to load LoRA adapter %s because it is already loaded", ref.name))
	}
	if ref.pinned && r.numPinned >= r.maxLorasPerBatch-1 {
		return r.failedUpdateLocked(fmt.Sprintf("Failed to load LoRA adapter %s as a pinned adapter. "+
			"It is not allowed to pin all slots in the LoRA memory pool to avoid starvation for unpinned adapters "+
			"and base models. Please increase your `--max-loras-per-batch` or load it as unpinned LoRA adapters.",
			ref.name))
	}
	r.registered = append(r.registered, ref)
	r.remember(ref)
	if ref.pinned {
		r.numPinned++
	}
	return r.succeededUpdateLocked()
}

// unloadLocked unregisters the adapter, so no new request can acquire it, and
// then waits for the requests holding it. Caller must hold updateMu.
func (r *loraRegistry) unloadLocked(name string) loraUpdateOutput {
	r.mu.Lock()
	defer r.mu.Unlock()
	i := r.indexOf(name)
	if i < 0 {
		return failedUpdate(fmt.Sprintf("LoRA with name %s does not exist. Loaded LoRAs: %s",
			name, pyKeys("odict_keys", r.registeredNames())))
	}
	if r.registered[i].pinned {
		r.numPinned--
	}
	r.registered = slices.Delete(r.registered, i, i+1)
	for r.inFlight[name] > 0 {
		r.released.Wait()
	}
	return r.succeededUpdateLocked()
}

// remember records ref as known, replacing an earlier load of the same name
// in place. Caller must hold mu.
func (r *loraRegistry) remember(ref loraRef) {
	for i, known := range r.known {
		if known.name == ref.name {
			r.known[i] = ref
			return
		}
	}
	r.known = append(r.known, ref)
}

func (r *loraRegistry) knownRef(name string) (loraRef, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, ref := range r.known {
		if ref.name == name {
			return ref, true
		}
	}
	return loraRef{}, false
}

func (r *loraRegistry) knownNames() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	names := make([]string, 0, len(r.known))
	for _, ref := range r.known {
		names = append(names, ref.name)
	}
	return names
}

// touch makes the adapter the most recently used. Caller must hold mu.
func (r *loraRegistry) touch(name string) {
	if i := r.indexOf(name); i >= 0 {
		ref := r.registered[i]
		r.registered = append(slices.Delete(r.registered, i, i+1), ref)
	}
}

// indexOf returns the adapter's position in registered, -1 if not loaded.
// Caller must hold mu.
func (r *loraRegistry) indexOf(name string) int {
	return slices.IndexFunc(r.registered, func(ref loraRef) bool { return ref.name == name })
}

// registeredNames lists the loaded adapters, least recently used first.
// Caller must hold mu.
func (r *loraRegistry) registeredNames() []string {
	names := make([]string, 0, len(r.registered))
	for _, ref := range r.registered {
		names = append(names, ref.name)
	}
	return names
}

// loadedAdaptersLocked maps each loaded adapter to its path. Caller must hold mu.
func (r *loraRegistry) loadedAdaptersLocked() map[string]string {
	loaded := make(map[string]string, len(r.registered))
	for _, ref := range r.registered {
		loaded[ref.name] = ref.path
	}
	return loaded
}

// pyKeys renders names the way Python prints a dict's keys view, e.g.
// dict_keys(['a', 'b']), which is how sglang names adapters in its errors.
func pyKeys(view string, names []string) string {
	quoted := make([]string, len(names))
	for i, name := range names {
		quoted[i] = "'" + name + "'"
	}
	return view + "([" + strings.Join(quoted, ", ") + "])"
}

func badRequest(message string) *api.Error {
	err := api.NewError(message, fasthttp.StatusBadRequest, nil)
	return &err
}
