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
	"encoding/json"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/lora"
)

func ptr[T any](v T) *T {
	return &v
}

const baseModel = "base-model"

func newRegistry(enabled bool, maxPerBatch, maxLoaded int, modules ...common.LoraModule) lora.Registry {
	config := common.NewConfig()
	New().ApplyDefaults(config)
	config.ServedModelNames = []string{baseModel}
	config.Lora.EnableLora = ptr(enabled)
	config.Lora.MaxLoras = maxPerBatch
	config.Lora.MaxCPULoras = maxLoaded
	config.Lora.LoraModules = modules
	return New().NewLoraRegistry(*config)
}

func loadBody(name, path string, pinned bool) []byte {
	body, err := json.Marshal(map[string]any{"lora_name": name, "lora_path": path, "pinned": pinned})
	Expect(err).NotTo(HaveOccurred())
	return body
}

func unloadBody(name string) []byte {
	return []byte(`{"lora_name":"` + name + `"}`)
}

// update runs a load or unload and returns its status and LoRAUpdateOutput.
func update(call func([]byte) (int, any, *api.Error), body []byte) (int, loraUpdateOutput) {
	status, response, err := call(body)
	Expect(err).To(BeNil())
	output, ok := response.(loraUpdateOutput)
	Expect(ok).To(BeTrue(), "response %#v is not a LoRAUpdateOutput", response)
	return status, output
}

// loraPath decodes a lora_path field from its JSON.
func loraPath(field string) *api.LoraPath {
	var path api.LoraPath
	Expect(json.Unmarshal([]byte(field), &path)).To(Succeed())
	return &path
}

// acquireOne acquires the adapters of a single-prompt request.
func acquireOne(r lora.Registry, model string, path *api.LoraPath) (string, *api.Error) {
	adapters, err := r.Acquire(api.ErrorRouteDefault, model, []*api.LoraPath{path})
	if err != nil {
		return "", err
	}
	return adapters[0], nil
}

func cardIDs(r lora.Registry) []string {
	cards := r.ModelCards(baseModel)
	ids := make([]string, 0, len(cards))
	for _, card := range cards {
		ids = append(ids, card.ID)
	}
	return ids
}

var _ = Describe("LoRA registry", func() {
	adapters := []common.LoraModule{{Name: "lora1", Path: "/p1"}, {Name: "lora2", Path: "/p2"}}

	Describe("adapter selection", func() {
		DescribeTable("should take the adapter from base:adapter, then from lora_path",
			func(model string, path *api.LoraPath, expected string) {
				adapter, err := acquireOne(newRegistry(true, 8, 0, adapters...), model, path)
				Expect(err).To(BeNil())
				Expect(adapter).To(Equal(expected))
			},
			Entry("base:adapter", baseModel+":lora1", nil, "lora1"),
			Entry("base:adapter, trimmed", baseModel+" : lora1 ", nil, "lora1"),
			Entry("lora_path", baseModel, loraPath(`"lora2"`), "lora2"),
			Entry("base:adapter over lora_path", baseModel+":lora1", loraPath(`"lora2"`), "lora1"),
			Entry("a bare adapter name, which runs the base model", "lora1", nil, ""),
			Entry("an unknown model, which runs the base model", "no-such-model", nil, ""),
		)

		It("should run each prompt on its own lora_path entry", func() {
			r := newRegistry(true, 8, 0, adapters...)
			acquired, err := r.Acquire(api.ErrorRouteDefault, baseModel, []*api.LoraPath{loraPath(`"lora1"`), nil, loraPath(`"lora2"`)})
			Expect(err).To(BeNil())
			Expect(acquired).To(Equal([]string{"lora1", "", "lora2"}))
		})

		It("should run a Responses API request on the base model", func() {
			acquired, err := newRegistry(true, 8, 0, adapters...).Acquire(api.ErrorRouteResponses,
				baseModel+":lora1", []*api.LoraPath{nil})
			Expect(err).To(BeNil())
			Expect(acquired).To(Equal([]string{""}))
		})

		DescribeTable("should refuse a lora_path of neither form as pydantic does",
			func(field string, errors ...string) {
				_, err := acquireOne(newRegistry(true, 8, 0, adapters...), baseModel+":lora1", loraPath(field))
				Expect(err).NotTo(BeNil())
				Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
				Expect(err.Message).To(HavePrefix(fmt.Sprintf("%d validation errors:\n  %s", len(errors), errors[0])))
				Expect(err.Message).To(HaveSuffix("[" + strings.Join(errors, ", ") + "]"))
			},
			Entry("a number", `5`,
				"{'type': 'list_type', 'loc': ('body', 'lora_path', 'list[nullable[str]]'), "+
					"'msg': 'Input should be a valid list', 'input': 5}",
				"{'type': 'string_type', 'loc': ('body', 'lora_path', 'str'), "+
					"'msg': 'Input should be a valid string', 'input': 5}"),
			Entry("a list holding a non-string", `["lora1", null, 1]`,
				"{'type': 'string_type', 'loc': ('body', 'lora_path', 'list[nullable[str]]', 2), "+
					"'msg': 'Input should be a valid string', 'input': 1}",
				"{'type': 'string_type', 'loc': ('body', 'lora_path', 'str'), "+
					"'msg': 'Input should be a valid string', 'input': ['lora1', None, 1]}"),
		)

		It("should refuse more unique adapters than max-loaded-loras, counting a null entry", func() {
			r := newRegistry(true, 2, 2, adapters...)
			_, err := r.Acquire(api.ErrorRouteDefault, baseModel, []*api.LoraPath{loraPath(`"lora1"`), nil, loraPath(`"lora2"`)})
			Expect(err).NotTo(BeNil())
			Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
			Expect(err.Message).To(Equal("Received request with 3 unique loras requested but max loaded loras is 2"))
		})
	})

	Describe("model names", func() {
		It("should report any model on a non-Responses route verbatim", func() {
			displayed, err := newRegistry(true, 8, 0).ResolveModel(api.ErrorRouteDefault, ptr("no-such-model"))
			Expect(err).To(BeNil())
			Expect(displayed).To(Equal("no-such-model"))
		})

		It("should not check an unset model on the Responses API", func() {
			displayed, err := newRegistry(true, 8, 0).ResolveModel(api.ErrorRouteResponses, nil)
			Expect(err).To(BeNil())
			Expect(displayed).To(BeEmpty())
		})

		DescribeTable("should check the model on the Responses API",
			func(model string, found bool) {
				displayed, err := newRegistry(true, 8, 0, adapters...).ResolveModel(api.ErrorRouteResponses, &model)
				if found {
					Expect(err).To(BeNil())
					Expect(displayed).To(Equal(model))
				} else {
					Expect(err).NotTo(BeNil())
					Expect(err.Code).To(Equal(fasthttp.StatusNotFound))
					Expect(err.Message).To(Equal("The model `" + model + "` does not exist."))
				}
			},
			Entry("a served name", baseModel, true),
			Entry("an adapter", "lora1", true),
			Entry("base:adapter with a served base", baseModel+":anything", true),
			Entry("an unknown model", "no-such-model", false),
			Entry("an empty model", "", false),
			Entry("base:adapter with an unknown base", "no-such-model:lora1", false),
		)
	})

	Describe("with LoRA disabled", func() {
		It("should refuse a request naming an adapter", func() {
			_, err := acquireOne(newRegistry(false, 8, 0), baseModel+":lora1", nil)
			Expect(err).NotTo(BeNil())
			Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
			Expect(err.Message).To(HavePrefix("LoRA adapter 'lora1' was requested, but LoRA is not enabled."))
		})

		It("should run a request naming no adapter", func() {
			adapter, err := acquireOne(newRegistry(false, 8, 0), baseModel, nil)
			Expect(err).To(BeNil())
			Expect(adapter).To(BeEmpty())
		})

		DescribeTable("should refuse loads and unloads",
			func(call func(lora.Registry) func([]byte) (int, any, *api.Error), body []byte) {
				status, output := update(call(newRegistry(false, 8, 0)), body)
				Expect(status).To(Equal(fasthttp.StatusBadRequest))
				Expect(output.Success).To(BeFalse())
				Expect(output.ErrorMessage).To(HaveValue(Equal(loraNotEnabled)))
				Expect(output.LoadedAdapters).To(BeNil())
			},
			Entry("load", func(r lora.Registry) func([]byte) (int, any, *api.Error) { return r.Load },
				loadBody("lora1", "/p1", false)),
			Entry("unload", func(r lora.Registry) func([]byte) (int, any, *api.Error) { return r.Unload },
				unloadBody("lora1")),
		)

		It("should list no adapters", func() {
			Expect(newRegistry(false, 8, 0).ModelCards(baseModel)).To(BeEmpty())
		})
	})

	Describe("load and unload", func() {
		It("should answer a load with sglang's LoRAUpdateOutput", func() {
			status, response, err := newRegistry(true, 8, 0).Load(loadBody("lora1", "/p1", false))
			Expect(err).To(BeNil())
			Expect(status).To(Equal(fasthttp.StatusOK))
			body, marshalErr := json.Marshal(response)
			Expect(marshalErr).NotTo(HaveOccurred())
			Expect(body).To(MatchJSON(`{"rid":null,"http_worker_ipc":null,"success":true,` +
				`"error_message":"","loaded_adapters":{"lora1":"/p1"}}`))
		})

		It("should refuse to load an adapter already loaded, listing the adapters", func() {
			status, output := update(newRegistry(true, 8, 0, adapters...).Load, loadBody("lora1", "/other", false))
			Expect(status).To(Equal(fasthttp.StatusBadRequest))
			Expect(output.ErrorMessage).To(HaveValue(Equal(
				"Failed to load LoRA adapter lora1 because it is already loaded")))
			Expect(output.LoadedAdapters).To(Equal(map[string]string{"lora1": "/p1", "lora2": "/p2"}))
		})

		It("should refuse to unload an adapter not loaded", func() {
			status, output := update(newRegistry(true, 8, 0, adapters...).Unload, unloadBody("lora3"))
			Expect(status).To(Equal(fasthttp.StatusBadRequest))
			Expect(output.ErrorMessage).To(HaveValue(Equal(
				"LoRA with name lora3 does not exist. Loaded LoRAs: odict_keys(['lora1', 'lora2'])")))
			Expect(output.LoadedAdapters).To(BeNil())
		})

		It("should unload an adapter, listing the rest", func() {
			r := newRegistry(true, 8, 0, adapters...)
			status, output := update(r.Unload, unloadBody("lora1"))
			Expect(status).To(Equal(fasthttp.StatusOK))
			Expect(output.LoadedAdapters).To(Equal(map[string]string{"lora2": "/p2"}))
			Expect(r.IsAdapter("lora1")).To(BeFalse())
		})

		It("should refuse a request naming an adapter never loaded", func() {
			_, err := acquireOne(newRegistry(true, 8, 0, adapters...), baseModel+":lora3", nil)
			Expect(err).NotTo(BeNil())
			Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
			Expect(err.Message).To(Equal("Got LoRA adapter that has never been loaded: lora3\n" +
				"All loaded adapters: dict_keys(['lora1', 'lora2'])."))
		})

		It("should load an unloaded adapter again for a request naming it", func() {
			r := newRegistry(true, 8, 0, adapters...)
			update(r.Unload, unloadBody("lora1"))
			adapter, err := acquireOne(r, baseModel+":lora1", nil)
			Expect(err).To(BeNil())
			Expect(adapter).To(Equal("lora1"))
			Expect(r.IsAdapter("lora1")).To(BeTrue())
			r.Release(adapter)
		})

		It("should wait for the requests holding an adapter before unloading it", func() {
			r := newRegistry(true, 8, 0, adapters...)
			adapter, err := acquireOne(r, baseModel+":lora1", nil)
			Expect(err).To(BeNil())

			done := make(chan int)
			go func() {
				defer GinkgoRecover()
				status, _ := update(r.Unload, unloadBody("lora1"))
				done <- status
			}()
			Eventually(func() bool { return r.IsAdapter("lora1") }).Should(BeFalse())
			Consistently(done, 100*time.Millisecond).ShouldNot(Receive())

			r.Release(adapter)
			Eventually(done).Should(Receive(Equal(fasthttp.StatusOK)))
		})

		It("should list the adapters least recently used first, with no max_model_len", func() {
			r := newRegistry(true, 8, 0, adapters...)
			adapter, err := acquireOne(r, baseModel+":lora1", nil)
			Expect(err).To(BeNil())
			r.Release(adapter)

			cards := r.ModelCards(baseModel)
			Expect(cards).To(HaveLen(2))
			Expect(cards[0].ID).To(Equal("lora2"))
			Expect(cards[1].ID).To(Equal("lora1"))
			Expect(cards[1].Root).To(Equal("/p1"))
			Expect(cards[1].Parent).To(HaveValue(Equal(baseModel)))
			Expect(cards[1].OwnedBy).To(Equal("sglang"))
			Expect(cards[1].MaxModelLen).To(BeNil())
		})
	})

	Describe("max-loaded-loras", func() {
		It("should evict the least recently used adapter", func() {
			r := newRegistry(true, 2, 2, adapters...)
			adapter, err := acquireOne(r, baseModel+":lora1", nil)
			Expect(err).To(BeNil())
			r.Release(adapter)

			status, output := update(r.Load, loadBody("lora3", "/p3", false))
			Expect(status).To(Equal(fasthttp.StatusOK))
			Expect(output.LoadedAdapters).To(Equal(map[string]string{"lora1": "/p1", "lora3": "/p3"}))
			Expect(cardIDs(r)).To(Equal([]string{"lora1", "lora3"}))
		})

		It("should not evict a pinned adapter", func() {
			r := newRegistry(true, 2, 2, common.LoraModule{Name: "lora1", Path: "/p1", Pinned: true},
				common.LoraModule{Name: "lora2", Path: "/p2"})
			update(r.Load, loadBody("lora3", "/p3", false))
			Expect(cardIDs(r)).To(Equal([]string{"lora1", "lora3"}))
		})

		It("should fail a load that finds no unpinned adapter to evict", func() {
			r := newRegistry(true, 8, 2, common.LoraModule{Name: "lora1", Path: "/p1", Pinned: true},
				common.LoraModule{Name: "lora2", Path: "/p2", Pinned: true})
			status, output := update(r.Load, loadBody("lora3", "/p3", true))
			Expect(status).To(Equal(fasthttp.StatusBadRequest))
			Expect(output.ErrorMessage).To(HaveValue(HavePrefix(
				"Didn't find any LoRA adapters when trying to evict LRU LoRA adapter.")))
		})

		It("should evict when a request loads an adapter again", func() {
			r := newRegistry(true, 2, 2, adapters...)
			update(r.Load, loadBody("lora3", "/p3", false))
			Expect(cardIDs(r)).To(Equal([]string{"lora2", "lora3"}))

			adapter, err := acquireOne(r, baseModel+":lora1", nil)
			Expect(err).To(BeNil())
			r.Release(adapter)
			Expect(cardIDs(r)).To(Equal([]string{"lora3", "lora1"}))
		})
	})

	It("should refuse to pin every slot of max-loras-per-batch", func() {
		r := newRegistry(true, 2, 0)
		status, _ := update(r.Load, loadBody("lora1", "/p1", true))
		Expect(status).To(Equal(fasthttp.StatusOK))

		status, output := update(r.Load, loadBody("lora2", "/p2", true))
		Expect(status).To(Equal(fasthttp.StatusBadRequest))
		Expect(output.ErrorMessage).To(HaveValue(HavePrefix("Failed to load LoRA adapter lora2 as a pinned adapter.")))
		Expect(output.LoadedAdapters).To(Equal(map[string]string{"lora1": "/p1"}))
	})

	// sglang turns FastAPI's 422 into a 400 whose message is str(exc) and
	// str(exc.errors()).
	DescribeTable("should answer a body pydantic rejects with sglang's 400",
		func(body string, expected string) {
			_, _, err := newRegistry(true, 8, 0).Load([]byte(body))
			Expect(err).NotTo(BeNil())
			Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
			Expect(err.Type).To(Equal("Bad Request"))
			Expect(err.Message).To(Equal(expected))
		},
		Entry("no body", ``,
			"1 validation error:\n  {'type': 'missing', 'loc': ('body',), 'msg': 'Field required', 'input': None} "+
				"[{'type': 'missing', 'loc': ('body',), 'msg': 'Field required', 'input': None}]"),
		Entry("not an object", `[1, 2.5, "x"]`,
			"1 validation error:\n  {'type': 'dict_type', 'loc': ('body',), 'msg': 'Input should be a valid dictionary', "+
				"'input': [1, 2.5, 'x']} [{'type': 'dict_type', 'loc': ('body',), "+
				"'msg': 'Input should be a valid dictionary', 'input': [1, 2.5, 'x']}]"),
		Entry("missing fields, echoing the body in its key order", `{"z":null,"pinned":true}`,
			"2 validation errors:\n"+
				"  {'type': 'missing', 'loc': ('body', 'lora_name'), 'msg': 'Field required', 'input': {'z': None, 'pinned': True}}\n"+
				"  {'type': 'missing', 'loc': ('body', 'lora_path'), 'msg': 'Field required', 'input': {'z': None, 'pinned': True}} "+
				"[{'type': 'missing', 'loc': ('body', 'lora_name'), 'msg': 'Field required', 'input': {'z': None, 'pinned': True}}, "+
				"{'type': 'missing', 'loc': ('body', 'lora_path'), 'msg': 'Field required', 'input': {'z': None, 'pinned': True}}]"),
		Entry("a name that is not a string", `{"lora_name":1,"lora_path":"/p1"}`,
			"1 validation error:\n  {'type': 'string_type', 'loc': ('body', 'lora_name'), "+
				"'msg': 'Input should be a valid string', 'input': 1} "+
				"[{'type': 'string_type', 'loc': ('body', 'lora_name'), 'msg': 'Input should be a valid string', 'input': 1}]"),
		Entry("an unparsable pinned", `{"lora_name":"lora1","lora_path":"/p1","pinned":"it's"}`,
			"1 validation error:\n  {'type': 'bool_parsing', 'loc': ('body', 'pinned'), "+
				"'msg': 'Input should be a valid boolean, unable to interpret input', 'input': \"it's\"} "+
				"[{'type': 'bool_parsing', 'loc': ('body', 'pinned'), "+
				"'msg': 'Input should be a valid boolean, unable to interpret input', 'input': \"it's\"}]"),
	)

	It("should report invalid JSON as json_invalid", func() {
		_, _, err := newRegistry(true, 8, 0).Unload([]byte(`{"lora_name":`))
		Expect(err).NotTo(BeNil())
		Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
		Expect(err.Message).To(HavePrefix("1 validation error:\n  {'type': 'json_invalid', 'loc': ('body', 13), " +
			"'msg': 'JSON decode error', 'input': {}, 'ctx': {'error': "))
	})

	It("should coerce pinned as pydantic's lax mode does", func() {
		// With one pinnable slot, a second pinned load fails only if the first
		// was pinned.
		r := newRegistry(true, 2, 0)
		status, _ := update(r.Load, []byte(`{"lora_name":"lora1","lora_path":"/p1","pinned":"yes"}`))
		Expect(status).To(Equal(fasthttp.StatusOK))
		status, output := update(r.Load, []byte(`{"lora_name":"lora2","lora_path":"/p2","pinned":1}`))
		Expect(status).To(Equal(fasthttp.StatusBadRequest))
		Expect(output.ErrorMessage).To(HaveValue(ContainSubstring("as a pinned adapter")))
	})
})
