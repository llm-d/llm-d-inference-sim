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

package tokenizer

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

// Runs against a real vLLM render service. The suite's shared renderer model
// (Qwen2-VL) has a chat template that ignores tools, so a dropped tools array
// would render identically there; this uses a model whose template renders them.
var _ = Describe("HF tokenizer tools against a real vLLM renderer", Ordered, func() {
	const model = "Qwen/Qwen2.5-0.5B-Instruct"
	const weather = `[{"type":"function","function":{"name":"get_weather","description":"Get the weather",` +
		`"parameters":{"type":"object","properties":{"zeta":{"type":"string"},"city":{"type":"string"}},"required":["city"]}}}]`
	const search = `[{"type":"function","function":{"name":"search","description":"Search the web",` +
		`"parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]`
	messages := []api.Message{{Role: api.RoleUser, Content: api.ChatComplContent{Raw: "What is the weather in Paris?"}}}

	var (
		renderURL string
		cleanup   func()
		tk        *HFTokenizer
	)

	BeforeAll(func() {
		var err error
		renderURL, cleanup, err = tokenizerMngr.startRenderContainer(context.Background(), model, RenderToolArgs...)
		Expect(err).NotTo(HaveOccurred())
		tk, err = NewHFTokenizer(context.Background(), klog.Background(), renderURL, model, 30*time.Second, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
	})
	AfterAll(func() {
		if cleanup != nil {
			cleanup()
		}
	})

	// post asks a render service directly, with the client's own fields.
	post := func(url string, tools api.RenderTools) *http.Response {
		body := map[string]json.RawMessage{"model": json.RawMessage(`"` + model + `"`)}
		msgs, err := json.Marshal(messages)
		Expect(err).NotTo(HaveOccurred())
		body["messages"] = msgs
		if tools.Tools != nil {
			body["tools"] = tools.Tools
		}
		if tools.ToolChoice != nil {
			body["tool_choice"] = tools.ToolChoice
		}
		payload, err := json.Marshal(body)
		Expect(err).NotTo(HaveOccurred())
		resp, err := http.Post(url+"/v1/chat/completions/render", "application/json", bytes.NewReader(payload))
		Expect(err).NotTo(HaveOccurred())
		return resp
	}
	direct := func(tools api.RenderTools) []uint32 {
		resp := post(renderURL, tools)
		defer func() { _ = resp.Body.Close() }()
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		var out api.RenderResponse
		Expect(json.NewDecoder(resp.Body).Decode(&out)).To(Succeed())
		Expect(out.TokenIDs).NotTo(BeEmpty())
		return out.TokenIDs
	}
	viaSimulator := func(tools api.RenderTools) []uint32 {
		tokens, _, _, err := tk.RenderMessages(messages, tools)
		Expect(err).NotTo(HaveOccurred())
		return tokens
	}

	DescribeTable("renders the same tokens as a direct request",
		func(tools, choice string) {
			rt := api.RenderTools{}
			if tools != "" {
				rt.Tools = json.RawMessage(tools)
			}
			if choice != "" {
				rt.ToolChoice = json.RawMessage(choice)
			}
			Expect(viaSimulator(rt)).To(Equal(direct(rt)))
		},
		Entry("no tools", "", ""),
		Entry("tools, tool_choice omitted", weather, ""),
		Entry("tools, tool_choice none", weather, `"none"`),
		Entry("tools, tool_choice auto", weather, `"auto"`),
		Entry("tools, named function", weather, `{"type":"function","function":{"name":"get_weather"}}`),
		Entry("different tools", search, ""),
	)

	It("renders the tool definitions into the prompt", func() {
		none := viaSimulator(api.RenderTools{})
		w := viaSimulator(api.RenderTools{Tools: json.RawMessage(weather)})
		s := viaSimulator(api.RenderTools{Tools: json.RawMessage(search)})
		Expect(w).NotTo(Equal(none))
		Expect(s).NotTo(Equal(w))
	})

	// With this model and parser, dropping tool_choice is observable through
	// validation rather than token differences: without tool calling configured,
	// vLLM accepts tools only with tool_choice "none". This is how the render
	// service is started by default.
	Context("on a renderer without tool calling configured", Ordered, func() {
		var (
			plainURL     string
			plainCleanup func()
			plain        *HFTokenizer
		)
		BeforeAll(func() {
			var err error
			plainURL, plainCleanup, err = tokenizerMngr.startRenderContainer(context.Background(), model)
			Expect(err).NotTo(HaveOccurred())
			plain, err = NewHFTokenizer(context.Background(), klog.Background(), plainURL, model, 30*time.Second, 60*time.Second)
			Expect(err).NotTo(HaveOccurred())
		})
		AfterAll(func() {
			if plainCleanup != nil {
				plainCleanup()
			}
		})

		It("accepts tools with tool_choice none, as vLLM does", func() {
			rt := api.RenderTools{Tools: json.RawMessage(weather), ToolChoice: json.RawMessage(`"none"`)}
			resp := post(plainURL, rt)
			defer func() { _ = resp.Body.Close() }()
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			var want api.RenderResponse
			Expect(json.NewDecoder(resp.Body).Decode(&want)).To(Succeed())

			got, _, _, err := plain.RenderMessages(messages, rt)
			Expect(err).NotTo(HaveOccurred())
			Expect(got).To(Equal(want.TokenIDs))
		})

		It("rejects tools without tool_choice, as vLLM does", func() {
			rt := api.RenderTools{Tools: json.RawMessage(weather)}
			resp := post(plainURL, rt)
			_ = resp.Body.Close()
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

			_, _, _, err := plain.RenderMessages(messages, rt)
			Expect(err).To(HaveOccurred())
		})
	})
})
