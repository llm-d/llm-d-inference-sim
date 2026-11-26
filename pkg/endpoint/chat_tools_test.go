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

package endpoint

import (
	"encoding/json"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
)

// recordingTokenizer captures what each RenderMessages call receives.
type recordingTokenizer struct {
	tools api.RenderTools
	calls int
}

func (r *recordingTokenizer) RenderText(string) ([]uint32, []string, error) { return nil, nil, nil }
func (r *recordingTokenizer) Detokenize([]uint32) (string, error)           { return "", nil }
func (r *recordingTokenizer) Tokenize(api.TokenizeRequest) ([]uint32, []string, error) {
	return nil, nil, nil
}
func (r *recordingTokenizer) RenderOutputText(string) ([]uint32, []string, error) {
	return nil, nil, nil
}
func (r *recordingTokenizer) RenderMessages(_ []api.Message, tools api.RenderTools, _ string) ([]uint32, []string, *api.RenderMMFeatures, error) {
	r.calls++
	r.tools = tools
	return []uint32{7}, nil, nil, nil
}

var _ = Describe("Chat completions raw tools", func() {
	const tools = `[{"type":"function","function":{"name":"f","strict":true,"parameters":{"properties":{"z":{},"a":{}}}}}]`
	const messages = `"messages":[{"role":"user","content":"hi"}]`

	render := func(body string) api.RenderTools {
		var req ChatCompletionsRequest
		Expect(req.Unmarshal([]byte(body))).To(Succeed())
		tk := &recordingTokenizer{}
		_, _, err := req.Render(tk, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(tk.calls).To(Equal(1))
		return tk.tools
	}

	It("gives normal chat tokenization the same tools and tool_choice", func() {
		var req ChatCompletionsRequest
		Expect(req.Unmarshal([]byte(`{"model":"m",` + messages + `,"tools":` + tools + `,"tool_choice":"none"}`))).To(Succeed())
		tk := &recordingTokenizer{}
		reqCtx := req.BuildRequestContext(&fakeRuntime{tokenizer: tk}, common.Channel[*ResponseInfo]{}, 0, func() {})
		_, _, _, err := reqCtx.(*chatCompletionReqCtx).encode()
		Expect(err).NotTo(HaveOccurred())
		Expect(tk.calls).To(Equal(1))
		Expect(string(tk.tools.Tools)).To(Equal(tools))
		Expect(string(tk.tools.ToolChoice)).To(Equal(`"none"`))
	})

	It("keeps the tools' fields and key order", func() {
		got := render(`{"model":"m",` + messages + `,"tools":` + tools + `}`)
		Expect(string(got.Tools)).To(Equal(tools))
		Expect(got.ToolChoice).To(BeNil())
	})

	DescribeTable("forwards tool_choice with its original meaning",
		func(choice string) {
			got := render(`{"model":"m",` + messages + `,"tools":` + tools + `,"tool_choice":` + choice + `}`)
			Expect(string(got.Tools)).To(Equal(tools))
			Expect(string(got.ToolChoice)).To(Equal(choice))
		},
		Entry("none", `"none"`),
		Entry("auto", `"auto"`),
		Entry("named function", `{"type":"function","function":{"name":"f"}}`),
	)

	// tool_choice:null is not covered: the typed ToolChoice rejects it before
	// rendering, independently of this forwarding.
	It("passes neither when the request has none, or null tools", func() {
		for _, body := range []string{
			`{"model":"m",` + messages + `}`,
			`{"model":"m",` + messages + `,"tools":null}`,
		} {
			got := render(body)
			Expect(got.Tools).To(BeNil())
			Expect(got.ToolChoice).To(BeNil())
		}
	})

	DescribeTable("creates no tool calls for a request without tools",
		func(fields string, matchErr OmegaMatcher) {
			var req ChatCompletionsRequest
			Expect(req.Unmarshal([]byte(`{"model":"m",` + messages + fields + `}`))).To(Succeed())
			Expect(req.Validate(nil)).To(BeNil())
			runtime := &fakeRuntime{config: common.NewConfig(), random: common.NewRandom(1, 0)}
			reqCtx := req.BuildRequestContext(runtime, common.Channel[*ResponseInfo]{}, 0, func() {})
			toolCalls, _, _, err := reqCtx.createToolCalls()
			Expect(err).To(matchErr)
			Expect(toolCalls).To(BeNil())
		},
		Entry("empty tools, no tool_choice", `,"tools":[]`, BeNil()),
		Entry("empty tools, auto", `,"tools":[],"tool_choice":"auto"`, BeNil()),
		Entry("empty tools, none", `,"tools":[],"tool_choice":"none"`, BeNil()),
		Entry("no tools, auto", `,"tool_choice":"auto"`, BeNil()),
		Entry("empty tools, a named tool is still looked up",
			`,"tools":[],"tool_choice":{"type":"function","function":{"name":"f"}}`,
			MatchError(ContainSubstring("not found in the tools list"))),
	)

	DescribeTable("rejects tool_choice required without tools",
		func(fields string) {
			var req ChatCompletionsRequest
			Expect(req.Unmarshal([]byte(`{"model":"m",` + messages + fields + `}`))).To(Succeed())
			apiErr := req.Validate(nil)
			Expect(apiErr).NotTo(BeNil())
			Expect(apiErr.Code).To(Equal(fasthttp.StatusBadRequest))
			Expect(apiErr.Message).To(Equal("When using `tool_choice`, `tools` must be set."))
			Expect(apiErr.Param).To(HaveValue(Equal("tool_choice")))
		},
		Entry("required", `,"tools":[],"tool_choice":"required"`),
		Entry("required, tools omitted", `,"tool_choice":"required"`),
	)

	It("accepts tool_choice required with a tool", func() {
		var req ChatCompletionsRequest
		Expect(req.Unmarshal([]byte(`{"model":"m",` + messages + `,"tools":` + tools + `,"tool_choice":"required"}`))).To(Succeed())
		Expect(req.Validate(&ToolsValidator{Skip: func() bool { return true }})).To(BeNil())
	})
})

var _ = Describe("Chat completions template options", func() {
	const messages = `"messages":[{"role":"user","content":"hi"}]`
	const options = `"add_generation_prompt":false,"continue_final_message":true,"add_special_tokens":true,` +
		`"documents":[{"title":"t","text":"x"}],"chat_template":"{{ messages }}",` +
		`"chat_template_kwargs":{"enable_thinking":false,"z":1,"a":2}`

	It("builds the chat render request with the options as the client sent them", func() {
		var req ChatCompletionsRequest
		Expect(req.Unmarshal([]byte(`{"model":"m",` + messages + `,` + options + `}`))).To(Succeed())
		tk := &recordingTokenizer{}
		reqCtx := req.BuildRequestContext(&fakeRuntime{tokenizer: tk}, common.Channel[*ResponseInfo]{}, 0, func() {})
		_, _, _, err := reqCtx.(*chatCompletionReqCtx).encode()
		Expect(err).NotTo(HaveOccurred())

		rendered, err := json.Marshal(api.NewChatCompletionsRenderRequest("m", nil, tk.tools, ""))
		Expect(err).NotTo(HaveOccurred())
		var sent map[string]json.RawMessage
		Expect(json.Unmarshal(rendered, &sent)).To(Succeed())
		Expect(string(sent["add_generation_prompt"])).To(Equal(`false`))
		Expect(string(sent["continue_final_message"])).To(Equal(`true`))
		Expect(string(sent["add_special_tokens"])).To(Equal(`true`))
		Expect(string(sent["documents"])).To(Equal(`[{"title":"t","text":"x"}]`))
		Expect(string(sent["chat_template"])).To(Equal(`"{{ messages }}"`))
		Expect(string(sent["chat_template_kwargs"])).To(Equal(`{"enable_thinking":false,"z":1,"a":2}`))
	})

	It("leaves the options out when the request has none, or sets them to null", func() {
		for _, body := range []string{
			`{"model":"m",` + messages + `}`,
			`{"model":"m",` + messages + `,"add_generation_prompt":null,"chat_template_kwargs":null}`,
		} {
			var req ChatCompletionsRequest
			Expect(req.Unmarshal([]byte(body))).To(Succeed())
			tk := &recordingTokenizer{}
			_, _, err := req.Render(tk, "")
			Expect(err).NotTo(HaveOccurred())

			rendered, err := json.Marshal(api.NewChatCompletionsRenderRequest("m", nil, tk.tools, ""))
			Expect(err).NotTo(HaveOccurred())
			var sent map[string]json.RawMessage
			Expect(json.Unmarshal(rendered, &sent)).To(Succeed())
			for _, key := range []string{"add_generation_prompt", "continue_final_message", "add_special_tokens",
				"documents", "chat_template", "chat_template_kwargs"} {
				Expect(sent).NotTo(HaveKey(key))
			}
		}
	})
})
