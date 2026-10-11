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
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// recordingTokenizer captures what each RenderMessages call receives.
type recordingTokenizer struct {
	tools api.RenderTools
	calls int
}

func (r *recordingTokenizer) RenderText(string) ([]uint32, []string, error) { return nil, nil, nil }
func (r *recordingTokenizer) Detokenize([]uint32) (string, error)           { return "", nil }
func (r *recordingTokenizer) RenderMessages(_ []api.Message, tools api.RenderTools) ([]uint32, []string, *api.RenderMMFeatures, error) {
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
		_, _, err := req.Render(tk)
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
})
