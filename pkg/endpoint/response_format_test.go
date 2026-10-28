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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

var _ = Describe("Chat completions structured output", func() {
	structuredOutput := func(configure func(*common.Configuration)) []string {
		config := common.NewConfig()
		configure(config)
		var req ChatCompletionsRequest
		Expect(req.Unmarshal([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],` +
			`"response_format":{"type":"json_object"}}`))).To(Succeed())
		runtime := &fakeRuntime{config: config, random: common.NewRandom(1, 0), tokenizer: tokenizer.NewSimpleTokenizer()}
		reqCtx := req.BuildRequestContext(runtime, common.Channel[*ResponseInfo]{}, 0, func() {})
		Expect(reqCtx.tokenize()).To(BeNil())
		tokens, _, err := reqCtx.(*chatCompletionReqCtx).createStructuredOutput()
		Expect(err).NotTo(HaveOccurred())
		if tokens == nil {
			return nil
		}
		return tokens.Strings
	}

	It("generates JSON in random mode", func() {
		Expect(structuredOutput(func(*common.Configuration) {})).NotTo(BeEmpty())
	})

	DescribeTable("leaves the response to the configured source",
		func(configure func(*common.Configuration)) {
			Expect(structuredOutput(configure)).To(BeNil())
		},
		Entry("echo mode", func(c *common.Configuration) { c.Mode = common.ModeEcho }),
		Entry("dataset path", func(c *common.Configuration) { c.Dataset.DatasetPath = "responses.sqlite3" }),
		Entry("dataset URL", func(c *common.Configuration) { c.Dataset.DatasetURL = "https://example.com/responses.sqlite3" }),
		Entry("mm-encoder-only", func(c *common.Configuration) { c.MMEncoderOnly = true }),
	)
})
