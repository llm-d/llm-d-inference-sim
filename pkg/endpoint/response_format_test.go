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
	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

var _ = Describe("Chat completions structured output", func() {
	structuredOutput := func(nonRandomResponses bool) []string {
		var req ChatCompletionsRequest
		Expect(req.Unmarshal([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],` +
			`"response_format":{"type":"json_object"}}`))).To(Succeed())
		runtime := &fakeRuntime{config: common.NewConfig(), random: common.NewRandom(1, 0),
			tokenizer: tokenizer.NewSimpleTokenizer(), nonRandomResponses: nonRandomResponses}
		reqCtx := req.BuildRequestContext(runtime, common.Channel[*ResponseInfo]{}, 0, func() {})
		Expect(reqCtx.tokenize()).To(BeNil())
		tokens, _, err := reqCtx.(*chatCompletionReqCtx).createStructuredOutput()
		Expect(err).NotTo(HaveOccurred())
		if tokens == nil {
			return nil
		}
		return tokens.Strings
	}

	It("generates JSON when responses are random text", func() {
		Expect(structuredOutput(false)).NotTo(BeEmpty())
	})

	It("leaves the response to the dataset otherwise", func() {
		Expect(structuredOutput(true)).To(BeNil())
	})
})

var _ = Describe("Chat completions response_format validation", func() {
	validate := func(responseFormat string, skipToolValidation bool) *api.Error {
		v, err := NewToolsValidator()
		Expect(err).NotTo(HaveOccurred())
		v.Skip = func() bool { return skipToolValidation }
		var req ChatCompletionsRequest
		Expect(req.Unmarshal([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],` +
			`"response_format":` + responseFormat + `}`))).To(Succeed())
		return req.Validate(v)
	}

	DescribeTable("accepts",
		func(responseFormat string) {
			Expect(validate(responseFormat, false)).To(BeNil())
		},
		Entry("null", `null`),
		Entry("text", `{"type":"text"}`),
		Entry("json_object", `{"type":"json_object"}`),
		Entry("structural_tag", `{"type":"structural_tag","format":{}}`),
		Entry("json_schema", `{"type":"json_schema","json_schema":{"name":"city","schema":`+
			`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}`),
	)

	DescribeTable("rejects with 400",
		func(responseFormat string, skipToolValidation bool) {
			apiErr := validate(responseFormat, skipToolValidation)
			Expect(apiErr).NotTo(BeNil())
			Expect(apiErr.Code).To(Equal(fasthttp.StatusBadRequest))
		},
		Entry("a string", `"json_object"`, false),
		Entry("an unknown type", `{"type":"xml"}`, false),
		Entry("json_schema without json_schema", `{"type":"json_schema"}`, false),
		Entry("json_schema that is not an object", `{"type":"json_schema","json_schema":[]}`, false),
		Entry("json_schema without schema", `{"type":"json_schema","json_schema":{"name":"x"}}`, false),
		Entry("a schema without type", `{"type":"json_schema","json_schema":{"name":"x","schema":{"properties":{}}}}`, false),
		Entry("a schema with $ref", `{"type":"json_schema","json_schema":{"name":"x","schema":{"$ref":"#/$defs/a"}}}`, false),
		Entry("a nested model with $defs and $ref", `{"type":"json_schema","json_schema":{"name":"x","schema":`+
			`{"$defs":{"City":{"type":"object","properties":{"name":{"type":"string"}}}},`+
			`"type":"object","properties":{"city":{"$ref":"#/$defs/City"}},"required":["city"]}}}`, false),
		Entry("an optional property with anyOf", `{"type":"json_schema","json_schema":{"name":"x","schema":`+
			`{"type":"object","properties":{"age":{"anyOf":[{"type":"integer"},{"type":"null"}]}},"required":["age"]}}}`, false),
		Entry("a schema with oneOf", `{"type":"json_schema","json_schema":{"name":"x","schema":`+
			`{"oneOf":[{"type":"string"},{"type":"integer"}]}}}`, false),
		Entry("a string minItems", `{"type":"json_schema","json_schema":{"name":"x","schema":`+
			`{"type":"array","items":{"type":"string"},"minItems":"2"}}}`, false),
		Entry("a negative maxItems", `{"type":"json_schema","json_schema":{"name":"x","schema":`+
			`{"type":"array","items":{"type":"string"},"maxItems":-1}}}`, false),
		Entry("a string minItems with tool validation skipped", `{"type":"json_schema","json_schema":{"name":"x","schema":`+
			`{"type":"array","items":{"type":"string"},"minItems":"2"}}}`, true),
	)
})
