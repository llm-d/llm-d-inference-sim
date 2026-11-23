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
)

var _ = Describe("responses tool_choice validation", func() {
	const notInTools = "Tool choice 'function' not found in 'tools' parameter."
	const requiredNoTools = "Tool choice 'required' must be specified with 'tools' parameter."
	const weatherTool = `[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{}}}]`

	DescribeTable("validate",
		func(fields string, wantErr string) {
			req := &ResponsesRequest{}
			Expect(req.Unmarshal([]byte(`{"model":"m","input":"hi"` + fields + `}`))).To(Succeed())
			validator, verr := NewToolsValidator()
			Expect(verr).NotTo(HaveOccurred())
			err := req.Validate(validator)
			if wantErr == "" {
				Expect(err).To(BeNil())
				return
			}
			Expect(err).NotTo(BeNil())
			Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
			Expect(err.Message).To(Equal(wantErr))
			Expect(err.Param).To(HaveValue(Equal("tool_choice")))
		},
		Entry("accepts a named tool that is in tools",
			`,"tools":`+weatherTool+`,"tool_choice":{"type":"function","name":"get_weather"}`, ""),
		Entry("rejects a named tool that is not in tools",
			`,"tools":`+weatherTool+`,"tool_choice":{"type":"function","name":"get_time"}`, notInTools),
		Entry("rejects a named tool with an empty tools array",
			`,"tools":[],"tool_choice":{"type":"function","name":"get_time"}`, notInTools),
		Entry("rejects a named tool with tools omitted",
			`,"tool_choice":{"type":"function","name":"get_time"}`, notInTools),
		Entry("rejects required with tools omitted", `,"tool_choice":"required"`, requiredNoTools),
		Entry("rejects required with an empty tools array", `,"tools":[],"tool_choice":"required"`, requiredNoTools),
		Entry("accepts required with tools", `,"tools":`+weatherTool+`,"tool_choice":"required"`, ""),
		Entry("accepts auto with tools omitted", `,"tool_choice":"auto"`, ""),
		Entry("accepts none with tools omitted", `,"tool_choice":"none"`, ""),
		Entry("accepts no tool_choice", `,"tools":`+weatherTool, ""),
	)
})
