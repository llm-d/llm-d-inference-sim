/*
Copyright 2025 The llm-d-inference-sim Authors.

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

package simulator

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("tools schema validator", func() {
	DescribeTable("accepts common JSON schema keywords on parameters",
		func(toolJSON string) {
			v, err := createToolsValidator()
			Expect(err).NotTo(HaveOccurred())

			var value any
			Expect(json.Unmarshal([]byte(toolJSON), &value)).To(Succeed())
			Expect(v.validateTool([]byte(toolJSON))).To(Succeed())
		},
		Entry(nil, `{
			"name": "get_weather",
			"description": "Get the weather",
			"parameters": {
				"type": "object",
				"properties": {
					"location": {"type": "string"},
					"unit": {"type": "string", "enum": ["C", "F"], "default": "C"}
				},
				"required": ["location"]
			}
		}`),
		Entry(nil, `{
			"name": "get_weather",
			"description": "Get the weather",
			"parameters": {
				"type": "object",
				"properties": {
					"unit": {"type": "string", "default": "C",
						"minLength": 1, "maxLength": 3, "pattern": "^[CF]$",
						"format": "temperature", "title": "Unit"}
				},
				"required": ["unit"]
			}
		}`),
		Entry(nil, `{
			"name": "get_count",
			"description": "Get a count",
			"parameters": {
				"type": "object",
				"properties": {
					"count": {"type": "integer", "minimum": 0, "maximum": 100,
						"exclusiveMinimum": 0, "exclusiveMaximum": 100, "default": 5}
				},
				"required": ["count"]
			}
		}`),
	)
})
