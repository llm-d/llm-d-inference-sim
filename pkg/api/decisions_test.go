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

package api

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("SystemOneResponse", func() {
	It("serializes required fields even at zero values", func() {
		response := SystemOneResponse{
			Answers:     SystemOneAnswers{},
			Diagnostics: map[string]DecisionDiagnostics{},
		}
		raw, err := json.Marshal(response)
		Expect(err).NotTo(HaveOccurred())
		Expect(raw).To(MatchJSON(`{
			"id":"","object":"","created":0,"model":"","answers":{},
			"usage":{"input_tokens":0,"output_tokens":0},"diagnostics":{}
		}`))
	})
})
