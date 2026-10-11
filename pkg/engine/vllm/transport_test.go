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

package vllm

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

// vLLM frames an error the same way everywhere except the Messages API, where it
// translates the same error object into the Anthropic envelope field by field,
// keeping its own error-type spelling. The OpenAI envelope pinned here is the
// shape every existing client of this simulator already parses.
var _ = Describe("error bodies", func() {
	err := api.NewError("bad request", 400, nil)
	const (
		openAIBody    = `{"error":{"message":"bad request","type":"BadRequestError","param":null,"code":400}}`
		anthropicBody = `{"type":"error","error":{"type":"BadRequestError","message":"bad request"}}`
	)

	DescribeTable("should frame a whole-response body",
		func(route api.ErrorRoute, expected string) {
			data, marshalErr := json.Marshal(New().ErrorBody(err, route))
			Expect(marshalErr).NotTo(HaveOccurred())
			Expect(string(data)).To(Equal(expected))
		},
		Entry("an OpenAI-shaped route", api.ErrorRouteDefault, openAIBody),
		Entry("the Responses API", api.ErrorRouteResponses, openAIBody),
		Entry("the Messages API", api.ErrorRouteMessages, anthropicBody),
	)

	DescribeTable("should frame a streaming frame the same way",
		func(route api.ErrorRoute, expected string) {
			data, marshalErr := json.Marshal(New().StreamErrorBody(err, route))
			Expect(marshalErr).NotTo(HaveOccurred())
			Expect(string(data)).To(Equal(expected))
		},
		Entry("an OpenAI-shaped route", api.ErrorRouteDefault, openAIBody),
		Entry("the Responses API", api.ErrorRouteResponses, openAIBody),
		Entry("the Messages API", api.ErrorRouteMessages, anthropicBody),
	)
})
