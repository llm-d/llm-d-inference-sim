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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

// sglang frames an error by three separate helpers, one per API family, and no
// two agree. Each spec below pins the exact body one of them produces.
var _ = Describe("error bodies", func() {
	badRequest := api.NewError("bad request", 400, nil)

	DescribeTable("should frame a whole-response body",
		func(route api.ErrorRoute, err api.Error, expected string) {
			data, marshalErr := json.Marshal(New().ErrorBody(err, route))
			Expect(marshalErr).NotTo(HaveOccurred())
			Expect(string(data)).To(Equal(expected))
		},
		// The error's own fields at the top level, tagged as an error object,
		// where vLLM wraps them under an "error" key.
		Entry("an OpenAI-shaped route", api.ErrorRouteDefault, badRequest,
			`{"object":"error","message":"bad request","type":"BadRequestError","param":null,"code":400}`),
		// Nested, and every type on this route is spelled the same way whatever
		// the status: one helper builds them all and never varies it.
		Entry("the Responses API", api.ErrorRouteResponses, badRequest,
			`{"error":{"message":"bad request","type":"invalid_request_error","param":null,"code":400}}`),
		Entry("the Responses API, on a status of its own", api.ErrorRouteResponses,
			api.NewError("The model 'x' does not exist", 404, nil),
			`{"error":{"message":"The model 'x' does not exist","type":"invalid_request_error","param":null,"code":404}}`),
		// The Anthropic envelope, which has no field for the param or the status.
		Entry("the Messages API", api.ErrorRouteMessages, badRequest,
			`{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`),
	)

	DescribeTable("should frame a streaming frame",
		func(route api.ErrorRoute, expected string) {
			data, marshalErr := json.Marshal(New().StreamErrorBody(badRequest, route))
			Expect(marshalErr).NotTo(HaveOccurred())
			Expect(string(data)).To(Equal(expected))
		},
		// Wrapped here, unlike the whole-response body above.
		Entry("an OpenAI-shaped route", api.ErrorRouteDefault,
			`{"error":{"object":"error","message":"bad request","type":"BadRequestError","param":null,"code":400}}`),
		// The error's own type survives here, unlike in a whole-response body.
		Entry("the Responses API", api.ErrorRouteResponses,
			`{"error":{"message":"bad request","type":"BadRequestError","param":null,"code":400}}`),
		Entry("the Messages API", api.ErrorRouteMessages,
			`{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`),
	)

	// On /v1/messages the type is drawn from Anthropic's own vocabulary rather
	// than this engine's usual spelling, and a server error's message is replaced.
	DescribeTable("should map a Messages API error to Anthropic's vocabulary",
		func(code int, message, expectedType, expectedMessage string) {
			data, marshalErr := json.Marshal(New().ErrorBody(api.NewError(message, code, nil),
				api.ErrorRouteMessages))
			Expect(marshalErr).NotTo(HaveOccurred())

			var body api.MessagesErrorResponse
			Expect(json.Unmarshal(data, &body)).To(Succeed())
			Expect(body.Type).To(Equal(api.MessagesTypeError))
			Expect(body.Error.Type).To(Equal(expectedType))
			Expect(body.Error.Message).To(Equal(expectedMessage))
		},
		Entry("bad request", 400, "messages must not be empty", "invalid_request_error", "messages must not be empty"),
		Entry("unauthorized", 401, "no key", "authentication_error", "no key"),
		Entry("not found", 404, "no such model", "not_found_error", "no such model"),
		Entry("too many requests", 429, "slow down", "rate_limit_error", "slow down"),
		// No server error's own message reaches the client, whichever 5xx it is.
		Entry("service unavailable", 503, "loading", "overloaded_error", "Internal server error"),
		Entry("server error", 500, "tokenizer blew up", "api_error", "Internal server error"),
		// A status the map does not name still gets a documented type.
		Entry("unnamed status", 418, "teapot", "api_error", "teapot"),
	)
})
