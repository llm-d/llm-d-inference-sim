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

package communication

import (
	"encoding/json"
	"testing"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
)

var _ = Describe("strict vLLM request validation", func() {
	Describe("content type", func() {
		It("accepts JSON with parameters", func() {
			Expect(validateStrictContentType("application/json; charset=utf-8")).To(BeNil())
		})

		It("matches vLLM's unsupported media type response", func() {
			Expect(validateStrictContentType("text/plain")).To(Equal(&api.Error{
				Message: "1 validation error:\n  Unsupported Media Type: Only 'application/json' is allowed " +
					"[\"Unsupported Media Type: Only 'application/json' is allowed\"]",
				Type: "Bad Request",
				Code: fasthttp.StatusBadRequest,
			}))
		})
	})

	type invalidCase struct {
		body    string
		path    string
		message string
		param   *string
	}

	DescribeTable("matches vLLM sampling errors",
		func(tc invalidCase) {
			path := tc.path
			if path == "" {
				path = "/v1/chat/completions"
			}
			err := validateStrictCompletionBody([]byte(tc.body), path)
			Expect(err).To(Equal(&api.Error{
				Message: tc.message,
				Type:    "BadRequestError",
				Param:   tc.param,
				Code:    fasthttp.StatusBadRequest,
			}))
		},
		Entry("n below one", invalidCase{
			body:    `{"n":0}`,
			message: "n must be at least 1, got 0.",
		}),
		Entry("max_tokens below one", invalidCase{
			body:    `{"max_tokens":0}`,
			message: "max_tokens must be at least 1, got 0. (parameter=max_tokens, value=0)",
			param:   ptrTo("max_tokens"),
		}),
		Entry("max_completion_tokens maps to vLLM max_tokens error", invalidCase{
			body:    `{"max_completion_tokens":-1}`,
			message: "max_tokens must be at least 1, got -1. (parameter=max_tokens, value=-1)",
			param:   ptrTo("max_tokens"),
		}),
		Entry("negative temperature", invalidCase{
			body:    `{"temperature":-0.1}`,
			message: "temperature must be non-negative, got -0.1. (parameter=temperature, value=-0.1)",
			param:   ptrTo("temperature"),
		}),
		Entry("top_p above one", invalidCase{
			body:    `{"top_p":1.1}`,
			message: "top_p must be in (0, 1], got 1.1. (parameter=top_p, value=1.1)",
			param:   ptrTo("top_p"),
		}),
		Entry("top_p equal to zero", invalidCase{
			body:    `{"top_p":0}`,
			message: "top_p must be in (0, 1], got 0.0. (parameter=top_p, value=0.0)",
			param:   ptrTo("top_p"),
		}),
		Entry("presence penalty outside range", invalidCase{
			body:    `{"presence_penalty":2.1}`,
			message: "presence_penalty must be in [-2, 2], got 2.1.",
		}),
		Entry("frequency penalty outside range", invalidCase{
			body:    `{"frequency_penalty":-2.1}`,
			message: "frequency_penalty must be in [-2, 2], got -2.1.",
		}),
		Entry("non-positive repetition penalty", invalidCase{
			body:    `{"repetition_penalty":0}`,
			message: "repetition_penalty must be greater than zero, got 0.0.",
		}),
		Entry("min_p outside range", invalidCase{
			body:    `{"min_p":-0.1}`,
			message: "min_p must be in [0, 1], got -0.1.",
		}),
		Entry("top_k below disabled sentinel", invalidCase{
			body:    `{"top_k":-2}`,
			message: "top_k must be 0 (disable), or at least 1, got -2.",
		}),
		Entry("chat top_logprobs above model limit", invalidCase{
			body:    `{"logprobs":true,"top_logprobs":21}`,
			message: "Requested sample logprobs of 21, which is greater than max allowed: 20 (parameter=logprobs, value=21)",
			param:   ptrTo("logprobs"),
		}),
		Entry("text completion logprobs above model limit", invalidCase{
			body:    `{"logprobs":21}`,
			path:    "/v1/completions",
			message: "Requested sample logprobs of 21, which is greater than max allowed: 20 (parameter=logprobs, value=21)",
			param:   ptrTo("logprobs"),
		}),
		Entry("negative min_tokens", invalidCase{
			body:    `{"min_tokens":-1}`,
			message: "min_tokens must be greater than or equal to 0, got -1.",
		}),
		Entry("min_tokens above max_tokens", invalidCase{
			body:    `{"min_tokens":5,"max_tokens":4}`,
			message: "min_tokens must be less than or equal to max_tokens=4, got 5.",
		}),
		Entry("empty stop string", invalidCase{
			body:    `{"stop":""}`,
			message: "stop cannot contain an empty string.",
		}),
		Entry("empty stop string in array", invalidCase{
			body:    `{"stop":["done",""]}`,
			message: "stop cannot contain an empty string.",
		}),
	)

	It("accepts vLLM boundary values", func() {
		body := []byte(`{
			"n":1,
			"max_tokens":1,
			"max_completion_tokens":1,
			"temperature":0,
			"top_p":1,
			"presence_penalty":-2,
			"frequency_penalty":2,
			"repetition_penalty":0.1,
			"min_p":0,
			"min_tokens":1,
			"stop":["done"]
		}`)
		Expect(validateStrictCompletionBody(body, "/v1/chat/completions")).To(BeNil())
	})

	It("defers malformed field types to normal request decoding", func() {
		Expect(validateStrictCompletionBody([]byte(`{"temperature":"cold"}`), "/v1/chat/completions")).To(BeNil())
	})

	It("serializes stable validation errors byte-for-byte like vLLM", func() {
		err := validateStrictCompletionBody([]byte(`{"top_p":0}`), "/v1/chat/completions")
		body, marshalErr := json.Marshal(api.ErrorResponse{Error: *err})
		Expect(marshalErr).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal(
			`{"error":{"message":"top_p must be in (0, 1], got 0.0. (parameter=top_p, value=0.0)",` +
				`"type":"BadRequestError","param":"top_p","code":400}}`,
		))
	})
})

func ptrTo(value string) *string {
	return &value
}

func BenchmarkStrictCompletionValidation(b *testing.B) {
	valid := []byte(`{"n":1,"max_completion_tokens":32,"temperature":0.7,"top_p":0.95,"top_k":50,"min_p":0.05,"stop":["done"]}`)
	invalid := []byte(`{"n":1,"max_completion_tokens":32,"temperature":0.7,"top_p":0}`)

	b.Run("valid", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			if err := validateStrictCompletionBody(valid, "/v1/chat/completions"); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("invalid", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			if err := validateStrictCompletionBody(invalid, "/v1/chat/completions"); err == nil {
				b.Fatal("expected a validation error")
			}
		}
	})
}
