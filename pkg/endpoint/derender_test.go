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
	"fmt"
	"strconv"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
	"k8s.io/klog/v2"
)

func tokenIDsAsJSON(ids []uint32) string {
	strs := make([]string, len(ids))
	for i, id := range ids {
		strs[i] = strconv.FormatUint(uint64(id), 10)
	}
	return "[" + strings.Join(strs, ",") + "]"
}

var _ = Describe("derender requests", func() {
	const displayModel = "display-model"
	logger := klog.Background()

	var tk tokenizer.Tokenizer

	BeforeEach(func() {
		tk = tokenizer.NewSimpleTokenizer()
	})

	Describe("chat", func() {
		unmarshalChat := func(body string) *DerenderChatRequest {
			req := &DerenderChatRequest{}
			Expect(req.Unmarshal([]byte(body))).To(Succeed())
			return req
		}

		It("round-trips tokens produced by the tokenizer", func() {
			text := "The quick brown fox jumps over the lazy dog."
			ids, _, err := tk.RenderText(text)
			Expect(err).NotTo(HaveOccurred())

			req := unmarshalChat(fmt.Sprintf(
				`{"model":"m","prompt_tokens":7,"generate_response":{"request_id":"req-1",`+
					`"kv_transfer_params":{"remote_host":"h","remote_port":42},`+
					`"choices":[{"index":0,"finish_reason":"length","token_ids":%s}]}}`,
				tokenIDsAsJSON(ids)))
			Expect(req.ValidateBody()).To(BeNil())

			result, apiErr := req.Derender(tk, displayModel, logger)
			Expect(apiErr).To(BeNil())
			resp, ok := result.(*api.ChatCompletionsResponse)
			Expect(ok).To(BeTrue())

			Expect(resp.ID).To(Equal("req-1"))
			Expect(resp.Model).To(Equal("m"))
			Expect(resp.Object).To(Equal(api.ChatCompletionObject))
			Expect(resp.KVParams).NotTo(BeNil())
			Expect(resp.KVParams.RemoteHost).To(Equal("h"))
			Expect(resp.KVParams.RemotePort).To(Equal(42))
			Expect(resp.Choices).To(HaveLen(1))
			Expect(resp.Choices[0].Index).To(Equal(0))
			Expect(*resp.Choices[0].FinishReason).To(Equal("length"))
			Expect(resp.Choices[0].Message.Role).To(Equal(api.RoleAssistant))
			Expect(resp.Choices[0].Message.Content.Raw).To(Equal(text))
			Expect(resp.Usage.PromptTokens).To(Equal(7))
			Expect(resp.Usage.CompletionTokens).To(Equal(len(ids)))
			Expect(resp.Usage.TotalTokens).To(Equal(7 + len(ids)))
		})

		It("uses the display model when the request has no model", func() {
			ids, _, err := tk.RenderText("hi")
			Expect(err).NotTo(HaveOccurred())
			req := unmarshalChat(fmt.Sprintf(
				`{"generate_response":{"request_id":"r","choices":[{"index":0,"token_ids":%s}]}}`,
				tokenIDsAsJSON(ids)))
			Expect(req.ValidateBody()).To(BeNil())
			result, apiErr := req.Derender(tk, displayModel, logger)
			Expect(apiErr).To(BeNil())
			Expect(result.(*api.ChatCompletionsResponse).Model).To(Equal(displayModel))
		})

		It("renders unknown token ids as placeholders", func() {
			req := unmarshalChat(`{"generate_response":{"request_id":"r","choices":[{"index":0,"token_ids":[123]}]}}`)
			Expect(req.ValidateBody()).To(BeNil())
			result, apiErr := req.Derender(tk, displayModel, logger)
			Expect(apiErr).To(BeNil())
			Expect(result.(*api.ChatCompletionsResponse).Choices[0].Message.Content.Raw).To(Equal("<unk_123>"))
		})

		DescribeTable("body validation",
			func(body string, expectedMsg string) {
				req := unmarshalChat(body)
				err := req.ValidateBody()
				Expect(err).NotTo(BeNil())
				Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
				Expect(err.Message).To(ContainSubstring(expectedMsg))
			},
			Entry("rejects streaming",
				`{"stream":true,"generate_response":{"choices":[{"index":0,"token_ids":[1]}]}}`,
				"streaming derender is not supported"),
			Entry("rejects a missing generate_response",
				`{"model":"m"}`,
				"generate_response is required"),
			Entry("rejects empty token_ids",
				`{"generate_response":{"choices":[{"index":2,"token_ids":[]}]}}`,
				"choice 2 has empty or null token_ids"),
		)
	})

	Describe("completions", func() {
		unmarshalCompletion := func(body string) *DerenderCompletionRequest {
			req := &DerenderCompletionRequest{}
			Expect(req.Unmarshal([]byte(body))).To(Succeed())
			return req
		}

		It("round-trips tokens with a flat choice index across responses", func() {
			ids1, _, err := tk.RenderText("first prompt answer")
			Expect(err).NotTo(HaveOccurred())
			ids2, _, err := tk.RenderText("second one")
			Expect(err).NotTo(HaveOccurred())

			req := unmarshalCompletion(fmt.Sprintf(
				`{"model":"m","prompt_tokens":[3,4],"generate_responses":[`+
					`{"request_id":"req-1","choices":[{"index":0,"finish_reason":"stop","token_ids":%s}]},`+
					`{"request_id":"req-2","choices":[{"index":0,"finish_reason":"stop","token_ids":%s}]}]}`,
				tokenIDsAsJSON(ids1), tokenIDsAsJSON(ids2)))
			Expect(req.ValidateBody()).To(BeNil())

			result, apiErr := req.Derender(tk, displayModel, logger)
			Expect(apiErr).To(BeNil())
			resp, ok := result.(*api.TextCompletionsResponse)
			Expect(ok).To(BeTrue())

			Expect(resp.ID).To(Equal("req-1"))
			Expect(resp.Object).To(Equal(api.TextCompletionObject))
			Expect(resp.Choices).To(HaveLen(2))
			Expect(resp.Choices[0].Index).To(Equal(0))
			Expect(resp.Choices[0].Text).To(Equal("first prompt answer"))
			Expect(resp.Choices[1].Index).To(Equal(1))
			Expect(resp.Choices[1].Text).To(Equal("second one"))
			Expect(resp.Usage.PromptTokens).To(Equal(7))
			Expect(resp.Usage.CompletionTokens).To(Equal(len(ids1) + len(ids2)))
			Expect(resp.Usage.TotalTokens).To(Equal(7 + len(ids1) + len(ids2)))
		})

		It("keeps kv_transfer_params shared by all responses", func() {
			req := unmarshalCompletion(
				`{"generate_responses":[` +
					`{"request_id":"a","kv_transfer_params":{"remote_host":"h"},"choices":[{"index":0,"token_ids":[1]}]},` +
					`{"request_id":"b","kv_transfer_params":{"remote_host":"h"},"choices":[{"index":0,"token_ids":[2]}]}]}`)
			result, apiErr := req.Derender(tk, displayModel, logger)
			Expect(apiErr).To(BeNil())
			resp := result.(*api.TextCompletionsResponse)
			Expect(resp.KVParams).NotTo(BeNil())
			Expect(resp.KVParams.RemoteHost).To(Equal("h"))
		})

		It("drops differing kv_transfer_params", func() {
			req := unmarshalCompletion(
				`{"generate_responses":[` +
					`{"request_id":"a","kv_transfer_params":{"remote_host":"h1"},"choices":[{"index":0,"token_ids":[1]}]},` +
					`{"request_id":"b","kv_transfer_params":{"remote_host":"h2"},"choices":[{"index":0,"token_ids":[2]}]}]}`)
			result, apiErr := req.Derender(tk, displayModel, logger)
			Expect(apiErr).To(BeNil())
			Expect(result.(*api.TextCompletionsResponse).KVParams).To(BeNil())
		})

		DescribeTable("body validation",
			func(body string, expectedMsg string) {
				req := unmarshalCompletion(body)
				err := req.ValidateBody()
				Expect(err).NotTo(BeNil())
				Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
				Expect(err.Message).To(ContainSubstring(expectedMsg))
			},
			Entry("rejects streaming",
				`{"stream":true,"generate_responses":[{"choices":[{"index":0,"token_ids":[1]}]}]}`,
				"streaming derender is not supported"),
			Entry("rejects empty generate_responses",
				`{"model":"m","generate_responses":[]}`,
				"generate_responses must not be empty"),
			Entry("rejects a prompt_tokens length mismatch",
				`{"prompt_tokens":[1,2],"generate_responses":[{"choices":[{"index":0,"token_ids":[1]}]}]}`,
				"prompt_tokens length (2) must equal generate_responses length (1)"),
			Entry("rejects empty token_ids",
				`{"generate_responses":[{"choices":[{"index":1,"token_ids":[]}]}]}`,
				"choice 1 has empty or null token_ids"),
		)
	})
})
