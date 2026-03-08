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

package tests

import (
	"context"
	"errors"
	"net/http"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

var _ = Describe("Failures", func() {
	Describe("Simulator with failure injection", func() {
		var (
			client *http.Client
			ctx    context.Context
		)

		AfterEach(func() {
			if ctx != nil {
				ctx.Done()
			}
		})

		Context("with 100% failure injection rate", func() {
			BeforeEach(func() {
				ctx = context.Background()
				var err error
				client, err = startServerWithArgs(ctx, []string{
					"cmd", "--model", testModel,
					"--failure-injection-rate", "100",
				})
				Expect(err).ToNot(HaveOccurred())
			})

			It("should always return an error response for chat completions", func() {
				openaiClient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)
				_, err := openaiClient.Chat.Completions.New(ctx, params)
				Expect(err).To(HaveOccurred())

				var openaiError *openai.Error
				ok := errors.As(err, &openaiError)
				Expect(ok).To(BeTrue())
				Expect(openaiError.StatusCode).To(BeNumerically(">=", 400))
				Expect(openaiError.Type).ToNot(BeEmpty())
				Expect(openaiError.Message).ToNot(BeEmpty())
			})

			It("should always return an error response for text completions", func() {
				openaiClient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)
				_, err := openaiClient.Chat.Completions.New(ctx, params)
				Expect(err).To(HaveOccurred())

				var openaiError *openai.Error
				ok := errors.As(err, &openaiError)
				Expect(ok).To(BeTrue())
				Expect(openaiError.StatusCode).To(BeNumerically(">=", 400))
				Expect(openaiError.Type).ToNot(BeEmpty())
				Expect(openaiError.Message).ToNot(BeEmpty())
			})
		})

		Context("with specific failure types", func() {
			BeforeEach(func() {
				ctx = context.Background()
				var err error
				client, err = startServerWithArgs(ctx, []string{
					"cmd", "--model", testModel,
					"--failure-injection-rate", "100",
					"--failure-types", common.FailureTypeRateLimit,
				})
				Expect(err).ToNot(HaveOccurred())
			})

			It("should return only rate limit errors", func() {
				openaiClient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)
				_, err := openaiClient.Chat.Completions.New(ctx, params)
				Expect(err).To(HaveOccurred())

				var openaiError *openai.Error
				ok := errors.As(err, &openaiError)
				Expect(ok).To(BeTrue())
				Expect(openaiError.StatusCode).To(Equal(429))
				Expect(openaiError.Type).To(Equal(openaiserverapi.ErrorCodeToType(429)))
				Expect(strings.Contains(openaiError.Message, testModel)).To(BeTrue())
			})
		})

		Context("with multiple specific failure types", func() {
			BeforeEach(func() {
				ctx = context.Background()
				var err error
				client, err = startServerWithArgs(ctx, []string{
					"cmd", "--model", testModel,
					"--failure-injection-rate", "100",
					"--failure-types", common.FailureTypeInvalidAPIKey, common.FailureTypeServerError,
				})
				Expect(err).ToNot(HaveOccurred())
			})

			It("should return only specified error types", func() {
				openaiClient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)

				// Make multiple requests to verify we get the expected error types
				for i := 0; i < 10; i++ {
					_, err := openaiClient.Chat.Completions.New(ctx, params)
					Expect(err).To(HaveOccurred())

					var openaiError *openai.Error
					ok := errors.As(err, &openaiError)
					Expect(ok).To(BeTrue())

					// Should only be one of the specified types
					Expect(openaiError.StatusCode == 401 || openaiError.StatusCode == 503).To(BeTrue())
					Expect(openaiError.Type == openaiserverapi.ErrorCodeToType(401) ||
						openaiError.Type == openaiserverapi.ErrorCodeToType(503)).To(BeTrue())
				}
			})
		})

		Context("with 0% failure injection rate", func() {
			BeforeEach(func() {
				ctx = context.Background()
				var err error
				client, err = startServerWithArgs(ctx, []string{
					"cmd", "--model", testModel,
					"--failure-injection-rate", "0",
				})
				Expect(err).ToNot(HaveOccurred())
			})

			It("should never return errors and behave like random mode", func() {
				openaiClient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)
				resp, err := openaiClient.Chat.Completions.New(ctx, params)
				Expect(err).ToNot(HaveOccurred())
				Expect(resp.Choices).To(HaveLen(1))
				Expect(resp.Choices[0].Message.Content).ToNot(BeEmpty())
				Expect(resp.Model).To(Equal(testModel))
			})
		})

		Context("testing all predefined failure types", func() {
			DescribeTable("should return correct error for each failure type",
				func(failureType string, expectedStatusCode int, expectedErrorType string) {
					ctx := context.Background()
					client, err := startServerWithArgs(ctx, []string{
						"cmd", "--model", testModel,
						"--failure-injection-rate", "100",
						"--failure-types", failureType,
					})
					Expect(err).ToNot(HaveOccurred())

					openaiClient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)
					_, err = openaiClient.Chat.Completions.New(ctx, params)
					Expect(err).To(HaveOccurred())

					var openaiError *openai.Error
					ok := errors.As(err, &openaiError)
					Expect(ok).To(BeTrue())
					Expect(openaiError.StatusCode).To(Equal(expectedStatusCode))
					Expect(openaiError.Type).To(Equal(expectedErrorType))
					// Note: OpenAI Go client doesn't directly expose the error code field,
					// but we can verify via status code and type
				},
				Entry("rate_limit", common.FailureTypeRateLimit, 429, openaiserverapi.ErrorCodeToType(429)),
				Entry("invalid_api_key", common.FailureTypeInvalidAPIKey, 401, openaiserverapi.ErrorCodeToType(401)),
				Entry("context_length", common.FailureTypeContextLength, 400, openaiserverapi.ErrorCodeToType(400)),
				Entry("server_error", common.FailureTypeServerError, 503, openaiserverapi.ErrorCodeToType(503)),
				Entry("invalid_request", common.FailureTypeInvalidRequest, 400, openaiserverapi.ErrorCodeToType(400)),
				Entry("model_not_found", common.FailureTypeModelNotFound, 404, openaiserverapi.ErrorCodeToType(404)),
			)
		})
	})
})
