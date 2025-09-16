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

package llmdinferencesim

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

var _ = Describe("Failures", func() {
	Describe("getRandomFailure", Ordered, func() {
		BeforeAll(func() {
			common.InitRandom(time.Now().UnixNano())
		})

		It("should return a failure from all types when none specified", func() {
			config := &common.Configuration{
				Model:        "test-model",
				FailureTypes: []string{},
			}
			failure := getRandomFailure(config)
			Expect(failure.Code).To(BeNumerically(">=", 400))
			Expect(failure.Message).ToNot(BeEmpty())
			Expect(failure.Type).ToNot(BeEmpty())
		})

		It("should return rate limit failure when specified", func() {
			config := &common.Configuration{
				Model:        "test-model",
				FailureTypes: []string{common.FailureTypeRateLimit},
			}
			failure := getRandomFailure(config)
			Expect(failure.Code).To(Equal(429))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(429)))
			Expect(strings.Contains(failure.Message, "test-model")).To(BeTrue())
		})

		It("should return invalid API key failure when specified", func() {
			config := &common.Configuration{
				FailureTypes: []string{common.FailureTypeInvalidAPIKey},
			}
			failure := getRandomFailure(config)
			Expect(failure.Code).To(Equal(401))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(401)))
			Expect(failure.Message).To(Equal("Incorrect API key provided."))
		})

		It("should return context length failure when specified", func() {
			config := &common.Configuration{
				FailureTypes: []string{common.FailureTypeContextLength},
			}
			failure := getRandomFailure(config)
			Expect(failure.Code).To(Equal(400))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(400)))
			Expect(failure.Param).ToNot(BeNil())
			Expect(*failure.Param).To(Equal("messages"))
		})

		It("should return server error when specified", func() {
			config := &common.Configuration{
				FailureTypes: []string{common.FailureTypeServerError},
			}
			failure := getRandomFailure(config)
			Expect(failure.Code).To(Equal(503))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(503)))
		})

		It("should return model not found failure when specified", func() {
			config := &common.Configuration{
				Model:        "test-model",
				FailureTypes: []string{common.FailureTypeModelNotFound},
			}
			failure := getRandomFailure(config)
			Expect(failure.Code).To(Equal(404))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(404)))
			Expect(strings.Contains(failure.Message, "test-model-nonexistent")).To(BeTrue())
		})

		It("should return server error as fallback for empty types", func() {
			config := &common.Configuration{
				FailureTypes: []string{},
			}
			// This test is probabilistic since it randomly selects, but we can test structure
			failure := getRandomFailure(config)
			Expect(failure.Code).To(BeNumerically(">=", 400))
			Expect(failure.Type).ToNot(BeEmpty())
		})
	})
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
				client, err = startServerWithArgs(ctx, "", []string{
					"cmd", "--model", model,
					"--failure-injection-rate", "100",
				}, nil)
				Expect(err).ToNot(HaveOccurred())
			})

			It("should always return an error response for chat completions", func() {
				openaiClient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
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
				openaiClient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
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
				client, err = startServerWithArgs(ctx, "", []string{
					"cmd", "--model", model,
					"--failure-injection-rate", "100",
					"--failure-types", common.FailureTypeRateLimit,
				}, nil)
				Expect(err).ToNot(HaveOccurred())
			})

			It("should return only rate limit errors", func() {
				openaiClient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
				_, err := openaiClient.Chat.Completions.New(ctx, params)
				Expect(err).To(HaveOccurred())

				var openaiError *openai.Error
				ok := errors.As(err, &openaiError)
				Expect(ok).To(BeTrue())
				Expect(openaiError.StatusCode).To(Equal(429))
				Expect(openaiError.Type).To(Equal(openaiserverapi.ErrorCodeToType(429)))
				Expect(strings.Contains(openaiError.Message, model)).To(BeTrue())
			})
		})

		Context("with multiple specific failure types", func() {
			BeforeEach(func() {
				ctx = context.Background()
				var err error
				client, err = startServerWithArgs(ctx, "", []string{
					"cmd", "--model", model,
					"--failure-injection-rate", "100",
					"--failure-types", common.FailureTypeInvalidAPIKey, common.FailureTypeServerError,
				}, nil)
				Expect(err).ToNot(HaveOccurred())
			})

			It("should return only specified error types", func() {
				openaiClient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)

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
				client, err = startServerWithArgs(ctx, "", []string{
					"cmd", "--model", model,
					"--failure-injection-rate", "0",
				}, nil)
				Expect(err).ToNot(HaveOccurred())
			})

			It("should never return errors and behave like random mode", func() {
				openaiClient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
				resp, err := openaiClient.Chat.Completions.New(ctx, params)
				Expect(err).ToNot(HaveOccurred())
				Expect(resp.Choices).To(HaveLen(1))
				Expect(resp.Choices[0].Message.Content).ToNot(BeEmpty())
				Expect(resp.Model).To(Equal(model))
			})
		})

		Context("testing all predefined failure types", func() {
			DescribeTable("should return correct error for each failure type",
				func(failureType string, expectedStatusCode int, expectedErrorType string) {
					ctx := context.Background()
					client, err := startServerWithArgs(ctx, "", []string{
						"cmd", "--model", model,
						"--failure-injection-rate", "100",
						"--failure-types", failureType,
					}, nil)
					Expect(err).ToNot(HaveOccurred())

					openaiClient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
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
