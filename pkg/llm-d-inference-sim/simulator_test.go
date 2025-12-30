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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
)

const invalidMaxTokensErrMsg = "Max completion tokens and max tokens should be positive"

var _ = Describe("Simulator", func() {

	DescribeTable("chat completions streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, true)
			stream := openaiclient.Chat.Completions.NewStreaming(ctx, params)
			defer func() {
				err := stream.Close()
				Expect(err).NotTo(HaveOccurred())
			}()
			tokens := []string{}
			role := ""
			var chunk openai.ChatCompletionChunk
			numberOfChunksWithUsage := 0
			for stream.Next() {
				chunk = stream.Current()
				for _, choice := range chunk.Choices {
					if choice.Delta.Role != "" {
						role = choice.Delta.Role
					} else if choice.FinishReason == "" {
						tokens = append(tokens, choice.Delta.Content)
					}
				}
				if chunk.Usage.CompletionTokens != 0 || chunk.Usage.PromptTokens != 0 || chunk.Usage.TotalTokens != 0 {
					numberOfChunksWithUsage++
				}
				Expect(string(chunk.Object)).To(Equal(chatCompletionChunkObject))
			}

			Expect(numberOfChunksWithUsage).To(Equal(1))
			Expect(chunk.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(chunk.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(chunk.Usage.TotalTokens).To(Equal(chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens))

			msg := strings.Join(tokens, "")
			if mode == common.ModeRandom {
				// in case of random mode ensure that the returned message could be output of the random text generator
				Expect(dataset.IsValidText(msg)).To(BeTrue())
			} else {
				// in case of echo mode check that the text is returned as-is
				Expect(msg).Should(Equal(testUserMessage))
			}
			Expect(role).Should(Equal("assistant"))
		},
		func(mode string) string {
			return "mode: " + mode
		},
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeEcho),
	)

	DescribeTable("text completions streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, testModel, testUserMessage, true)
			stream := openaiclient.Completions.NewStreaming(ctx, params)
			defer func() {
				err := stream.Close()
				Expect(err).NotTo(HaveOccurred())
			}()
			tokens := []string{}
			var chunk openai.Completion
			numberOfChunksWithUsage := 0
			for stream.Next() {
				chunk = stream.Current()
				for _, choice := range chunk.Choices {
					if choice.FinishReason == "" {
						tokens = append(tokens, choice.Text)
					}
				}
				if chunk.Usage.CompletionTokens != 0 || chunk.Usage.PromptTokens != 0 || chunk.Usage.TotalTokens != 0 {
					numberOfChunksWithUsage++
				}
				Expect(string(chunk.Object)).To(Equal(textCompletionObject))
			}
			Expect(numberOfChunksWithUsage).To(Equal(1))
			Expect(chunk.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(chunk.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(chunk.Usage.TotalTokens).To(Equal(chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens))

			text := strings.Join(tokens, "")
			if mode == common.ModeRandom {
				// in case of random mode ensure that the returned message could be output of the random text generator
				Expect(dataset.IsValidText(text)).To(BeTrue())
			} else {
				// in case of echo mode check that the text is returned as-is
				Expect(text).Should(Equal(testUserMessage))
			}
		},
		func(mode string) string {
			return "mode: " + mode
		},
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeEcho),
	)

	DescribeTable("chat completions",
		func(mode string, maxTokens int, maxCompletionTokens int) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)
			numTokens := 0
			// if maxTokens and maxCompletionTokens are passsed
			// maxCompletionTokens is used
			if maxTokens != 0 {
				params.MaxTokens = param.NewOpt(int64(maxTokens))
				numTokens = maxTokens
			}
			if maxCompletionTokens != 0 {
				params.MaxCompletionTokens = param.NewOpt(int64(maxCompletionTokens))
				numTokens = maxCompletionTokens
			}
			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			if err != nil {
				var openaiError *openai.Error
				if errors.As(err, &openaiError) {
					if openaiError.StatusCode == 400 {
						errMsg, err := io.ReadAll(openaiError.Response.Body)
						Expect(err).NotTo(HaveOccurred())
						if strings.Contains(string(errMsg), invalidMaxTokensErrMsg) {
							return
						}
					}
				}
			}
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			msg := resp.Choices[0].Message.Content
			Expect(msg).ShouldNot(BeEmpty())

			if mode == common.ModeEcho {
				// in case of echo mode check that the text is returned as-is
				Expect(msg).Should(Equal(testUserMessage))
			} else {
				if numTokens > 0 {
					tokens := common.Tokenize(msg)
					Expect(int64(len(tokens))).Should(BeNumerically("<=", numTokens))
				} else {
					// in case of random mode ensure that the returned message could be output of the random text generator
					Expect(dataset.IsValidText(msg)).To(BeTrue())
				}
			}
		},
		func(mode string, maxTokens int, maxCompletionTokens int) string {
			return fmt.Sprintf("mode: %s max_tokens: %d max_completion_tokens: %d", mode, maxTokens, maxCompletionTokens)
		},
		Entry(nil, common.ModeRandom, 2, 0),
		Entry(nil, common.ModeEcho, 2, 0),
		Entry(nil, common.ModeRandom, 1000, 0),
		Entry(nil, common.ModeEcho, 1000, 0),
		Entry(nil, common.ModeRandom, 1000, 2),
		Entry(nil, common.ModeEcho, 1000, 2),
		Entry(nil, common.ModeRandom, 0, 2),
		Entry(nil, common.ModeEcho, 0, 2),
		Entry(nil, common.ModeRandom, 0, 1000),
		Entry(nil, common.ModeEcho, 0, 1000),
		Entry(nil, common.ModeRandom, 0, 0),
		Entry(nil, common.ModeEcho, 0, 0),
		Entry(nil, common.ModeRandom, -1, 0),
		Entry(nil, common.ModeEcho, -1, 0),
		Entry(nil, common.ModeRandom, 0, -1),
		Entry(nil, common.ModeEcho, 0, -1),
	)

	DescribeTable("text completions",
		// use a function so that httpClient is captured when running
		func(mode string, maxTokens int) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, testModel, testUserMessage, false)
			numTokens := 0
			if maxTokens != 0 {
				params.MaxTokens = param.NewOpt(int64(maxTokens))
				numTokens = maxTokens
			}
			resp, err := openaiclient.Completions.New(ctx, params)
			if err != nil {
				var openaiError *openai.Error
				if errors.As(err, &openaiError) {
					if openaiError.StatusCode == 400 {
						errMsg, err := io.ReadAll(openaiError.Response.Body)
						Expect(err).NotTo(HaveOccurred())
						if strings.Contains(string(errMsg), invalidMaxTokensErrMsg) {
							return
						}
					}
				}
			}
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(textCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			text := resp.Choices[0].Text
			Expect(text).ShouldNot(BeEmpty())

			if mode == common.ModeEcho {
				// in case of echo mode check that the text is returned as-is
				Expect(text).Should(Equal(testUserMessage))
			} else {
				if numTokens != 0 {
					tokens := common.Tokenize(text)
					Expect(int64(len(tokens))).Should(BeNumerically("<=", numTokens))
				} else {
					// in case of random mode ensure that the returned message could be output of the random text generator
					Expect(dataset.IsValidText(text)).To(BeTrue())
				}
			}
		},
		func(mode string, maxTokens int) string {
			return fmt.Sprintf("mode: %s max_tokens: %d", mode, maxTokens)
		},
		Entry(nil, common.ModeRandom, 2),
		Entry(nil, common.ModeEcho, 2),
		Entry(nil, common.ModeRandom, 1000),
		Entry(nil, common.ModeEcho, 1000),
		Entry(nil, common.ModeRandom, 0),
		Entry(nil, common.ModeEcho, 0),
		Entry(nil, common.ModeRandom, -1),
		Entry(nil, common.ModeEcho, -1),
	)

	Context("namespace and pod headers", func() {
		It("Should not include namespace, pod and port headers in chat completion response when env is not set", func() {
			httpResp := sendSimpleChatRequest(nil, false)

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)
			portHeader := httpResp.Header.Get(portHeader)

			Expect(namespaceHeader).To(BeEmpty(), "Expected namespace header not to be present")
			Expect(podHeader).To(BeEmpty(), "Expected pod header not to be present")
			Expect(portHeader).To(BeEmpty(), "Expected port header not to be present")
		})

		It("Should include namespace, pod and port headers in chat completion response", func() {
			testNamespace := "test-namespace"
			testPod := "test-pod"
			envs := map[string]string{
				podNameEnv: testPod,
				podNsEnv:   testNamespace,
			}
			httpResp := sendSimpleChatRequest(envs, false)

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)
			portHeader := httpResp.Header.Get(portHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
			Expect(portHeader).To(Equal("8000"), "Expected port header to be present")
		})

		It("Should include namespace, pod and port headers in chat completion streaming response", func() {
			testNamespace := "stream-test-namespace"
			testPod := "stream-test-pod"
			envs := map[string]string{
				podNameEnv: testPod,
				podNsEnv:   testNamespace,
			}
			httpResp := sendSimpleChatRequest(envs, true)

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)
			portHeader := httpResp.Header.Get(portHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
			Expect(portHeader).To(Equal("8000"), "Expected port header to be present")
		})

		It("Should not include namespace, pod and port headers in chat completion streaming response when env is not set", func() {
			httpResp := sendSimpleChatRequest(nil, true)

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)
			portHeader := httpResp.Header.Get(portHeader)

			Expect(namespaceHeader).To(BeEmpty(), "Expected namespace header not to be present")
			Expect(podHeader).To(BeEmpty(), "Expected pod header not to be present")
			Expect(portHeader).To(BeEmpty(), "Expected port header not to be present")
		})

		It("Should include namespace, pod and port headers in completion response", func() {
			ctx := context.TODO()

			testNamespace := "test-namespace"
			testPod := "test-pod"
			envs := map[string]string{
				podNameEnv: testPod,
				podNsEnv:   testNamespace,
			}
			client, err := startServerWithEnv(ctx, common.ModeRandom, envs)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, testModel, testUserMessage, false)
			var httpResp *http.Response
			resp, err := openaiclient.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)
			portHeader := httpResp.Header.Get(portHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
			Expect(portHeader).To(Equal("8000"), "Expected port header to be present")
		})

		It("Should include namespace, pod and port headers in completion streaming response", func() {
			ctx := context.TODO()

			testNamespace := "stream-test-namespace"
			testPod := "stream-test-pod"
			envs := map[string]string{
				podNameEnv: testPod,
				podNsEnv:   testNamespace,
			}
			client, err := startServerWithEnv(ctx, common.ModeRandom, envs)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, testModel, testUserMessage, true)
			var httpResp *http.Response
			resp, err := openaiclient.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)
			portHeader := httpResp.Header.Get(portHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
			Expect(portHeader).To(Equal("8000"), "Expected port header to be present")
		})
	})

	Context("logprobs functionality", func() {
		DescribeTable("streaming chat completions with logprobs",
			func(mode string, logprobs bool, topLogprobs int) {
				ctx := context.TODO()
				client, err := startServer(ctx, mode)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, true)
				params.Logprobs = param.NewOpt(logprobs)
				if logprobs && topLogprobs > 0 {
					params.TopLogprobs = param.NewOpt(int64(topLogprobs))
				}

				stream := openaiclient.Chat.Completions.NewStreaming(ctx, params)
				defer func() {
					err := stream.Close()
					Expect(err).NotTo(HaveOccurred())
				}()

				tokens := []string{}
				chunksWithLogprobs := 0

				for stream.Next() {
					chunk := stream.Current()
					for _, choice := range chunk.Choices {
						if choice.FinishReason == "" && choice.Delta.Content != "" {
							tokens = append(tokens, choice.Delta.Content)

							// Check logprobs in streaming chunks
							if logprobs && len(choice.Logprobs.Content) > 0 {
								chunksWithLogprobs++
								logprobContent := choice.Logprobs.Content[0]
								Expect(logprobContent.Token).To(Equal(choice.Delta.Content))
								Expect(logprobContent.Logprob).To(BeNumerically("<=", 0))

								if topLogprobs > 0 {
									Expect(logprobContent.TopLogprobs).To(HaveLen(topLogprobs))
									Expect(logprobContent.TopLogprobs[0].Token).To(Equal(choice.Delta.Content))
								}
							}
						}
					}
				}

				msg := strings.Join(tokens, "")
				if mode == common.ModeRandom {
					Expect(dataset.IsValidText(msg)).To(BeTrue())
				} else {
					Expect(msg).Should(Equal(testUserMessage))
				}

				// Verify logprobs behaviour
				if logprobs {
					Expect(chunksWithLogprobs).To(BeNumerically(">", 0), "Should have chunks with logprobs")
				} else {
					Expect(chunksWithLogprobs).To(Equal(0), "Should not have chunks with logprobs when not requested")
				}
			},
			func(mode string, logprobs bool, topLogprobs int) string {
				return fmt.Sprintf("mode: %s logprobs: %t top_logprobs: %d", mode, logprobs, topLogprobs)
			},
			Entry(nil, common.ModeEcho, true, 0),  // logprobs=true, default top_logprobs
			Entry(nil, common.ModeEcho, true, 2),  // logprobs=true, top_logprobs=2
			Entry(nil, common.ModeEcho, false, 0), // logprobs=false
		)

		DescribeTable("streaming text completions with logprobs",
			func(mode string, logprobsCount int) {
				ctx := context.TODO()
				client, err := startServer(ctx, mode)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClentAndCompletionParams(client, testModel, testUserMessage, true)
				if logprobsCount > 0 {
					params.Logprobs = param.NewOpt(int64(logprobsCount))
				}

				stream := openaiclient.Completions.NewStreaming(ctx, params)
				defer func() {
					err := stream.Close()
					Expect(err).NotTo(HaveOccurred())
				}()

				tokens := []string{}
				chunksWithLogprobs := 0

				for stream.Next() {
					chunk := stream.Current()
					for _, choice := range chunk.Choices {
						if choice.FinishReason == "" && choice.Text != "" {
							tokens = append(tokens, choice.Text)

							// Check logprobs in streaming chunks
							if logprobsCount > 0 && len(choice.Logprobs.Tokens) > 0 {
								chunksWithLogprobs++
								Expect(choice.Logprobs.Tokens[0]).To(Equal(choice.Text))
								Expect(choice.Logprobs.TokenLogprobs[0]).To(BeNumerically("<=", 0))
								Expect(choice.Logprobs.TopLogprobs[0]).To(HaveLen(logprobsCount))
								Expect(choice.Logprobs.TopLogprobs[0]).To(HaveKey(choice.Text))
							}
						}
					}
				}

				text := strings.Join(tokens, "")
				if mode == common.ModeRandom {
					Expect(dataset.IsValidText(text)).To(BeTrue())
				} else {
					Expect(text).Should(Equal(testUserMessage))
				}

				// Verify logprobs behaviour
				if logprobsCount > 0 {
					Expect(chunksWithLogprobs).To(BeNumerically(">", 0), "Should have chunks with logprobs")
				} else {
					Expect(chunksWithLogprobs).To(Equal(0), "Should not have chunks with logprobs when not requested")
				}
			},
			func(mode string, logprobsCount int) string {
				return fmt.Sprintf("mode: %s logprobs: %d", mode, logprobsCount)
			},
			Entry(nil, common.ModeEcho, 0), // No logprobs
			Entry(nil, common.ModeEcho, 2), // logprobs=2
		)

		DescribeTable("non-streaming completions with logprobs",
			func(isChat bool, mode string, logprobsParam interface{}) {
				ctx := context.TODO()
				client, err := startServer(ctx, mode)
				Expect(err).NotTo(HaveOccurred())

				var resp interface{}

				if isChat {
					openaiclient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)
					if logprobsParam != nil {
						if logprobs, ok := logprobsParam.(bool); ok && logprobs {
							params.Logprobs = param.NewOpt(true)
							params.TopLogprobs = param.NewOpt(int64(2))
						}
					}
					resp, err = openaiclient.Chat.Completions.New(ctx, params)
				} else {
					openaiclient, params := getOpenAIClentAndCompletionParams(client, testModel, testUserMessage, false)
					if logprobsParam != nil {
						if logprobsCount, ok := logprobsParam.(int); ok && logprobsCount > 0 {
							params.Logprobs = param.NewOpt(int64(logprobsCount))
						}
					}
					resp, err = openaiclient.Completions.New(ctx, params)
				}

				Expect(err).NotTo(HaveOccurred())

				// Verify logprobs in non-streaming response
				if isChat {
					chatResp := resp.(*openai.ChatCompletion)
					Expect(chatResp.Choices).ShouldNot(BeEmpty())

					if logprobsParam != nil {
						// When logprobs requested, Content should be populated
						Expect(chatResp.Choices[0].Logprobs.Content).NotTo(BeEmpty())

						tokens := common.Tokenize(chatResp.Choices[0].Message.Content)
						Expect(chatResp.Choices[0].Logprobs.Content).To(HaveLen(len(tokens)))
					} else {
						// When logprobs not requested, Content should be empty/nil
						// Note: SDK uses nullable types, so we check the Content field
						Expect(chatResp.Choices[0].Logprobs.Content).To(BeNil())
					}
				} else {
					textResp := resp.(*openai.Completion)
					Expect(textResp.Choices).ShouldNot(BeEmpty())

					if logprobsParam != nil {
						// When logprobs requested, fields should be populated
						Expect(textResp.Choices[0].Logprobs.Tokens).NotTo(BeNil())

						tokens := common.Tokenize(textResp.Choices[0].Text)
						Expect(textResp.Choices[0].Logprobs.Tokens).To(HaveLen(len(tokens)))
					} else {
						// When logprobs not requested, all fields should be empty/nil
						Expect(textResp.Choices[0].Logprobs.Tokens).To(BeNil())
					}
				}
			},
			func(isChat bool, mode string, logprobsParam interface{}) string {
				apiType := "text"
				if isChat {
					apiType = "chat"
				}
				return fmt.Sprintf("%s mode: %s logprobs: %v", apiType, mode, logprobsParam)
			},
			Entry(nil, true, common.ModeEcho, true), // Chat with logprobs
			Entry(nil, true, common.ModeEcho, nil),  // Chat without logprobs
			Entry(nil, false, common.ModeEcho, 2),   // Text with logprobs=2
			Entry(nil, false, common.ModeEcho, nil), // Text without logprobs
		)

	})

	Context("max-model-len context window validation", func() {
		It("Should reject requests exceeding context window", func() {
			ctx := context.TODO()
			// Start server with max-model-len=10
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom, "--max-model-len", "10"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			// Test with raw HTTP to verify the error response format
			reqBody := `{
				"messages": [{"role": "user", "content": "This is a test message"}],
				"model": "testmodel",
				"max_tokens": 8
			}`

			resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			Expect(resp.StatusCode).To(Equal(400))
			Expect(string(body)).To(ContainSubstring("This model's maximum context length is 10 tokens"))
			Expect(string(body)).To(ContainSubstring("However, you requested 13 tokens"))
			Expect(string(body)).To(ContainSubstring("5 in the messages, 8 in the completion"))
			Expect(string(body)).To(ContainSubstring("BadRequestError"))

			// Also test with OpenAI client to ensure it gets an error
			openaiclient, params := getOpenAIClientAndChatParams(client, testModel, "This is a test message", false)
			params.MaxTokens = openai.Int(8)

			_, err = openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).To(HaveOccurred())
			var apiErr *openai.Error
			Expect(errors.As(err, &apiErr)).To(BeTrue())
			Expect(apiErr.StatusCode).To(Equal(400))
		})

		It("Should accept requests within context window", func() {
			ctx := context.TODO()
			// Start server with max-model-len=50
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeEcho, "--max-model-len", "50"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, testModel, "Hello", false)
			params.MaxTokens = openai.Int(5)

			// Send a request within the context window
			resp, err := openaiclient.Chat.Completions.New(ctx, params)

			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).To(HaveLen(1))
			Expect(resp.Model).To(Equal(testModel))
		})

		It("Should handle text completion requests exceeding context window", func() {
			ctx := context.TODO()
			// Start server with max-model-len=10
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom, "--max-model-len", "10"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			// Test with raw HTTP for text completion
			reqBody := `{
				"prompt": "This is a long test prompt with many words",
				"model": "testmodel",
				"max_tokens": 5
			}`

			resp, err := client.Post("http://localhost/v1/completions", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			Expect(resp.StatusCode).To(Equal(400))
			Expect(string(body)).To(ContainSubstring("This model's maximum context length is 10 tokens"))
			Expect(string(body)).To(ContainSubstring("BadRequestError"))
		})
	})

	// Add in simulator_test.go, alongside other Context blocks

	Context("cache threshold header", func() {
		It("Should return cache_threshold finish reason when header is set", func() {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			// Create request with cache threshold header
			reqBody := `{
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "` + testModel + `",
            "max_tokens": 5
        }`

			req, err := http.NewRequest("POST", "http://localhost/v1/chat/completions", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Cache-Threshold", "true") // Your new header

			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var chatResp map[string]interface{}
			err = json.Unmarshal(body, &chatResp)
			Expect(err).NotTo(HaveOccurred())

			// Verify finish_reason is "cache_threshold"
			choices := chatResp["choices"].([]interface{})
			Expect(choices).To(HaveLen(1))
			firstChoice := choices[0].(map[string]interface{})
			Expect(firstChoice["finish_reason"]).To(Equal("cache_threshold"))
		})

		It("Should return normal finish reason when header is not set", func() {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			// Create request WITHOUT cache threshold header
			reqBody := `{
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "` + testModel + `",
            "max_tokens": 5
        }`

			req, err := http.NewRequest("POST", "http://localhost/v1/chat/completions", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			// No X-Cache-Threshold header

			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var chatResp map[string]interface{}
			err = json.Unmarshal(body, &chatResp)
			Expect(err).NotTo(HaveOccurred())

			// Verify finish_reason is NOT "cache_threshold"
			choices := chatResp["choices"].([]interface{})
			Expect(choices).To(HaveLen(1))
			firstChoice := choices[0].(map[string]interface{})
			Expect(firstChoice["finish_reason"]).To(Or(Equal("stop"), Equal("length")))
		})
	})
})
