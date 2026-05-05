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
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/valyala/fasthttp"
)

const prompt1 = "What is the weather like in New York today?"
const prompt2 = "I hear it's very cold."

var _ = Describe("Simulator", func() {

	DescribeTable("chat completions streaming",
		func(model string, mode string) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", mode}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, model, testUserMessage, true)
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
				Expect(string(chunk.Object)).To(Equal(openaiserverapi.ChatCompletionChunkObject))
			}

			Expect(numberOfChunksWithUsage).To(Equal(1))
			Expect(err).NotTo(HaveOccurred())
			Expect(chunk.Usage.PromptTokens).To(Equal(userMsgChatTokens))
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
		func(model string, mode string) string {
			return "model: " + model + " mode: " + mode
		},
		Entry(nil, common.TestModelName, common.ModeRandom),
		Entry(nil, common.TestModelName, common.ModeEcho),
		Entry(nil, common.QwenModelName, common.ModeEcho),
	)

	DescribeTable("text completions streaming",
		func(model string, mode string) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", mode}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, model, testUserMessage, true)
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
				Expect(string(chunk.Object)).To(Equal(openaiserverapi.TextCompletionObject))
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
		func(model string, mode string) string {
			return "model: " + model + " mode: " + mode
		},
		Entry(nil, common.TestModelName, common.ModeRandom),
		Entry(nil, common.TestModelName, common.ModeEcho),
		Entry(nil, common.QwenModelName, common.ModeEcho),
		Entry(nil, common.QwenModelName, common.ModeRandom),
	)

	DescribeTable("chat completions",
		func(model string, mode string, maxTokens int, maxCompletionTokens int) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", mode}
			server, _, client, err := startServerHandle(ctx, mode, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, model, testUserMessage, false)
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
						if strings.Contains(string(errMsg), common.InvalidMaxTokensErrMsg) {
							return
						}
					}
				}
			}
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(openaiserverapi.ChatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(userMsgChatTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			msg := resp.Choices[0].Message.Content
			Expect(msg).ShouldNot(BeEmpty())

			if mode == common.ModeEcho {
				// in case of echo mode check that the text is returned as-is
				Expect(msg).Should(Equal(testUserMessage))
			} else {
				if numTokens > 0 {
					_, tokens, err := server.Context.Tokenizer.RenderText(msg)
					Expect(err).NotTo(HaveOccurred())
					Expect(int64(len(tokens))).Should(BeNumerically("<=", numTokens))
				} else {
					// in case of random mode ensure that the returned message could be output of the random text generator
					Expect(dataset.IsValidText(msg)).To(BeTrue())
				}
			}
		},
		func(model string, mode string, maxTokens int, maxCompletionTokens int) string {
			return fmt.Sprintf("model: %s mode: %s max_tokens: %d max_completion_tokens: %d",
				model, mode, maxTokens, maxCompletionTokens)
		},
		Entry(nil, common.TestModelName, common.ModeRandom, 2, 0),
		Entry(nil, common.TestModelName, common.ModeEcho, 2, 0),
		Entry(nil, common.TestModelName, common.ModeRandom, 1000, 0),
		Entry(nil, common.TestModelName, common.ModeEcho, 1000, 0),
		Entry(nil, common.TestModelName, common.ModeRandom, 1000, 2),
		Entry(nil, common.TestModelName, common.ModeEcho, 1000, 2),
		Entry(nil, common.TestModelName, common.ModeRandom, 0, 2),
		Entry(nil, common.TestModelName, common.ModeEcho, 0, 2),
		Entry(nil, common.TestModelName, common.ModeRandom, 0, 1000),
		Entry(nil, common.TestModelName, common.ModeEcho, 0, 1000),
		Entry(nil, common.TestModelName, common.ModeRandom, 0, 0),
		Entry(nil, common.TestModelName, common.ModeEcho, 0, 0),
		Entry(nil, common.TestModelName, common.ModeRandom, -1, 0),
		Entry(nil, common.TestModelName, common.ModeEcho, -1, 0),
		Entry(nil, common.TestModelName, common.ModeRandom, 0, -1),
		Entry(nil, common.QwenModelName, common.ModeEcho, 1000, 0),
		Entry(nil, common.QwenModelName, common.ModeRandom, 1000, 0),
	)

	DescribeTable("text completions",
		// use a function so that httpClient is captured when running
		func(model string, mode string, maxTokens int) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", mode}
			server, _, client, err := startServerHandle(ctx, mode, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, model, testUserMessage, false)
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
						if strings.Contains(string(errMsg), common.InvalidMaxTokensErrMsg) {
							return
						}
					}
				}
			}
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(openaiserverapi.TextCompletionObject))

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
					_, tokens, err := server.Context.Tokenizer.RenderText(text)
					Expect(err).NotTo(HaveOccurred())
					Expect(int64(len(tokens))).Should(BeNumerically("<=", numTokens))
				} else {
					// in case of random mode ensure that the returned message could be output of the random text generator
					Expect(dataset.IsValidText(text)).To(BeTrue())
				}
			}
		},
		func(model string, mode string, maxTokens int) string {
			return fmt.Sprintf("model: %s mode: %s max_tokens: %d", model, mode, maxTokens)
		},
		Entry(nil, common.TestModelName, common.ModeRandom, 2),
		Entry(nil, common.TestModelName, common.ModeEcho, 2),
		Entry(nil, common.TestModelName, common.ModeRandom, 1000),
		Entry(nil, common.TestModelName, common.ModeEcho, 1000),
		Entry(nil, common.TestModelName, common.ModeRandom, 0),
		Entry(nil, common.TestModelName, common.ModeEcho, 0),
		Entry(nil, common.TestModelName, common.ModeRandom, -1),
		Entry(nil, common.TestModelName, common.ModeEcho, -1),
		Entry(nil, common.QwenModelName, common.ModeEcho, 1000),
		Entry(nil, common.QwenModelName, common.ModeRandom, 1000),
	)

	DescribeTable("mm encoder only",
		func(model string, mode string, maxTokens int) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mm-encoder-only", "--mm-processor-kwargs", "args",
				"--ec-transfer-config", "cfg", "--enforce-eager", "--no-enable-prefix-caching"}
			server, _, client, err := startServerHandle(ctx, mode, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client),
				option.WithMaxRetries(0))

			params := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage(
						[]openai.ChatCompletionContentPartUnionParam{
							{
								OfImageURL: &openai.ChatCompletionContentPartImageParam{
									ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
										URL: "https://github.com/llm-d/llm-d-inference-sim/blob/main/test/images/llmd.png?raw=true",
									},
								},
							},
						},
					),
				},
				Model:     model,
				MaxTokens: param.NewOpt(int64(maxTokens)),
			}

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(openaiserverapi.ChatCompletionObject))

			Expect(resp.Usage.CompletionTokens).To(BeNumerically("<=", maxTokens))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			msg := resp.Choices[0].Message.Content
			Expect(msg).ShouldNot(BeEmpty())
			for _, t := range msg {
				Expect(t).To(Equal('!'))
			}
			_, tokens, err := server.Context.Tokenizer.RenderText(msg)
			Expect(err).NotTo(HaveOccurred())
			Expect(int64(len(tokens))).Should(BeNumerically("<=", maxTokens))
		},
		func(model string, mode string, maxTokens int) string {
			return fmt.Sprintf("model: %s max_tokens: %d",
				model, maxTokens)
		},
		Entry(nil, common.TestModelName, common.ModeEcho, 1),
		Entry(nil, common.QwenModelName, common.ModeRandom, 1),
		Entry(nil, common.TestModelName, common.ModeRandom, 10),
		Entry(nil, common.QwenModelName, common.ModeEcho, 10),
	)

	It("echo mode with structured content blocks", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client),
			option.WithMaxRetries(0))

		params := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(
					[]openai.ChatCompletionContentPartUnionParam{
						openai.TextContentPart("Describe this"),
						openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
							URL: "https://example.com/img.png",
						}),
					},
				),
			},
			Model: common.TestModelName,
		}

		resp, err := openaiclient.Chat.Completions.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Choices).ShouldNot(BeEmpty())

		msg := resp.Choices[0].Message.Content
		Expect(msg).To(Equal("Describe this\nimage: https://example.com/img.png"))
	})

	It("echo mode with structured content blocks streaming", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client),
			option.WithMaxRetries(0))

		params := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(
					[]openai.ChatCompletionContentPartUnionParam{
						openai.TextContentPart("Describe this"),
						openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
							URL: "https://example.com/img.png",
						}),
					},
				),
			},
			Model:         common.TestModelName,
			StreamOptions: openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)},
		}

		stream := openaiclient.Chat.Completions.NewStreaming(ctx, params)
		defer func() {
			err := stream.Close()
			Expect(err).NotTo(HaveOccurred())
		}()

		var tokens []string
		for stream.Next() {
			chunk := stream.Current()
			for _, choice := range chunk.Choices {
				if choice.Delta.Content != "" {
					tokens = append(tokens, choice.Delta.Content)
				}
			}
		}
		Expect(stream.Err()).NotTo(HaveOccurred())

		msg := strings.Join(tokens, "")
		Expect(msg).To(Equal("Describe this\nimage: https://example.com/img.png"))
	})

	Context("namespace and pod headers", func() {
		It("Should not include namespace, pod and port headers in chat completion response when env is not set", func() {
			httpResp := sendSimpleChatRequest(nil, false)

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(communication.NamespaceHeader)
			podHeader := httpResp.Header.Get(communication.PodHeader)
			portHeader := httpResp.Header.Get(communication.PortHeader)

			Expect(namespaceHeader).To(BeEmpty(), "Expected namespace header not to be present")
			Expect(podHeader).To(BeEmpty(), "Expected pod header not to be present")
			Expect(portHeader).To(BeEmpty(), "Expected port header not to be present")
		})

		It("Should include namespace, pod and port headers in chat completion response", func() {
			testNamespace := "test-namespace"
			testPod := "test-pod"
			envs := map[string]string{
				common.PodNameEnv: testPod,
				common.PodNsEnv:   testNamespace,
			}
			httpResp := sendSimpleChatRequest(envs, false)

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(communication.NamespaceHeader)
			podHeader := httpResp.Header.Get(communication.PodHeader)
			portHeader := httpResp.Header.Get(communication.PortHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
			Expect(portHeader).To(Equal("8000"), "Expected port header to be present")
		})

		It("Should include namespace, pod and port headers in chat completion streaming response", func() {
			testNamespace := "stream-test-namespace"
			testPod := "stream-test-pod"
			envs := map[string]string{
				common.PodNameEnv: testPod,
				common.PodNsEnv:   testNamespace,
			}
			httpResp := sendSimpleChatRequest(envs, true)

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(communication.NamespaceHeader)
			podHeader := httpResp.Header.Get(communication.PodHeader)
			portHeader := httpResp.Header.Get(communication.PortHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
			Expect(portHeader).To(Equal("8000"), "Expected port header to be present")
		})

		It("Should not include namespace, pod and port headers in chat completion streaming response when env is not set", func() {
			httpResp := sendSimpleChatRequest(nil, true)

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(communication.NamespaceHeader)
			podHeader := httpResp.Header.Get(communication.PodHeader)
			portHeader := httpResp.Header.Get(communication.PortHeader)

			Expect(namespaceHeader).To(BeEmpty(), "Expected namespace header not to be present")
			Expect(podHeader).To(BeEmpty(), "Expected pod header not to be present")
			Expect(portHeader).To(BeEmpty(), "Expected port header not to be present")
		})

		It("Should include namespace, pod and port headers in completion response", func() {
			ctx := context.TODO()

			testNamespace := "test-namespace"
			testPod := "test-pod"
			envs := map[string]string{
				common.PodNameEnv: testPod,
				common.PodNsEnv:   testNamespace,
			}
			client, err := startServerWithEnv(ctx, common.ModeRandom, envs)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, common.TestModelName, testUserMessage, false)
			var httpResp *http.Response
			resp, err := openaiclient.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(communication.NamespaceHeader)
			podHeader := httpResp.Header.Get(communication.PodHeader)
			portHeader := httpResp.Header.Get(communication.PortHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
			Expect(portHeader).To(Equal("8000"), "Expected port header to be present")
		})

		It("Should include namespace, pod and port headers in completion streaming response", func() {
			ctx := context.TODO()

			testNamespace := "stream-test-namespace"
			testPod := "stream-test-pod"
			envs := map[string]string{
				common.PodNameEnv: testPod,
				common.PodNsEnv:   testNamespace,
			}
			client, err := startServerWithEnv(ctx, common.ModeRandom, envs)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, common.TestModelName, testUserMessage, true)
			var httpResp *http.Response
			resp, err := openaiclient.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace, pod and port headers
			namespaceHeader := httpResp.Header.Get(communication.NamespaceHeader)
			podHeader := httpResp.Header.Get(communication.PodHeader)
			portHeader := httpResp.Header.Get(communication.PortHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
			Expect(portHeader).To(Equal("8000"), "Expected port header to be present")
		})

		It("Should not include namespace, pod and port headers in embeddings response when env is not set", func() {
			httpResp := sendSimpleEmbeddingsRequest(nil)

			namespaceHeader := httpResp.Header.Get(communication.NamespaceHeader)
			podHeader := httpResp.Header.Get(communication.PodHeader)
			portHeader := httpResp.Header.Get(communication.PortHeader)

			Expect(namespaceHeader).To(BeEmpty(), "Expected namespace header not to be present")
			Expect(podHeader).To(BeEmpty(), "Expected pod header not to be present")
			Expect(portHeader).To(BeEmpty(), "Expected port header not to be present")
		})

		It("Should include namespace, pod and port headers in embeddings response", func() {
			testNamespace := "emb-test-namespace"
			testPod := "emb-test-pod"
			envs := map[string]string{
				common.PodNameEnv: testPod,
				common.PodNsEnv:   testNamespace,
			}
			httpResp := sendSimpleEmbeddingsRequest(envs)

			namespaceHeader := httpResp.Header.Get(communication.NamespaceHeader)
			podHeader := httpResp.Header.Get(communication.PodHeader)
			portHeader := httpResp.Header.Get(communication.PortHeader)

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

				openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, true)
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

				openaiclient, params := getOpenAIClentAndCompletionParams(client, common.TestModelName, testUserMessage, true)
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
				server, _, client, err := startServerHandle(ctx, mode, nil, nil)
				Expect(err).NotTo(HaveOccurred())

				var resp interface{}

				if isChat {
					openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, false)
					if logprobsParam != nil {
						if logprobs, ok := logprobsParam.(bool); ok && logprobs {
							params.Logprobs = param.NewOpt(true)
							params.TopLogprobs = param.NewOpt(int64(2))
						}
					}
					resp, err = openaiclient.Chat.Completions.New(ctx, params)
				} else {
					openaiclient, params := getOpenAIClentAndCompletionParams(client, common.TestModelName, testUserMessage, false)
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

						_, tokens, err := server.Context.Tokenizer.RenderText(chatResp.Choices[0].Message.Content)
						Expect(err).NotTo(HaveOccurred())
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

						_, tokens, err := server.Context.Tokenizer.RenderText(textResp.Choices[0].Text)
						Expect(err).NotTo(HaveOccurred())
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
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--max-model-len", "10"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			maxTokens := 8
			prompt := "This is a test message"
			promptChatTokens := getChatPromptTokensCountForTestModel(prompt)

			// Test with raw HTTP to verify the error response format
			reqBody := fmt.Sprintf(`{
				"messages": [{"role": "user", "content": "%s"}],
				"model": "%s",
				"max_tokens": %d
			}`, prompt, common.TestModelName, maxTokens)

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
			Expect(string(body)).To(ContainSubstring(fmt.Sprintf("However, you requested %d tokens", promptChatTokens+int64(maxTokens))))
			Expect(string(body)).To(ContainSubstring(fmt.Sprintf("%d in the messages, %d in the completion", promptChatTokens, maxTokens)))
			Expect(string(body)).To(ContainSubstring("BadRequestError"))

			// Also test with OpenAI client to ensure it gets an error
			openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, prompt, false)
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
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho, "--max-model-len", "50"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, "Hello", false)
			params.MaxTokens = openai.Int(5)

			// Send a request within the context window
			resp, err := openaiclient.Chat.Completions.New(ctx, params)

			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).To(HaveLen(1))
			Expect(resp.Model).To(Equal(common.TestModelName))
		})

		It("Should handle text completion requests exceeding context window", func() {
			ctx := context.TODO()
			// Start server with max-model-len=10
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--max-model-len", "10"}
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

	Context("cache threshold finish reason header", func() {
		testCacheThresholdFinishReasonHeader := func(setHeader bool, expectedFinishReasons []string) {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			reqBody := `{
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "` + common.TestModelName + `",
            "max_tokens": 5
        }`

			req, err := http.NewRequest("POST", "http://localhost/v1/chat/completions", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			if setHeader {
				req.Header.Set(communication.CacheThresholdFinishReasonHeader, "true")
			}

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

			choices := chatResp["choices"].([]interface{})
			Expect(choices).To(HaveLen(1))
			firstChoice := choices[0].(map[string]interface{})
			Expect(firstChoice["finish_reason"]).To(BeElementOf(expectedFinishReasons))

		}

		It("Should return cache_threshold finish reason when header is set", func() {
			testCacheThresholdFinishReasonHeader(true, []string{common.CacheThresholdFinishReason})
		})

		It("Should return normal finish reason when header is not set", func() {
			testCacheThresholdFinishReasonHeader(false, []string{common.StopFinishReason, common.LengthFinishReason})
		})
	})

	Context("X-Return-Error header", func() {
		It("Should return the specified HTTP error code", func() {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			reqBody := `{
				"messages": [{"role": "user", "content": "Hello"}],
				"model": "` + common.TestModelName + `",
				"max_tokens": 5
			}`

			req, err := http.NewRequest("POST", "http://localhost/v1/chat/completions", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(communication.XReturnErrorHeader, "422")

			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(422))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var errResp openaiserverapi.ErrorResponse
			err = json.Unmarshal(body, &errResp)
			Expect(err).NotTo(HaveOccurred())
			Expect(errResp.Error.Code).To(Equal(422))
			Expect(errResp.Error.Message).To(ContainSubstring("X-Return-Error"))
		})

		It("Should return 400 when header value is not a valid integer", func() {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			reqBody := `{
				"messages": [{"role": "user", "content": "Hello"}],
				"model": "` + common.TestModelName + `",
				"max_tokens": 5
			}`

			req, err := http.NewRequest("POST", "http://localhost/v1/chat/completions", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(communication.XReturnErrorHeader, "abc")

			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(400))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var errResp openaiserverapi.ErrorResponse
			err = json.Unmarshal(body, &errResp)
			Expect(err).NotTo(HaveOccurred())
			Expect(errResp.Error.Code).To(Equal(400))
			Expect(errResp.Error.Message).To(ContainSubstring("Invalid X-Return-Error"))
		})

	})

	Context("cache hit threshold", func() {
		type completionRequestParams struct {
			Prompt            string   `json:"prompt"`
			Model             string   `json:"model"`
			MaxTokens         int      `json:"max_tokens"`
			CacheHitThreshold *float64 `json:"cache_hit_threshold,omitempty"`
			Stream            bool     `json:"stream,omitempty"`
		}

		createCompletionRequest := func(params completionRequestParams) *http.Request {
			reqBodyBytes, err := json.Marshal(params)
			Expect(err).NotTo(HaveOccurred())

			req, err := http.NewRequest("POST", "http://localhost/v1/completions", strings.NewReader(string(reqBodyBytes)))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")

			return req
		}

		setupKVCacheServer := func(enableKVCache bool, globalThreshold *float64, model string) *http.Client {
			ctx := context.TODO()

			args := []string{"cmd", "--model", model, "--mode", common.ModeEcho}

			if enableKVCache {
				args = append(args, "--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8")
			}
			if globalThreshold != nil {
				args = append(args, "--global-cache-hit-threshold", fmt.Sprintf("%f", *globalThreshold))
			}
			client, err := startServerWithArgsAndEnv(ctx, common.ModeEcho, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			return client
		}

		populateCache := func(client *http.Client) {
			req1 := createCompletionRequest(completionRequestParams{
				Prompt:    prompt1,
				Model:     common.QwenModelName,
				MaxTokens: 5,
			})
			resp1, err := client.Do(req1)
			Expect(err).NotTo(HaveOccurred())
			err = resp1.Body.Close()
			Expect(err).NotTo(HaveOccurred())
		}

		testCacheHitThreshold := func(secondPrompt string, cacheHitThreshold float64, expectCacheThresholdFinishReason bool, checkImmediateResponse bool) {
			client := setupKVCacheServer(true, nil, common.QwenModelName)

			populateCache(client)

			// Second request: test cache hit threshold
			req2 := createCompletionRequest(completionRequestParams{
				Prompt:            secondPrompt,
				Model:             common.QwenModelName,
				MaxTokens:         5,
				CacheHitThreshold: &cacheHitThreshold,
			})
			var startTime time.Time
			if checkImmediateResponse {
				startTime = time.Now()
			}
			resp2, err := client.Do(req2)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp2.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			if checkImmediateResponse {
				elapsed := time.Since(startTime)
				Expect(elapsed).To(BeNumerically("<", 100*time.Millisecond), "Response should be immediate")
			}

			Expect(resp2.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())

			var completionResp map[string]interface{}
			err = json.Unmarshal(body, &completionResp)
			Expect(err).NotTo(HaveOccurred())

			choices := completionResp["choices"].([]interface{})
			Expect(choices).To(HaveLen(1))
			firstChoice := choices[0].(map[string]interface{})

			if expectCacheThresholdFinishReason {
				Expect(firstChoice["finish_reason"]).To(Equal(common.CacheThresholdFinishReason))

				// Verify response is empty (no tokens generated)
				text := firstChoice["text"].(string)
				Expect(text).To(BeEmpty())

				// Verify usage data
				usage := completionResp["usage"].(map[string]interface{})
				Expect(usage["completion_tokens"]).To(Equal(float64(0)))
				Expect(usage["prompt_tokens"]).To(BeNumerically(">", 0))
			} else {
				// Should have normal finish reason, not cache_threshold
				finishReason := firstChoice["finish_reason"].(string)
				Expect(finishReason).To(Equal(common.LengthFinishReason))

				// Should have generated tokens
				text := firstChoice["text"].(string)
				Expect(text).NotTo(BeEmpty())
			}
		}

		It("Should return cache_threshold finish reason when hit rate is below threshold", func() {
			testCacheHitThreshold(prompt2, 0.9, true, true)
		})

		It("Should proceed with normal processing when hit rate is at or above threshold", func() {
			testCacheHitThreshold(prompt1+prompt2, 0.3, false, false)
		})

		It("Should return cache_threshold finish reason in streaming response when threshold not met", func() {
			globalCacheHitThreshold := 0.9
			client := setupKVCacheServer(true, &globalCacheHitThreshold, common.QwenModelName)

			populateCache(client)

			req2 := createCompletionRequest(completionRequestParams{
				Prompt:    prompt2,
				Model:     common.QwenModelName,
				MaxTokens: 5,
				Stream:    true,
			})
			startTime := time.Now()
			resp2, err := client.Do(req2)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp2.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			elapsed := time.Since(startTime)
			Expect(elapsed).To(BeNumerically("<", 100*time.Millisecond))

			Expect(resp2.StatusCode).To(Equal(http.StatusOK))

			// Read streaming response
			reader := bufio.NewReader(resp2.Body)
			chunksWithFinishReason := 0
			hasCacheThreshold := false

			for {
				line, err := reader.ReadString('\n')
				if err == io.EOF {
					break
				}
				Expect(err).NotTo(HaveOccurred())

				if strings.HasPrefix(line, "data: ") {
					data := strings.TrimPrefix(line, "data: ")
					if strings.TrimSpace(data) == "[DONE]" {
						break
					}

					var chunk map[string]interface{}
					err = json.Unmarshal([]byte(data), &chunk)
					if err != nil {
						continue
					}

					choices, ok := chunk["choices"].([]interface{})
					if !ok || len(choices) == 0 {
						continue
					}

					firstChoice, ok := choices[0].(map[string]interface{})
					if !ok {
						continue
					}

					finishReason, ok := firstChoice["finish_reason"].(string)
					if ok && finishReason != "" {
						chunksWithFinishReason++
						if finishReason == common.CacheThresholdFinishReason {
							hasCacheThreshold = true
						}
					}
				}
			}

			Expect(hasCacheThreshold).To(BeTrue(), "Should have cache_threshold finish reason in streaming response")
			Expect(chunksWithFinishReason).To(BeNumerically(">", 0), "Should have at least one chunk with finish reason")
		})

		It("Should use global cache hit threshold when request doesn't specify cache_hit_threshold", func() {
			globalThreshold := 0.9
			client := setupKVCacheServer(true, &globalThreshold, common.QwenModelName)

			populateCache(client)

			// Second request: test global cache hit threshold
			req2 := createCompletionRequest(completionRequestParams{
				Prompt:    prompt2,
				Model:     common.QwenModelName,
				MaxTokens: 5,
			})
			startTime := time.Now()
			resp2, err := client.Do(req2)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp2.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			elapsed := time.Since(startTime)
			Expect(elapsed).To(BeNumerically("<", 100*time.Millisecond), "Response should be immediate")

			Expect(resp2.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())

			var completionResp map[string]interface{}
			err = json.Unmarshal(body, &completionResp)
			Expect(err).NotTo(HaveOccurred())

			choices := completionResp["choices"].([]interface{})
			Expect(choices).To(HaveLen(1))
			firstChoice := choices[0].(map[string]interface{})

			Expect(firstChoice["finish_reason"]).To(Equal(common.CacheThresholdFinishReason))
		})

		It("Should use request cache_hit_threshold over global threshold when both are set", func() {
			// Set global threshold to 1.0 (very high, would fail for any request with < 100% cache hit)
			globalThreshold := 1.0
			client := setupKVCacheServer(true, &globalThreshold, common.QwenModelName)

			// Request with global threshold 1.0 (would fail with 0% cache hit) but request threshold 0.0
			// This demonstrates that request threshold takes precedence over global threshold
			threshold := 0.0
			req := createCompletionRequest(completionRequestParams{
				Prompt:            prompt1,
				Model:             common.QwenModelName,
				MaxTokens:         5,
				CacheHitThreshold: &threshold,
			})
			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var completionResp map[string]interface{}
			err = json.Unmarshal(body, &completionResp)
			Expect(err).NotTo(HaveOccurred())

			choices := completionResp["choices"].([]interface{})
			Expect(choices).To(HaveLen(1))
			firstChoice := choices[0].(map[string]interface{})
			finishReason, ok := firstChoice["finish_reason"].(string)
			Expect(ok).To(BeTrue())
			// Should proceed normally because request threshold (0.0) is used, not global (1.0)
			// With 0% cache hit rate initially:
			// - Global threshold 1.0 would fail (0% < 1.0) → cache_threshold
			// - Request threshold 0.0 passes (0% >= 0.0) → normal finish reason
			// This proves request threshold takes precedence over global threshold
			Expect(finishReason).To(Or(Equal(common.StopFinishReason), Equal(common.LengthFinishReason)))
			Expect(finishReason).NotTo(Equal(common.CacheThresholdFinishReason))
		})

		testSimpleRequestWithKVCacheDisabled := func(cacheHitThreshold *float64, globalThreshold *float64) {
			client := setupKVCacheServer(false, globalThreshold, common.TestModelName)

			req := createCompletionRequest(completionRequestParams{
				Prompt:            "Hello world",
				Model:             common.TestModelName,
				MaxTokens:         5,
				CacheHitThreshold: cacheHitThreshold,
			})
			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var completionResp map[string]interface{}
			err = json.Unmarshal(body, &completionResp)
			Expect(err).NotTo(HaveOccurred())

			choices := completionResp["choices"].([]interface{})
			Expect(choices).To(HaveLen(1))
			firstChoice := choices[0].(map[string]interface{})

			finishReason, ok := firstChoice["finish_reason"].(string)
			Expect(ok).To(BeTrue())
			Expect(finishReason).To(Or(Equal(common.StopFinishReason), Equal(common.LengthFinishReason)))
		}

		Context("When KV cache is disabled", func() {
			It("Should ignore cache_hit_threshold defined in the request", func() {
				threshold := 1.0
				testSimpleRequestWithKVCacheDisabled(&threshold, nil)
			})

			It("Should ignore global_cache_hit_threshold command line argument", func() {
				globalThreshold := 0.9
				testSimpleRequestWithKVCacheDisabled(nil, &globalThreshold)
			})
		})
	})

	Context("errors", func() {
		It("Should return error for invalid model", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiClient := openai.NewClient(option.WithBaseURL(baseURL), option.WithHTTPClient(client),
				option.WithMaxRetries(0))

			params := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage(testUserMessage),
				},
				Model: "some-other-model",
			}

			_, err = openaiClient.Chat.Completions.New(ctx, params)
			Expect(err).To(HaveOccurred())
			var openaiError *openai.Error
			ok := errors.As(err, &openaiError)
			Expect(ok).To(BeTrue())
			Expect(openaiError.StatusCode).To(BeNumerically("==", fasthttp.StatusNotFound))
			Expect(openaiError.Type).ToNot(BeEmpty())
			Expect(openaiError.Message).To(ContainSubstring("The model `some-other-model` does not exist"))
		})

		It("Should return error for negative MaxCompletionTokens", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiClient := openai.NewClient(option.WithBaseURL(baseURL), option.WithHTTPClient(client),
				option.WithMaxRetries(0))

			params := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage(testUserMessage),
				},
				Model:               common.TestModelName,
				MaxCompletionTokens: openai.Int(-5),
			}

			_, err = openaiClient.Chat.Completions.New(ctx, params)
			Expect(err).To(HaveOccurred())
			var openaiError *openai.Error
			ok := errors.As(err, &openaiError)
			Expect(ok).To(BeTrue())
			Expect(openaiError.StatusCode).To(BeNumerically("==", fasthttp.StatusBadRequest))
			Expect(openaiError.Type).ToNot(BeEmpty())
			Expect(openaiError.Message).To(ContainSubstring("Max completion tokens and max tokens should be positive"))
		})
	})

	Context("OpenRequests counter", func() {
		It("Should reflect in-flight requests and return to zero after completion", func() {
			ctx := context.TODO()
			// 1 worker, queue capacity 2, 500ms TTFT so requests stay in-flight long enough to inspect
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
				"--time-to-first-token", "500", "--max-num-seqs", "1", "--max-waiting-queue-length", "2"}
			server, _, client, err := startServerHandle(ctx, common.ModeEcho, args, nil)
			Expect(err).NotTo(HaveOccurred())

			// Before any request the counter must be zero
			Expect(server.OpenRequests()).To(Equal(int64(0)))

			var wg sync.WaitGroup
			wg.Add(2)

			// Send two requests concurrently: one will be processed by the single
			// worker, the other will sit in the waiting queue.
			for range 2 {
				go func() {
					defer GinkgoRecover()
					defer wg.Done()
					openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, false)
					_, err := openaiclient.Chat.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
				}()
			}

			// Give the goroutines time to reach the server and enter the worker / queue
			time.Sleep(200 * time.Millisecond)
			Expect(server.OpenRequests()).To(Equal(int64(2)))

			// Wait for both requests to finish — counter must return to zero
			wg.Wait()
			time.Sleep(200 * time.Millisecond)
			Expect(server.OpenRequests()).To(Equal(int64(0)))
		})
	})

	Context("responses API", func() {
		DescribeTable("responses with string and array input",
			func(model string, mode string, useStringInput bool) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", model, "--mode", mode}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClientAndResponsesParams(client, model, testUserMessage)
				if !useStringInput {
					params.Input = responses.ResponseNewParamsInputUnion{
						OfInputItemList: responses.ResponseInputParam{
							responses.ResponseInputItemUnionParam{
								OfMessage: &responses.EasyInputMessageParam{
									Role:    responses.EasyInputMessageRoleUser,
									Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(testUserMessage)},
								},
							},
						},
					}
				}

				resp, err := openaiclient.Responses.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).NotTo(BeNil())

				outputText := resp.OutputText()
				Expect(outputText).NotTo(BeEmpty())
				if mode == common.ModeEcho {
					Expect(outputText).To(Equal(testUserMessage))
				} else {
					Expect(dataset.IsValidText(outputText)).To(BeTrue())
				}

				Expect(resp.Usage.InputTokens).To(BeNumerically(">", 0))
				Expect(resp.Usage.OutputTokens).To(BeNumerically(">", 0))
				Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.InputTokens + resp.Usage.OutputTokens))

				Expect(resp.ID).To(HavePrefix(openaiserverapi.ResponsesIDPrefix))
				Expect(resp.Status).To(Equal(responses.ResponseStatusCompleted))
				Expect(resp.Instructions.AsString()).To(BeEmpty())

				Expect(resp.Output).NotTo(BeEmpty())
				firstItem := resp.Output[0]
				Expect(string(firstItem.Role)).To(Equal("assistant"))
				Expect(firstItem.Content).NotTo(BeEmpty())
				Expect(firstItem.Content[0].Type).To(Equal(openaiserverapi.ResponsesOutputText))
			},
			func(model string, mode string, useStringInput bool) string {
				inputType := "array"
				if useStringInput {
					inputType = "string"
				}
				return fmt.Sprintf("model: %s mode: %s input: %s", model, mode, inputType)
			},
			Entry(nil, common.TestModelName, common.ModeRandom, true),
			Entry(nil, common.TestModelName, common.ModeEcho, true),
			Entry(nil, common.TestModelName, common.ModeRandom, false),
			Entry(nil, common.TestModelName, common.ModeEcho, false),
			Entry(nil, common.QwenModelName, common.ModeRandom, true),
			Entry(nil, common.QwenModelName, common.ModeEcho, true),
			Entry(nil, common.QwenModelName, common.ModeRandom, false),
			Entry(nil, common.QwenModelName, common.ModeEcho, false),
		)

		It("Should echo instructions in the response", func() {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			const testInstructions = "Reply in French"
			openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, testUserMessage, testInstructions)

			resp, err := openaiclient.Responses.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Instructions.AsString()).To(Equal(testInstructions))
		})

		It("Should respect max_output_tokens", func() {
			ctx := context.TODO()
			server, _, client, err := startServerHandle(ctx, common.ModeRandom, nil, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, testUserMessage)
			params.MaxOutputTokens = param.NewOpt(int64(2))

			resp, err := openaiclient.Responses.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())

			outputText := resp.OutputText()
			_, tokens, err := server.Context.Tokenizer.RenderText(outputText)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically("<=", 2))
		})

		It("Should return error for invalid model", func() {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndResponsesParams(client, "nonexistent-model", testUserMessage)
			_, err = openaiclient.Responses.New(ctx, params)
			Expect(err).To(HaveOccurred())
			var openaiError *openai.Error
			Expect(errors.As(err, &openaiError)).To(BeTrue())
			Expect(openaiError.StatusCode).To(Equal(fasthttp.StatusNotFound))
		})

		DescribeTable("responses streaming",
			func(model string, mode string) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", model, "--mode", mode}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClientAndResponsesParams(client, model, testUserMessage)

				stream := openaiclient.Responses.NewStreaming(ctx, params)
				defer func() {
					Expect(stream.Close()).NotTo(HaveOccurred())
				}()

				var eventTypes []string
				var deltas []string

				for stream.Next() {
					event := stream.Current()
					eventTypes = append(eventTypes, event.Type)
					switch event.Type {
					case openaiserverapi.ResponsesEventCreated:
						created := event.AsResponseCreated()
						Expect(string(created.Response.Status)).To(Equal(openaiserverapi.ResponsesStatusInProgress))
					case openaiserverapi.ResponsesEventOutputItemAdded:
						added := event.AsResponseOutputItemAdded()
						Expect(added.OutputIndex).To(Equal(int64(0)))
					case openaiserverapi.ResponsesEventTextDelta:
						delta := event.AsResponseOutputTextDelta()
						Expect(delta.Delta).NotTo(BeEmpty())
						deltas = append(deltas, delta.Delta)
					case openaiserverapi.ResponsesEventTextDone:
						done := event.AsResponseOutputTextDone()
						Expect(done.Text).NotTo(BeEmpty())
						Expect(done.Text).To(Equal(strings.Join(deltas, "")))
					case openaiserverapi.ResponsesEventCompleted:
						completed := event.AsResponseCompleted()
						Expect(completed.Response.Usage.InputTokens).To(BeNumerically(">", 0))
						Expect(completed.Response.Usage.OutputTokens).To(BeNumerically(">", 0))
						Expect(completed.Response.Usage.TotalTokens).To(Equal(
							completed.Response.Usage.InputTokens + completed.Response.Usage.OutputTokens))
						Expect(string(completed.Response.Status)).To(Equal(openaiserverapi.ResponsesStatusCompleted))
					}
				}
				Expect(stream.Err()).NotTo(HaveOccurred())

				// Verify the mandatory fixed positions in the event sequence:
				// [0] created, [1] in_progress, [2] output_item.added, [3] content_part.added,
				// [4..n-5] deltas, [n-4] text.done, [n-3] content_part.done,
				// [n-2] output_item.done, [n-1] completed
				Expect(len(eventTypes)).To(BeNumerically(">=", 9), "expected at least 9 events")
				Expect(eventTypes[0]).To(Equal(openaiserverapi.ResponsesEventCreated))
				Expect(eventTypes[1]).To(Equal(openaiserverapi.ResponsesEventInProgress))
				Expect(eventTypes[2]).To(Equal(openaiserverapi.ResponsesEventOutputItemAdded))
				Expect(eventTypes[3]).To(Equal(openaiserverapi.ResponsesEventContentPartAdded))
				// deltas occupy positions [4 .. len-5]
				nDeltas := len(eventTypes) - 8
				Expect(nDeltas).To(BeNumerically(">=", 1), "expected at least one delta event")
				for i := 4; i < 4+nDeltas; i++ {
					Expect(eventTypes[i]).To(Equal(openaiserverapi.ResponsesEventTextDelta))
				}
				Expect(eventTypes[len(eventTypes)-4]).To(Equal(openaiserverapi.ResponsesEventTextDone))
				Expect(eventTypes[len(eventTypes)-3]).To(Equal(openaiserverapi.ResponsesEventContentPartDone))
				Expect(eventTypes[len(eventTypes)-2]).To(Equal(openaiserverapi.ResponsesEventOutputItemDone))
				Expect(eventTypes[len(eventTypes)-1]).To(Equal(openaiserverapi.ResponsesEventCompleted))

				fullText := strings.Join(deltas, "")
				if mode == common.ModeEcho {
					Expect(fullText).To(Equal(testUserMessage))
				} else {
					Expect(dataset.IsValidText(fullText)).To(BeTrue())
				}
			},
			func(model string, mode string) string {
				return fmt.Sprintf("model: %s mode: %s", model, mode)
			},
			Entry(nil, common.TestModelName, common.ModeRandom),
			Entry(nil, common.TestModelName, common.ModeEcho),
			// Entry(nil, common.QwenModelName, common.ModeRandom),
			// Entry(nil, common.QwenModelName, common.ModeEcho),
		)
	})

	Context("kv-events for requests", func() {
		ctx := context.TODO()
		model := common.QwenModelName
		mode := common.ModeRandom
		longPrompt := "This is a test message for kv cache events, has to be long enough to be tokenized into multiple blocks."

		It("chat completions", func() {
			// create kv events listener
			topic := kvcache.CreateKVEventsTopic("localhost", model)
			sub, zmqEndpoint := common.CreateSub(ctx, topic)
			//nolint
			defer sub.Close()

			// start the server
			args := []string{"cmd", "--model", model, "--mode", mode,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--event-batch-size", "1", "--zmq-endpoint", zmqEndpoint}
			client, err := startServerWithArgsAndEnv(ctx, mode, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			go func() {
				time.Sleep(200 * time.Millisecond)

				openaiclient, params := getOpenAIClientAndChatParams(client, model, longPrompt, false)
				resp, err := openaiclient.Chat.Completions.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.Choices).ShouldNot(BeEmpty())
			}()

			// read one event
			msg, err := sub.Recv()
			Expect(err).NotTo(HaveOccurred())
			storedCount, removedCount, _ := kvcache.CountKVEventBlocks(msg.Frames, topic, 1)
			Expect(storedCount).To(Equal(4))
			Expect(removedCount).To(Equal(0))
		})

		It("completions", func() {
			// create kv events listener
			topic := kvcache.CreateKVEventsTopic("localhost", model)
			sub, zmqEndpoint := common.CreateSub(ctx, topic)
			//nolint
			defer sub.Close()

			// start the server
			args := []string{"cmd", "--model", model, "--mode", mode,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--event-batch-size", "1", "--zmq-endpoint", zmqEndpoint}
			client, err := startServerWithArgsAndEnv(ctx, mode, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			go func() {
				time.Sleep(200 * time.Millisecond)

				openaiclient, params := getOpenAIClentAndCompletionParams(client, model, longPrompt, false)
				resp, err := openaiclient.Completions.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.Choices).ShouldNot(BeEmpty())
			}()

			// read one event
			msg, err := sub.Recv()
			Expect(err).NotTo(HaveOccurred())
			storedCount, removedCount, _ := kvcache.CountKVEventBlocks(msg.Frames, topic, 1)
			Expect(storedCount).To(Equal(2))
			Expect(removedCount).To(Equal(0))
		})
	})
})
