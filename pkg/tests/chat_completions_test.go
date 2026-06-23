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

package tests

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
)

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
				Expect(string(chunk.Object)).To(Equal(api.ChatCompletionChunkObject))
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

	It("Should send length finish_reason chunk in chat completions streaming", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, true)
		params.MaxTokens = param.NewOpt(int64(1))
		stream := openaiclient.Chat.Completions.NewStreaming(ctx, params)
		defer func() {
			err := stream.Close()
			Expect(err).NotTo(HaveOccurred())
		}()

		var finishReason string
		for stream.Next() {
			for _, choice := range stream.Current().Choices {
				if choice.FinishReason != "" {
					finishReason = choice.FinishReason
				}
			}
		}
		Expect(stream.Err()).NotTo(HaveOccurred())
		Expect(finishReason).To(Equal(common.LengthFinishReason))
	})

	DescribeTable("chat completions",
		func(model string, mode string, maxTokens int, maxCompletionTokens int) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", mode}
			server, _, client, err := startServerHandle(ctx, mode, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, model, testUserMessage, false)
			numTokens := 0
			// if maxTokens and maxCompletionTokens are passed
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
			Expect(string(resp.Object)).To(Equal(api.ChatCompletionObject))

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

	DescribeTable("chat completions with n parameter",
		func(mode string, n int) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", mode, "--max-num-seqs", "10"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, false)
			params.N = param.NewOpt(int64(n))
			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())

			// Exact number of choices must match n
			Expect(resp.Choices).To(HaveLen(n))
			Expect(string(resp.Object)).To(Equal(api.ChatCompletionObject))

			// Prompt tokens should be counted once, not n times
			Expect(resp.Usage.PromptTokens).To(Equal(userMsgChatTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			// Each choice must have valid content and a sequential index
			for i, choice := range resp.Choices {
				Expect(choice.Index).To(BeEquivalentTo(i))
				msg := choice.Message.Content
				Expect(msg).ShouldNot(BeEmpty())

				if mode == common.ModeEcho {
					Expect(msg).Should(Equal(testUserMessage))
				} else {
					Expect(dataset.IsValidText(msg)).To(BeTrue())
				}
			}
		},
		func(mode string, n int) string {
			return fmt.Sprintf("mode: %s n: %d", mode, n)
		},
		Entry(nil, common.ModeRandom, 1),
		Entry(nil, common.ModeEcho, 1),
		Entry(nil, common.ModeRandom, 3),
		Entry(nil, common.ModeEcho, 3),
		Entry(nil, common.ModeRandom, 5),
	)

	DescribeTable("chat completions streaming with n parameter",
		func(mode string, n int) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", mode, "--max-num-seqs", "10"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, true)
			params.N = param.NewOpt(int64(n))
			stream := openaiclient.Chat.Completions.NewStreaming(ctx, params)
			defer func() {
				err := stream.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			tokensPerChoice := make(map[int64][]string)
			roles := make(map[int64]string)
			var chunk openai.ChatCompletionChunk
			numberOfChunksWithUsage := 0
			for stream.Next() {
				chunk = stream.Current()
				for _, choice := range chunk.Choices {
					if choice.Delta.Role != "" {
						roles[choice.Index] = choice.Delta.Role
					} else if choice.FinishReason == "" {
						tokensPerChoice[choice.Index] = append(tokensPerChoice[choice.Index], choice.Delta.Content)
					}
				}
				if chunk.Usage.CompletionTokens != 0 || chunk.Usage.PromptTokens != 0 || chunk.Usage.TotalTokens != 0 {
					numberOfChunksWithUsage++
				}
			}
			Expect(stream.Err()).NotTo(HaveOccurred())

			Expect(numberOfChunksWithUsage).To(Equal(1))
			// Prompt tokens counted once
			Expect(chunk.Usage.PromptTokens).To(Equal(userMsgChatTokens))
			Expect(chunk.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(chunk.Usage.TotalTokens).To(Equal(chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens))

			// Exactly n choices must have been seen
			Expect(tokensPerChoice).To(HaveLen(n))
			for i := int64(0); i < int64(n); i++ {
				Expect(roles[i]).To(Equal("assistant"), "choice %d missing role", i)
				msg := strings.Join(tokensPerChoice[i], "")
				Expect(msg).ShouldNot(BeEmpty(), "choice %d has empty content", i)
				if mode == common.ModeEcho {
					Expect(msg).Should(Equal(testUserMessage))
				} else {
					Expect(dataset.IsValidText(msg)).To(BeTrue())
				}
			}
		},
		func(mode string, n int) string {
			return fmt.Sprintf("mode: %s n: %d", mode, n)
		},
		Entry(nil, common.ModeRandom, 1),
		Entry(nil, common.ModeEcho, 1),
		Entry(nil, common.ModeRandom, 3),
		Entry(nil, common.ModeEcho, 3),
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

	DescribeTable("non-streaming chat completions with logprobs",
		func(mode string, logprobsParam interface{}) {
			ctx := context.TODO()
			server, _, client, err := startServerHandle(ctx, mode, nil, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, false)
			if logprobsParam != nil {
				if logprobs, ok := logprobsParam.(bool); ok && logprobs {
					params.Logprobs = param.NewOpt(true)
					params.TopLogprobs = param.NewOpt(int64(2))
				}
			}
			chatResp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(chatResp.Choices).ShouldNot(BeEmpty())

			if logprobsParam != nil {
				Expect(chatResp.Choices[0].Logprobs.Content).NotTo(BeEmpty())
				_, tokens, err := server.Context.Tokenizer.RenderText(chatResp.Choices[0].Message.Content)
				Expect(err).NotTo(HaveOccurred())
				Expect(chatResp.Choices[0].Logprobs.Content).To(HaveLen(len(tokens)))
			} else {
				Expect(chatResp.Choices[0].Logprobs.Content).To(BeNil())
			}
		},
		func(mode string, logprobsParam interface{}) string {
			return fmt.Sprintf("mode: %s logprobs: %v", mode, logprobsParam)
		},
		Entry(nil, common.ModeEcho, true), // with logprobs
		Entry(nil, common.ModeEcho, nil),  // without logprobs
	)

	Context("chat completions with image output (omni mode)", func() {
		const syntheticImagePrefix = "data:image/png;base64,"

		newOmniClient := func(client *http.Client) openai.Client {
			return openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client),
				option.WithMaxRetries(0))
		}

		omniParams := func() openai.ChatCompletionNewParams {
			return openai.ChatCompletionNewParams{
				Messages:  []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
				Model:     common.TestModelName,
				MaxTokens: param.NewOpt(int64(5)),
			}
		}

		assertNoImage := func(ctx context.Context, openaiclient openai.Client, opts ...option.RequestOption) {
			resp, err := openaiclient.Chat.Completions.New(ctx, omniParams(), opts...)
			Expect(err).NotTo(HaveOccurred())
			var rawResp map[string]any
			Expect(json.Unmarshal([]byte(resp.RawJSON()), &rawResp)).To(Succeed())
			choices := rawResp["choices"].([]any)
			message := choices[0].(map[string]any)["message"].(map[string]any)
			_, isString := message["content"].(string)
			Expect(isString).To(BeTrue(), "expected plain string content, not image blocks")
		}

		assertImage := func(ctx context.Context, openaiclient openai.Client, opts ...option.RequestOption) {
			resp, err := openaiclient.Chat.Completions.New(ctx, omniParams(), opts...)
			Expect(err).NotTo(HaveOccurred())
			var rawResp map[string]any
			Expect(json.Unmarshal([]byte(resp.RawJSON()), &rawResp)).To(Succeed())
			choices := rawResp["choices"].([]any)
			message := choices[0].(map[string]any)["message"].(map[string]any)
			contentBlocks, isArray := message["content"].([]any)
			Expect(isArray).To(BeTrue(), "expected structured content array")
			Expect(contentBlocks).To(HaveLen(2))
			Expect(contentBlocks[1].(map[string]any)["type"]).To(Equal("image_url"))
		}

		It("Should include image content blocks in non-streaming response when omni + X-Send-Image: true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--omni"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := newOmniClient(client)
			resp, err := openaiclient.Chat.Completions.New(ctx, omniParams(),
				option.WithHeader(communication.XSendImageHeader, "true"))
			Expect(err).NotTo(HaveOccurred())

			var rawResp map[string]any
			Expect(json.Unmarshal([]byte(resp.RawJSON()), &rawResp)).To(Succeed())

			choices := rawResp["choices"].([]any)
			message := choices[0].(map[string]any)["message"].(map[string]any)
			contentBlocks := message["content"].([]any)
			Expect(contentBlocks).To(HaveLen(2))

			textBlock := contentBlocks[0].(map[string]any)
			Expect(textBlock["type"]).To(Equal("text"))
			Expect(textBlock["text"]).NotTo(BeEmpty())

			imageBlock := contentBlocks[1].(map[string]any)
			Expect(imageBlock["type"]).To(Equal("image_url"))
			imageURL := imageBlock["image_url"].(map[string]any)
			Expect(imageURL["url"].(string)).To(HavePrefix(syntheticImagePrefix))
		})

		It("Should include image chunk in streaming response when omni + X-Send-Image: true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--omni"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := newOmniClient(client)
			stream := openaiclient.Chat.Completions.NewStreaming(ctx, omniParams(),
				option.WithHeader(communication.XSendImageHeader, "true"))
			defer func() { Expect(stream.Close()).To(Succeed()) }()

			var imageChunk map[string]any
			for stream.Next() {
				var rawChunk map[string]any
				Expect(json.Unmarshal([]byte(stream.Current().RawJSON()), &rawChunk)).To(Succeed())
				if rawChunk["modality"] == "image" {
					imageChunk = rawChunk
				}
			}
			Expect(stream.Err()).NotTo(HaveOccurred())

			Expect(imageChunk).NotTo(BeNil(), "expected an image modality chunk in the stream")
			choices := imageChunk["choices"].([]any)
			delta := choices[0].(map[string]any)["delta"].(map[string]any)
			contentBlocks := delta["content"].([]any)
			Expect(contentBlocks).To(HaveLen(1))
			imageBlock := contentBlocks[0].(map[string]any)
			Expect(imageBlock["type"]).To(Equal("image_url"))
			imageURL := imageBlock["image_url"].(map[string]any)
			Expect(imageURL["url"].(string)).To(HavePrefix(syntheticImagePrefix))
		})

		It("Should use structured text blocks for token chunks in streaming when omni + X-Send-Image: true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--omni"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := newOmniClient(client)
			stream := openaiclient.Chat.Completions.NewStreaming(ctx, omniParams(),
				option.WithHeader(communication.XSendImageHeader, "true"))
			defer func() { Expect(stream.Close()).To(Succeed()) }()

			var tokenChunks []map[string]any
			for stream.Next() {
				var rawChunk map[string]any
				Expect(json.Unmarshal([]byte(stream.Current().RawJSON()), &rawChunk)).To(Succeed())
				if rawChunk["modality"] == "image" {
					continue
				}
				choices, ok := rawChunk["choices"].([]any)
				if !ok || len(choices) == 0 {
					continue
				}
				delta := choices[0].(map[string]any)["delta"].(map[string]any)
				if _, isArray := delta["content"].([]any); isArray {
					tokenChunks = append(tokenChunks, rawChunk)
				}
			}
			Expect(stream.Err()).NotTo(HaveOccurred())

			Expect(tokenChunks).NotTo(BeEmpty(), "expected token chunks with structured content")
			for _, chunk := range tokenChunks {
				choices := chunk["choices"].([]any)
				delta := choices[0].(map[string]any)["delta"].(map[string]any)
				blocks := delta["content"].([]any)
				Expect(blocks).To(HaveLen(1))
				block := blocks[0].(map[string]any)
				Expect(block["type"]).To(Equal("text"))
				Expect(block["text"]).NotTo(BeEmpty())
			}
		})

		It("Should NOT include image when omni mode is disabled even with X-Send-Image: true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			assertNoImage(ctx, newOmniClient(client), option.WithHeader(communication.XSendImageHeader, "true"))
		})

		It("Should NOT include image when X-Send-Image header is absent even in omni mode", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--omni"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			assertNoImage(ctx, newOmniClient(client))
		})

		It("Should emit image in non-streaming response when omni + image-emission-rate 100, no header", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--omni", "--image-emission-rate", "100"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			assertImage(ctx, newOmniClient(client))
		})

		It("Should NOT emit image when omni + image-emission-rate 0, no header", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--omni", "--image-emission-rate", "0"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			assertNoImage(ctx, newOmniClient(client))
		})

		It("Should emit image via X-Send-Image header even when image-emission-rate is 0", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--omni", "--image-emission-rate", "0"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			assertImage(ctx, newOmniClient(client), option.WithHeader(communication.XSendImageHeader, "true"))
		})

		Context("with admin config update", func() {
			It("toggles image emission by updating image-emission-rate via /admin/config", func() {
				ctx := context.TODO()
				args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--omni", "--image-emission-rate", "0"}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				openaiclient := newOmniClient(client)

				// rate=0: no image emitted without header.
				assertNoImage(ctx, openaiclient)

				// Enable 100%: image always emitted.
				resp := postAdminConfig(client, `{"image-emission-rate":100}`)
				Expect(resp.Body.Close()).To(Succeed())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))
				assertImage(ctx, openaiclient)

				// Back to 0: no image again.
				resp = postAdminConfig(client, `{"image-emission-rate":0}`)
				Expect(resp.Body.Close()).To(Succeed())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))
				assertNoImage(ctx, openaiclient)
			})

			It("rejects an out-of-range image-emission-rate", func() {
				ctx := context.TODO()
				args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--omni"}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				resp := postAdminConfig(client, `{"image-emission-rate":101}`)
				defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
				Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
			})
		})
	})
})
