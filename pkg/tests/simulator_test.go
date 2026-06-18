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

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
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

	DescribeTable("text completions streaming",
		func(model string, mode string) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", mode}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndCompletionParams(client, model, testUserMessage, true)
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
				Expect(string(chunk.Object)).To(Equal(api.TextCompletionObject))
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

	It("Should send length finish_reason chunk in text completions streaming", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, testUserMessage, true)
		params.MaxTokens = param.NewOpt(int64(1))
		stream := openaiclient.Completions.NewStreaming(ctx, params)
		defer func() {
			err := stream.Close()
			Expect(err).NotTo(HaveOccurred())
		}()

		var finishReason string
		for stream.Next() {
			for _, choice := range stream.Current().Choices {
				if choice.FinishReason != "" {
					finishReason = string(choice.FinishReason)
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

	DescribeTable("text completions",
		// use a function so that httpClient is captured when running
		func(model string, mode string, maxTokens int) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", mode}
			server, _, client, err := startServerHandle(ctx, mode, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndCompletionParams(client, model, testUserMessage, false)
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
			Expect(string(resp.Object)).To(Equal(api.TextCompletionObject))

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

	DescribeTable("text completions with n parameter",
		func(mode string, n int) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", mode, "--max-num-seqs", "10"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, testUserMessage, false)
			params.N = param.NewOpt(int64(n))
			resp, err := openaiclient.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())

			// Exact number of choices must match n
			Expect(resp.Choices).To(HaveLen(n))
			Expect(string(resp.Object)).To(Equal(api.TextCompletionObject))

			// Prompt tokens should be counted once
			Expect(resp.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			// Each choice must have valid content and a sequential index
			for i, choice := range resp.Choices {
				Expect(choice.Index).To(BeEquivalentTo(i))
				Expect(choice.Text).ShouldNot(BeEmpty())

				if mode == common.ModeEcho {
					Expect(choice.Text).Should(Equal(testUserMessage))
				} else {
					Expect(dataset.IsValidText(choice.Text)).To(BeTrue())
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

	DescribeTable("text completions streaming with n parameter",
		func(mode string, n int) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", mode, "--max-num-seqs", "10"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, testUserMessage, true)
			params.N = param.NewOpt(int64(n))
			stream := openaiclient.Completions.NewStreaming(ctx, params)
			defer func() {
				err := stream.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			tokensPerChoice := make(map[int64][]string)
			var chunk openai.Completion
			numberOfChunksWithUsage := 0
			for stream.Next() {
				chunk = stream.Current()
				for _, choice := range chunk.Choices {
					if choice.FinishReason == "" {
						tokensPerChoice[choice.Index] = append(tokensPerChoice[choice.Index], choice.Text)
					}
				}
				if chunk.Usage.CompletionTokens != 0 || chunk.Usage.PromptTokens != 0 || chunk.Usage.TotalTokens != 0 {
					numberOfChunksWithUsage++
				}
			}
			Expect(stream.Err()).NotTo(HaveOccurred())

			Expect(numberOfChunksWithUsage).To(Equal(1))
			Expect(chunk.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(chunk.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(chunk.Usage.TotalTokens).To(Equal(chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens))

			// Exactly n choices must have been seen
			Expect(tokensPerChoice).To(HaveLen(n))
			for i := int64(0); i < int64(n); i++ {
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

	It("text completions with array prompt and n parameter", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho, "--max-num-seqs", "10"}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		prompts := []string{prompt1, prompt2}
		n := 3

		var expectedPromptTokens int64
		for _, p := range prompts {
			tokens, _, err := tokenizerMngr.TestTokenizer().RenderText(p)
			Expect(err).NotTo(HaveOccurred())
			expectedPromptTokens += int64(len(tokens))
		}

		resp, err := openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: prompts},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
			N:      param.NewOpt(int64(n)),
		})
		Expect(err).NotTo(HaveOccurred())

		// Total choices = len(prompts) * n
		totalChoices := len(prompts) * n
		Expect(resp.Choices).To(HaveLen(totalChoices))

		// In echo mode, each group of n choices for a prompt should echo that prompt.
		// Prompt 0 → choices 0..n-1, Prompt 1 → choices n..2n-1.
		for i, c := range resp.Choices {
			Expect(c.Index).To(BeEquivalentTo(i))
			promptIdx := int(c.Index) / n
			Expect(c.Text).To(Equal(prompts[promptIdx]),
				"choice %d should echo prompt %d (%q)", c.Index, promptIdx, prompts[promptIdx])
		}

		// Prompt tokens counted once per prompt, not once per choice.
		Expect(resp.Usage.PromptTokens).To(Equal(expectedPromptTokens))
		// In echo mode completion tokens equal the sum of prompt tokens across
		// all choices: each of the n copies for each prompt echoes the full prompt.
		Expect(resp.Usage.CompletionTokens).To(Equal(expectedPromptTokens * int64(n)))
		Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))
	})

	DescribeTable("text completions with array prompt",
		func(streaming bool) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
				"--time-to-first-token", "500ms", "--time-to-first-token-std-dev", "100ms"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			prompts := []string{prompt1, prompt2, "How about tomorrow?"}
			const logprobsCount = 2
			params := openai.CompletionNewParams{
				Prompt:   openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: prompts},
				Model:    openai.CompletionNewParamsModel(common.TestModelName),
				Logprobs: param.NewOpt(int64(logprobsCount)),
			}

			// In echo mode each sub-request's completion equals its prompt, so the
			// aggregated usage is 2× the sum of per-prompt token counts.
			var expectedPromptTokens int64
			for _, p := range prompts {
				tokens, _, err := tokenizerMngr.TestTokenizer().RenderText(p)
				Expect(err).NotTo(HaveOccurred())
				expectedPromptTokens += int64(len(tokens))
			}

			if streaming {
				params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
				stream := openaiclient.Completions.NewStreaming(ctx, params)
				defer func() {
					Expect(stream.Close()).To(Succeed())
				}()
				// Collect streamed text per choice index
				texts := make([]string, len(prompts))
				chunksWithLogprobs := make([]int, len(prompts))
				var chunk openai.Completion
				var usage openai.CompletionUsage
				for stream.Next() {
					chunk = stream.Current()
					for _, choice := range chunk.Choices {
						texts[choice.Index] += choice.Text
						if choice.FinishReason == "" && choice.Text != "" && len(choice.Logprobs.Tokens) > 0 {
							chunksWithLogprobs[choice.Index]++
							Expect(choice.Logprobs.Tokens[0]).To(Equal(choice.Text))
							Expect(choice.Logprobs.TokenLogprobs[0]).To(BeNumerically("<=", 0))
							Expect(choice.Logprobs.TopLogprobs[0]).To(HaveLen(logprobsCount))
							Expect(choice.Logprobs.TopLogprobs[0]).To(HaveKey(choice.Text))
						}
					}
					if chunk.Usage.TotalTokens != 0 {
						usage = chunk.Usage
					}
				}
				Expect(stream.Err()).NotTo(HaveOccurred())
				Expect(string(chunk.Object)).To(Equal(api.TextCompletionObject))
				for i, prompt := range prompts {
					Expect(texts[i]).To(Equal(prompt))
					// Every choice must carry its own logprobs stream.
					Expect(chunksWithLogprobs[i]).To(BeNumerically(">", 0),
						"choice %d should have logprobs chunks", i)
				}
				Expect(usage.PromptTokens).To(Equal(expectedPromptTokens))
				Expect(usage.CompletionTokens).To(Equal(expectedPromptTokens))
				Expect(usage.TotalTokens).To(Equal(expectedPromptTokens * 2))
			} else {
				resp, err := openaiclient.Completions.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(string(resp.Object)).To(Equal(api.TextCompletionObject))
				Expect(resp.Choices).To(HaveLen(len(prompts)))
				// Each choice should echo the corresponding prompt and carry its own index and logprobs.
				for i, prompt := range prompts {
					Expect(resp.Choices[i].Index).To(BeEquivalentTo(i))
					Expect(resp.Choices[i].Text).To(Equal(prompt))
					Expect(resp.Choices[i].Logprobs.Tokens).NotTo(BeNil())
					_, tokens, err := tokenizerMngr.TestTokenizer().RenderText(prompt)
					Expect(err).NotTo(HaveOccurred())
					Expect(resp.Choices[i].Logprobs.Tokens).To(HaveLen(len(tokens)))
				}
				Expect(resp.Usage.PromptTokens).To(Equal(expectedPromptTokens))
				Expect(resp.Usage.CompletionTokens).To(Equal(expectedPromptTokens))
				Expect(resp.Usage.TotalTokens).To(Equal(expectedPromptTokens * 2))
			}
		},
		Entry("non-streaming", false),
		Entry("streaming", true),
	)

	// Token-id prompts: /completions accepts the prompt as []uint32 (a single
	// pre-tokenized prompt) or [][]uint32 (an array of pre-tokenized prompts).
	// In echo mode the simulator replays the ids back as a comma-separated
	// decimal string ("1,2,3"), so prompt_tokens stays equal to the input
	// length and the tokenizer is never invoked on the prompt — which is why
	// the same expectations hold for both the simulated and real-model tokenizers.
	DescribeTable("text completions with token-id prompt",
		func(model string, mode string, streaming bool) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", mode}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			tokens := []int64{1, 2, 3, 4}
			expectedPromptTokens := int64(len(tokens))
			expectedEcho := "1,2,3,4"
			params := openai.CompletionNewParams{
				Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfTokens: tokens},
				Model:  openai.CompletionNewParamsModel(model),
			}

			var text string
			var usage openai.CompletionUsage
			if streaming {
				params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
				stream := openaiclient.Completions.NewStreaming(ctx, params)
				defer func() { Expect(stream.Close()).To(Succeed()) }()
				var b strings.Builder
				for stream.Next() {
					chunk := stream.Current()
					for _, choice := range chunk.Choices {
						b.WriteString(choice.Text)
					}
					if chunk.Usage.TotalTokens != 0 {
						usage = chunk.Usage
					}
				}
				Expect(stream.Err()).NotTo(HaveOccurred())
				text = b.String()
			} else {
				resp, err := openaiclient.Completions.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.Choices).To(HaveLen(1))
				text = resp.Choices[0].Text
				usage = resp.Usage
			}

			Expect(usage.PromptTokens).To(Equal(expectedPromptTokens))
			if mode == common.ModeEcho {
				Expect(text).To(Equal(expectedEcho))
			} else {
				Expect(text).NotTo(BeEmpty())
				Expect(dataset.IsValidText(text)).To(BeTrue())
			}
		},
		func(model string, mode string, streaming bool) string {
			return fmt.Sprintf("model: %s mode: %s streaming: %v", model, mode, streaming)
		},
		Entry(nil, common.TestModelName, common.ModeEcho, false),
		Entry(nil, common.TestModelName, common.ModeEcho, true),
		Entry(nil, common.TestModelName, common.ModeRandom, false),
		Entry(nil, common.TestModelName, common.ModeRandom, true),
		Entry(nil, common.QwenModelName, common.ModeEcho, false),
		Entry(nil, common.QwenModelName, common.ModeEcho, true),
		Entry(nil, common.QwenModelName, common.ModeRandom, false),
		Entry(nil, common.QwenModelName, common.ModeRandom, true),
	)

	DescribeTable("text completions with token-id arrays prompt",
		func(model string, mode string, streaming bool) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", mode}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			promptTokens := [][]int64{{1, 2, 3}, {10, 20}, {7}}
			expectedEcho := []string{"1,2,3", "10,20", "7"}
			var expectedPromptTokens int64
			for _, ids := range promptTokens {
				expectedPromptTokens += int64(len(ids))
			}

			params := openai.CompletionNewParams{
				Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfTokenArrays: promptTokens},
				Model:  openai.CompletionNewParamsModel(model),
			}

			texts := make([]string, len(promptTokens))
			var usage openai.CompletionUsage
			if streaming {
				params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
				stream := openaiclient.Completions.NewStreaming(ctx, params)
				defer func() { Expect(stream.Close()).To(Succeed()) }()
				for stream.Next() {
					chunk := stream.Current()
					for _, choice := range chunk.Choices {
						texts[choice.Index] += choice.Text
					}
					if chunk.Usage.TotalTokens != 0 {
						usage = chunk.Usage
					}
				}
				Expect(stream.Err()).NotTo(HaveOccurred())
			} else {
				resp, err := openaiclient.Completions.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.Choices).To(HaveLen(len(promptTokens)))
				for _, choice := range resp.Choices {
					texts[choice.Index] = choice.Text
				}
				usage = resp.Usage
			}

			Expect(usage.PromptTokens).To(Equal(expectedPromptTokens))
			for i, text := range texts {
				if mode == common.ModeEcho {
					Expect(text).To(Equal(expectedEcho[i]))
				} else {
					Expect(text).NotTo(BeEmpty())
					Expect(dataset.IsValidText(text)).To(BeTrue())
				}
			}
		},
		func(model string, mode string, streaming bool) string {
			return fmt.Sprintf("model: %s mode: %s streaming: %v", model, mode, streaming)
		},
		Entry(nil, common.TestModelName, common.ModeEcho, false),
		Entry(nil, common.TestModelName, common.ModeEcho, true),
		Entry(nil, common.TestModelName, common.ModeRandom, false),
		Entry(nil, common.TestModelName, common.ModeRandom, true),
		Entry(nil, common.QwenModelName, common.ModeEcho, false),
		Entry(nil, common.QwenModelName, common.ModeEcho, true),
		Entry(nil, common.QwenModelName, common.ModeRandom, false),
		Entry(nil, common.QwenModelName, common.ModeRandom, true),
	)

	DescribeTable("text completions with array prompt fail-fast when one sub-request errors",
		func(streaming bool) {
			ctx := context.TODO()
			// max-num-seqs=1 + max-waiting-queue-length=1 means only the first two
			// sub-requests fit (1 running + 1 waiting). The third hits the queue-full
			// error. TTFT is high enough that the queue-full error arrives before any
			// token chunks would.
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--time-to-first-token", "3s",
				"--max-num-seqs", "1", "--max-waiting-queue-length", "1"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.CompletionNewParams{
				Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{prompt1, prompt2, "a third prompt"}},
				Model:  openai.CompletionNewParamsModel(common.TestModelName),
			}

			if streaming {
				stream := openaiclient.Completions.NewStreaming(ctx, params)
				defer func() { Expect(stream.Close()).To(Succeed()) }()

				// Fail-fast contract: no token chunks leak through before the error.
				// Every chunk we observe must have empty Text on every choice.
				for stream.Next() {
					for _, c := range stream.Current().Choices {
						Expect(c.Text).To(BeEmpty(),
							"no token chunk should appear before the fail-fast error")
					}
				}
				Expect(stream.Err()).To(HaveOccurred())
				// TODO: check after fixing inconsistency in error responses in HTTP
				// var oaiErr *openai.Error
				// Expect(errors.As(stream.Err(), &oaiErr)).To(BeTrue())
				// Expect(oaiErr.StatusCode).To(Equal(fasthttp.StatusTooManyRequests))
				// Expect(oaiErr.Message).To(ContainSubstring("waiting requests queue is full"))
			} else {
				_, err := openaiclient.Completions.New(ctx, params)
				Expect(err).To(HaveOccurred())
				var oaiErr *openai.Error
				Expect(errors.As(err, &oaiErr)).To(BeTrue())
				Expect(oaiErr.StatusCode).To(Equal(fasthttp.StatusTooManyRequests))
				Expect(oaiErr.Message).To(ContainSubstring("waiting requests queue is full"))
			}
		},
		Entry("non-streaming", false),
		Entry("streaming", true),
	)

	It("text completions single-element array prompt behaves like a single-prompt request", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		params := openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{prompt1}},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		}

		resp, err := openaiclient.Completions.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(resp.Object)).To(Equal(api.TextCompletionObject))
		Expect(resp.Choices).To(HaveLen(1))
		Expect(resp.Choices[0].Index).To(BeEquivalentTo(0))
		Expect(resp.Choices[0].Text).To(Equal(prompt1))
	})

	It("text completions wire form accepts both string and array prompts", func() {
		// This test sends raw JSON (bypassing the OpenAI SDK's encoding) to pin
		// down the dual-form contract on the `prompt` field directly:
		//   - "prompt": "..."  → single-choice response.
		//   - "prompt": [...]  → one choice per element, in order.
		// The X-Request-ID response header echoes the parent request id (the
		// "-i" suffix is stamped on internal sub-request ids only).
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
			"--enable-request-id-headers"}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		post := func(body, requestID string) *http.Response {
			req, err := http.NewRequest("POST", "http://localhost/v1/completions", strings.NewReader(body))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			if requestID != "" {
				req.Header.Set(communication.RequestIDHeader, requestID)
			}
			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			return resp
		}

		decode := func(resp *http.Response) openai.Completion {
			defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
			Expect(resp.StatusCode).To(Equal(200))
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			var out openai.Completion
			Expect(json.Unmarshal(body, &out)).To(Succeed())
			return out
		}

		// Single-string prompt — wire form `"prompt": "..."`.
		strResp := post(fmt.Sprintf(`{"model":%q,"prompt":%q}`, common.TestModelName, prompt1), "rid-string")
		Expect(strResp.Header.Get(communication.RequestIDHeader)).To(Equal("rid-string"))
		strBody := decode(strResp)
		Expect(strBody.Choices).To(HaveLen(1))
		Expect(strBody.Choices[0].Index).To(BeEquivalentTo(0))
		Expect(strBody.Choices[0].Text).To(Equal(prompt1))

		// Two-element array prompt — wire form `"prompt": ["...", "..."]`.
		arrBody := fmt.Sprintf(`{"model":%q,"prompt":[%q,%q]}`, common.TestModelName, prompt1, prompt2)
		arrResp := post(arrBody, "rid-array")
		Expect(arrResp.Header.Get(communication.RequestIDHeader)).To(Equal("rid-array"))
		arr := decode(arrResp)
		Expect(arr.Choices).To(HaveLen(2))
		Expect(arr.Choices[0].Index).To(BeEquivalentTo(0))
		Expect(arr.Choices[0].Text).To(Equal(prompt1))
		Expect(arr.Choices[1].Index).To(BeEquivalentTo(1))
		Expect(arr.Choices[1].Text).To(Equal(prompt2))

		// Single-element array — equivalent to the string form.
		oneResp := post(fmt.Sprintf(`{"model":%q,"prompt":[%q]}`, common.TestModelName, prompt1), "rid-onearr")
		Expect(oneResp.Header.Get(communication.RequestIDHeader)).To(Equal("rid-onearr"))
		oneBody := decode(oneResp)
		Expect(oneBody.Choices).To(HaveLen(1))
		Expect(oneBody.Choices[0].Index).To(BeEquivalentTo(0))
		Expect(oneBody.Choices[0].Text).To(Equal(prompt1))

		// Invalid prompt type (number) — must be rejected at JSON unmarshalling.
		badResp := post(fmt.Sprintf(`{"model":%q,"prompt":123}`, common.TestModelName), "")
		defer func() { Expect(badResp.Body.Close()).To(Succeed()) }()
		Expect(badResp.StatusCode).To(Equal(400))
		badBytes, err := io.ReadAll(badResp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(badBytes)).To(ContainSubstring("prompt must be a string, an array of strings, an array of token ids, or an array of arrays of token ids"))
	})

	It("text completions empty array prompt is rejected with 400", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		_, err = openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{}},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		})
		Expect(err).To(HaveOccurred())
		var oaiErr *openai.Error
		Expect(errors.As(err, &oaiErr)).To(BeTrue())
		Expect(oaiErr.StatusCode).To(Equal(fasthttp.StatusBadRequest))
		Expect(oaiErr.Message).To(ContainSubstring("prompt array must contain at least one prompt"))

		// Follow-up single-prompt request must still succeed — proves rejecting the
		// bad request didn't affect the worker pool or response channel machinery.
		followUp, err := openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfString: param.NewOpt(prompt1)},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(followUp.Choices).To(HaveLen(1))
		Expect(followUp.Choices[0].Text).To(Equal(prompt1))
	})

	It("text completions array containing an empty string is rejected with 400", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		_, err = openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{"", prompt1, ""}},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		})
		Expect(err).To(HaveOccurred())
		var oaiErr *openai.Error
		Expect(errors.As(err, &oaiErr)).To(BeTrue())
		Expect(oaiErr.StatusCode).To(Equal(fasthttp.StatusBadRequest))
		Expect(oaiErr.Message).To(ContainSubstring("prompt must not contain an empty string"))
	})

	It("text completions array containing an empty token-id array is rejected with 400", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		_, err = openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfTokenArrays: [][]int64{{1, 2}, {}, {3}}},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		})
		Expect(err).To(HaveOccurred())
		var oaiErr *openai.Error
		Expect(errors.As(err, &oaiErr)).To(BeTrue())
		Expect(oaiErr.StatusCode).To(Equal(fasthttp.StatusBadRequest))
		Expect(oaiErr.Message).To(ContainSubstring("prompt must not contain an empty token-id array"))
	})

	It("text completions array prompt in random mode yields per-choice content and aggregated usage", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--max-num-seqs", "3"}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		prompts := []string{prompt1, prompt2, "a third prompt"}

		var expectedPromptTokens int64
		for _, p := range prompts {
			tokens, _, err := tokenizerMngr.TestTokenizer().RenderText(p)
			Expect(err).NotTo(HaveOccurred())
			expectedPromptTokens += int64(len(tokens))
		}

		resp, err := openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: prompts},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Choices).To(HaveLen(len(prompts)))

		// Indexes must be 0..N-1 with no duplicates, regardless of worker completion order.
		seen := make(map[int64]bool, len(prompts))
		for _, c := range resp.Choices {
			Expect(seen[c.Index]).To(BeFalse(), "duplicate choice index %d", c.Index)
			seen[c.Index] = true
			// Random mode content is non-deterministic but must be non-empty and the
			// finish reason must be a recognized terminal state.
			Expect(c.Text).NotTo(BeEmpty())
			Expect(string(c.FinishReason)).To(BeElementOf(common.StopFinishReason, common.LengthFinishReason))
		}
		for i := int64(0); i < int64(len(prompts)); i++ {
			Expect(seen[i]).To(BeTrue(), "missing choice index %d", i)
		}

		Expect(resp.Usage.PromptTokens).To(Equal(expectedPromptTokens))
		Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))
	})

	It("text completions array prompt with low max-tokens produces length finish reasons", func() {
		ctx := context.TODO()
		// max-tokens=1 forces every sub-request to finish with "length" (except any
		// that happens to generate an EOS at position 0 — so the assertion tolerates
		// both, but at least one must be "length").
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--max-num-seqs", "3"}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		prompts := []string{prompt1, prompt2, "third"}
		resp, err := openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt:    openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: prompts},
			Model:     openai.CompletionNewParamsModel(common.TestModelName),
			MaxTokens: param.NewOpt(int64(1)),
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Choices).To(HaveLen(len(prompts)))
		sawLength := false
		for _, c := range resp.Choices {
			Expect(string(c.FinishReason)).To(BeElementOf(common.StopFinishReason, common.LengthFinishReason))
			if c.FinishReason == common.LengthFinishReason {
				sawLength = true
			}
		}
		Expect(sawLength).To(BeTrue(), "expected at least one choice to hit max_tokens")
		// With max-tokens=1 each choice contributes at most 1 completion token.
		Expect(resp.Usage.CompletionTokens).To(BeNumerically("<=", int64(len(prompts))))
	})

	It("text completions array prompt without logprobs returns nil logprobs on every choice", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		prompts := []string{prompt1, prompt2}

		// Non-streaming: the openai-go type is a value, not a pointer, so we check
		// that its fields are all zero — that's how "no logprobs" manifests.
		resp, err := openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: prompts},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Choices).To(HaveLen(len(prompts)))
		for i, c := range resp.Choices {
			Expect(c.Logprobs.Tokens).To(BeEmpty(), "choice %d should have no logprobs tokens", i)
			Expect(c.Logprobs.TokenLogprobs).To(BeEmpty(), "choice %d should have no logprobs token_logprobs", i)
			Expect(c.Logprobs.TopLogprobs).To(BeEmpty(), "choice %d should have no top_logprobs", i)
		}

		// Streaming: no chunk for any choice should carry logprobs content.
		streamParams := openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: prompts},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		}
		stream := openaiclient.Completions.NewStreaming(ctx, streamParams)
		defer func() { Expect(stream.Close()).To(Succeed()) }()
		for stream.Next() {
			chunk := stream.Current()
			for _, c := range chunk.Choices {
				Expect(c.Logprobs.Tokens).To(BeEmpty(),
					"streaming choice %d unexpectedly has logprobs tokens", c.Index)
			}
		}
		Expect(stream.Err()).NotTo(HaveOccurred())
	})

	It("text completions array prompt still serves new requests after a fail-fast abort", func() {
		// The existing fail-fast test verifies the client sees an error + [DONE].
		// This test covers the *follow-up*: after fail-fast triggers `drainResponseChannel`
		// and the original request's remaining sub-requests drain in the background,
		// the simulator must be ready to serve another request. Regression guard for
		// wg leaks, dangling worker state, or permanently-stuck waiting queue.
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--time-to-first-token", "1s",
			"--max-num-seqs", "1", "--max-waiting-queue-length", "1"}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		// First: trigger fail-fast via queue overflow on a 3-prompt array.
		_, err = openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: []string{prompt1, prompt2, "third"}},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		})
		Expect(err).To(HaveOccurred())
		var oaiErr *openai.Error
		Expect(errors.As(err, &oaiErr)).To(BeTrue())
		Expect(oaiErr.StatusCode).To(Equal(fasthttp.StatusTooManyRequests))

		// Give the background drain/wg bookkeeping time to complete before we probe.
		// 2× TTFT covers the worst case where the queued sub-request was already past TTFT.
		time.Sleep(2500 * time.Millisecond)

		// Follow-up: a single-prompt request must succeed end-to-end.
		followUp, err := openaiclient.Completions.New(ctx, openai.CompletionNewParams{
			Prompt: openai.CompletionNewParamsPromptUnion{OfString: param.NewOpt(prompt1)},
			Model:  openai.CompletionNewParamsModel(common.TestModelName),
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(followUp.Choices).To(HaveLen(1))
		Expect(followUp.Choices[0].Text).NotTo(BeEmpty())
	})

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
			Expect(string(resp.Object)).To(Equal(api.ChatCompletionObject))

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

	It("Should return ec_transfer_params on chat completions in MMEncoderOnly mode when messages contain images", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mm-encoder-only",
			"--mm-processor-kwargs", "args", "--ec-transfer-config", "cfg",
			"--enforce-eager", "--no-enable-prefix-caching"}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		reqBody := fmt.Sprintf(`{
				"model": "%s",
				"messages": [
					{"role": "user", "content": [
						{"type": "text", "text": "describe"},
						{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
						{"type": "image_url", "image_url": {"url": "https://example.com/b.png"}}
					]}
				],
				"max_tokens": 1
			}`, common.TestModelName)

		resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(reqBody))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			err := resp.Body.Close()
			Expect(err).NotTo(HaveOccurred())
		}()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var chatResp api.ChatCompletionsResponse
		Expect(json.Unmarshal(body, &chatResp)).To(Succeed())
		Expect(chatResp.Choices).To(HaveLen(1))
		Expect(chatResp.ECTransferParams).To(HaveLen(2))
		for _, params := range chatResp.ECTransferParams {
			Expect(params.PeerHost).NotTo(BeEmpty())
			Expect(params.PeerPort).To(BeNumerically(">", 0))
			Expect(params.SizeBytes).To(BeNumerically(">", 0))
			Expect(params.NixlAgentData).NotTo(BeEmpty())
		}
	})

	It("Should not return ec_transfer_params on chat completions when no images in request", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mm-encoder-only",
			"--mm-processor-kwargs", "args", "--ec-transfer-config", "cfg",
			"--enforce-eager", "--no-enable-prefix-caching"}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		reqBody := fmt.Sprintf(`{
				"model": "%s",
				"messages": [{"role": "user", "content": "hi"}],
				"max_tokens": 1
			}`, common.TestModelName)

		resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(reqBody))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			err := resp.Body.Close()
			Expect(err).NotTo(HaveOccurred())
		}()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var chatResp api.ChatCompletionsResponse
		Expect(json.Unmarshal(body, &chatResp)).To(Succeed())
		Expect(chatResp.ECTransferParams).To(BeNil())
	})

	It("Should not return ec_transfer_params on chat completions when MMEncoderOnly mode is disabled", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		reqBody := fmt.Sprintf(`{
				"model": "%s",
				"messages": [
					{"role": "user", "content": [
						{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
					]}
				],
				"max_tokens": 1
			}`, common.TestModelName)

		resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(reqBody))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			err := resp.Body.Close()
			Expect(err).NotTo(HaveOccurred())
		}()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var chatResp api.ChatCompletionsResponse
		Expect(json.Unmarshal(body, &chatResp)).To(Succeed())
		Expect(chatResp.ECTransferParams).To(BeNil())
	})

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

			openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, testUserMessage, false)
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

			openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, testUserMessage, true)
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

				openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, testUserMessage, true)
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
					openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, testUserMessage, false)
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
			model := common.TestModelName
			// Start server with max-model-len=10
			args := []string{"cmd", "--model", model, "--mode", common.ModeRandom, "--max-model-len", "10"}
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
			}`, prompt, model, maxTokens)

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
			openaiclient, params := getOpenAIClientAndChatParams(client, model, prompt, false)
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

			var errResp api.ErrorResponse
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

			var errResp api.ErrorResponse
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

				if strings.HasPrefix(line, api.SSEDataPrefix) {
					data := strings.TrimPrefix(line, api.SSEDataPrefix)
					if strings.TrimSpace(data) == api.SSEDoneMarker {
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
				"--time-to-first-token", "500ms", "--max-num-seqs", "1", "--max-waiting-queue-length", "2"}
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
		responseParts := []string{
			api.ResponsesEventCreated,
			api.ResponsesEventInProgress,
			api.ResponsesEventOutputItemAdded,
			api.ResponsesEventContentPartAdded,
			api.ResponsesEventTextDelta,
			api.ResponsesEventTextDone,
			api.ResponsesEventContentPartDone,
			api.ResponsesEventOutputItemDone,
			api.ResponsesEventCompleted}

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

				Expect(resp.ID).To(HavePrefix(api.ResponsesIDPrefix))
				Expect(resp.Status).To(Equal(responses.ResponseStatusCompleted))
				Expect(resp.Instructions.AsString()).To(BeEmpty())

				Expect(resp.Output).NotTo(BeEmpty())
				firstItem := resp.Output[0]
				Expect(string(firstItem.Role)).To(Equal("assistant"))
				Expect(firstItem.Content).NotTo(BeEmpty())
				Expect(firstItem.Content[0].Type).To(Equal(api.ResponsesOutputText))
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
					case api.ResponsesEventCreated:
						created := event.AsResponseCreated()
						Expect(string(created.Response.Status)).To(Equal(api.ResponsesStatusInProgress))
					case api.ResponsesEventOutputItemAdded:
						added := event.AsResponseOutputItemAdded()
						Expect(added.OutputIndex).To(Equal(int64(0)))
					case api.ResponsesEventTextDelta:
						delta := event.AsResponseOutputTextDelta()
						deltas = append(deltas, delta.Delta)
					case api.ResponsesEventTextDone:
						done := event.AsResponseOutputTextDone()
						Expect(done.Text).NotTo(BeEmpty())
						Expect(done.Text).To(Equal(strings.Join(deltas, "")))
					case api.ResponsesEventCompleted:
						completed := event.AsResponseCompleted()
						Expect(completed.Response.Usage.InputTokens).To(BeNumerically(">", 0))
						Expect(completed.Response.Usage.OutputTokens).To(BeNumerically(">", 0))
						Expect(completed.Response.Usage.TotalTokens).To(Equal(
							completed.Response.Usage.InputTokens + completed.Response.Usage.OutputTokens))
						Expect(string(completed.Response.Status)).To(Equal(api.ResponsesStatusCompleted))
					}
				}
				Expect(stream.Err()).NotTo(HaveOccurred())

				// Verify the mandatory fixed positions in the event sequence:
				// [0] created, [1] in_progress, [2] output_item.added, [3] content_part.added,
				// [4..n-5] deltas, [n-4] text.done, [n-3] content_part.done,
				// [n-2] output_item.done, [n-1] completed
				Expect(len(eventTypes)).To(BeNumerically(">=", 9), "expected at least 9 events")
				Expect(eventTypes[0]).To(Equal(api.ResponsesEventCreated))
				Expect(eventTypes[1]).To(Equal(api.ResponsesEventInProgress))
				Expect(eventTypes[2]).To(Equal(api.ResponsesEventOutputItemAdded))
				Expect(eventTypes[3]).To(Equal(api.ResponsesEventContentPartAdded))
				// deltas occupy positions [4 .. len-5]
				nDeltas := len(eventTypes) - 8
				Expect(nDeltas).To(BeNumerically(">=", 1), "expected at least one delta event")
				for i := 4; i < 4+nDeltas; i++ {
					Expect(eventTypes[i]).To(Equal(api.ResponsesEventTextDelta))
				}
				Expect(eventTypes[len(eventTypes)-4]).To(Equal(api.ResponsesEventTextDone))
				Expect(eventTypes[len(eventTypes)-3]).To(Equal(api.ResponsesEventContentPartDone))
				Expect(eventTypes[len(eventTypes)-2]).To(Equal(api.ResponsesEventOutputItemDone))
				Expect(eventTypes[len(eventTypes)-1]).To(Equal(api.ResponsesEventCompleted))

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
			Entry(nil, common.QwenModelName, common.ModeRandom),
			Entry(nil, common.QwenModelName, common.ModeEcho),
		)

		DescribeTable("responses with logprobs",
			func(includeLogprobs bool, topLogprobs int) {
				ctx := context.TODO()
				client, err := startServer(ctx, common.ModeEcho)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, testUserMessage)
				if includeLogprobs {
					params.Include = []responses.ResponseIncludable{responses.ResponseIncludableMessageOutputTextLogprobs}
					if topLogprobs > 0 {
						params.TopLogprobs = param.NewOpt(int64(topLogprobs))
					}
				}

				resp, err := openaiclient.Responses.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).NotTo(BeNil())
				Expect(resp.Output).NotTo(BeEmpty())

				contentItem := resp.Output[0].Content[0]
				if includeLogprobs {
					Expect(contentItem.Logprobs).NotTo(BeEmpty())
					Expect(contentItem.Logprobs[0].Token).NotTo(BeEmpty())
					Expect(contentItem.Logprobs[0].Logprob).To(BeNumerically("<=", 0))
					if topLogprobs > 0 {
						Expect(contentItem.Logprobs[0].TopLogprobs).To(HaveLen(topLogprobs))
						Expect(contentItem.Logprobs[0].TopLogprobs[0].Token).To(Equal(contentItem.Logprobs[0].Token))
					} else {
						Expect(contentItem.Logprobs[0].TopLogprobs).To(BeEmpty())
					}
				} else {
					Expect(contentItem.Logprobs).To(BeEmpty())
				}
			},
			func(includeLogprobs bool, topLogprobs int) string {
				return fmt.Sprintf("includeLogprobs: %t top_logprobs: %d", includeLogprobs, topLogprobs)
			},
			Entry(nil, true, 0),  // logprobs requested, no top alternatives
			Entry(nil, true, 2),  // logprobs requested, 2 top alternatives
			Entry(nil, false, 0), // logprobs not requested
		)

		DescribeTable("responses streaming with logprobs",
			func(includeLogprobs bool, topLogprobs int) {
				ctx := context.TODO()
				client, err := startServer(ctx, common.ModeEcho)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, testUserMessage)
				if includeLogprobs {
					params.Include = []responses.ResponseIncludable{responses.ResponseIncludableMessageOutputTextLogprobs}
					if topLogprobs > 0 {
						params.TopLogprobs = param.NewOpt(int64(topLogprobs))
					}
				}

				stream := openaiclient.Responses.NewStreaming(ctx, params)
				defer func() {
					Expect(stream.Close()).NotTo(HaveOccurred())
				}()

				deltaCount := 0
				deltaLogprobsCount := 0

				for stream.Next() {
					event := stream.Current()
					switch event.Type {
					case api.ResponsesEventTextDelta:
						delta := event.AsResponseOutputTextDelta()
						Expect(delta.Delta).NotTo(BeEmpty())
						deltaCount++
						if includeLogprobs {
							Expect(delta.JSON.Logprobs.Valid()).To(BeTrue(),
								"delta event should have logprobs field present")
							Expect(delta.Logprobs).NotTo(BeEmpty(),
								"delta event should have non-empty logprobs when requested")
							for _, lp := range delta.Logprobs {
								Expect(lp.Token).NotTo(BeEmpty())
								Expect(lp.Logprob).To(BeNumerically("<=", 0))
								if topLogprobs > 0 {
									Expect(lp.TopLogprobs).To(HaveLen(topLogprobs))
								}
							}
							deltaLogprobsCount++
						} else {
							Expect(delta.Logprobs).To(BeEmpty(),
								"delta event should have no logprobs when not requested")
						}
					case api.ResponsesEventTextDone:
						done := event.AsResponseOutputTextDone()
						Expect(done.Text).NotTo(BeEmpty())
						if includeLogprobs {
							Expect(done.JSON.Logprobs.Valid()).To(BeTrue(),
								"done event should have logprobs field present (as [])")
							Expect(done.Logprobs).To(BeEmpty(),
								"done event logprobs should be empty array, not populated")
						} else {
							Expect(done.JSON.Logprobs.Valid()).To(BeFalse(),
								"done event should not have logprobs field when not requested")
						}
					}
				}
				Expect(stream.Err()).NotTo(HaveOccurred())

				Expect(deltaCount).To(BeNumerically(">", 0), "should have received delta events")
				if includeLogprobs {
					Expect(deltaLogprobsCount).To(Equal(deltaCount),
						"all delta events should have logprobs when requested")
				}
			},
			func(includeLogprobs bool, topLogprobs int) string {
				return fmt.Sprintf("includeLogprobs: %t top_logprobs: %d", includeLogprobs, topLogprobs)
			},
			Entry(nil, true, 0),  // logprobs requested, no top alternatives
			Entry(nil, true, 2),  // logprobs requested, 2 top alternatives
			Entry(nil, false, 0), // logprobs not requested
		)

		DescribeTable("responses streaming logprobs per chunk type",
			func(includeLogprobs bool, topLogprobs int) {
				ctx := context.TODO()
				client, err := startServer(ctx, common.ModeEcho)
				Expect(err).NotTo(HaveOccurred())

				reqBody := fmt.Sprintf(`{"model":%q,"input":%q,"stream":true`, common.TestModelName, testUserMessage)
				if includeLogprobs {
					reqBody += `,"include":["message.output_text.logprobs"]`
					if topLogprobs > 0 {
						reqBody += fmt.Sprintf(`,"top_logprobs":%d`, topLogprobs)
					}
				}
				reqBody += "}"

				req, err := http.NewRequest("POST", "http://localhost/v1/responses", strings.NewReader(reqBody))
				Expect(err).NotTo(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				httpResp, err := client.Do(req)
				Expect(err).NotTo(HaveOccurred())
				defer func() { Expect(httpResp.Body.Close()).To(Succeed()) }()
				Expect(httpResp.StatusCode).To(Equal(http.StatusOK))

				checkLogprobsMissing := func(part map[string]any, partType string) {
					_, ok := part["logprobs"]
					Expect(ok).To(BeFalse(), partType+": logprobs must be absent when not requested")
				}
				checkLogprobEmpty := func(partObj map[string]any, partType string, isNullExpected bool) {
					if includeLogprobs {
						logprobs, ok := partObj["logprobs"]
						Expect(ok).To(BeTrue(), partType+": part.logprobs must be present when requested")
						if isNullExpected {
							Expect(logprobs).To(BeNil(), partType+": part.logprobs must be null")
						} else {
							Expect(logprobs.([]any)).To(BeEmpty(), partType+": part.logprobs must be empty []")
						}
					} else {
						checkLogprobsMissing(partObj, partType)
					}
				}

				seenTypes := map[string]bool{}
				var deltaLogprobs []any
				reader := bufio.NewReader(httpResp.Body)
				for {
					line, err := reader.ReadString('\n')
					if err == io.EOF {
						break
					}
					Expect(err).NotTo(HaveOccurred())
					if !strings.HasPrefix(line, api.SSEDataPrefix) {
						continue
					}
					data := strings.TrimSpace(strings.TrimPrefix(line, api.SSEDataPrefix))
					if data == api.SSEDoneMarker {
						break
					}
					var event map[string]any
					Expect(json.Unmarshal([]byte(data), &event)).To(Succeed())
					eventType, _ := event["type"].(string)
					seenTypes[eventType] = true

					switch eventType {
					case api.ResponsesEventContentPartAdded:
						// part.logprobs: [] when requested, absent when not
						partObj, _ := event["part"].(map[string]any)
						checkLogprobEmpty(partObj, eventType, false)

					case api.ResponsesEventTextDelta:
						// logprobs: populated when requested, absent when not
						if includeLogprobs {
							logprobsArr, _ := event["logprobs"].([]any)
							Expect(logprobsArr).NotTo(BeEmpty(), "text.delta: logprobs must be non-empty")
							for _, lp := range logprobsArr {
								lpMap, _ := lp.(map[string]any)
								Expect(lpMap["token"]).NotTo(BeEmpty())
								Expect(lpMap["logprob"].(float64)).To(BeNumerically("<=", 0))
								if topLogprobs > 0 {
									Expect(lpMap["top_logprobs"].([]any)).To(HaveLen(topLogprobs))
								}
							}
							deltaLogprobs = append(deltaLogprobs, logprobsArr...)
						} else {
							checkLogprobsMissing(event, eventType)
						}

					case api.ResponsesEventTextDone:
						checkLogprobEmpty(event, eventType, false)

					case api.ResponsesEventContentPartDone:
						// part.logprobs: null when requested (signals per-token entries already streamed), absent when not
						partObj, _ := event["part"].(map[string]any)
						checkLogprobEmpty(partObj, eventType, true)

					case api.ResponsesEventOutputItemDone:
						// item.content[0].logprobs: null when requested, absent when not
						item, ok := event["item"].(map[string]any)
						Expect(ok).To(BeTrue(), "output_item.done: event.item must be a map")
						contentArr, ok := item["content"].([]any)
						Expect(ok).To(BeTrue(), "output_item.done: item.content must be an array")
						Expect(contentArr).NotTo(BeEmpty(), "output_item.done: item.content must not be empty")
						firstContent, ok := contentArr[0].(map[string]any)
						Expect(ok).To(BeTrue(), "output_item.done: item.content[0] must be a map")
						checkLogprobEmpty(firstContent, eventType, true)

					case api.ResponsesEventCompleted:
						// response.output[0].content[0].logprobs: accumulated entries when requested, absent when not
						response, ok := event["response"].(map[string]any)
						Expect(ok).To(BeTrue(), "completed: event.response must be a map")
						outputArr, ok := response["output"].([]any)
						Expect(ok).To(BeTrue(), "completed: response.output must be an array")
						Expect(outputArr).NotTo(BeEmpty(), "completed: response.output must not be empty")
						firstOutput, ok := outputArr[0].(map[string]any)
						Expect(ok).To(BeTrue(), "completed: response.output[0] must be a map")
						contentArr, ok := firstOutput["content"].([]any)
						Expect(ok).To(BeTrue(), "completed: response.output[0].content must be an array")
						Expect(contentArr).NotTo(BeEmpty(), "completed: response.output[0].content must not be empty")
						firstContent, ok := contentArr[0].(map[string]any)
						Expect(ok).To(BeTrue(), "completed: response.output[0].content[0] must be a map")
						if includeLogprobs {
							logprobsArr, _ := firstContent["logprobs"].([]any)
							Expect(logprobsArr).NotTo(BeEmpty(), "completed: content[0].logprobs must have accumulated entries")
							Expect(logprobsArr).To(HaveLen(len(deltaLogprobs)),
								"completed: accumulated logprobs count must equal sum of all delta logprobs")
							for i, lp := range logprobsArr {
								Expect(lp).To(Equal(deltaLogprobs[i]),
									"completed: logprobs[%d] must match the corresponding delta logprob entry", i)
							}
						} else {
							checkLogprobsMissing(firstContent, eventType)
						}
					}
				}

				// check that all chunk types were received
				for _, et := range responseParts {
					Expect(seenTypes[et]).To(BeTrue(), "event type %q was not received in the stream", et)
				}
			},
			func(includeLogprobs bool, topLogprobs int) string {
				return fmt.Sprintf("includeLogprobs: %t top_logprobs: %d", includeLogprobs, topLogprobs)
			},
			Entry(nil, true, 2),  // logprobs with 2 top alternatives
			Entry(nil, true, 0),  // logprobs with no top alternatives
			Entry(nil, false, 0), // no logprobs: logprobs fields must be absent in all chunks
		)
	})

	Context("generate API", func() {
		DescribeTable("Should return correct response to /inference/v1/generate",
			func(model string) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", model, "--mode", common.ModeRandom}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				reqBody := fmt.Sprintf(`{
					"model": "%s",
					"token_ids": [1, 2, 3, 4],
					"sampling_params": {"max_tokens": 5}
				}`, model)
				resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					err := resp.Body.Close()
					Expect(err).NotTo(HaveOccurred())
				}()

				Expect(resp.StatusCode).To(Equal(http.StatusOK))

				body, err := io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())

				var generateResp api.GenerateResponse
				Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
				Expect(generateResp.GenRequestID).NotTo(BeEmpty())
				Expect(generateResp.Choices).To(HaveLen(1))
				Expect(generateResp.Choices[0].FinishReason).NotTo(BeNil())
				Expect(generateResp.Choices[0].TokenIDs).NotTo(BeEmpty())
				Expect(int64(len(generateResp.Choices[0].TokenIDs))).To(BeNumerically("<=", 5))
			},
			func(model string) string {
				return "model: " + model
			},
			Entry(nil, common.TestModelName),
			Entry(nil, common.QwenModelName),
		)

		DescribeTable("Should return 400 when required fields are missing",
			func(reqBody string, expectedErrMsg string) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					err := resp.Body.Close()
					Expect(err).NotTo(HaveOccurred())
				}()

				Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

				body, err := io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())
				Expect(string(body)).To(ContainSubstring(expectedErrMsg))
			},
			Entry("missing token_ids",
				fmt.Sprintf(`{"model": "%s", "sampling_params": {"max_tokens": 5}}`, common.TestModelName),
				"Missing input token_ids",
			),
			Entry("missing sampling_params",
				fmt.Sprintf(`{"model": "%s", "token_ids": [1, 2, 3]}`, common.TestModelName),
				"Missing sampling_params field",
			),
		)

		It("Should return ec_transfer_params in MMEncoderOnly mode when features are present", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mm-encoder-only"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 32000, 32000, 32000],
				"features": {
					"mm_hashes": {"image": ["abc123hash", "def456hash"]},
					"mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
					"kwargs_data": {"image": ["<base64-encoded-pixel-tensor-1>"]}
				},
				"sampling_params": {"max_tokens": 1}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.GenRequestID).NotTo(BeEmpty())
			Expect(generateResp.Choices).To(HaveLen(1))
			Expect(generateResp.ECTransferParams).To(HaveLen(2))
			Expect(generateResp.ECTransferParams).To(HaveKey("abc123hash"))
			Expect(generateResp.ECTransferParams).To(HaveKey("def456hash"))
			Expect(generateResp.ECTransferParams["abc123hash"].PeerPort).To(BeNumerically(">", 0))
		})

		It("Should not return ec_transfer_params when features are nil", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mm-encoder-only"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 1}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.ECTransferParams).To(BeNil())
		})

		It("Should return kv_transfer_params when do_remote_decode is true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5},
				"kv_transfer_params": {"do_remote_decode": true}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.KVParams).NotTo(BeNil())
			Expect(generateResp.KVParams.DoRemotePrefill).To(BeTrue())
			Expect(generateResp.KVParams.DoRemoteDecode).To(BeFalse())
			Expect(generateResp.KVParams.RemoteHost).NotTo(BeEmpty())
			Expect(generateResp.KVParams.RemotePort).To(BeNumerically(">", 0))
			Expect(generateResp.KVParams.RemoteBlockIds).NotTo(BeEmpty())
			Expect(generateResp.KVParams.RemoteEngineId).NotTo(BeEmpty())
		})

		It("Should not return kv_transfer_params when do_remote_decode is absent", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.KVParams).To(BeNil())
		})

		It("Should return kv_transfer_params when do_remote_decode is true inside sampling_params.extra_args", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5, "extra_args": {"kv_transfer_params": {"do_remote_decode": true}}}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.KVParams).NotTo(BeNil())
			Expect(generateResp.KVParams.DoRemotePrefill).To(BeTrue())
			Expect(generateResp.KVParams.DoRemoteDecode).To(BeFalse())
			Expect(generateResp.KVParams.RemoteHost).NotTo(BeEmpty())
			Expect(generateResp.KVParams.RemotePort).To(BeNumerically(">", 0))
			Expect(generateResp.KVParams.RemoteBlockIds).NotTo(BeEmpty())
			Expect(generateResp.KVParams.RemoteEngineId).NotTo(BeEmpty())
		})

		It("Should stream SSE chunks for /inference/v1/generate with stream=true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5},
				"stream": true
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			Expect(resp.Header.Get("Content-Type")).To(Equal("text/event-stream"))

			reader := bufio.NewReader(resp.Body)
			var tokenChunks []api.GenerateStreamResponse
			var finishChunk *api.GenerateStreamResponse
			var usageChunk *api.GenerateStreamResponse
			gotDone := false

			for {
				line, err := reader.ReadString('\n')
				if err == io.EOF {
					break
				}
				Expect(err).NotTo(HaveOccurred())

				if !strings.HasPrefix(line, api.SSEDataPrefix) {
					continue
				}
				data := strings.TrimSpace(strings.TrimPrefix(line, api.SSEDataPrefix))
				if data == api.SSEDoneMarker {
					gotDone = true
					break
				}

				var streamResp api.GenerateStreamResponse
				Expect(json.Unmarshal([]byte(data), &streamResp)).To(Succeed(), "failed to parse SSE chunk: %s", data)
				if len(streamResp.Choices) == 0 {
					Expect(streamResp.Usage).NotTo(BeNil(), "empty choices chunk must carry usage")
					usageChunk = &streamResp
					continue
				}
				choice := streamResp.Choices[0]
				if choice.TokenIDs != nil {
					tokenChunks = append(tokenChunks, streamResp)
				}
				if choice.FinishReason != nil {
					finishChunk = &streamResp
				}
			}

			Expect(tokenChunks).NotTo(BeEmpty(), "should have received at least one streaming chunk with token_ids")
			for _, tc := range tokenChunks {
				Expect(tc.RequestID).NotTo(BeEmpty())
				Expect(tc.Choices[0].TokenIDs).NotTo(BeEmpty())
			}

			Expect(finishChunk).NotTo(BeNil(), "should have received a chunk with finish_reason")
			Expect(finishChunk.RequestID).NotTo(BeEmpty())
			Expect(*finishChunk.Choices[0].FinishReason).NotTo(BeEmpty())
			Expect(finishChunk.Choices[0].TokenIDs).NotTo(BeNil(), "finish_reason must be in the last token chunk, not a separate empty chunk")
			Expect(finishChunk.Usage).To(BeNil(), "finish chunk should not carry usage")

			Expect(usageChunk).To(BeNil(), "should not receive usage chunk without stream_options.include_usage")

			Expect(gotDone).To(BeTrue(), "stream should end with [DONE]")
		})

		It("Should send length finish_reason in last token chunk for generate streaming", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			// max_tokens: 1 forces length finish reason
			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 1},
				"stream": true
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			reader := bufio.NewReader(resp.Body)
			var lastTokenChunk *api.GenerateStreamResponse
			gotDone := false

			for {
				line, err := reader.ReadString('\n')
				if err == io.EOF {
					break
				}
				Expect(err).NotTo(HaveOccurred())

				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				data := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
				if data == api.SSEDoneMarker {
					gotDone = true
					break
				}

				var streamResp api.GenerateStreamResponse
				Expect(json.Unmarshal([]byte(data), &streamResp)).To(Succeed(), "failed to parse SSE chunk: %s", data)
				if len(streamResp.Choices) == 0 {
					continue
				}
				choice := streamResp.Choices[0]
				if choice.TokenIDs != nil {
					lastTokenChunk = &streamResp
				}
				// finish_reason must never appear in a separate empty chunk
				if choice.FinishReason != nil {
					Expect(choice.TokenIDs).NotTo(BeNil(), "finish_reason must be carried by a token chunk, not a separate empty chunk")
				}
			}

			Expect(lastTokenChunk).NotTo(BeNil(), "should have received at least one token chunk")
			Expect(lastTokenChunk.Choices[0].FinishReason).NotTo(BeNil(), "last token chunk should carry finish_reason")
			Expect(*lastTokenChunk.Choices[0].FinishReason).To(Equal(common.LengthFinishReason))
			Expect(gotDone).To(BeTrue(), "stream should end with [DONE]")
		})

		It("Should include usage chunk when stream_options.include_usage is true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5},
				"stream": true,
				"stream_options": {"include_usage": true}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			reader := bufio.NewReader(resp.Body)
			var usageChunk *api.GenerateStreamResponse
			gotDone := false

			for {
				line, err := reader.ReadString('\n')
				if err == io.EOF {
					break
				}
				Expect(err).NotTo(HaveOccurred())

				if !strings.HasPrefix(line, api.SSEDataPrefix) {
					continue
				}
				data := strings.TrimSpace(strings.TrimPrefix(line, api.SSEDataPrefix))
				if data == api.SSEDoneMarker {
					gotDone = true
					break
				}

				var streamResp api.GenerateStreamResponse
				Expect(json.Unmarshal([]byte(data), &streamResp)).To(Succeed(), "failed to parse SSE chunk: %s", data)
				if len(streamResp.Choices) == 0 {
					Expect(streamResp.Usage).NotTo(BeNil(), "empty choices chunk must carry usage")
					usageChunk = &streamResp
					continue
				}
			}

			Expect(usageChunk).NotTo(BeNil(), "should have received a usage chunk with choices:[]")
			Expect(usageChunk.Choices).To(BeEmpty(), "usage chunk should have empty choices")
			Expect(usageChunk.Usage).NotTo(BeNil())
			Expect(usageChunk.Usage.PromptTokens).To(BeNumerically(">", 0))
			Expect(usageChunk.Usage.CompletionTokens).To(BeNumerically(">", 0))

			Expect(gotDone).To(BeTrue(), "stream should end with [DONE]")
		})
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
			Expect(storedCount).To(Equal(5))
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

				openaiclient, params := getOpenAIClientAndCompletionParams(client, model, longPrompt, false)
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

		Context("force-dummy-tokenizer flag", func() {
			It("should use dummy tokenizer when flag is set with real model", func() {
				ctx := context.TODO()
				// Use a real model name but force dummy tokenizer
				args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom, "--force-dummy-tokenizer"}
				simulator, _, _, err := startServerHandle(ctx, "", args, nil)
				Expect(err).NotTo(HaveOccurred())

				// Verify that the dummy tokenizer was actually created
				Expect(simulator.Context.Tokenizer).To(BeAssignableToTypeOf(&tokenizer.SimpleTokenizer{}))
			})

			It("should work with YAML config file", func() {
				ctx := context.TODO()
				// Create a temporary config file with force-dummy-tokenizer set
				configContent := `model: ` + common.QwenModelName + `
mode: random
force-dummy-tokenizer: true
`
				configFile := "/tmp/test-tokenizer-config.yaml"
				err := writeTestConfig(configFile, configContent)
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					err := removeTestConfig(configFile)
					Expect(err).NotTo(HaveOccurred())
				}()

				args := []string{"cmd", "--config", configFile}
				simulator, _, _, err := startServerHandle(ctx, "", args, nil)
				Expect(err).NotTo(HaveOccurred())

				// Verify that the dummy tokenizer was actually created
				Expect(simulator.Context.Tokenizer).To(BeAssignableToTypeOf(&tokenizer.SimpleTokenizer{}))
			})

			It("should override YAML config with command line flag", func() {
				ctx := context.TODO()
				// Create a config file with force-dummy-tokenizer set to false
				configContent := `model: ` + common.QwenModelName + `
mode: random
force-dummy-tokenizer: false
`
				configFile := "/tmp/test-tokenizer-override-config.yaml"
				err := writeTestConfig(configFile, configContent)
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					err := removeTestConfig(configFile)
					Expect(err).NotTo(HaveOccurred())
				}()

				// Override with command line flag
				args := []string{"cmd", "--config", configFile, "--force-dummy-tokenizer"}
				simulator, _, _, err := startServerHandle(ctx, "", args, nil)
				Expect(err).NotTo(HaveOccurred())

				// Verify that the dummy tokenizer was actually created
				Expect(simulator.Context.Tokenizer).To(BeAssignableToTypeOf(&tokenizer.SimpleTokenizer{}))
			})
		})
	})

	Context("Mooncake bootstrap query", func() {
		queryEngines := func(client *http.Client) map[string]map[string]string {
			resp, err := client.Get("http://localhost/query")
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				Expect(resp.Body.Close()).To(Succeed())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			engines := map[string]map[string]string{}
			Expect(json.Unmarshal(body, &engines)).To(Succeed())
			return engines
		}

		It("Should return a dp_rank to engine_id map on /query", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			engines := queryEngines(client)
			Expect(engines).To(HaveKey("0"))
			Expect(engines["0"]).To(HaveKey("engine_id"))
			Expect(engines["0"]["engine_id"]).NotTo(BeEmpty())
		})

		It("Should return the same engine ids by multiple calls", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			firstCall := queryEngines(client)
			secondCall := queryEngines(client)
			Expect(secondCall).To(Equal(firstCall))
		})
	})
})
