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
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/valyala/fasthttp"
)

const prompt1 = "What is the weather like in New York today?"
const prompt2 = "I hear it's very cold."

var _ = Describe("Simulator", func() {

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

	DescribeTable("non-streaming text completions with logprobs",
		func(mode string, logprobsParam interface{}) {
			ctx := context.TODO()
			server, _, client, err := startServerHandle(ctx, mode, nil, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, testUserMessage, false)
			if logprobsParam != nil {
				if logprobsCount, ok := logprobsParam.(int); ok && logprobsCount > 0 {
					params.Logprobs = param.NewOpt(int64(logprobsCount))
				}
			}
			textResp, err := openaiclient.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(textResp.Choices).ShouldNot(BeEmpty())

			if logprobsParam != nil {
				Expect(textResp.Choices[0].Logprobs.Tokens).NotTo(BeNil())
				_, tokens, err := server.Context.Tokenizer.RenderText(textResp.Choices[0].Text)
				Expect(err).NotTo(HaveOccurred())
				Expect(textResp.Choices[0].Logprobs.Tokens).To(HaveLen(len(tokens)))
			} else {
				Expect(textResp.Choices[0].Logprobs.Tokens).To(BeNil())
			}
		},
		func(mode string, logprobsParam interface{}) string {
			return fmt.Sprintf("mode: %s logprobs: %v", mode, logprobsParam)
		},
		Entry(nil, common.ModeEcho, 2),   // with logprobs=2
		Entry(nil, common.ModeEcho, nil), // without logprobs
	)
})
