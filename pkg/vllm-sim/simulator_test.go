/*
Copyright 2025 The vLLM-Sim Authors.

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

package vllmsim

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/valyala/fasthttp/fasthttputil"
	"k8s.io/klog/v2"
)

const model = "my_model"
const baseURL = "http://localhost/v1"
const textPrompt = "This is a test."

var chatPrompt = map[string]string{
	"system": "You are a helpful assistant.",
	"user":   "What is the best pie?",
}

func startServer(ctx context.Context, mode string) (*http.Client, error) {
	oldArgs := os.Args
	defer func() {
		os.Args = oldArgs
	}()
	os.Args = []string{"cmd", "--model", model, "--mode", mode}
	logger := klog.Background()

	s := New(logger)
	// parse command line parameters
	if err := s.parseCommandParams(); err != nil {
		return nil, err
	}

	// run request processing workers
	for i := 1; i <= int(s.maxRunningReqs); i++ {
		go s.reqProcessingWorker(ctx, i)
	}

	listener := fasthttputil.NewInmemoryListener()

	// start the http server
	go func() {
		if err := s.startServer(listener); err != nil {
			logger.Error(err, "error starting server")
		}
	}()

	return &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				return listener.Dial()
			},
		},
	}, nil
}

var _ = Describe("Simulator", func() {

	DescribeTable("chat completions streaming",
		func(mode string, includeUsage bool) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.SystemMessage(chatPrompt["system"]),
					openai.UserMessage(chatPrompt["user"]),
				},
				Model: model,
			}
			if includeUsage {
				params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
			}
			stream := openaiclient.Chat.Completions.NewStreaming(ctx, params)
			defer func() {
				err := stream.Close()
				Expect(err).NotTo(HaveOccurred())
			}()
			chunks := []openai.ChatCompletionChunk{}
			for stream.Next() {
				chunks = append(chunks, stream.Current())
			}
			Expect(len(chunks)).Should(BeNumerically(">", 2))

			tokens := []string{}
			role := ""
			finishReason := ""
			for idx, chunk := range chunks {
				Expect(chunk.ID).Should(HavePrefix(chatComplIdPrefix))
				usage := chunk.Usage
				Expect(usage.TotalTokens).Should(Equal(usage.PromptTokens + usage.CompletionTokens))
				if includeUsage && idx == len(chunks)-1 {
					// chunk with the usage result
					Expect(len(chunk.Choices)).Should(Equal(0))
					Expect(usage.TotalTokens).Should(BeNumerically(">", int64(0)))
					fmt.Printf("prompt_tokens %d completion_tokens %d total_tokens %d",
						usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
				} else {
					Expect(len(chunk.Choices)).Should(BeNumerically(">", 0))
					Expect(usage.TotalTokens).Should(Equal(int64(0)))
				}
				for _, choice := range chunk.Choices {
					if choice.Delta.Role != "" {
						role = choice.Delta.Role
					} else if choice.FinishReason == "" {
						tokens = append(tokens, choice.Delta.Content)
					} else {
						finishReason = choice.FinishReason
					}
				}
			}
			Expect(finishReason).Should(Equal(stopFinishReason))
			msg := strings.Join(tokens, " ")
			expectedMsg := ""
			if mode == modeEcho {
				expectedMsg = chatPrompt["user"]
			} else {
				expectedMsg = strings.Trim(getFullTextFromPartialString(msg), " ")
			}
			Expect(role).Should(Equal("assistant"))
			Expect(msg).Should(Equal(expectedMsg))
		},
		func(mode string, includeUsage bool) string {
			return fmt.Sprintf("mode: %s include usage: %t", mode, includeUsage)
		},
		Entry(nil, modeRandom, false),
		Entry(nil, modeRandom, true),
		Entry(nil, modeEcho, false),
		Entry(nil, modeEcho, true),
	)

	DescribeTable("text completions streaming",
		func(mode string, includeUsage bool) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.CompletionNewParams{
				Prompt: openai.CompletionNewParamsPromptUnion{
					OfString: openai.String(textPrompt),
				},
				Model: openai.CompletionNewParamsModel(model),
			}
			if includeUsage {
				params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
			}
			stream := openaiclient.Completions.NewStreaming(ctx, params)
			defer func() {
				err := stream.Close()
				Expect(err).NotTo(HaveOccurred())
			}()
			chunks := []openai.Completion{}
			for stream.Next() {
				chunks = append(chunks, stream.Current())
			}
			Expect(len(chunks)).Should(BeNumerically(">", 2))

			tokens := []string{}
			var finishReason openai.CompletionChoiceFinishReason
			for idx, chunk := range chunks {
				Expect(chunk.ID).Should(HavePrefix(textComplIdPrefix))
				usage := chunk.Usage
				Expect(usage.TotalTokens).Should(Equal(usage.PromptTokens + usage.CompletionTokens))
				if includeUsage && idx == len(chunks)-1 {
					// chunk with the usage result
					Expect(len(chunk.Choices)).Should(Equal(0))
					Expect(usage.TotalTokens).Should(BeNumerically(">", int64(0)))
					fmt.Printf("prompt_tokens %d completion_tokens %d total_tokens %d",
						usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
				} else {
					Expect(len(chunk.Choices)).Should(BeNumerically(">", 0))
					Expect(usage.TotalTokens).Should(Equal(int64(0)))
				}
				for _, choice := range chunk.Choices {
					if choice.FinishReason == "" {
						tokens = append(tokens, choice.Text)
					} else {
						finishReason = choice.FinishReason
					}
				}
			}
			Expect(string(finishReason)).Should(Equal(stopFinishReason))
			text := strings.Join(tokens, " ")
			expectedText := ""
			if mode == modeEcho {
				expectedText = textPrompt
			} else {
				expectedText = strings.Trim(getFullTextFromPartialString(text), " ")
			}
			Expect(text).Should(Equal(expectedText))
		},
		func(mode string, includeUsage bool) string {
			return fmt.Sprintf("mode: %s include usage: %t", mode, includeUsage)
		},
		Entry(nil, modeRandom, false),
		Entry(nil, modeRandom, true),
		Entry(nil, modeEcho, false),
		Entry(nil, modeEcho, true),
	)

	DescribeTable("chat completions",
		func(mode string, maxTokens int, maxCompletionTokens int) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.SystemMessage(chatPrompt["system"]),
					openai.UserMessage(chatPrompt["user"]),
				},
				Model: model,
			}
			num_tokens := 0
			partialErrMsg := ""
			// if maxTokens and maxCompletionTokens are passsed
			// maxCompletionTokens is used
			if maxTokens != 0 {
				params.MaxTokens = param.NewOpt(int64(maxTokens))
				num_tokens = maxTokens
				partialErrMsg = "max_tokens must be at least 1, got -1"
			}
			if maxCompletionTokens != 0 {
				params.MaxCompletionTokens = param.NewOpt(int64(maxCompletionTokens))
				num_tokens = maxCompletionTokens
				partialErrMsg = "max_completion_tokens must be at least 1, got -1"
			}
			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			if err != nil {
				var openaiError *openai.Error
				if errors.As(err, &openaiError) {
					if openaiError.StatusCode == 400 {
						errMsg, err := io.ReadAll(openaiError.Response.Body)
						Expect(err).NotTo(HaveOccurred())
						if strings.Contains(string(errMsg), partialErrMsg) {
							return
						}
					}
				}
			}
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.ID).Should(HavePrefix(chatComplIdPrefix))
			Expect(len(resp.Choices)).Should(BeNumerically(">", 0))

			msg := resp.Choices[0].Message.Content
			Expect(msg).ShouldNot(BeEmpty())

			if num_tokens > 0 {
				tokens := strings.Fields(msg)
				Expect(int64(len(tokens))).Should(BeNumerically("<=", num_tokens))
			} else {
				expectedMsg := ""
				if mode == modeEcho {
					expectedMsg = chatPrompt["user"]
				} else {
					expectedMsg = getFullTextFromPartialString(msg)
				}
				Expect(msg).Should(Equal(expectedMsg))
			}
		},
		func(mode string, maxTokens int, maxCompletionTokens int) string {
			return fmt.Sprintf("mode: %s max_tokens: %d max_completion_tokens: %d", mode, maxTokens, maxCompletionTokens)
		},
		Entry(nil, modeRandom, 2, 0),
		Entry(nil, modeEcho, 2, 0),
		Entry(nil, modeRandom, 1000, 0),
		Entry(nil, modeEcho, 1000, 0),
		Entry(nil, modeRandom, 1000, 2),
		Entry(nil, modeEcho, 1000, 2),
		Entry(nil, modeRandom, 0, 2),
		Entry(nil, modeEcho, 0, 2),
		Entry(nil, modeRandom, 0, 1000),
		Entry(nil, modeEcho, 0, 1000),
		Entry(nil, modeRandom, 0, 0),
		Entry(nil, modeEcho, 0, 0),
		Entry(nil, modeRandom, -1, 0),
		Entry(nil, modeEcho, -1, 0),
		Entry(nil, modeRandom, 0, -1),
		Entry(nil, modeEcho, 0, -1),
	)

	DescribeTable("text completions",
		// use a function so that httpClient is captured when running
		func(mode string, maxTokens int) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))
			params := openai.CompletionNewParams{
				Prompt: openai.CompletionNewParamsPromptUnion{
					OfString: openai.String(textPrompt),
				},
				Model: openai.CompletionNewParamsModel(model),
			}
			num_tokens := 0
			partialErrMsg := "max_tokens must be at least 1, got -1"
			if maxTokens != 0 {
				params.MaxTokens = param.NewOpt(int64(maxTokens))
				num_tokens = maxTokens
			}
			resp, err := openaiclient.Completions.New(ctx, params)
			if err != nil {
				var openaiError *openai.Error
				if errors.As(err, &openaiError) {
					if openaiError.StatusCode == 400 {
						errMsg, err := io.ReadAll(openaiError.Response.Body)
						Expect(err).NotTo(HaveOccurred())
						if strings.Contains(string(errMsg), partialErrMsg) {
							return
						}
					}
				}
			}
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.ID).Should(HavePrefix(textComplIdPrefix))
			Expect(len(resp.Choices)).Should(BeNumerically(">", 0))

			text := resp.Choices[0].Text
			Expect(text).ShouldNot(BeEmpty())

			if num_tokens != 0 {
				tokens := strings.Fields(text)
				Expect(int64(len(tokens))).Should(BeNumerically("<=", num_tokens))
			} else {
				expectedText := ""
				if mode == modeEcho {
					expectedText = textPrompt
				} else {
					expectedText = getFullTextFromPartialString(text)
				}
				Expect(text).Should(Equal(expectedText))
			}
		},
		func(mode string, maxTokens int) string {
			return fmt.Sprintf("mode: %s max_tokens: %d", mode, maxTokens)
		},
		Entry(nil, modeRandom, 2),
		Entry(nil, modeEcho, 2),
		Entry(nil, modeRandom, 1000),
		Entry(nil, modeEcho, 1000),
		Entry(nil, modeRandom, 0),
		Entry(nil, modeEcho, 0),
		Entry(nil, modeRandom, -1),
		Entry(nil, modeEcho, -1),
	)
})
