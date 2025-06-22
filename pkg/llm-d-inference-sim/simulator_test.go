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
const userMessage = "This is a test."

func startServer(ctx context.Context, mode string) (*http.Client, error) {
	oldArgs := os.Args
	defer func() {
		os.Args = oldArgs
	}()
	os.Args = []string{"cmd", "--model", model, "--mode", mode}
	logger := klog.Background()

	s, err := New(logger)
	if err != nil {
		return nil, err
	}
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
			DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
				return listener.Dial()
			},
		},
	}, nil
}

var _ = Describe("Simulator", func() {

	DescribeTable("chat completions streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage(userMessage),
				},
				Model:         model,
				StreamOptions: openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)},
			}
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
			Expect(chunk.Usage.PromptTokens).To(Equal(int64(4)))
			Expect(chunk.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(chunk.Usage.TotalTokens).To(Equal(chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens))

			msg := strings.Join(tokens, "")
			expectedMsg := userMessage
			if mode == modeRandom {
				expectedMsg = getFullTextFromPartialString(msg)
			}
			Expect(role).Should(Equal("assistant"))
			Expect(msg).Should(Equal(expectedMsg))
		},
		func(mode string) string {
			return "mode: " + mode
		},
		Entry(nil, modeRandom),
		Entry(nil, modeEcho),
	)

	DescribeTable("text completions streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.CompletionNewParams{
				Prompt: openai.CompletionNewParamsPromptUnion{
					OfString: openai.String(userMessage),
				},
				Model:         openai.CompletionNewParamsModel(model),
				StreamOptions: openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)},
			}
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
			Expect(chunk.Usage.PromptTokens).To(Equal(int64(4)))
			Expect(chunk.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(chunk.Usage.TotalTokens).To(Equal(chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens))

			text := strings.Join(tokens, "")
			expectedText := userMessage
			if mode == modeRandom {
				expectedText = getFullTextFromPartialString(text)
			}
			Expect(text).Should(Equal(expectedText))
		},
		func(mode string) string {
			return "mode: " + mode
		},
		Entry(nil, modeRandom),
		Entry(nil, modeEcho),
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
					openai.UserMessage(userMessage),
				},
				Model: model,
			}
			numTokens := 0
			partialErrMsg := ""
			// if maxTokens and maxCompletionTokens are passsed
			// maxCompletionTokens is used
			if maxTokens != 0 {
				params.MaxTokens = param.NewOpt(int64(maxTokens))
				numTokens = maxTokens
				partialErrMsg = "max_tokens must be at least 1, got -1"
			}
			if maxCompletionTokens != 0 {
				params.MaxCompletionTokens = param.NewOpt(int64(maxCompletionTokens))
				numTokens = maxCompletionTokens
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
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(int64(4)))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			msg := resp.Choices[0].Message.Content
			Expect(msg).ShouldNot(BeEmpty())

			if numTokens > 0 {
				tokens := strings.Fields(msg)
				Expect(int64(len(tokens))).Should(BeNumerically("<=", numTokens))
			} else {
				expectedMsg := userMessage
				if mode == modeRandom {
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
					OfString: openai.String(userMessage),
				},
				Model: openai.CompletionNewParamsModel(model),
			}
			numTokens := 0
			partialErrMsg := "max_tokens must be at least 1, got -1"
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
						if strings.Contains(string(errMsg), partialErrMsg) {
							return
						}
					}
				}
			}
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(textCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(int64(4)))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			text := resp.Choices[0].Text
			Expect(text).ShouldNot(BeEmpty())

			if numTokens != 0 {
				tokens := strings.Fields(text)
				Expect(int64(len(tokens))).Should(BeNumerically("<=", numTokens))
			} else {
				expectedText := userMessage
				if mode == modeRandom {
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

	It("Should respond to /health", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, modeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Get("http://localhost/health")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
	})

	It("Should respond to /ready", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, modeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Get("http://localhost/ready")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
	})
})
