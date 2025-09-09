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
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
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
const invalidMaxTokensErrMsg = "Max completion tokens and max tokens should be positive"

var userMsgTokens int64

func startServer(ctx context.Context, mode string) (*http.Client, error) {
	return startServerWithArgs(ctx, mode, nil, nil)
}

func startServerWithArgs(ctx context.Context, mode string, args []string, envs map[string]string) (*http.Client, error) {
	oldArgs := os.Args
	defer func() {
		os.Args = oldArgs
	}()

	if args != nil {
		os.Args = args
	} else {
		os.Args = []string{"cmd", "--model", model, "--mode", mode}
	}

	if envs != nil {
		for k, v := range envs {
			err := os.Setenv(k, v)
			Expect(err).NotTo(HaveOccurred())
		}

		defer func() {
			for k := range envs {
				err := os.Unsetenv(k)
				Expect(err).NotTo(HaveOccurred())
			}
		}()
	}

	logger := klog.Background()

	s, err := New(logger)
	if err != nil {
		return nil, err
	}
	config, err := common.ParseCommandParamsAndLoadConfig()
	if err != nil {
		return nil, err
	}
	s.config = config

	for _, lora := range config.LoraModules {
		s.loraAdaptors.Store(lora.Name, "")
	}

	common.InitRandom(s.config.Seed)

	if err := s.createAndRegisterPrometheus(); err != nil {
		return nil, err
	}

	// calculate number of tokens for user message,
	// must be activated after parseCommandParamsAndLoadConfig since it initializes the random engine
	userMsgTokens = int64(len(common.Tokenize(userMessage)))

	// run request processing workers
	for i := 1; i <= s.config.MaxNumSeqs; i++ {
		go s.reqProcessingWorker(ctx, i)
	}

	s.startMetricsUpdaters(ctx)

	listener := fasthttputil.NewInmemoryListener()

	// start the http server
	go func() {
		if err := s.startServer(ctx, listener); err != nil {
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
			Expect(chunk.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(chunk.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(chunk.Usage.TotalTokens).To(Equal(chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens))

			msg := strings.Join(tokens, "")
			if mode == common.ModeRandom {
				// in case of random mode ensure that the returned message could be output of the random text generator
				Expect(common.IsValidText(msg)).To(BeTrue())
			} else {
				// in case of echo mode check that the text is returned as-is
				Expect(msg).Should(Equal(userMessage))
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
			Expect(chunk.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(chunk.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(chunk.Usage.TotalTokens).To(Equal(chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens))

			text := strings.Join(tokens, "")
			if mode == common.ModeRandom {
				// in case of random mode ensure that the returned message could be output of the random text generator
				Expect(common.IsValidText(text)).To(BeTrue())
			} else {
				// in case of echo mode check that the text is returned as-is
				Expect(text).Should(Equal(userMessage))
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

			if numTokens > 0 {
				tokens := common.Tokenize(msg)
				Expect(int64(len(tokens))).Should(BeNumerically("<=", numTokens))
			} else {
				if mode == common.ModeRandom {
					// in case of random mode ensure that the returned message could be output of the random text generator
					Expect(common.IsValidText(msg)).To(BeTrue())
				} else {
					// in case of echo mode check that the text is returned as-is
					Expect(msg).Should(Equal(userMessage))
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

			if numTokens != 0 {
				tokens := common.Tokenize(text)
				Expect(int64(len(tokens))).Should(BeNumerically("<=", numTokens))
			} else {
				if mode == common.ModeRandom {
					// in case of random mode ensure that the returned message could be output of the random text generator
					Expect(common.IsValidText(text)).To(BeTrue())
				} else {
					// in case of echo mode check that the text is returned as-is
					Expect(text).Should(Equal(userMessage))
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

	It("Should respond to /health", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Get("http://localhost/health")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
	})

	It("Should respond to /ready", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Get("http://localhost/ready")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
	})

	Context("namespace and pod headers", func() {
		It("Should not include namespace and pod headers in chat completion response when env is not set", func() {
			ctx := context.TODO()

			client, err := startServer(ctx, common.ModeRandom)
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

			var httpResp *http.Response
			resp, err := openaiclient.Chat.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace and pod headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)

			Expect(namespaceHeader).To(BeEmpty(), "Expected namespace header not to be present")
			Expect(podHeader).To(BeEmpty(), "Expected pod header not to be present")
		})

		It("Should include namespace and pod headers in chat completion response", func() {
			ctx := context.TODO()

			testNamespace := "test-namespace"
			testPod := "test-pod"
			envs := map[string]string{
				podNameEnv: testPod,
				podNsEnv:   testNamespace,
			}
			client, err := startServerWithArgs(ctx, common.ModeRandom, nil, envs)
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

			var httpResp *http.Response
			resp, err := openaiclient.Chat.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace and pod headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
		})

		It("Should include namespace and pod headers in chat completion streaming response", func() {
			ctx := context.TODO()

			testNamespace := "stream-test-namespace"
			testPod := "stream-test-pod"
			envs := map[string]string{
				podNameEnv: testPod,
				podNsEnv:   testNamespace,
			}
			client, err := startServerWithArgs(ctx, common.ModeRandom, nil, envs)
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

			var httpResp *http.Response
			resp, err := openaiclient.Chat.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace and pod headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
		})

		It("Should not include namespace and pod headers in chat completion streaming response when env is not set", func() {
			ctx := context.TODO()

			client, err := startServer(ctx, common.ModeRandom)
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

			var httpResp *http.Response
			resp, err := openaiclient.Chat.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace and pod headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)

			Expect(namespaceHeader).To(BeEmpty(), "Expected namespace header not to be present")
			Expect(podHeader).To(BeEmpty(), "Expected pod header not to be present")
		})

		It("Should include namespace and pod headers in completion response", func() {
			ctx := context.TODO()

			testNamespace := "test-namespace"
			testPod := "test-pod"
			envs := map[string]string{
				podNameEnv: testPod,
				podNsEnv:   testNamespace,
			}
			client, err := startServerWithArgs(ctx, common.ModeRandom, nil, envs)
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
			var httpResp *http.Response
			resp, err := openaiclient.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace and pod headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
		})

		It("Should include namespace and pod headers in completion streaming response", func() {
			ctx := context.TODO()

			testNamespace := "stream-test-namespace"
			testPod := "stream-test-pod"
			envs := map[string]string{
				podNameEnv: testPod,
				podNsEnv:   testNamespace,
			}
			client, err := startServerWithArgs(ctx, common.ModeRandom, nil, envs)
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
			var httpResp *http.Response
			resp, err := openaiclient.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp).NotTo(BeNil())

			// Check for namespace and pod headers
			namespaceHeader := httpResp.Header.Get(namespaceHeader)
			podHeader := httpResp.Header.Get(podHeader)

			Expect(namespaceHeader).To(Equal(testNamespace), "Expected namespace header to be present")
			Expect(podHeader).To(Equal(testPod), "Expected pod header to be present")
		})
	})

	Context("max-model-len context window validation", func() {
		It("Should reject requests exceeding context window", func() {
			ctx := context.TODO()
			// Start server with max-model-len=10
			args := []string{"cmd", "--model", model, "--mode", common.ModeRandom, "--max-model-len", "10"}
			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
			Expect(err).NotTo(HaveOccurred())

			// Test with raw HTTP to verify the error response format
			reqBody := `{
				"messages": [{"role": "user", "content": "This is a test message"}],
				"model": "my_model",
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
			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client),
			)

			_, err = openaiclient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("This is a test message"),
				},
				Model:     model,
				MaxTokens: openai.Int(8),
			})

			Expect(err).To(HaveOccurred())
			var apiErr *openai.Error
			Expect(errors.As(err, &apiErr)).To(BeTrue())
			Expect(apiErr.StatusCode).To(Equal(400))
		})

		It("Should accept requests within context window", func() {
			ctx := context.TODO()
			// Start server with max-model-len=50
			args := []string{"cmd", "--model", model, "--mode", common.ModeEcho, "--max-model-len", "50"}
			client, err := startServerWithArgs(ctx, common.ModeEcho, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client),
			)

			// Send a request within the context window
			resp, err := openaiclient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Hello"),
				},
				Model:     model,
				MaxTokens: openai.Int(5),
			})

			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).To(HaveLen(1))
			Expect(resp.Model).To(Equal(model))
		})

		It("Should handle text completion requests exceeding context window", func() {
			ctx := context.TODO()
			// Start server with max-model-len=10
			args := []string{"cmd", "--model", model, "--mode", common.ModeRandom, "--max-model-len", "10"}
			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
			Expect(err).NotTo(HaveOccurred())

			// Test with raw HTTP for text completion
			reqBody := `{
				"prompt": "This is a long test prompt with many words",
				"model": "my_model",
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

	Describe("Check random latencies", Ordered, func() {
		var simulator *VllmSimulator

		BeforeAll(func() {
			var err error
			simulator, err = New(klog.Background())
			Expect(err).NotTo(HaveOccurred())

			simulator.config = &common.Configuration{
				TimeToFirstToken:             2048,
				TimeToFirstTokenStdDev:       2048,
				KVCacheTransferLatency:       2048,
				KVCacheTransferLatencyStdDev: 2048,
			}

			common.InitRandom(time.Now().UnixNano())
		})

		DescribeTable("should calculate inter token latency correctly",
			func(interTokenLatency int, stddev int) {
				simulator.config.InterTokenLatency = interTokenLatency
				simulator.config.InterTokenLatencyStdDev = stddev
				interToken := simulator.getInterTokenLatency()
				Expect(interToken).To(BeNumerically(">=", int(float32(interTokenLatency)*0.3)))
				Expect(interToken).To(BeNumerically("<=", int(float32(interTokenLatency)*1.7)))
			},
			func(interTokenLatency int, stddev int) string {
				return fmt.Sprintf("interTokenLatency: %d stddev: %d", interTokenLatency, stddev)
			},
			Entry(nil, 1000, 300),
			Entry(nil, 1000, 800), // invalid std dev, used for testing purposes
			Entry(nil, 1000, 900), // invalid std dev, used for testing purposes
			Entry(nil, 1000, 0),
		)

		DescribeTable("should calculate total inter token latency correctly",
			func(interTokenLatency int, stddev int, numberOfTokens int) {
				simulator.config.InterTokenLatency = interTokenLatency
				simulator.config.InterTokenLatencyStdDev = stddev

				latency := 0
				for range numberOfTokens - 1 {
					latency += simulator.getInterTokenLatency()
				}

				Expect(latency).To(BeNumerically(">=", int(float32(interTokenLatency)*0.3*float32(numberOfTokens))))
				Expect(latency).To(BeNumerically("<=", int(float32(interTokenLatency)*1.7*float32(numberOfTokens))))
			},
			func(interTokenLatency int, stddev int, numberOfTokens int) string {
				return fmt.Sprintf("interTokenLatency: %d stddev: %d, numberOfTokens: %d", interTokenLatency,
					stddev, numberOfTokens)
			},
			Entry(nil, 1000, 30, 100),
			Entry(nil, 1000, 800, 20), // invalid std dev, used for testing purposes
			Entry(nil, 1000, 900, 5),  // invalid std dev, used for testing purposes
			Entry(nil, 1000, 0, 50),
		)

		DescribeTable("should calculate time to first token correctly",
			func(timeToFirstToken int, timeToFirstTokenStdDev int,
				kvCacheLatency int, kvCacheLatencyStdDev int, doREmotePrefill bool) {
				simulator.config.TimeToFirstToken = timeToFirstToken
				simulator.config.TimeToFirstTokenStdDev = timeToFirstTokenStdDev
				simulator.config.KVCacheTransferLatency = kvCacheLatency
				simulator.config.KVCacheTransferLatencyStdDev = kvCacheLatencyStdDev
				timeToFirst := simulator.getTimeToFirstToken(1, 0, doREmotePrefill)
				if doREmotePrefill {
					Expect(timeToFirst).To(BeNumerically(">=", int(float32(kvCacheLatency)*0.3)))
					Expect(timeToFirst).To(BeNumerically("<=", int(float32(kvCacheLatency)*1.7)))
				} else {
					Expect(timeToFirst).To(BeNumerically(">=", int(float32(timeToFirstToken)*0.3)))
					Expect(timeToFirst).To(BeNumerically("<=", int(float32(timeToFirstToken)*1.7)))
				}
			},
			func(timeToFirstToken int, timeToFirstTokenStdDev int,
				kvCacheLatency int, kvCacheLatencyStdDev int, doREmotePrefill bool) string {
				return fmt.Sprintf("timeToFirstToken: %d stddev: %d kvCacheLatency: %d stddev: %d doREmotePrefill: %t",
					timeToFirstToken, timeToFirstTokenStdDev, kvCacheLatency, kvCacheLatencyStdDev, doREmotePrefill)
			},
			Entry(nil, 10000, 300, 1000, 200, true),
			Entry(nil, 10000, 300, 1000, 200, false),
			Entry(nil, 10000, 9000, 1000, 800, true),  // invalid std dev, used for testing purposes
			Entry(nil, 10000, 8000, 1000, 900, false), // invalid std dev, used for testing purposes
			Entry(nil, 10000, 0, 1000, 0, true),
			Entry(nil, 10000, 0, 1000, 0, false),
		)

		It("when <time-to-first-token> is not 0, ignore <prefill-overhead>", func() {
			timeToFirstToken := 1000
			simulator.config.TimeToFirstToken = timeToFirstToken
			simulator.config.TimeToFirstTokenStdDev = 0

			simulator.config.PrefillOverhead = 100
			simulator.config.PrefillTimePerToken = 200
			simulator.config.PrefillTimeStdDev = 80

			ttft := simulator.getTimeToFirstToken(128, 0, false)

			Expect(ttft).To(BeNumerically("==", timeToFirstToken))
		})

		It("when <time-to-first-token> is 0, and <prefill-overhead> is not 0, use <prefill-overhead>", func() {
			simulator.config.TimeToFirstToken = 0
			simulator.config.TimeToFirstTokenStdDev = 0

			simulator.config.PrefillOverhead = 100
			simulator.config.PrefillTimePerToken = 200
			simulator.config.PrefillTimeStdDev = 80

			ttft := simulator.getTimeToFirstToken(128, 0, false)
			Expect(ttft).NotTo(BeNumerically("==", 0))
		})

		DescribeTable("time to first token is against number of prompt tokens with std",
			func(prefillOverhead int, prefillTimePerToken int, stdDev int, nTokens int, nCachedTokens int) {
				simulator.config.TimeToFirstToken = 0
				simulator.config.PrefillOverhead = prefillOverhead
				simulator.config.PrefillTimePerToken = prefillTimePerToken
				simulator.config.PrefillTimeStdDev = stdDev

				ttft := simulator.getTimeToFirstToken(nTokens, nCachedTokens, false)

				expectedTTFT := prefillOverhead + prefillTimePerToken*(nTokens-nCachedTokens)
				Expect(ttft).To(BeNumerically(">=", int(float64(expectedTTFT)*0.3)))
				Expect(ttft).To(BeNumerically("<=", int(float64(expectedTTFT)*1.7)))
			},
			func(prefillOverhead int, prefillTimePerToken, stdDev int, nTokens int, nCachedTokens int) string {
				return fmt.Sprintf("prefillOverhead: %d, prefillTimePerToken: %d, stdDev: %d, nTokens: %d nCachedTokens: %d",
					prefillOverhead, prefillTimePerToken, stdDev, nTokens, nCachedTokens)
			},
			Entry("single token", 100, 50, 10, 1, 0),
			Entry("single token big std", 100, 50, 70, 1, 0),
			Entry("stddev is 0", 100, 50, 0, 1, 0),
			Entry("medium overhead, 512 tokens", 200, 1000, 150, 512, 0),
			Entry("large overhead, 1024 tokens", 2000, 3000, 800, 1024, 0),
			Entry("very long prompt", 150, 200, 70, 20000, 0),
			Entry("medium overhead, 512 tokens, 256 cached", 200, 1000, 150, 512, 256),
			Entry("large overhead, 1024 tokens, 1008 cached", 2000, 3000, 800, 1024, 1008),
			Entry("very long prompt, 1024 cached", 150, 200, 70, 20000, 1024),
		)

		DescribeTable("time to first token is against number of prompt tokens",
			func(prefillOverhead int, prefillTimePerToken int, nTokens int, nCachedTokens int) {
				simulator.config.TimeToFirstToken = 0
				simulator.config.PrefillOverhead = prefillOverhead
				simulator.config.PrefillTimePerToken = prefillTimePerToken
				simulator.config.PrefillTimeStdDev = 0

				ttft := simulator.getTimeToFirstToken(nTokens, nCachedTokens, false)
				expectedTTFT := prefillOverhead + prefillTimePerToken*(nTokens-nCachedTokens)
				Expect(ttft).To(Equal(expectedTTFT))
			},
			func(prefillOverhead int, prefillTimePerToken, nTokens int, nCachedTokens int) string {
				return fmt.Sprintf("prefillOverhead: %d, prefillTimePerToken: %d, nTokens: %d nCachedTokens: %d",
					prefillOverhead, prefillTimePerToken, nTokens, nCachedTokens)
			},
			Entry("single token", 100, 50, 1, 0),
			Entry("medium overhead, 512 tokens", 200, 1000, 512, 0),
			Entry("large overhead, 1024 tokens", 2000, 3000, 1024, 0),
			Entry("very long prompt", 150, 200, 20000, 0),
			Entry("medium overhead, 512 tokens, 256 cached", 200, 1000, 512, 256),
			Entry("large overhead, 1024 tokens, 128 cached", 2000, 3000, 1024, 128),
			Entry("very long prompt, 1024 cached", 150, 200, 20000, 1024),
		)

		It("when <kv-cache-transfer-latency> not 0, ignore <kv-cache-transfer-overhead>", func() {
			simulator.config.KVCacheTransferLatency = 200
			simulator.config.KVCacheTransferLatencyStdDev = 0

			simulator.config.KVCacheTransferTimePerToken = 100
			simulator.config.KVCacheTransferTimeStdDev = 0

			ttft := simulator.getTimeToFirstToken(128, 0, true)
			Expect(ttft).To(BeNumerically("==", 200))
		})

		It("when <kv-cache-transfer-latency> is 0, and <kv-cache-transfer-overhead> is not 0, use <kv-cache-transfer-overhead>", func() {
			simulator.config.KVCacheTransferLatency = 0
			simulator.config.KVCacheTransferLatencyStdDev = 0

			simulator.config.KVCacheTransferTimePerToken = 100
			simulator.config.KVCacheTransferTimeStdDev = 0

			ttft := simulator.getTimeToFirstToken(128, 0, true)
			Expect(ttft).To(BeNumerically("==", 12800))
		})

		DescribeTable("kv cache transfer time against number of prompt tokens",
			func(kvCacheTransTPT int, stddev int, nTokens int) {
				simulator.config.TimeToFirstToken = 0
				simulator.config.PrefillOverhead = 1
				simulator.config.KVCacheTransferTimePerToken = kvCacheTransTPT
				simulator.config.KVCacheTransferTimeStdDev = stddev

				ttft := simulator.getTimeToFirstToken(nTokens, 0, true)

				expectedTTFT := kvCacheTransTPT * nTokens
				Expect(ttft).To(BeNumerically(">=", int(float64(expectedTTFT)*0.3)))
				Expect(ttft).To(BeNumerically("<=", int(float64(expectedTTFT)*1.7)))

			},
			func(kvCacheTransferTimePerToken int, stddev int, nTokens int) string {
				return fmt.Sprintf("kvCacheTransferTimePerToken: %d stddev: %d nTokens: %d",
					kvCacheTransferTimePerToken, stddev, nTokens)
			},
			Entry("single token", 100, 70, 1),
			Entry("stddev is 0", 100, 0, 1),
			Entry("medium overhead, 512 tokens", 200, 150, 512),
			Entry("large overhead, 1024 tokens", 2000, 1800, 1024),
			Entry("very long prompt", 150, 100, 20000),
		)

		It("when time-factor-under-load is 1, the time to first token should be equal to time-to-first-token", func() {
			simulator.config.TimeToFirstToken = 42
			simulator.config.TimeToFirstTokenStdDev = 0
			simulator.config.TimeFactorUnderLoad = 1.0

			simulator.runReqChan <- 100

			ttft := simulator.getTimeToFirstToken(128, 0, false)
			Expect(ttft).To(Equal(42))
		})

		It("when time-factor-under-load is > 1, but max-num-seqs is 1, the factor will not take effect", func() {
			simulator.config.TimeToFirstToken = 42
			simulator.config.TimeToFirstTokenStdDev = 0
			simulator.config.TimeFactorUnderLoad = 100.0
			simulator.config.MaxNumSeqs = 1

			for len(simulator.runReqChan) > 0 {
				<-simulator.runReqChan
			}

			simulator.runReqChan <- 1

			ttft := simulator.getTimeToFirstToken(128, 0, false)
			Expect(ttft).To(Equal(42))
		})

		DescribeTable("when time-factor-under-load is > 1, and the sim is fully loaded, the time to first token should be time-factor-under-load * time-to-first-token",
			func(timeFactorUnderLoad float64, maxNumOfReq int) {
				simulator.config.TimeToFirstToken = 42
				simulator.config.TimeToFirstTokenStdDev = 0
				simulator.config.TimeFactorUnderLoad = timeFactorUnderLoad
				simulator.config.MaxNumSeqs = maxNumOfReq
				simulator.nRunningReqs = int64(maxNumOfReq)

				ttft := simulator.getTimeToFirstToken(128, 0, false)
				Expect(ttft).To(Equal(int(float64(42) * timeFactorUnderLoad)))

			},
			func(timeFactorUnderLoad float64, maxNumOfReq int64) string {
				return fmt.Sprintf("timeFactorUnderLoad: %f maxNumOfReq: %d",
					timeFactorUnderLoad, maxNumOfReq)
			},

			Entry("factor: 1.5", 1.5, 70),
			Entry("factor: 2.0", 2.0, 2),
			Entry("factor: 100.0", 100.0, 150),
			Entry("factor: 20000.0", 20000.0, 310),
		)

		DescribeTable("when time-factor-under-load is > 1, and the sim is partially loaded, the time to first token should be linear interpolation between time-to-first-token and time-factor-under-load * time-to-first-token",
			func(timeFactorUnderLoad float64, maxNumOfReq int, nCurrNumOfReq int) {
				simulator.config.TimeToFirstToken = 42
				simulator.config.TimeToFirstTokenStdDev = 0
				simulator.config.TimeFactorUnderLoad = timeFactorUnderLoad
				simulator.config.MaxNumSeqs = maxNumOfReq
				simulator.nRunningReqs = int64(nCurrNumOfReq)

				ttft := simulator.getTimeToFirstToken(128, 0, false)
				max := timeFactorUnderLoad * float64(42)
				Expect(ttft).To(BeNumerically(">=", 42))
				Expect(ttft).To(BeNumerically("<=", max))

			},
			func(timeFactorUnderLoad float64, maxNumOfReq int, nCurrNumOfReq int) string {
				return fmt.Sprintf("timeFactorUnderLoad: %f maxNumOfReq: %d nCurrNumOfReq: %d",
					timeFactorUnderLoad, maxNumOfReq, nCurrNumOfReq)
			},

			Entry("factor: 1.5", 1.5, 70, 35),
			Entry("factor: 2.0", 2.0, 2, 1),
			Entry("factor: 100.0", 100.0, 150, 75),
			Entry("factor: 20000.0", 20000.0, 310, 155),
		)

		It("when TimeFactorUnderLoad is 1.0, calcLoadFactor should give 1", func() {
			simulator.config.TimeFactorUnderLoad = 1.0
			simulator.config.MaxNumSeqs = 11
			simulator.nRunningReqs = 3

			factor := simulator.getCurrFactor()
			Expect(factor).To(BeNumerically("==", 1.0))
		})

		It("when TimeFactorUnderLoad is > 1.0, and sim is fully loaded, calcLoadFactor should give TimeFactorUnderLoad", func() {
			simulator.config.TimeFactorUnderLoad = 2.0
			simulator.config.MaxNumSeqs = 11
			simulator.nRunningReqs = 11

			factor := simulator.getCurrFactor()
			Expect(factor).To(BeNumerically("==", simulator.config.TimeFactorUnderLoad))

		})

		It("when TimeFactorUnderLoad is > 1.0, and sim is partially loaded, calcLoadFactor should give a value between 1 and TimeFactorUnderLoad", func() {
			simulator.config.TimeFactorUnderLoad = 2.0
			simulator.config.MaxNumSeqs = 11
			simulator.nRunningReqs = 6

			factor := simulator.getCurrFactor()
			Expect(factor).To(BeNumerically(">", 1.0))
			Expect(factor).To(BeNumerically("<", simulator.config.TimeFactorUnderLoad))
		})

	})

})
