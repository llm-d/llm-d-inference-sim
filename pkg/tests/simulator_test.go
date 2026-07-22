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
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/valyala/fasthttp"
)

var _ = Describe("Simulator", func() {

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

		// extendedPrompt shares all blocks of longPrompt and adds enough text to push past the next block boundary.
		extendedPrompt := longPrompt + " This extra sentence pushes it past the next block boundary for the test."

		DescribeTable("parent_block_hash on second request with shared prefix",
			func(useMapFormat string) {
				// Two consecutive completions: the first stores all blocks with no parent
				// (ParentHash == 0 / EmptyBlockHash). The second extends the prompt by one
				// extra block; its store event must carry the hash of the last block from
				// the first request as parent.
				topic := kvcache.CreateKVEventsTopic("localhost", model)
				sub, zmqEndpoint := common.CreateSub(ctx, topic)
				//nolint
				defer sub.Close()

				args := []string{"cmd", "--model", model, "--mode", mode,
					"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
					"--event-batch-size", "1", "--zmq-endpoint", zmqEndpoint,
					"--use-vllm-map-event-format", useMapFormat}

				client, err := startServerWithArgsAndEnv(ctx, mode, args, map[string]string{"POD_IP": "localhost"})
				Expect(err).NotTo(HaveOccurred())

				go func() {
					time.Sleep(200 * time.Millisecond)

					// First request — all blocks new, no cached prefix
					openaiclient, params := getOpenAIClientAndCompletionParams(client, model, longPrompt, false)
					resp, err := openaiclient.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
					Expect(resp.Choices).ShouldNot(BeEmpty())

					// Second request — shares the prefix, stores at least one new block
					openaiclient2, params2 := getOpenAIClientAndCompletionParams(client, model, extendedPrompt, false)
					resp2, err := openaiclient2.Completions.New(ctx, params2)
					Expect(err).NotTo(HaveOccurred())
					Expect(resp2.Choices).ShouldNot(BeEmpty())
				}()

				// First event: all blocks new → parent must be EmptyBlockHash (0)
				msg1, err := sub.Recv()
				Expect(err).NotTo(HaveOccurred())
				events1, _, _ := kvcache.ParseKVEvent(msg1.Frames, topic, 1)
				Expect(events1).NotTo(BeEmpty())
				Expect(events1[0].ParentHash).To(Equal(uint64(0)))
				lastHashFromFirst := events1[0].BlockHashes[len(events1[0].BlockHashes)-1]

				// Second event: only the extra block(s) are new → parent == last block of first request
				msg2, err := sub.Recv()
				Expect(err).NotTo(HaveOccurred())
				events2, _, _ := kvcache.ParseKVEvent(msg2.Frames, topic, 2)
				Expect(events2).NotTo(BeEmpty())
				Expect(events2[0].ParentHash).To(Equal(lastHashFromFirst))
			},
			Entry("array format (legacy)", "false"),
			Entry("map format (vLLM PR #42892)", "true"),
		)

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

	Context("kv-events replay", func() {
		const (
			replayEndpoint = "tcp://127.0.0.1:15559"
			replayModel    = common.QwenModelName
			replayMode     = common.ModeRandom
			replayPrompt   = "This is a test message for kv cache events, has to be long enough to be tokenized into multiple blocks."
		)

		// setupReplayServer starts the simulator with replay enabled.
		// Returns the HTTP client, the PUB subscriber (for watching live events), and the topic.
		setupReplayServer := func(ctx context.Context) (*http.Client, zmq4.Socket, string) {
			topic := kvcache.CreateKVEventsTopic("localhost", replayModel)
			sub, zmqEndpoint := common.CreateSub(ctx, topic)

			args := []string{"cmd", "--model", replayModel, "--mode", replayMode,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--event-batch-size", "1", "--zmq-endpoint", zmqEndpoint,
				"--kv-events-replay-endpoint", replayEndpoint,
				"--kv-events-replay-queue-size", "64",
			}
			client, err := startServerWithArgsAndEnv(ctx, replayMode, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())
			return client, sub, topic
		}

		// sendReplayRequestAndRecv connects a REQ socket to the ROUTER replay endpoint,
		// sends startSeq, and collects all reply batches until the sentinel.
		// Replies are [topic, seq(8B), payload] frames (REQ strips identity+delimiter),
		// the same shape as the live PUB stream.
		// Returns (totalStoredBlocks, lastSeq).
		sendReplayRequestAndRecv := func(ctx context.Context, startSeq uint64) (int, uint64) {
			req := zmq4.NewReq(ctx)
			DeferCleanup(req.Close)

			Expect(req.Dial(replayEndpoint)).To(Succeed())
			frame := make([]byte, 8)
			binary.BigEndian.PutUint64(frame, startSeq)
			Expect(req.Send(zmq4.NewMsg(frame))).To(Succeed())

			totalStored := 0
			var lastSeq uint64
			for {
				msg, err := req.Recv()
				Expect(err).NotTo(HaveOccurred())
				// Each reply: [topic, seq(8B), payload]
				Expect(msg.Frames).To(HaveLen(3))
				seq := binary.BigEndian.Uint64(msg.Frames[1])
				if seq == ^uint64(0) {
					// End-of-replay sentinel
					break
				}
				stored, _, _ := kvcache.CountKVEventBlocks(msg.Frames, kvcache.CreateKVEventsTopic("localhost", replayModel), seq)
				totalStored += stored
				lastSeq = seq
			}
			return totalStored, lastSeq
		}

		// startSubReceiver pumps all PUB messages from sub into a channel.
		startSubReceiver := func(sub zmq4.Socket) chan zmq4.Msg {
			ch := make(chan zmq4.Msg, 64)
			go func() {
				for {
					msg, err := sub.Recv()
					if err != nil {
						return
					}
					ch <- msg
				}
			}()
			return ch
		}

		// drainPubUntilQuiet reads live PUB events until idle, returning
		// total stored blocks and last sequence number seen.
		drainPubUntilQuiet := func(msgCh chan zmq4.Msg, topic string, timeout time.Duration) (totalStored int, lastSeq uint64) {
			for {
				select {
				case msg := <-msgCh:
					if len(msg.Frames) != 3 {
						continue
					}
					seq := binary.BigEndian.Uint64(msg.Frames[1])
					stored, _, _ := kvcache.CountKVEventBlocks(msg.Frames, topic, seq)
					totalStored += stored
					lastSeq = seq
				case <-time.After(timeout):
					return
				}
			}
		}

		It("stores published batches and replays them from a given sequence number", func() {
			ctx := context.TODO()
			client, sub, topic := setupReplayServer(ctx)
			defer sub.Close() //nolint:errcheck

			msgCh := startSubReceiver(sub)

			// Allow replayer socket to bind
			time.Sleep(300 * time.Millisecond)

			// Trigger two chat completions
			for range 2 {
				openaiclient, params := getOpenAIClientAndChatParams(client, replayModel, replayPrompt, false)
				resp, err := openaiclient.Chat.Completions.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.Choices).ShouldNot(BeEmpty())
			}

			// Drain all live events; record last seq published
			_, lastSeq := drainPubUntilQuiet(msgCh, topic, 500*time.Millisecond)
			Expect(lastSeq).To(BeNumerically(">", 0))

			// Request replay from the last seq — exactly that one batch should come back
			replayedTotal, replayedLastSeq := sendReplayRequestAndRecv(ctx, lastSeq)
			Expect(replayedTotal).To(BeNumerically(">", 0))
			Expect(replayedLastSeq).To(Equal(lastSeq))
		})

		It("replays all stored batches when startSeq is 1", func() {
			ctx := context.TODO()
			client, sub, topic := setupReplayServer(ctx)
			defer sub.Close() //nolint:errcheck

			msgCh := startSubReceiver(sub)

			time.Sleep(300 * time.Millisecond)

			// Trigger one completion
			openaiclient, params := getOpenAIClientAndChatParams(client, replayModel, replayPrompt, false)
			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())

			// Count blocks from the live PUB stream
			origTotal, _ := drainPubUntilQuiet(msgCh, topic, 500*time.Millisecond)
			Expect(origTotal).To(BeNumerically(">", 0))

			// Replay from seq 1 — all stored batches must be returned
			replayedTotal, _ := sendReplayRequestAndRecv(ctx, 1)
			Expect(replayedTotal).To(Equal(origTotal))
		})

		It("returns only the sentinel when startSeq is beyond all stored batches", func() {
			ctx := context.TODO()
			client, sub, topic := setupReplayServer(ctx)
			defer sub.Close() //nolint:errcheck

			msgCh := startSubReceiver(sub)

			time.Sleep(300 * time.Millisecond)

			// Trigger one completion
			openaiclient, params := getOpenAIClientAndChatParams(client, replayModel, replayPrompt, false)
			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())

			// Drain live events
			drainPubUntilQuiet(msgCh, topic, 500*time.Millisecond)

			// Ask for a seq far beyond what was stored — only the sentinel comes back
			replayedTotal, _ := sendReplayRequestAndRecv(ctx, 999999)
			Expect(replayedTotal).To(Equal(0))
		})
	})
})
