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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/valyala/fasthttp"
)

// The simulator's own behavior on the OpenAI-compatible routes: request
// validation, the headers it adds, its error bodies, and its request
// accounting. None of it is an engine's to decide, so every spec here runs
// under each registered engine. What an engine does own lives in
// simulator_test.go alongside the features it belongs to.
var _ = Describe("Simulator core", forEachEngine(func() {
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
			Expect(portHeader).To(Equal(strconv.Itoa(common.DefaultPort)), "Expected port header to be present")
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
			Expect(portHeader).To(Equal(strconv.Itoa(common.DefaultPort)), "Expected port header to be present")
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
			Expect(portHeader).To(Equal(strconv.Itoa(common.DefaultPort)), "Expected port header to be present")
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
			Expect(portHeader).To(Equal(strconv.Itoa(common.DefaultPort)), "Expected port header to be present")
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
			Expect(portHeader).To(Equal(strconv.Itoa(common.DefaultPort)), "Expected port header to be present")
		})
	})

	Context("max-model-len context window validation", func() {
		const contextWindowTestPrompt = "This is a test message"

		It("Should reject requests exceeding context window in random mode, regardless of max_tokens", func() {
			ctx := context.TODO()
			model := common.TestModelName
			prompt := contextWindowTestPrompt
			promptChatTokens := getChatPromptTokensCountForTestModel(prompt)

			// random mode no longer considers max_tokens - only that the prompt
			// leaves room for at least one response token. Size max-model-len so
			// the prompt alone fills it, leaving no such room.
			maxModelLen := promptChatTokens
			args := []string{"cmd", "--model", model, "--mode", common.ModeRandom, "--max-model-len", strconv.FormatInt(maxModelLen, 10)}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			// max_tokens is huge to demonstrate it is irrelevant to this check
			// Test with raw HTTP to verify the error response format
			reqBody := fmt.Sprintf(`{
				"messages": [{"role": "user", "content": "%s"}],
				"model": "%s",
				"max_tokens": 1000000
			}`, prompt, model)

			resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			Expect(resp.StatusCode).To(Equal(400))
			Expect(string(body)).To(ContainSubstring(fmt.Sprintf("This model's maximum context length is %d tokens", maxModelLen)))
			Expect(string(body)).To(ContainSubstring(fmt.Sprintf("you requested %d tokens in the messages", promptChatTokens)))
			Expect(string(body)).To(ContainSubstring("BadRequestError"))

			// Also test with OpenAI client to ensure it gets an error
			openaiclient, params := getOpenAIClientAndChatParams(client, model, prompt, false)
			params.MaxTokens = openai.Int(1000000)

			_, err = openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).To(HaveOccurred())
			var apiErr *openai.Error
			Expect(errors.As(err, &apiErr)).To(BeTrue())
			Expect(apiErr.StatusCode).To(Equal(400))
		})

		It("Should accept requests in random mode with a huge max_tokens, as long as the prompt fits", func() {
			ctx := context.TODO()
			model := common.TestModelName
			prompt := contextWindowTestPrompt
			promptChatTokens := getChatPromptTokensCountForTestModel(prompt)

			// leave room for at least one response token
			maxModelLen := promptChatTokens + 1
			args := []string{"cmd", "--model", model, "--mode", common.ModeRandom, "--max-model-len", strconv.FormatInt(maxModelLen, 10)}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, model, prompt, false)
			// would have been rejected under the old prompt+max_tokens<=max-model-len check
			params.MaxTokens = openai.Int(1000000)

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).To(HaveLen(1))
			Expect(resp.Model).To(Equal(model))
		})

		It("Should accept requests within context window in echo mode", func() {
			ctx := context.TODO()
			prompt := "Hello"
			promptChatTokens := getChatPromptTokensCountForTestModel(prompt)

			// Start server with max-model-len=50
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho, "--max-model-len", "50"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, prompt, false)
			// max_tokens must be at least the prompt length in echo mode
			params.MaxTokens = openai.Int(promptChatTokens)

			// Send a request within the context window
			resp, err := openaiclient.Chat.Completions.New(ctx, params)

			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).To(HaveLen(1))
			Expect(resp.Model).To(Equal(common.TestModelName))
		})

		It("Should reject echo mode requests exceeding context window", func() {
			ctx := context.TODO()
			model := common.TestModelName
			prompt := contextWindowTestPrompt
			promptChatTokens := getChatPromptTokensCountForTestModel(prompt)

			// in echo mode the prompt is echoed back as the response, so it must fit
			// twice within the context window; one below that boundary must be rejected
			maxModelLen := promptChatTokens*2 - 1
			args := []string{"cmd", "--model", model, "--mode", common.ModeEcho, "--max-model-len", strconv.FormatInt(maxModelLen, 10)}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"messages": [{"role": "user", "content": "%s"}],
				"model": "%s",
				"max_tokens": %d
			}`, prompt, model, promptChatTokens)

			resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			Expect(resp.StatusCode).To(Equal(400))
			Expect(string(body)).To(ContainSubstring(fmt.Sprintf("This model's maximum context length is %d tokens", maxModelLen)))
			Expect(string(body)).To(ContainSubstring("BadRequestError"))
		})

		It("Should reject echo mode requests whose prompt exceeds max_tokens", func() {
			ctx := context.TODO()
			model := common.TestModelName
			prompt := contextWindowTestPrompt
			promptChatTokens := getChatPromptTokensCountForTestModel(prompt)

			args := []string{"cmd", "--model", model, "--mode", common.ModeEcho, "--max-model-len", "1000"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"messages": [{"role": "user", "content": "%s"}],
				"model": "%s",
				"max_tokens": 1
			}`, prompt, model)

			resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			Expect(resp.StatusCode).To(Equal(400))
			Expect(string(body)).To(ContainSubstring(fmt.Sprintf("max_tokens is 1, but the prompt has %d tokens", promptChatTokens)))
			Expect(string(body)).To(ContainSubstring("BadRequestError"))
		})

		It("Should handle text completion requests exceeding context window", func() {
			ctx := context.TODO()
			prompt := "This is a long test prompt with many words"
			promptTokens := getTextPromptTokensCountForTestModel(prompt)

			// random mode: size max-model-len so the prompt alone fills it
			maxModelLen := promptTokens
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--max-model-len", strconv.FormatInt(maxModelLen, 10)}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"prompt": "%s",
				"model": "%s",
				"max_tokens": 5
			}`, prompt, common.TestModelName)

			resp, err := client.Post("http://localhost/v1/completions", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			Expect(resp.StatusCode).To(Equal(400))
			Expect(string(body)).To(ContainSubstring(fmt.Sprintf("This model's maximum context length is %d tokens", maxModelLen)))
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

			apiErr := apiErrorFromBody(body)
			Expect(apiErr.Code).To(Equal(422))
			Expect(apiErr.Message).To(ContainSubstring("X-Return-Error"))
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

			apiErr := apiErrorFromBody(body)
			Expect(apiErr.Code).To(Equal(400))
			Expect(apiErr.Message).To(ContainSubstring("Invalid X-Return-Error"))
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
			Expect(errorType(openaiError)).ToNot(BeEmpty())
			Expect(errorMessage(openaiError)).To(ContainSubstring("The model `some-other-model` does not exist"))
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
			Expect(errorType(openaiError)).ToNot(BeEmpty())
			Expect(errorMessage(openaiError)).To(ContainSubstring("Max completion tokens and max tokens should be positive"))
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

	Context("Response channel buffer sizing", func() {
		// buildWordPrompt returns a prompt of n space-separated distinct words, which the
		// SimpleTokenizer used in tests renders as exactly n tokens.
		buildWordPrompt := func(n int) string {
			words := make([]string, n)
			for i := range words {
				words[i] = fmt.Sprintf("tok%d", i)
			}
			return strings.Join(words, " ")
		}

		It("Should not drop tokens across many concurrent requests (echo mode)", func() {
			ctx := context.TODO()
			// echo mode requires max-model-len >= 2*prompt tokens (the prompt is echoed
			// back as the response), so max-model-len is set just above 2*15 to stay
			// as tight as that constraint allows.
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
				"--max-num-seqs", "100", "--max-model-len", "30", "--max-waiting-queue-length", "1"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			prompt := buildWordPrompt(15)
			openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, prompt, false)
			n := 10
			params.N = param.NewOpt(int64(n))

			var wg sync.WaitGroup
			numberOfRequests := 10
			wg.Add(numberOfRequests)
			for range numberOfRequests {
				go func() {
					defer GinkgoRecover()
					defer wg.Done()
					resp, err := openaiclient.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
					Expect(resp.Choices).To(HaveLen(n))
					for i := range n {
						Expect(resp.Choices[i].Text).To(Equal(prompt))
					}
				}()
			}
			wg.Wait()
		})

		It("Should not drop tokens across multiple prompts each with n>1 choices (echo mode)", func() {
			ctx := context.TODO()
			// echo mode requires max-model-len >= 2*prompt tokens (the prompt is echoed
			// back as the response), so max-model-len is set just above 2*15 to stay
			// as tight as that constraint allows.
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
				"--max-num-seqs", "100", "--max-model-len", "30", "--max-waiting-queue-length", "1"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			prompts := []string{buildWordPrompt(15), buildWordPrompt(14), buildWordPrompt(13)}
			openaiclient, params := getOpenAIClientAndCompletionParams(client, common.TestModelName, prompts[0], false)
			params.Prompt = openai.CompletionNewParamsPromptUnion{OfArrayOfStrings: prompts}
			n := 3
			params.N = param.NewOpt(int64(n))

			resp, err := openaiclient.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			// len(prompts) * n choices, one respCtx per choice, must all have arrived
			// on the single response channel shared by this request.
			Expect(resp.Choices).To(HaveLen(len(prompts) * n))
			for i, choice := range resp.Choices {
				Expect(choice.Text).To(Equal(prompts[i/n]))
			}
		})
	})
}))
