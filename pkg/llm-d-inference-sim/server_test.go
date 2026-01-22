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
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

var _ = Describe("Server", func() {

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

	Context("tokenize", Ordered, func() {
		It("Should return correct response to /tokenize chat", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", qwenModelName, "--mode", common.ModeRandom,
				"--max-model-len", "2048"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := `{
				"messages": [{"role": "user", "content": "This is a test"}],
				"model": "Qwen/Qwen2-0.5B"
			}`
			resp, err := client.Post("http://localhost/tokenize", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var tokenizeResp vllmapi.TokenizeResponse
			err = json.Unmarshal(body, &tokenizeResp)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokenizeResp.Count).To(Equal(4))
			Expect(tokenizeResp.Tokens).To(HaveLen(4))
			Expect(tokenizeResp.MaxModelLen).To(Equal(2048))
		})

		It("Should return correct response to /tokenize text", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", qwenModelName, "--mode", common.ModeRandom,
				"--max-model-len", "2048"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := `{
				"prompt": "This is a test",
				"model": "Qwen/Qwen2-0.5B"
			}`
			resp, err := client.Post("http://localhost/tokenize", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var tokenizeResp vllmapi.TokenizeResponse
			err = json.Unmarshal(body, &tokenizeResp)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokenizeResp.Count).To(Equal(4))
			Expect(tokenizeResp.Tokens).To(HaveLen(4))
			Expect(tokenizeResp.MaxModelLen).To(Equal(2048))
		})
	})

	Context("SSL/HTTPS Configuration", func() {
		It("Should parse SSL certificate configuration correctly", func() {
			tempDir := GinkgoT().TempDir()
			certFile, keyFile, err := GenerateTempCerts(tempDir)
			Expect(err).NotTo(HaveOccurred())

			oldArgs := os.Args
			defer func() {
				os.Args = oldArgs
			}()

			os.Args = []string{"cmd", "--model", testModel, "--ssl-certfile", certFile, "--ssl-keyfile", keyFile}
			config, err := common.ParseCommandParamsAndLoadConfig()
			Expect(err).NotTo(HaveOccurred())
			Expect(config.SSLEnabled()).To(BeTrue())
			Expect(config.SSLCertFile).To(Equal(certFile))
			Expect(config.SSLKeyFile).To(Equal(keyFile))
		})

		It("Should parse self-signed certificate configuration correctly", func() {
			oldArgs := os.Args
			defer func() {
				os.Args = oldArgs
			}()

			os.Args = []string{"cmd", "--model", testModel, "--self-signed-certs"}
			config, err := common.ParseCommandParamsAndLoadConfig()
			Expect(err).NotTo(HaveOccurred())
			Expect(config.SSLEnabled()).To(BeTrue())
			Expect(config.SelfSignedCerts).To(BeTrue())
		})

		It("Should create self-signed TLS certificate successfully", func() {
			cert, err := CreateSelfSignedTLSCertificate()
			Expect(err).NotTo(HaveOccurred())
			Expect(cert.Certificate).To(HaveLen(1))
			Expect(cert.PrivateKey).NotTo(BeNil())
		})

		It("Should validate SSL configuration - both cert and key required", func() {
			tempDir := GinkgoT().TempDir()

			oldArgs := os.Args
			defer func() {
				os.Args = oldArgs
			}()

			certFile, _, err := GenerateTempCerts(tempDir)
			Expect(err).NotTo(HaveOccurred())

			os.Args = []string{"cmd", "--model", testModel, "--ssl-certfile", certFile}
			_, err = common.ParseCommandParamsAndLoadConfig()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("both ssl-certfile and ssl-keyfile must be provided together"))

			_, keyFile, err := GenerateTempCerts(tempDir)
			Expect(err).NotTo(HaveOccurred())

			os.Args = []string{"cmd", "--model", testModel, "--ssl-keyfile", keyFile}
			_, err = common.ParseCommandParamsAndLoadConfig()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("both ssl-certfile and ssl-keyfile must be provided together"))
		})

		It("Should start HTTPS server with provided SSL certificates", func(ctx SpecContext) {
			tempDir := GinkgoT().TempDir()
			certFile, keyFile, err := GenerateTempCerts(tempDir)
			Expect(err).NotTo(HaveOccurred())

			args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom,
				"--ssl-certfile", certFile, "--ssl-keyfile", keyFile}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get("https://localhost/health")
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
		})

		It("Should start HTTPS server with self-signed certificates", func(ctx SpecContext) {
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom, "--self-signed-certs"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get("https://localhost/health")
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
		})

	})

	Context("request ID headers", func() {
		testRequestIDHeader := func(enableRequestID bool, endpoint, reqBody, inputRequestID string, expectRequestID *string, validateBody func([]byte)) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeEcho}
			if enableRequestID {
				args = append(args, "--enable-request-id-headers")
			}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			req, err := http.NewRequest("POST", "http://localhost"+endpoint, strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set(fasthttp.HeaderContentType, "application/json")
			if inputRequestID != "" {
				req.Header.Set(requestIDHeader, inputRequestID)
			}

			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			if expectRequestID != nil {
				actualRequestID := resp.Header.Get(requestIDHeader)
				if *expectRequestID != "" {
					// When a request ID is provided, it should be echoed back
					Expect(actualRequestID).To(Equal(*expectRequestID))
				} else {
					// When no request ID is provided, a UUID should be generated
					Expect(actualRequestID).NotTo(BeEmpty())
					Expect(len(actualRequestID)).To(BeNumerically(">", 30))
				}
			} else {
				// When request ID headers are disabled, the header should be empty
				Expect(resp.Header.Get(requestIDHeader)).To(BeEmpty())
			}

			if validateBody != nil {
				body, err := io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())
				validateBody(body)
			}
		}

		DescribeTable("request ID behavior",
			testRequestIDHeader,
			Entry("includes X-Request-Id when enabled",
				true,
				"/v1/chat/completions",
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+testModel+`", "max_tokens": 5}`,
				"test-request-id-123",
				ptr("test-request-id-123"),
				nil,
			),
			Entry("excludes X-Request-Id when disabled",
				false,
				"/v1/chat/completions",
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+testModel+`", "max_tokens": 5}`,
				"test-request-id-456",
				nil,
				nil,
			),
			Entry("includes X-Request-Id in streaming response",
				true,
				"/v1/chat/completions",
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+testModel+`", "max_tokens": 5, "stream": true}`,
				"test-streaming-789",
				ptr("test-streaming-789"),
				nil,
			),
			Entry("works with text completions endpoint",
				true,
				"/v1/completions",
				`{"prompt": "Hello world", "model": "`+testModel+`", "max_tokens": 5}`,
				"text-request-111",
				ptr("text-request-111"),
				nil,
			),
			Entry("generates UUID when no request ID provided",
				true,
				"/v1/chat/completions",
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+testModel+`", "max_tokens": 5}`,
				"",
				ptr(""),
				nil,
			),
			Entry("uses request ID in response body ID field",
				true,
				"/v1/chat/completions",
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+testModel+`", "max_tokens": 5}`,
				"body-test-999",
				ptr("body-test-999"),
				func(body []byte) {
					var resp map[string]any
					Expect(json.Unmarshal(body, &resp)).To(Succeed())
					Expect(resp["id"]).To(Equal("chatcmpl-body-test-999"))
				},
			),
		)
	})

	Context("sleep mode", Ordered, func() {
		It("Should respond to /is_sleeping", func() {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			checkSimSleeping(client, false)
		})

		It("Should not enter sleep mode without the flag", func() {
			ctx := context.TODO()
			client, err := startServerWithEnv(ctx, common.ModeRandom, map[string]string{"VLLM_SERVER_DEV_MODE": "1"})
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Post("http://localhost/sleep", "", nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			checkSimSleeping(client, false)
		})

		It("Should not enter sleep mode without the env var", func() {
			ctx := context.TODO()
			client, err := startServerWithArgs(ctx,
				[]string{"cmd", "--model", qwenModelName, "--mode", common.ModeRandom, "--enable-sleep-mode"})
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Post("http://localhost/sleep", "", nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			checkSimSleeping(client, false)
		})

		It("Should enter sleep mode and wake up", func() {
			topic := kvcache.CreateKVEventsTopic("localhost", qwenModelName)
			sub, endpoint := common.CreateSub(topic)

			ctx := context.TODO()
			client, err := startServerWithArgsAndEnv(ctx, common.ModeRandom,
				[]string{"cmd", "--model", qwenModelName, "--mode", common.ModeRandom, "--enable-sleep-mode",
					"--enable-kvcache", "--v", "5", "--port", "8000", "--zmq-endpoint", endpoint},
				map[string]string{"VLLM_SERVER_DEV_MODE": "1", "POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			//nolint
			defer sub.Close()

			// Send a request, check that a kv event BlockStored was sent
			go func() {
				time.Sleep(200 * time.Millisecond)
				sendTextCompletionRequest(ctx, client)
			}()
			parts, err := sub.RecvMessageBytes(0)
			Expect(err).NotTo(HaveOccurred())
			stored, _, _ := kvcache.ParseKVEvent(parts, topic, uint64(1))
			Expect(stored).To(HaveLen(1))

			// Sleep and check that AllBlocksCleared event was sent
			go func() {
				time.Sleep(200 * time.Millisecond)
				resp, err := client.Post("http://localhost/sleep", "", nil)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))
			}()
			parts, err = sub.RecvMessageBytes(0)
			Expect(err).NotTo(HaveOccurred())
			_, _, allCleared := kvcache.ParseKVEvent(parts, topic, uint64(2))
			Expect(allCleared).To(BeTrue())

			checkSimSleeping(client, true)

			// Send a request
			go sendTextCompletionRequest(ctx, client)

			resp, err := client.Post("http://localhost/wake_up", "", nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			checkSimSleeping(client, false)

			// Send a request, check that a kv event BlockStored was sent,
			// this checks that in sleep mode the kv cache was disabled.
			// The sequence number of the event is an addition check.
			go func() {
				time.Sleep(200 * time.Millisecond)
				sendTextCompletionRequest(ctx, client)
			}()
			parts, err = sub.RecvMessageBytes(0)
			Expect(err).NotTo(HaveOccurred())
			stored, _, _ = kvcache.ParseKVEvent(parts, topic, uint64(3))
			Expect(stored).To(HaveLen(1))

			// Sleep again and wait for AllBlocksCleared
			go func() {
				time.Sleep(200 * time.Millisecond)
				resp, err := client.Post("http://localhost/sleep", "", nil)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))
			}()

			parts, err = sub.RecvMessageBytes(0)
			Expect(err).NotTo(HaveOccurred())
			_, _, allCleared = kvcache.ParseKVEvent(parts, topic, uint64(4))
			Expect(allCleared).To(BeTrue())

			checkSimSleeping(client, true)

			// Wake up the weghts only, kv cache shouldn't wake up yet
			resp, err = client.Post("http://localhost/wake_up?tags=weights", "", nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			checkSimSleeping(client, false)

			// Send a request
			go sendTextCompletionRequest(ctx, client)

			// Now wake up the cache
			resp, err = client.Post("http://localhost/wake_up?tags=kv_cache", "", nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			checkSimSleeping(client, false)

			// Send a request, check that a kv event BlockStored was sent,
			// this checks that the kv cache was disabled after waking up with weights.
			// The sequence number of the event is an addition check.
			go func() {
				time.Sleep(200 * time.Millisecond)
				sendTextCompletionRequest(ctx, client)
			}()
			parts, err = sub.RecvMessageBytes(0)
			Expect(err).NotTo(HaveOccurred())
			stored, _, _ = kvcache.ParseKVEvent(parts, topic, uint64(5))
			Expect(stored).To(HaveLen(1))
		})
	})
})
