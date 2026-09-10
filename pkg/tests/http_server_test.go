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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
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

	It("Should respond to /health/ready", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Get("http://localhost/health/ready")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
	})

	It("Should return 503 on /health/ready during startup-duration", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--startup-duration", "10s"}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Get("http://localhost/health/ready")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusServiceUnavailable))
	})

	It("Should return 200 on /health/ready after startup-duration elapses", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--startup-duration", "100ms"}
		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() int {
			resp, err := client.Get("http://localhost/health/ready")
			if err != nil {
				return 0
			}
			return resp.StatusCode
		}, 5*time.Second, 50*time.Millisecond).Should(Equal(http.StatusOK))
	})

	Context("tokenize", Ordered, func() {
		It("Should return correct response to /tokenize chat", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom,
				"--max-model-len", "2048"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
    			"messages": [{"role": "user", "content": "This is a test"}],
    			"model": "%s"
			}`, common.QwenModelName)
			resp, err := client.Post("http://localhost/tokenize", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			expectedTokens := 23
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var tokenizeResp api.TokenizeResponse
			err = json.Unmarshal(body, &tokenizeResp)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokenizeResp.Count).To(Equal(expectedTokens))
			Expect(tokenizeResp.Tokens).To(HaveLen(expectedTokens))
			Expect(tokenizeResp.MaxModelLen).To(Equal(2048))
		})

		It("Should return correct response to /tokenize text", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom,
				"--max-model-len", "2048"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"prompt": "This is a test",
				"model": "%s"
			}`, common.QwenModelName)
			resp, err := client.Post("http://localhost/tokenize", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var tokenizeResp api.TokenizeResponse
			err = json.Unmarshal(body, &tokenizeResp)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokenizeResp.Count).To(Equal(4))
			Expect(tokenizeResp.Tokens).To(HaveLen(4))
			Expect(tokenizeResp.MaxModelLen).To(Equal(2048))
		})
	})

	DescribeTable("render endpoints",
		func(model, endpoint, reqBody string, assert func(body []byte)) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Post("http://localhost"+endpoint, "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			assert(body)
		},
		Entry("simulate /v1/completions/render with a string prompt",
			common.TestModelName, "/v1/completions/render",
			fmt.Sprintf(`{"model":"%s","prompt":"hello world"}`, common.TestModelName),
			func(body []byte) {
				var arr []api.RenderResponse
				Expect(json.Unmarshal(body, &arr)).To(Succeed())
				Expect(arr).To(HaveLen(1))
				Expect(arr[0].TokenIDs).NotTo(BeEmpty())
				Expect(arr[0].Features).To(BeNil())
			}),
		Entry("simulate /v1/completions/render with an array prompt",
			common.TestModelName, "/v1/completions/render",
			fmt.Sprintf(`{"model":"%s","prompt":["hello","world"]}`, common.TestModelName),
			func(body []byte) {
				var arr []api.RenderResponse
				Expect(json.Unmarshal(body, &arr)).To(Succeed())
				Expect(arr).To(HaveLen(2))
				Expect(arr[0].TokenIDs).NotTo(BeEmpty())
				Expect(arr[0].Features).To(BeNil())
				Expect(arr[1].TokenIDs).NotTo(BeEmpty())
				Expect(arr[1].Features).To(BeNil())
			}),
		Entry("simulate /v1/completions/render with a token-id prompt copies tokens through",
			common.TestModelName, "/v1/completions/render",
			fmt.Sprintf(`{"model":"%s","prompt":[10,20,30]}`, common.TestModelName),
			func(body []byte) {
				var arr []api.RenderResponse
				Expect(json.Unmarshal(body, &arr)).To(Succeed())
				Expect(arr).To(HaveLen(1))
				Expect(arr[0].TokenIDs).To(Equal([]uint32{10, 20, 30}))
				Expect(arr[0].Features).To(BeNil())
			}),
		Entry("simulate /v1/completions/render with a token-id-array prompt copies tokens through",
			common.TestModelName, "/v1/completions/render",
			fmt.Sprintf(`{"model":"%s","prompt":[[1,2],[3,4,5]]}`, common.TestModelName),
			func(body []byte) {
				var arr []api.RenderResponse
				Expect(json.Unmarshal(body, &arr)).To(Succeed())
				Expect(arr).To(HaveLen(2))
				Expect(arr[0].TokenIDs).To(Equal([]uint32{1, 2}))
				Expect(arr[0].Features).To(BeNil())
				Expect(arr[1].TokenIDs).To(Equal([]uint32{3, 4, 5}))
				Expect(arr[1].Features).To(BeNil())
			}),
		Entry("simulate /v1/chat/completions/render with text-only messages",
			common.TestModelName, "/v1/chat/completions/render",
			fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":"This is a test"}]}`, common.TestModelName),
			func(body []byte) {
				var resp api.RenderResponse
				Expect(json.Unmarshal(body, &resp)).To(Succeed())
				Expect(resp.TokenIDs).NotTo(BeEmpty())
				Expect(resp.Features).To(BeNil())
			}),
		Entry("simulate /v1/chat/completions/render with image_url synthesizes mm features",
			common.TestModelName, "/v1/chat/completions/render",
			fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":[`+
				`{"type":"text","text":"describe this"},`+
				`{"type":"image_url","image_url":{"url":"http://example.com/a.png"}},`+
				`{"type":"image_url","image_url":{"url":"http://example.com/b.png"}}`+
				`]}]}`, common.TestModelName),
			func(body []byte) {
				var resp api.RenderResponse
				Expect(json.Unmarshal(body, &resp)).To(Succeed())
				Expect(resp.TokenIDs).NotTo(BeEmpty())
				Expect(resp.Features).NotTo(BeNil())
				Expect(resp.Features.MMHashes).To(HaveKey("image"))
				Expect(resp.Features.MMHashes["image"]).To(HaveLen(2))
				Expect(resp.Features.MMPlaceholders).To(HaveKey("image"))
				Expect(resp.Features.MMPlaceholders["image"]).To(HaveLen(2))
				Expect(resp.Features.KwargsData).To(HaveKey("image"))
				Expect(resp.Features.KwargsData["image"]).To(HaveLen(2))
			}),
		Entry("proxy /v1/completions/render to the upstream renderer (HF model)",
			common.QwenModelName, "/v1/completions/render",
			fmt.Sprintf(`{"model":"%s","prompt":"This is a test"}`, common.QwenModelName),
			func(body []byte) {
				var arr []api.RenderResponse
				Expect(json.Unmarshal(body, &arr)).To(Succeed())
				Expect(arr).NotTo(BeEmpty())
				Expect(arr[0].TokenIDs).NotTo(BeEmpty())
				Expect(arr[0].Features).To(BeNil())
			}),
		Entry("proxy /v1/chat/completions/render to the upstream renderer (HF model)",
			common.QwenModelName, "/v1/chat/completions/render",
			fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":"This is a test"}]}`, common.QwenModelName),
			func(body []byte) {
				var resp api.RenderResponse
				Expect(json.Unmarshal(body, &resp)).To(Succeed())
				Expect(resp.TokenIDs).NotTo(BeEmpty())
				Expect(resp.Features).To(BeNil())
			}),
		Entry("proxy /v1/chat/completions/render with image_url returns mm features (HF model)",
			common.QwenModelName, "/v1/chat/completions/render",
			// Two minimal 1x1 RGB PNGs inlined as data URLs — keeps the test
			// hermetic (no outbound fetch) and small enough to embed.
			// Pixel values: [0,255,0,0] (red) and [0,0,0,255] (blue).
			fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":[`+
				`{"type":"text","text":"describe this"},`+
				`{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"}},`+
				`{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"}}`+
				`]}]}`, common.QwenModelName),
			func(body []byte) {
				var resp api.RenderResponse
				Expect(json.Unmarshal(body, &resp)).To(Succeed())
				Expect(resp.TokenIDs).NotTo(BeEmpty())
				Expect(resp.Features).NotTo(BeNil())
				Expect(resp.Features.MMHashes).To(HaveKey("image"))
				Expect(resp.Features.MMHashes["image"]).To(HaveLen(2))
				Expect(resp.Features.MMPlaceholders).To(HaveKey("image"))
				Expect(resp.Features.MMPlaceholders["image"]).To(HaveLen(2))
				Expect(resp.Features.KwargsData).To(HaveKey("image"))
				Expect(resp.Features.KwargsData["image"]).To(HaveLen(2))
			}),
	)

	Describe("derender endpoints", func() {
		startSim := func(model string) *http.Client {
			client, err := startServerWithArgs(context.TODO(),
				[]string{"cmd", "--model", model, "--mode", common.ModeRandom})
			Expect(err).NotTo(HaveOccurred())
			return client
		}

		post := func(client *http.Client, endpoint, reqBody string) (int, []byte) {
			resp, err := client.Post("http://localhost"+endpoint, "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				Expect(resp.Body.Close()).To(Succeed())
			}()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			return resp.StatusCode, body
		}

		idsAsJSON := func(ids []uint32) string {
			data, err := json.Marshal(ids)
			Expect(err).NotTo(HaveOccurred())
			return string(data)
		}

		// skipWithoutUpstreamDerender skips proxy-mode specs when the render
		// container's vLLM does not serve the derender endpoints.
		skipWithoutUpstreamDerender := func(status int, body []byte) {
			if status == http.StatusInternalServerError && strings.Contains(string(body), "status 404") {
				Skip("render container does not serve /derender")
			}
		}

		chatDerenderRoundTrip := func(model string) {
			client := startSim(model)
			status, body := post(client, "/v1/chat/completions/render",
				fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":"This is a test"}]}`, model))
			Expect(status).To(Equal(http.StatusOK))
			var renderResp api.RenderResponse
			Expect(json.Unmarshal(body, &renderResp)).To(Succeed())
			Expect(renderResp.TokenIDs).NotTo(BeEmpty())

			status, body = post(client, "/v1/chat/completions/derender",
				fmt.Sprintf(`{"model":"%s","prompt_tokens":5,"generate_response":{"request_id":"req-42",`+
					`"choices":[{"index":0,"finish_reason":"stop","token_ids":%s}]}}`,
					model, idsAsJSON(renderResp.TokenIDs)))
			skipWithoutUpstreamDerender(status, body)
			Expect(status).To(Equal(http.StatusOK))

			var resp api.ChatCompletionsResponse
			Expect(json.Unmarshal(body, &resp)).To(Succeed())
			Expect(resp.ID).To(Equal("req-42"))
			Expect(resp.Model).To(Equal(model))
			Expect(resp.Object).To(Equal(api.ChatCompletionObject))
			Expect(resp.Choices).To(HaveLen(1))
			Expect(resp.Choices[0].Message.Role).To(Equal(api.RoleAssistant))
			Expect(resp.Choices[0].Message.Content.Raw).To(ContainSubstring("This is a test"))
			Expect(*resp.Choices[0].FinishReason).To(Equal("stop"))
			Expect(resp.Usage.PromptTokens).To(Equal(5))
			Expect(resp.Usage.CompletionTokens).To(Equal(len(renderResp.TokenIDs)))
			Expect(resp.Usage.TotalTokens).To(Equal(5 + len(renderResp.TokenIDs)))
		}

		textDerenderRoundTrip := func(model string) {
			client := startSim(model)
			status, body := post(client, "/v1/completions/render",
				fmt.Sprintf(`{"model":"%s","prompt":["hello world","good day"]}`, model))
			Expect(status).To(Equal(http.StatusOK))
			var renderResps []api.RenderResponse
			Expect(json.Unmarshal(body, &renderResps)).To(Succeed())
			Expect(renderResps).To(HaveLen(2))

			status, body = post(client, "/v1/completions/derender",
				fmt.Sprintf(`{"model":"%s","prompt_tokens":[3,4],"generate_responses":[`+
					`{"request_id":"req-1","choices":[{"index":0,"finish_reason":"stop","token_ids":%s}]},`+
					`{"request_id":"req-2","choices":[{"index":0,"finish_reason":"length","token_ids":%s}]}]}`,
					model, idsAsJSON(renderResps[0].TokenIDs), idsAsJSON(renderResps[1].TokenIDs)))
			skipWithoutUpstreamDerender(status, body)
			Expect(status).To(Equal(http.StatusOK))

			var resp api.TextCompletionsResponse
			Expect(json.Unmarshal(body, &resp)).To(Succeed())
			Expect(resp.ID).To(Equal("req-1"))
			Expect(resp.Model).To(Equal(model))
			Expect(resp.Object).To(Equal(api.TextCompletionObject))
			Expect(resp.Choices).To(HaveLen(2))
			Expect(resp.Choices[0].Index).To(Equal(0))
			Expect(resp.Choices[0].Text).To(Equal("hello world"))
			Expect(*resp.Choices[0].FinishReason).To(Equal("stop"))
			Expect(resp.Choices[1].Index).To(Equal(1))
			Expect(resp.Choices[1].Text).To(Equal("good day"))
			Expect(*resp.Choices[1].FinishReason).To(Equal("length"))
			totalTokens := len(renderResps[0].TokenIDs) + len(renderResps[1].TokenIDs)
			Expect(resp.Usage.PromptTokens).To(Equal(7))
			Expect(resp.Usage.CompletionTokens).To(Equal(totalTokens))
			Expect(resp.Usage.TotalTokens).To(Equal(7 + totalTokens))
		}

		It("simulate /v1/chat/completions/derender round-trips rendered tokens", func() {
			chatDerenderRoundTrip(common.TestModelName)
		})

		It("simulate /v1/completions/derender round-trips rendered tokens", func() {
			textDerenderRoundTrip(common.TestModelName)
		})

		It("simulate derender defaults to the served model when model is omitted", func() {
			client := startSim(common.TestModelName)
			status, body := post(client, "/v1/chat/completions/derender",
				`{"generate_response":{"request_id":"r","choices":[{"index":0,"token_ids":[1,2]}]}}`)
			Expect(status).To(Equal(http.StatusOK))
			var resp api.ChatCompletionsResponse
			Expect(json.Unmarshal(body, &resp)).To(Succeed())
			Expect(resp.Model).To(Equal(common.TestModelName))
		})

		It("proxy /v1/chat/completions/derender round-trips rendered tokens (HF model)", func() {
			chatDerenderRoundTrip(common.QwenModelName)
		})

		It("proxy /v1/completions/derender round-trips rendered tokens (HF model)", func() {
			textDerenderRoundTrip(common.QwenModelName)
		})

		DescribeTable("request validation",
			func(endpoint, reqBody string, expectedStatus int, expectedMsg string) {
				client := startSim(common.TestModelName)
				status, body := post(client, endpoint, reqBody)
				Expect(status).To(Equal(expectedStatus))
				Expect(string(body)).To(ContainSubstring(expectedMsg))
			},
			Entry("rejects streaming", "/v1/chat/completions/derender",
				`{"stream":true,"generate_response":{"choices":[{"index":0,"token_ids":[1]}]}}`,
				http.StatusBadRequest, "streaming derender is not supported"),
			Entry("rejects an unknown model", "/v1/chat/completions/derender",
				`{"model":"unknown-model","generate_response":{"choices":[{"index":0,"token_ids":[1]}]}}`,
				http.StatusNotFound, "does not exist"),
			Entry("rejects a missing generate_response", "/v1/chat/completions/derender",
				`{}`, http.StatusBadRequest, "generate_response is required"),
			Entry("rejects empty generate_responses", "/v1/completions/derender",
				`{"generate_responses":[]}`, http.StatusBadRequest, "generate_responses must not be empty"),
			Entry("rejects a prompt_tokens length mismatch", "/v1/completions/derender",
				`{"prompt_tokens":[1,2],"generate_responses":[{"choices":[{"index":0,"token_ids":[1]}]}]}`,
				http.StatusBadRequest, "prompt_tokens length (2) must equal generate_responses length (1)"),
			Entry("rejects empty token_ids", "/v1/completions/derender",
				`{"generate_responses":[{"choices":[{"index":0,"token_ids":[]}]}]}`,
				http.StatusBadRequest, "has empty or null token_ids"),
		)
	})

	Context("SSL/HTTPS Configuration", func() {
		It("Should start HTTPS server with provided SSL certificates", func(ctx SpecContext) {
			tempDir := GinkgoT().TempDir()
			certFile, keyFile, err := communication.GenerateTempCerts(tempDir)
			Expect(err).NotTo(HaveOccurred())

			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--ssl-certfile", certFile, "--ssl-keyfile", keyFile}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get("https://localhost/health")
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
		})

		It("Should start HTTPS server with self-signed certificates", func(ctx SpecContext) {
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--self-signed-certs"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get("https://localhost/health")
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
		})

		It("Should reload certificates when files are updated on disk", func(ctx SpecContext) {
			tempDir := GinkgoT().TempDir()
			certFile := filepath.Join(tempDir, "tls.crt")
			keyFile := filepath.Join(tempDir, "tls.key")

			certPEM1, keyPEM1, err := communication.CreateSelfSignedTLSCertificatePEM()
			Expect(err).NotTo(HaveOccurred())
			Expect(os.WriteFile(certFile, certPEM1, 0644)).To(Succeed())
			Expect(os.WriteFile(keyFile, keyPEM1, 0600)).To(Succeed())

			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--ssl-certfile", certFile, "--ssl-keyfile", keyFile}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get("https://localhost/health")
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = resp.Body.Close() }()
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			Expect(resp.TLS).NotTo(BeNil())
			oldSerial := resp.TLS.PeerCertificates[0].SerialNumber

			certPEM2, keyPEM2, err := communication.CreateSelfSignedTLSCertificatePEM()
			Expect(err).NotTo(HaveOccurred())

			// Write to temp files and rename atomically so the watcher never
			// sees a half-written cert/key pair.
			tmpCert := certFile + ".tmp"
			tmpKey := keyFile + ".tmp"
			Expect(os.WriteFile(tmpCert, certPEM2, 0644)).To(Succeed())
			Expect(os.WriteFile(tmpKey, keyPEM2, 0600)).To(Succeed())
			Expect(os.Rename(tmpKey, keyFile)).To(Succeed())
			Expect(os.Rename(tmpCert, certFile)).To(Succeed())

			// Poll until the reloader picks up the new cert (debounce is 250ms).
			Eventually(func(g Gomega) {
				client.CloseIdleConnections()
				r, err := client.Get("https://localhost/health")
				g.Expect(err).NotTo(HaveOccurred())
				defer func() { _ = r.Body.Close() }()
				g.Expect(r.StatusCode).To(Equal(http.StatusOK))
				g.Expect(r.TLS).NotTo(BeNil())
				g.Expect(r.TLS.PeerCertificates[0].SerialNumber).NotTo(Equal(oldSerial))
			}).WithTimeout(5 * time.Second).WithPolling(250 * time.Millisecond).Should(Succeed())
		})

	})

	Context("request ID headers", func() {
		testRequestIDHeader := func(enableRequestID bool, endpoint, reqBody, inputRequestID string, expectRequestID *string, validateBody func([]byte)) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}
			if enableRequestID {
				args = append(args, "--enable-request-id-headers")
			}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			req, err := http.NewRequest("POST", "http://localhost"+endpoint, strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set(fasthttp.HeaderContentType, "application/json")
			if inputRequestID != "" {
				req.Header.Set(communication.RequestIDHeader, inputRequestID)
			}

			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			if expectRequestID != nil {
				actualRequestID := resp.Header.Get(communication.RequestIDHeader)
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
				Expect(resp.Header.Get(communication.RequestIDHeader)).To(BeEmpty())
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
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+common.TestModelName+`", "max_tokens": 20}`,
				"test-request-id-123",
				ptr("test-request-id-123"),
				nil,
			),
			Entry("excludes X-Request-Id when disabled",
				false,
				"/v1/chat/completions",
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+common.TestModelName+`", "max_tokens": 20}`,
				"test-request-id-456",
				nil,
				nil,
			),
			Entry("includes X-Request-Id in streaming response",
				true,
				"/v1/chat/completions",
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+common.TestModelName+`", "max_tokens": 20, "stream": true}`,
				"test-streaming-789",
				ptr("test-streaming-789"),
				nil,
			),
			Entry("works with text completions endpoint",
				true,
				"/v1/completions",
				`{"prompt": "Hello world", "model": "`+common.TestModelName+`", "max_tokens": 5}`,
				"text-request-111",
				ptr("text-request-111"),
				nil,
			),
			Entry("generates UUID when no request ID provided",
				true,
				"/v1/chat/completions",
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+common.TestModelName+`", "max_tokens": 20}`,
				"",
				ptr(""),
				nil,
			),
			Entry("uses request ID in response body ID field",
				true,
				"/v1/chat/completions",
				`{"messages": [{"role": "user", "content": "Hello"}], "model": "`+common.TestModelName+`", "max_tokens": 20}`,
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
				[]string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom, "--enable-sleep-mode"})
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Post("http://localhost/sleep", "", nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			checkSimSleeping(client, false)
		})

		It("Should enter sleep mode and wake up", func() {
			ctx := context.TODO()

			topic := kvcache.CreateKVEventsTopic("localhost", 8000, common.QwenModelName)
			sub, endpoint := common.CreateSub(ctx, topic)

			client, err := startServerWithArgsAndEnv(ctx, common.ModeRandom,
				[]string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom, "--enable-sleep-mode",
					"--enable-kvcache", "--v", "5", "--port", "8000", "--zmq-endpoint", endpoint},
				map[string]string{"VLLM_SERVER_DEV_MODE": "1", "POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			//nolint
			defer sub.Close()

			// Send a request, check that a kv event BlockStored was sent
			go func() {
				time.Sleep(200 * time.Millisecond)
				sendTextCompletionsRequest(ctx, client)
			}()
			msg, err := sub.Recv()
			Expect(err).NotTo(HaveOccurred())
			storedCount, _, _ := kvcache.CountKVEventBlocks(msg.Frames, topic, uint64(1))
			Expect(storedCount).To(Equal(1))

			// Sleep and check that AllBlocksCleared event was sent
			go func() {
				time.Sleep(200 * time.Millisecond)
				resp, err := client.Post("http://localhost/sleep", "", nil)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))
			}()
			msg, err = sub.Recv()
			Expect(err).NotTo(HaveOccurred())
			_, _, allCleared := kvcache.CountKVEventBlocks(msg.Frames, topic, uint64(2))
			Expect(allCleared).To(BeTrue())

			checkSimSleeping(client, true)

			// Send a request
			go sendTextCompletionsRequest(ctx, client)

			resp, err := client.Post("http://localhost/wake_up", "", nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			checkSimSleeping(client, false)

			// Send a request, check that a kv event BlockStored was sent,
			// this checks that in sleep mode the kv cache was disabled.
			// The sequence number of the event is an addition check.
			go func() {
				time.Sleep(200 * time.Millisecond)
				sendTextCompletionsRequest(ctx, client)
			}()
			msg, err = sub.Recv()
			Expect(err).NotTo(HaveOccurred())
			storedCount, _, _ = kvcache.CountKVEventBlocks(msg.Frames, topic, uint64(3))
			Expect(storedCount).To(Equal(1))

			// Sleep again and wait for AllBlocksCleared
			go func() {
				time.Sleep(200 * time.Millisecond)
				resp, err := client.Post("http://localhost/sleep", "", nil)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))
			}()

			msg, err = sub.Recv()
			Expect(err).NotTo(HaveOccurred())
			_, _, allCleared = kvcache.CountKVEventBlocks(msg.Frames, topic, uint64(4))
			Expect(allCleared).To(BeTrue())

			checkSimSleeping(client, true)

			// Wake up the weights only, kv cache shouldn't wake up yet
			resp, err = client.Post("http://localhost/wake_up?tags=weights", "", nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			checkSimSleeping(client, false)

			// Send a request
			go sendTextCompletionsRequest(ctx, client)

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
				sendTextCompletionsRequest(ctx, client)
			}()
			msg, err = sub.Recv()
			Expect(err).NotTo(HaveOccurred())
			storedCount, _, _ = kvcache.CountKVEventBlocks(msg.Frames, topic, uint64(5))
			Expect(storedCount).To(Equal(1))
		})
	})
})
