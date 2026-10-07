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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/engine/sglang"
)

var _ = Describe("sglang engine", func() {
	var client *http.Client

	BeforeEach(func() {
		var err error
		client, err = startServerWithArgs(context.TODO(),
			[]string{"cmd", "--engine", "sglang", "--model", common.TestModelName, "--mode", common.ModeEcho})
		Expect(err).NotTo(HaveOccurred())
	})

	It("Should serve the OpenAI-compatible endpoints", func() {
		message := "This is a test."
		openaiclient, params := getOpenAIClientAndTextParams(client, common.TestModelName, message, false)
		resp, err := openaiclient.Completions.New(context.TODO(), params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Choices).To(HaveLen(1))
		Expect(resp.Choices[0].Text).To(Equal(message))
	})

	It("Should report itself as the owner of the model", func() {
		resp, err := client.Get(baseURL + "/models")
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		var models api.ModelsResponse
		Expect(json.Unmarshal(body, &models)).To(Succeed())
		Expect(models.Data).To(HaveLen(1))
		Expect(models.Data[0].OwnedBy).To(Equal("sglang"))
	})

	// The engine reports the sglang:-prefixed subset of sglang's metrics, with
	// the per-request ones split by request kind, so one non-streaming and one
	// streaming request are needed to see both halves of the split. Real sglang
	// mounts /metrics only under --enable-metrics; here the route always exists.
	It("Should expose the sglang metrics", func() {
		openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, false)
		_, err := openaiclient.Chat.Completions.New(context.TODO(), params)
		Expect(err).NotTo(HaveOccurred())

		_, streamParams := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, true)
		stream := openaiclient.Chat.Completions.NewStreaming(context.TODO(), streamParams)
		for stream.Next() {
		}
		Expect(stream.Err()).NotTo(HaveOccurred())
		Expect(stream.Close()).To(Succeed())

		eventuallyMetrics(client, func(g Gomega, metricsData string) {
			g.Expect(metricsData).To(ContainSubstring(
				getCountMetricLine(common.TestModelName, sglang.SGLangRunningReqsMetricName, 0)))
			g.Expect(metricsData).To(ContainSubstring(
				getCountMetricLine(common.TestModelName, sglang.SGLangQueueReqsMetricName, 0)))

			g.Expect(metricsData).To(ContainSubstring(
				getCountMetricLine(common.TestModelName, sglang.SGLangContextLenMetricName, 1024)))

			// Both requests reached first token, decode, and completion, so each
			// of the request-kind series carries exactly one of them.
			for _, isStreaming := range []string{"false", "true"} {
				for _, metric := range []string{
					sglang.SGLangTTFTMetricName,
					sglang.SGLangE2EReqLatencyMetricName,
					sglang.SGLangReqTPOTMetricName,
				} {
					g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`%s_count{is_streaming="%s",model_name="%s"} 1`,
						metric, isStreaming, common.TestModelName)))
				}
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`%s{is_streaming="%s",model_name="%s"} 1`,
					sglang.SGLangRequestsTotalMetricName, isStreaming, common.TestModelName)))
				g.Expect(metricsData).To(MatchRegexp(fmt.Sprintf(`%s{is_streaming="%s",model_name="%s"} [1-9]`,
					sglang.SGLangPromptTokensTotalMetricName, isStreaming, common.TestModelName)))
				g.Expect(metricsData).To(MatchRegexp(fmt.Sprintf(`%s{is_streaming="%s",model_name="%s"} [1-9]`,
					sglang.SGLangGenerationTokensTotalMetricName, isStreaming, common.TestModelName)))
			}

			// The token-length histograms record one observation per request, and
			// the inter-token latency one per generated token beyond the first.
			// sglang:queue_time_seconds is absent: a free worker takes a request
			// without it ever entering the waiting queue, so nothing was queued
			// here. Its observation is covered by the adapter's own suite.
			for _, metric := range []string{
				sglang.SGLangPromptTokensHistMetricName,
				sglang.SGLangUncachedPromptTokensHistMetricName,
				sglang.SGLangGenerationTokensHistMetricName,
			} {
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`%s_count{model_name="%s"} 2`,
					metric, common.TestModelName)))
			}

			g.Expect(metricsData).To(MatchRegexp(fmt.Sprintf(`%s_count{model_name="%s"} [1-9]`,
				sglang.SGLangInterTokenLatencyMetricName, common.TestModelName)))

			// With no KV cache, no prompt token is cached, so the uncached
			// histogram's sum matches the prompt histogram's.
			metricsLines := strings.Split(metricsData, "\n")
			promptSum := findFloatMetric(metricsLines, sglang.SGLangPromptTokensHistMetricName+"_sum")
			uncachedSum := findFloatMetric(metricsLines, sglang.SGLangUncachedPromptTokensHistMetricName+"_sum")
			g.Expect(promptSum).NotTo(BeNil())
			g.Expect(*promptSum).To(BeNumerically(">", 0))
			g.Expect(uncachedSum).To(HaveValue(Equal(*promptSum)))
		})
	})

	// No vLLM metric leaks into this engine's surface.
	It("Should expose no vllm metrics", func() {
		Expect(sampleMetrics(client)).NotTo(ContainSubstring("vllm:"))
	})

	// The Messages API's envelope is the same for every engine; the error type
	// inside it is Anthropic's own vocabulary under this engine, where vLLM keeps
	// its "BadRequestError" spelling.
	It("Should report a Messages API error in Anthropic's vocabulary", func() {
		body := `{"model":"` + common.TestModelName + `","max_tokens":100,"messages":[]}`
		resp, err := client.Post("http://localhost/v1/messages", "application/json",
			bytes.NewBufferString(body))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		respBody, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		var errResp api.MessagesErrorResponse
		Expect(json.Unmarshal(respBody, &errResp)).To(Succeed())
		Expect(errResp.Type).To(Equal(api.MessagesTypeError))
		Expect(errResp.Error.Type).To(Equal("invalid_request_error"))
		Expect(errResp.Error.Message).To(ContainSubstring("messages must not be empty"))
	})

	// The Responses API nests its errors whatever the engine, and this engine
	// spells every error type there the same way whatever the status.
	It("Should report a Responses API error in sglang's format", func() {
		body := `{"model":"no-such-model","input":"hello"}`
		resp, err := client.Post(baseURL+"/responses", "application/json", bytes.NewBufferString(body))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		Expect(resp.StatusCode).To(Equal(http.StatusNotFound))

		respBody, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		var errResp api.ErrorResponse
		Expect(json.Unmarshal(respBody, &errResp)).To(Succeed())
		Expect(errResp.Error.Type).To(Equal("invalid_request_error"))
		Expect(errResp.Error.Code).To(Equal(http.StatusNotFound))
		Expect(errResp.Error.Message).To(ContainSubstring("does not exist"))
	})

	// sglang puts the error's own fields at the top level, where vLLM wraps them
	// under an "error" key.
	It("Should report an error in sglang's format", func() {
		resp, err := client.Post(baseURL+"/completions", "application/json",
			bytes.NewBufferString(`{"model":"`+common.TestModelName+`","prompt":"hi","max_tokens":0}`))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		var errBody map[string]any
		Expect(json.Unmarshal(body, &errBody)).To(Succeed())
		Expect(errBody).To(HaveKeyWithValue("object", "error"))
		Expect(errBody).To(HaveKey("message"))
		Expect(errBody).NotTo(HaveKey("error"))
	})

	// Every route vLLM's BindHTTP registers, so that a route added there is
	// either implemented here or reported as absent rather than inherited.
	DescribeTable("should not register a vLLM-only route",
		func(method string, route string) {
			req, err := http.NewRequest(method, "http://localhost"+route, http.NoBody)
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")

			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				Expect(resp.Body.Close()).To(Succeed())
			}()
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound))
		},
		Entry("generate", http.MethodPost, "/inference/v1/generate"),
		Entry("load_lora_adapter", http.MethodPost, "/v1/load_lora_adapter"),
		Entry("unload_lora_adapter", http.MethodPost, "/v1/unload_lora_adapter"),
		Entry("mooncake query", http.MethodGet, "/query"),
		Entry("sleep", http.MethodPost, "/sleep"),
		Entry("wake_up", http.MethodPost, "/wake_up"),
		Entry("is_sleeping", http.MethodGet, "/is_sleeping"),
		Entry("completions render", http.MethodPost, "/v1/completions/render"),
		Entry("chat completions render", http.MethodPost, "/v1/chat/completions/render"),
		Entry("responses render", http.MethodPost, "/v1/responses/render"),
		Entry("messages render", http.MethodPost, "/v1/messages/render"),
		Entry("chat completions derender", http.MethodPost, "/v1/chat/completions/derender"),
		Entry("completions derender", http.MethodPost, "/v1/completions/derender"),
	)

	It("Should serve the routes sglang has: /tokenize and /v1/messages/count_tokens", func() {
		for path, body := range map[string]string{
			"/tokenize": `{"model":"` + common.TestModelName + `","prompt":"hi"}`,
			"/v1/messages/count_tokens": `{"model":"` + common.TestModelName +
				`","messages":[{"role":"user","content":"hi"}]}`,
		} {
			resp, err := client.Post("http://localhost"+path, "application/json", strings.NewReader(body))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), "path %s", path)
		}
	})
})

var _ = Describe("sglang engine startup", func() {
	It("Should refuse to start with the KV cache enabled, naming the feature", func() {
		_, err := startServerWithArgs(context.TODO(),
			[]string{"cmd", "--engine", "sglang", "--model", common.TestModelName, "--enable-kvcache"})
		Expect(err).To(MatchError(ContainSubstring("KV-cache simulation is not implemented by the sglang engine yet")))
	})

	It("Should refuse a config file written for vLLM", func() {
		_, err := startServerWithArgs(context.TODO(),
			[]string{"cmd", "--engine", "sglang", "--config", "../../manifests/vllm-config.yaml"})
		Expect(err).To(MatchError(ContainSubstring("does not recognize the following configuration key(s)")))
	})
})
