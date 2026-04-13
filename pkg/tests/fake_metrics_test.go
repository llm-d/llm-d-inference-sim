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
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

var _ = Describe("Fake metrics", Ordered, func() {
	Context("general fake metrics", func() {
		It("Should respond with fake metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{` +
					`"running-requests":10,` +
					`"waiting-requests":30,` +
					`"kv-cache-usage":0.4,` +
					`"request-success-total":{` +
					`"stop":20,` +
					`"length":0,` +
					`"tool_calls":0,` +
					`"remote_decode":0` +
					`},` +
					`"request-prompt-tokens":[10,20,30],` +
					`"request-generation-tokens":[10,20,30],` +
					`"request-max-generation-tokens":[10,20,30],` +
					`"request-params-max-tokens":[10,20,30],` +
					`"ttft-buckets-values":[1,2,3],` +
					`"tpot-buckets-values":[0,0,1,2,3],` +
					`"prefix-cache-hits":750,` +
					`"prefix-cache-queries":2000,` +
					`"loras":[` +
					`{` +
					`"running":"lora4,lora2",` +
					`"waiting":"lora3",` +
					`"timestamp":1257894567` +
					`},` +
					`{` +
					`"running":"lora4,lora3",` +
					`"waiting":"",` +
					`"timestamp":1257894569` +
					`}` +
					`]` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqRunningMetricName, 10)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqWaitingMetricName, 30)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.KVCacheUsageMetricName, 0.4)))
			Expect(metrics).To(ContainSubstring("vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora4,lora2\",waiting_lora_adapters=\"lora3\"} 1.257894567e+09"))
			Expect(metrics).To(ContainSubstring("vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora4,lora3\",waiting_lora_adapters=\"\"} 1.257894569e+09"))

			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, 0.001, 1)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, 0.005, 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, 0.01, 6)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, 0.02, 6)))

			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.01, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.025, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.05, 1)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.075, 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.1, 6)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.15, 6)))

			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.01, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.025, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.05, 1)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.075, 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.1, 6)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.15, 6)))

			buckets := vllmsim.Build125Buckets(1024)
			var expectedCount int

			for _, boundary := range buckets {
				switch boundary {
				case 1.0:
					expectedCount = 10
				case 2.0:
					expectedCount = 30
				default:
					expectedCount = 60
				}

				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.GenerationTokensMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.MaxNumGenerationTokensMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PromptTokensMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ParamMaxTokensMetricName, boundary, expectedCount)))

			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.GenerationTokensMetricName, math.Inf(1), expectedCount)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PromptTokensMetricName, math.Inf(1), expectedCount)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ParamMaxTokensMetricName, math.Inf(1), expectedCount)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:generation_tokens_total{model_name="%s"} 200`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:prompt_tokens_total{model_name="%s"} 200`, common.TestModelName)))

			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="length",model_name="%s"} 0`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="remote_decode",model_name="%s"} 0`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="stop",model_name="%s"} 20`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="tool_calls",model_name="%s"} 0`, common.TestModelName)))

			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.PrefixCacheHitsMetricName, 750)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.PrefixCacheQueriesMetricName, 2000)))
		})

		It("Should generate correct fake metrics using functions", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{` +
					`"running-requests":"oscillate:1:5:1s",` +
					`"waiting-requests":"squarewave:10:15:400ms",` +
					`"kv-cache-usage":"ramp:0:1:700ms"` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			var prevKVCacheUsage float64
			for i := 1; i <= 5; i++ {
				time.Sleep(200 * time.Millisecond)
				resp, err := client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))

				data, err := io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics := string(data)
				metricsLines := strings.Split(metrics, "\n")

				// Running requests: should be various values in [1, 5]
				count := findIntMetric(metricsLines, getCountMetricPrefix(common.TestModelName, vllmsim.ReqRunningMetricName))
				Expect(count).ToNot(BeNil())
				Expect(*count).To(BeNumerically(">=", 1))
				Expect(*count).To(BeNumerically("<=", 5))

				// Waiting requests: should be either 10 or 15
				Expect(metrics).To(Or(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqWaitingMetricName, 10)),
					ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqWaitingMetricName, 15))))

				// KV cache usage: should grow from 0 to 1, and reach 1 after 700ms (i >= 4)
				kvCacheUsage := findFloatMetric(metricsLines, getCountMetricPrefix(common.TestModelName, vllmsim.KVCacheUsageMetricName))
				Expect(kvCacheUsage).ToNot(BeNil())
				if i < 4 {
					Expect(*kvCacheUsage).To(BeNumerically("<", 1))
					Expect(*kvCacheUsage).To(BeNumerically(">", prevKVCacheUsage))
				} else {
					Expect(*kvCacheUsage).To(BeNumerically("==", 1))
				}
				prevKVCacheUsage = *kvCacheUsage
			}
		})

		It("Should generate correct fake metrics using rampreset function", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{` +
					`"kv-cache-usage":"rampreset:1:0:550ms"` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			prevKVCacheUsage := float64(1)
			for i := 1; i <= 5; i++ {
				time.Sleep(200 * time.Millisecond)
				resp, err := client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))

				data, err := io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics := string(data)
				metricsLines := strings.Split(metrics, "\n")

				// KV cache usage: should decrease from 1 towards 0, and reset at 550ms (i=3)
				kvCacheUsage := findFloatMetric(metricsLines, getCountMetricPrefix(common.TestModelName, vllmsim.KVCacheUsageMetricName))
				Expect(kvCacheUsage).ToNot(BeNil())
				if i != 3 {
					Expect(*kvCacheUsage).To(BeNumerically("<=", 1))
					Expect(*kvCacheUsage).To(BeNumerically(">=", 0))
					Expect(*kvCacheUsage).To(BeNumerically("<", prevKVCacheUsage))
				} else {
					Expect(*kvCacheUsage).To(BeNumerically("<=", 1))
					Expect(*kvCacheUsage).To(BeNumerically(">=", 0))
					Expect(*kvCacheUsage).To(BeNumerically(">", prevKVCacheUsage))
				}
				prevKVCacheUsage = *kvCacheUsage
			}
		})

		It("Should use TotalPromptTokens and TotalGenerationTokens if provided", func() {
			ctx := context.TODO()
			args := []string{
				"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{` +
					`"running-requests":5,` +
					`"waiting-requests":2,` +
					`"kv-cache-usage":0.1,` +
					`"request-prompt-tokens":[100,200],` +
					`"request-generation-tokens":[50,150],` +
					`"total-prompt-tokens":12345,` + // explicit total
					`"total-generation-tokens":67890,` + // explicit total
					`"request-success-total":{"stop":10}` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			// Verify that the explicit totals are used
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:prompt_tokens_total{model_name="%s"} 12345`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:generation_tokens_total{model_name="%s"} 67890`, common.TestModelName)))
		})
	})

	Context("fake prefix cache metrics", func() {
		It("Should respond with fake prefix cache metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{"prefix-cache-hits":500,"prefix-cache-queries":1000}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.PrefixCacheQueriesMetricName, 1000)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.PrefixCacheHitsMetricName, 500)))
		})

		It("Should not update prefix cache counters from real requests when fake metrics are set", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--fake-metrics",
				`{"prefix-cache-hits":100,"prefix-cache-queries":200}`,
			}

			client, err := startServerWithArgsAndEnv(ctx, common.ModeRandom, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			// Send a request — counters should not change from the fake values
			_, err = openaiclient.Completions.New(ctx, openai.CompletionNewParams{
				Prompt: openai.CompletionNewParamsPromptUnion{
					OfString: openai.String("What is the weather like in Haifa today?"),
				},
				Model: openai.CompletionNewParamsModel(common.QwenModelName),
			})
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(500 * time.Millisecond)

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)
			// Fake values should be unchanged — reportPrefixCacheStats returns early when FakeMetrics is set
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.PrefixCacheQueriesMetricName, 200)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.PrefixCacheHitsMetricName, 100)))
		})
	})

	Context("fake ttft metrics", func() {
		It("Should respond with fake ttft metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				"{\"ttft-buckets-values\":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}",
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			for _, boundary := range common.TTFTBucketsBoundaries {
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, boundary, 0)))
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, math.Inf(1), 1)))
		})
	})

	Context("fake latency metrics", func() {
		It("should respond with valid fake latency metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
				"--fake-metrics",
				`{` +
					`"e2erl-buckets-values":[0, 1, 2],` +
					`"queue-time-buckets-values":[0, 1, 2],` +
					`"inf-time-buckets-values":[0, 1, 2],` +
					`"prefill-time-buckets-values":[0, 1, 2],` +
					`"decode-time-buckets-values":[0, 1, 2]` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			// buckets counts should be 0, 1, 3, 3, 3, ...
			var expectedCount int

			for i, boundary := range common.RequestLatencyBucketsBoundaries {
				switch i {
				case 0:
					expectedCount = 0
				case 1:
					expectedCount = 1
				default:
					expectedCount = 3
				}

				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.E2EReqLatencyMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqInferenceTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqQueueTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PrefillTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.DecodeTimeMetricName, boundary, expectedCount)))
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.E2EReqLatencyMetricName, math.Inf(1), 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqInferenceTimeMetricName, math.Inf(1), 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqQueueTimeMetricName, math.Inf(1), 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PrefillTimeMetricName, math.Inf(1), 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.DecodeTimeMetricName, math.Inf(1), 3)))
		})
	})

	Context("update fake metrics", func() {
		It("Should update fake metrics with functions correctly", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{` +
					`"running-requests":"oscillate:1:5:1s",` +
					`"waiting-requests":30,` +
					`"kv-cache-usage":0.4` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(200 * time.Millisecond)

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)
			metricsLines := strings.Split(metrics, "\n")

			// Running requests: should be various values in [1, 5]
			count := findIntMetric(metricsLines, getCountMetricPrefix(common.TestModelName, vllmsim.ReqRunningMetricName))
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">=", 1))
			Expect(*count).To(BeNumerically("<=", 5))

			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqWaitingMetricName, 30)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.KVCacheUsageMetricName, 0.4)))

			// Update
			reqBody := `{
            "running-requests":15,
            "waiting-requests":0,
            "kv-cache-usage":0.9
        }`

			req, err := http.NewRequest("POST", updateFakeMetricsUrl, strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			resp, err = client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

			time.Sleep(200 * time.Millisecond)

			resp, err = client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err = io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics = string(data)

			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqRunningMetricName, 15)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqWaitingMetricName, 0)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.KVCacheUsageMetricName, 0.9)))

			// Update
			reqBody = `{
            "running-requests":"oscillate:10:50:1s",
            "waiting-requests":30,
            "kv-cache-usage":"ramp:0:1:150ms"
        }`

			req, err = http.NewRequest("POST", updateFakeMetricsUrl, strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			resp, err = client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

			time.Sleep(400 * time.Millisecond)

			resp, err = client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err = io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics = string(data)
			metricsLines = strings.Split(metrics, "\n")

			// Running requests: should be various values in [10, 50]
			count = findIntMetric(metricsLines, getCountMetricPrefix(common.TestModelName, vllmsim.ReqRunningMetricName))
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">=", 10))
			Expect(*count).To(BeNumerically("<=", 50))

			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqWaitingMetricName, 30)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.KVCacheUsageMetricName, 1)))

		})

		It("Should update fake ttft and tpot metrics correctly", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{` +
					`"ttft-buckets-values":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],` +
					`"tpot-buckets-values":[0,0,1,2,3]` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			for _, boundary := range common.TTFTBucketsBoundaries {
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, boundary, 0)))
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, math.Inf(1), 1)))

			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.01, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.025, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.05, 1)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.075, 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.1, 6)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TPOTMetricName, 0.15, 6)))

			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.01, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.025, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.05, 1)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.075, 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.1, 6)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.InterTokenLatencyMetricName, 0.15, 6)))

			// Update
			reqBody := `{
            "ttft-buckets-values":[1,2,3],
			"tpot-buckets-values":null
        }`

			req, err := http.NewRequest("POST", updateFakeMetricsUrl, strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			resp, err = client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

			resp, err = client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err = io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics = string(data)
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, 0.001, 1)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, 0.005, 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, 0.01, 6)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, 0.02, 6)))

			Expect(metrics).NotTo(ContainSubstring(vllmsim.TPOTMetricName))
			Expect(metrics).NotTo(ContainSubstring(vllmsim.InterTokenLatencyMetricName))
		})

		It("Should update fake latency and token-param histogram metrics correctly", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
				"--fake-metrics",
				`{` +
					`"e2erl-buckets-values":[0, 1, 2],` +
					`"queue-time-buckets-values":[0, 1, 2],` +
					`"inf-time-buckets-values":[0, 1, 2],` +
					`"prefill-time-buckets-values":[0, 1, 2],` +
					`"decode-time-buckets-values":[0, 1, 2],` +
					`"request-params-max-tokens":[10,20,30],` +
					`"request-max-generation-tokens":[10,20,30]` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			// Verify initial state
			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			// Initial latency buckets: counts should be 0, 1, 3, 3, 3, ...
			var expectedCount int
			for i, boundary := range common.RequestLatencyBucketsBoundaries {
				switch i {
				case 0:
					expectedCount = 0
				case 1:
					expectedCount = 1
				default:
					expectedCount = 3
				}
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.E2EReqLatencyMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqQueueTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqInferenceTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PrefillTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.DecodeTimeMetricName, boundary, expectedCount)))
			}

			// Initial token-param buckets: [10,20,30] on Build125Buckets
			buckets := vllmsim.Build125Buckets(1024)
			for _, boundary := range buckets {
				switch {
				case boundary <= 1:
					expectedCount = 10
				case boundary <= 2:
					expectedCount = 30
				default:
					expectedCount = 60
				}
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ParamMaxTokensMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.MaxNumGenerationTokensMetricName, boundary, expectedCount)))
			}

			// Update all histograms
			reqBody := `{
				"e2erl-buckets-values":[1, 0, 0, 1],
				"queue-time-buckets-values":[1, 0, 0, 1],
				"inf-time-buckets-values":[1, 0, 0, 1],
				"prefill-time-buckets-values":[1, 0, 0, 1],
				"decode-time-buckets-values":[1, 0, 0, 1],
				"request-params-max-tokens":[1,2,3],
				"request-max-generation-tokens":[1,2,3]
			}`

			req, err := http.NewRequest("POST", updateFakeMetricsUrl, strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			resp, err = client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

			resp, err = client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err = io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics = string(data)

			// After update: latency buckets [1, 0, 0, 1] → counts: 1, 1, 1, 2, 2, ...
			for i, boundary := range common.RequestLatencyBucketsBoundaries {
				switch {
				case i < 3:
					expectedCount = 1
				default:
					expectedCount = 2
				}
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.E2EReqLatencyMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqQueueTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqInferenceTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PrefillTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.DecodeTimeMetricName, boundary, expectedCount)))
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.E2EReqLatencyMetricName, math.Inf(1), 2)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqQueueTimeMetricName, math.Inf(1), 2)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqInferenceTimeMetricName, math.Inf(1), 2)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PrefillTimeMetricName, math.Inf(1), 2)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.DecodeTimeMetricName, math.Inf(1), 2)))

			// After update: token-param buckets [1,2,3]
			for _, boundary := range buckets {
				switch {
				case boundary <= 1:
					expectedCount = 1
				case boundary <= 2:
					expectedCount = 3
				default:
					expectedCount = 6
				}
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ParamMaxTokensMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.MaxNumGenerationTokensMetricName, boundary, expectedCount)))
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ParamMaxTokensMetricName, math.Inf(1), 6)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.MaxNumGenerationTokensMetricName, math.Inf(1), 6)))
		})

		It("Should update fake request-success-total, prefix-cache-hits and prefix-cache-queries correctly", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{` +
					`"request-success-total":{"stop":20,"length":5,"tool_calls":0,"remote_decode":0},` +
					`"prefix-cache-hits":500,` +
					`"prefix-cache-queries":1000` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			// Verify initial state
			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="stop",model_name="%s"} 20`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="length",model_name="%s"} 5`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.PrefixCacheHitsMetricName, 500)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.PrefixCacheQueriesMetricName, 1000)))

			// Update all three
			reqBody := `{
				"request-success-total":{"stop":100,"length":50,"tool_calls":10,"remote_decode":5},
				"prefix-cache-hits":750,
				"prefix-cache-queries":2000
			}`

			req, err := http.NewRequest("POST", updateFakeMetricsUrl, strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			resp, err = client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

			resp, err = client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err = io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics = string(data)

			// After update: values should be replaced, not accumulated
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="stop",model_name="%s"} 100`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="length",model_name="%s"} 50`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="tool_calls",model_name="%s"} 10`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="remote_decode",model_name="%s"} 5`, common.TestModelName)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.PrefixCacheHitsMetricName, 750)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.PrefixCacheQueriesMetricName, 2000)))
		})

		It("Should update fake lora metrics correctly", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{` +
					`"loras":[` +
					`{"running":"lora1,lora2","waiting":"lora3","timestamp":1000000001},` +
					`{"running":"lora1","waiting":"","timestamp":1000000002}` +
					`]` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			// Verify initial state
			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			Expect(metrics).To(ContainSubstring(`vllm:lora_requests_info{max_lora="1",running_lora_adapters="lora1,lora2",waiting_lora_adapters="lora3"} 1.000000001e+09`))
			Expect(metrics).To(ContainSubstring(`vllm:lora_requests_info{max_lora="1",running_lora_adapters="lora1",waiting_lora_adapters=""} 1.000000002e+09`))

			// Update lora metrics
			reqBody := `{
				"loras":[
					{"running":"lora4","waiting":"lora5,lora6","timestamp":2000000001}
				]
			}`

			req, err := http.NewRequest("POST", updateFakeMetricsUrl, strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			resp, err = client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

			resp, err = client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err = io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics = string(data)

			// Old lora entries should be gone, only new one present
			Expect(metrics).NotTo(ContainSubstring("lora1"))
			Expect(metrics).NotTo(ContainSubstring("lora2"))
			Expect(metrics).NotTo(ContainSubstring("lora3"))
			Expect(metrics).To(ContainSubstring(`vllm:lora_requests_info{max_lora="1",running_lora_adapters="lora4",waiting_lora_adapters="lora5,lora6"} 2.000000001e+09`))

			// Update lora metrics with an empty array
			reqBody = `{
				"loras":[]
			}`

			req, err = http.NewRequest("POST", updateFakeMetricsUrl, strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			resp, err = client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

			resp, err = client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err = io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics = string(data)

			// Old lora entries should be gone, only a new empty one present
			Expect(metrics).NotTo(ContainSubstring("lora4"))
			Expect(metrics).NotTo(ContainSubstring("lora5"))
			Expect(metrics).NotTo(ContainSubstring("lora6"))
			Expect(metrics).To(ContainSubstring(`vllm:lora_requests_info{max_lora="1",running_lora_adapters="",waiting_lora_adapters=""}`))
		})

		// This table tests the update logic for both prompt and generation token metrics.
		// Each entry exercises a 3-phase lifecycle: initial load → first POST update → second POST update.
		// For each phase, the test verifies the histogram buckets and total counter for both
		// request_prompt_tokens / prompt_tokens_total and request_generation_tokens / generation_tokens_total.
		//
		// Parameters per entry:
		//   initialMetrics   – JSON for --fake-metrics flag at startup
		//   initialPrompt/Gen – expected state after startup
		//   firstUpdate      – JSON body for the first POST to /fake_metrics
		//   firstPrompt/Gen  – expected state after first update
		//   secondUpdate     – JSON body for the second POST
		//   secondPrompt/Gen – expected state after second update
		//
		// tokenTestPhase.checkBuckets == nil means the histogram metric must be absent.
		// tokenTestPhase.total == nil means the total counter must be absent.
		DescribeTable("Should update fake request token metrics correctly",
			func(initialMetrics string, initialPrompt tokenTestPhase, initialGen tokenTestPhase,
				firstUpdate string, firstPrompt tokenTestPhase, firstGen tokenTestPhase,
				secondUpdate string, secondPrompt tokenTestPhase, secondGen tokenTestPhase) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
					"--fake-metrics", initialMetrics}

				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				// Verify initial state
				resp, err := client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))

				data, err := io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics := string(data)

				verifyTokenMetrics(metrics, vllmsim.PromptTokensMetricName, vllmsim.PromptTokensTotalMetricName, initialPrompt)
				verifyTokenMetrics(metrics, vllmsim.GenerationTokensMetricName, vllmsim.GenerationTokensTotalMetricName, initialGen)

				// First update: POST new fake metrics and verify
				req, err := http.NewRequest("POST", updateFakeMetricsUrl, strings.NewReader(firstUpdate))
				Expect(err).NotTo(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")
				resp, err = client.Do(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

				resp, err = client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))

				data, err = io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics = string(data)

				verifyTokenMetrics(metrics, vllmsim.PromptTokensMetricName, vllmsim.PromptTokensTotalMetricName, firstPrompt)
				verifyTokenMetrics(metrics, vllmsim.GenerationTokensMetricName, vllmsim.GenerationTokensTotalMetricName, firstGen)

				// Second update: POST new fake metrics and verify
				req, err = http.NewRequest("POST", updateFakeMetricsUrl, strings.NewReader(secondUpdate))
				Expect(err).NotTo(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")
				resp, err = client.Do(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

				resp, err = client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))

				data, err = io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics = string(data)

				verifyTokenMetrics(metrics, vllmsim.PromptTokensMetricName, vllmsim.PromptTokensTotalMetricName, secondPrompt)
				verifyTokenMetrics(metrics, vllmsim.GenerationTokensMetricName, vllmsim.GenerationTokensTotalMetricName, secondGen)
			},
			// Prompt tokens: hist+total → update hist → update total
			// Generated tokens: only total → add hist → update total
			Entry("#1 Prompt tokens: hist+total, Generated tokens: only total; then update hists, then update totals",
				`{"request-prompt-tokens":[1,2,3], "total-prompt-tokens":12345, "total-generation-tokens":54321}`,
				tokenTestPhase{checkBuckets123, intPtr(12345)}, tokenTestPhase{nil, intPtr(54321)},
				`{"request-prompt-tokens":[10, 20], "request-generation-tokens":[10, 20]}`,
				tokenTestPhase{checkBuckets10_20, intPtr(50)}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"total-prompt-tokens":58, "total-generation-tokens":99}`,
				tokenTestPhase{checkBuckets10_20, intPtr(58)}, tokenTestPhase{checkBuckets10_20, intPtr(99)}),
			// Prompt tokens: only hist → update hist → update total
			// Generated tokens:    only hist → update hist → update total
			Entry("#2 Both: only hist; then update hists, then set totals",
				`{"request-prompt-tokens":[1,2,3], "request-generation-tokens":[10, 20]}`,
				tokenTestPhase{checkBuckets123, intPtr(20)}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"request-prompt-tokens":[10, 20], "request-generation-tokens":[1,2,3]}`,
				tokenTestPhase{checkBuckets10_20, intPtr(50)}, tokenTestPhase{checkBuckets123, intPtr(20)},
				`{"total-prompt-tokens":58, "total-generation-tokens":99}`,
				tokenTestPhase{checkBuckets10_20, intPtr(58)}, tokenTestPhase{checkBuckets123, intPtr(99)}),
			// Prompt tokens: empty → set total → add hist
			// Generated tokens: empty → set total → add hist
			Entry("#3 Both: empty; then set totals, then add hists",
				`{}`,
				tokenTestPhase{nil, nil}, tokenTestPhase{nil, nil},
				`{"total-prompt-tokens":58, "total-generation-tokens":77}`,
				tokenTestPhase{nil, intPtr(58)}, tokenTestPhase{nil, intPtr(77)},
				`{"request-prompt-tokens":[10, 20], "request-generation-tokens":[1,2,3]}`,
				tokenTestPhase{checkBuckets10_20, intPtr(50)}, tokenTestPhase{checkBuckets123, intPtr(20)}),
			// Prompt tokens: hist+total → empty (no change) → null hist (remove)
			// Generated tokens: hist+total → empty (no change) → null hist (remove)
			Entry("#4 Both: hist+total; then empty, then null hists",
				`{"request-prompt-tokens":[1,2,3], "total-prompt-tokens":12345, "request-generation-tokens":[10,20], "total-generation-tokens":54321}`,
				tokenTestPhase{checkBuckets123, intPtr(12345)}, tokenTestPhase{checkBuckets10_20, intPtr(54321)},
				`{}`,
				tokenTestPhase{checkBuckets123, intPtr(12345)}, tokenTestPhase{checkBuckets10_20, intPtr(54321)},
				`{"request-prompt-tokens":null, "request-generation-tokens":null}`,
				tokenTestPhase{nil, nil}, tokenTestPhase{nil, nil}),
			// Prompt tokens: only hist → empty hist → update hist
			// Generated tokens: only hist → null total (no-op) → update hist
			Entry("#5 Both: only hist; then empty/null clears, then re-add hists",
				`{"request-prompt-tokens":[1,2,3], "request-generation-tokens":[10,20]}`,
				tokenTestPhase{checkBuckets123, intPtr(20)}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"request-prompt-tokens":[], "total-generation-tokens":null}`,
				tokenTestPhase{nil, nil}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"request-prompt-tokens":[10, 20], "request-generation-tokens":[1,2,3]}`,
				tokenTestPhase{checkBuckets10_20, intPtr(50)}, tokenTestPhase{checkBuckets123, intPtr(20)}),
			// Prompt tokens: empty → add hist only → update with hist+total simultaneously
			// Generated tokens: empty → add hist only → update with hist+total simultaneously
			Entry("#6 Both: empty; then add hists, then update with hist+total simultaneously",
				`{}`,
				tokenTestPhase{nil, nil}, tokenTestPhase{nil, nil},
				`{"request-prompt-tokens":[1,2,3], "request-generation-tokens":[10, 20]}`,
				tokenTestPhase{checkBuckets123, intPtr(20)}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"request-prompt-tokens":[10,20], "total-prompt-tokens":999, "request-generation-tokens":[1,2,3], "total-generation-tokens":888}`,
				tokenTestPhase{checkBuckets10_20, intPtr(999)}, tokenTestPhase{checkBuckets123, intPtr(888)}),
			// Prompt tokens: only total → update total → add hist+total simultaneously
			// Generated tokens: only total → update total → add hist+total simultaneously
			Entry("#7 Both: only total; then update totals, then add hist+total simultaneously",
				`{"total-prompt-tokens":100, "total-generation-tokens":200}`,
				tokenTestPhase{nil, intPtr(100)}, tokenTestPhase{nil, intPtr(200)},
				`{"total-prompt-tokens":300, "total-generation-tokens":400}`,
				tokenTestPhase{nil, intPtr(300)}, tokenTestPhase{nil, intPtr(400)},
				`{"request-prompt-tokens":[10,20], "total-prompt-tokens":999, "request-generation-tokens":[1,2,3], "total-generation-tokens":888}`,
				tokenTestPhase{checkBuckets10_20, intPtr(999)}, tokenTestPhase{checkBuckets123, intPtr(888)}),
		)
	})
})

type checkBucketsFunc func(metrics string, metricName string)

// tokenTestPhase describes the expected state of a token metric (prompt or generation)
// after a phase (initial load, first update, or second update).
type tokenTestPhase struct {
	checkBuckets checkBucketsFunc // nil means histogram should be absent
	total        *int             // nil means the total counter should be absent
}

func intPtr(v int) *int { return &v }

// verifyTokenMetrics checks the histogram and total counter for a single token type.
func verifyTokenMetrics(metrics string, histMetricName string, totalMetricName string, phase tokenTestPhase) {
	if phase.total != nil {
		Expect(metrics).To(ContainSubstring(fmt.Sprintf(`%s{model_name="%s"} %d`,
			totalMetricName, common.TestModelName, *phase.total)))
	} else {
		Expect(metrics).NotTo(ContainSubstring(totalMetricName))
	}
	if phase.checkBuckets != nil {
		phase.checkBuckets(metrics, histMetricName)
	} else {
		Expect(metrics).NotTo(ContainSubstring(histMetricName))
	}
}

func checkBuckets123(metrics string, metricName string) {
	buckets := vllmsim.Build125Buckets(1024)
	for _, boundary := range buckets {
		switch {
		case boundary <= 1:
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 1)))
		case boundary <= 2:
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 3)))
		default:
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 6)))
		}
	}
	Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
		math.Inf(1), 6)))
}

func checkBuckets10_20(metrics string, metricName string) {
	buckets := vllmsim.Build125Buckets(1024)
	for _, boundary := range buckets {
		switch {
		case boundary <= 1:
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 10)))
		default:
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 30)))
		}
	}
	Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
		math.Inf(1), 30)))
}
