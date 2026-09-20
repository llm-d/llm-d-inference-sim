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

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/engine/vllm"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
	"github.com/llm-d/llm-d-inference-sim/pkg/simulator"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/prometheus/client_golang/prometheus"
)

func postFakeMetricsUpdate(client *http.Client, body string) (*http.Response, error) {
	req, err := http.NewRequest(
		"POST",
		adminConfigURL,
		strings.NewReader(`{"fake-metrics":`+body+`}`),
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return client.Do(req)
}

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
					`"request-tpot-buckets-values":[0,2,4,6,8],` +
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

			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqRunningMetricName, 10)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqWaitingMetricName, 30)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMKVCacheUsageMetricName, 0.4)))
				g.Expect(metricsData).To(ContainSubstring("vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora4,lora2\",waiting_lora_adapters=\"lora3\"} 1.257894567e+09"))
				g.Expect(metricsData).To(ContainSubstring("vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora4,lora3\",waiting_lora_adapters=\"\"} 1.257894569e+09"))

				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, 0.001, 1)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, 0.005, 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, 0.01, 6)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, 0.02, 6)))

				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.01, 0)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.025, 0)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.05, 1)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.075, 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.1, 6)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.15, 6)))

				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.01, 0)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.025, 0)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.05, 1)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.075, 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.1, 6)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.15, 6)))

				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqTPOTMetricName, 0.01, 0)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqTPOTMetricName, 0.025, 2)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqTPOTMetricName, 0.05, 6)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqTPOTMetricName, 0.075, 12)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqTPOTMetricName, 0.1, 20)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqTPOTMetricName, 0.15, 20)))

				buckets := metrics.BuildBuckets(1024, vllm.TokenBucketMantissas)
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

					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMGenerationTokensMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMMaxNumGenerationTokensMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMPromptTokensMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMParamMaxTokensMetricName, boundary, expectedCount)))

				}
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMGenerationTokensMetricName, math.Inf(1), expectedCount)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMPromptTokensMetricName, math.Inf(1), expectedCount)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMParamMaxTokensMetricName, math.Inf(1), expectedCount)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:generation_tokens_total{model_name="%s"} 200`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:prompt_tokens_total{model_name="%s"} 200`, common.TestModelName)))

				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="length",model_name="%s"} 0`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="remote_decode",model_name="%s"} 0`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="stop",model_name="%s"} 20`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="tool_calls",model_name="%s"} 0`, common.TestModelName)))

				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheHitsTotalMetricName, 750)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheQueriesTotalMetricName, 2000)))
			})
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
				metricsData := sampleMetrics(client)
				metricsLines := strings.Split(metricsData, "\n")

				// Running requests: should be various values in [1, 5]
				count := findIntMetric(metricsLines, getCountMetricPrefix(common.TestModelName, vllm.VLLMReqRunningMetricName))
				Expect(count).ToNot(BeNil())
				Expect(*count).To(BeNumerically(">=", 1))
				Expect(*count).To(BeNumerically("<=", 5))

				// Waiting requests: should be either 10 or 15
				Expect(metricsData).To(Or(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqWaitingMetricName, 10)),
					ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqWaitingMetricName, 15))))

				// KV cache usage: should grow from 0 to 1, and reach 1 after 700ms (i >= 4)
				kvCacheUsage := findFloatMetric(metricsLines, getCountMetricPrefix(common.TestModelName, vllm.VLLMKVCacheUsageMetricName))
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
				metricsLines := strings.Split(sampleMetrics(client), "\n")

				// KV cache usage: should decrease from 1 towards 0, and reset at 550ms (i=3)
				kvCacheUsage := findFloatMetric(metricsLines, getCountMetricPrefix(common.TestModelName, vllm.VLLMKVCacheUsageMetricName))
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

			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				// Verify that the explicit totals are used
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:prompt_tokens_total{model_name="%s"} 12345`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:generation_tokens_total{model_name="%s"} 67890`, common.TestModelName)))
			})
		})
	})

	Context("fake metrics on /metrics", func() {
		It("Should respond with fake fixed-value metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{` +
					`"running-requests":10,` +
					`"waiting-requests":30,` +
					`"kv-cache-usage":0.4,` +
					`"prefix-cache-hits":750,` +
					`"prefix-cache-queries":2000` +
					`}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqRunningMetricName, 10)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqWaitingMetricName, 30)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMKVCacheUsageMetricName, 0.4)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheHitsTotalMetricName, 750)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheQueriesTotalMetricName, 2000)))
			})
		})

		It("Should drive generator-based fake metrics on /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{"kv-cache-usage":"ramp:0:1:700ms"}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			var prev float64
			for i := 1; i <= 5; i++ {
				time.Sleep(200 * time.Millisecond)
				lines := strings.Split(sampleMetrics(client), "\n")

				v := findFloatMetric(lines, getCountMetricPrefix(common.TestModelName, vllm.VLLMKVCacheUsageMetricName))
				Expect(v).ToNot(BeNil())
				if i < 4 {
					Expect(*v).To(BeNumerically("<", 1))
					Expect(*v).To(BeNumerically(">", prev))
				} else {
					Expect(*v).To(BeNumerically("==", 1))
				}
				prev = *v
			}
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

			consistentlyMetrics(client, func(g Gomega, metricsData string) {

				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheQueriesTotalMetricName, 1000)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheHitsTotalMetricName, 500)))
			})
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

			consistentlyMetrics(client, func(g Gomega, metricsData string) {

				// Fake values should be unchanged — reportPrefixCacheStats returns early when FakeMetrics is set
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllm.VLLMPrefixCacheQueriesTotalMetricName, 200)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllm.VLLMPrefixCacheHitsTotalMetricName, 100)))
			})
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

			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				for _, boundary := range common.TTFTBucketsBoundaries {
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, boundary, 0)))
				}
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, math.Inf(1), 1)))
			})
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

			eventuallyMetrics(client, func(g Gomega, metricsData string) {

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

					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLME2EReqLatencyMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqInferenceTimeMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqQueueTimeMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMPrefillTimeMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMDecodeTimeMetricName, boundary, expectedCount)))
				}
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLME2EReqLatencyMetricName, math.Inf(1), 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqInferenceTimeMetricName, math.Inf(1), 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqQueueTimeMetricName, math.Inf(1), 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMPrefillTimeMetricName, math.Inf(1), 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMDecodeTimeMetricName, math.Inf(1), 3)))
			})
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

			consistentlyMetrics(client, func(g Gomega, metricsData string) {
				metricsLines := strings.Split(metricsData, "\n")

				// Running requests: should be various values in [1, 5]
				count := findIntMetric(metricsLines, getCountMetricPrefix(common.TestModelName, vllm.VLLMReqRunningMetricName))
				g.Expect(count).ToNot(BeNil())
				g.Expect(*count).To(BeNumerically(">=", 1))
				g.Expect(*count).To(BeNumerically("<=", 5))

				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqWaitingMetricName, 30)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMKVCacheUsageMetricName, 0.4)))
			})

			// Update
			reqBody := `{
            "running-requests":15,
            "waiting-requests":0,
            "kv-cache-usage":0.9
        }`

			resp, err := postFakeMetricsUpdate(client, reqBody)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqRunningMetricName, 15)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqWaitingMetricName, 0)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMKVCacheUsageMetricName, 0.9)))
			})

			// Update
			reqBody = `{
            "running-requests":"oscillate:10:50:1s",
            "waiting-requests":30,
            "kv-cache-usage":"ramp:0:1:150ms"
        }`

			resp, err = postFakeMetricsUpdate(client, reqBody)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			// kv-cache-usage ramps to 1 over 150ms, so wait for it to top out first
			eventuallyMetrics(client, func(g Gomega, metricsData string) {
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqWaitingMetricName, 30)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMKVCacheUsageMetricName, 1)))
			})

			// Running requests: should be various values in [10, 50]
			consistentlyMetrics(client, func(g Gomega, metricsData string) {
				count := findIntMetric(strings.Split(metricsData, "\n"), getCountMetricPrefix(common.TestModelName, vllm.VLLMReqRunningMetricName))
				g.Expect(count).ToNot(BeNil())
				g.Expect(*count).To(BeNumerically(">=", 10))
				g.Expect(*count).To(BeNumerically("<=", 50))
			})

		})

		It("Should update fake metrics via POST /admin/config", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--fake-metrics",
				`{"running-requests":1,"waiting-requests":2,"kv-cache-usage":0.1}`,
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := `{"failure-injection-rate":42,"fake-metrics":{"running-requests":7,"waiting-requests":8,"kv-cache-usage":0.5}}`
			req, err := http.NewRequest("POST", "http://localhost/admin/config", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			Expect(resp.Body.Close()).To(Succeed())

			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqRunningMetricName, 7)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMReqWaitingMetricName, 8)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMKVCacheUsageMetricName, 0.5)))
			})

			// The non-fake-metrics field also took effect.
			resp, err = client.Get("http://localhost/admin/config")
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(data)).To(ContainSubstring(`"failure-injection-rate":42`))
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

			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				for _, boundary := range common.TTFTBucketsBoundaries {
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, boundary, 0)))
				}
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, math.Inf(1), 1)))

				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.01, 0)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.025, 0)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.05, 1)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.075, 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.1, 6)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTPOTMetricName, 0.15, 6)))

				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.01, 0)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.025, 0)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.05, 1)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.075, 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.1, 6)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMInterTokenLatencyMetricName, 0.15, 6)))
			})

			// Update
			reqBody := `{
            "ttft-buckets-values":[1,2,3],
			"tpot-buckets-values":[]
        }`

			resp, err := postFakeMetricsUpdate(client, reqBody)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			consistentlyMetrics(client, func(g Gomega, metricsData string) {
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, 0.001, 1)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, 0.005, 3)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, 0.01, 6)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMTTFTMetricName, 0.02, 6)))

				g.Expect(metricsData).NotTo(ContainSubstring(vllm.VLLMTPOTMetricName))
				g.Expect(metricsData).NotTo(ContainSubstring(vllm.VLLMInterTokenLatencyMetricName))
			})
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
			eventuallyMetrics(client, func(g Gomega, metricsData string) {

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
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLME2EReqLatencyMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqQueueTimeMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqInferenceTimeMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMPrefillTimeMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMDecodeTimeMetricName, boundary, expectedCount)))
				}

				// Initial token-param buckets: [10,20,30] on BuildBuckets
				buckets := metrics.BuildBuckets(1024, vllm.TokenBucketMantissas)
				for _, boundary := range buckets {
					switch {
					case boundary <= 1:
						expectedCount = 10
					case boundary <= 2:
						expectedCount = 30
					default:
						expectedCount = 60
					}
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMParamMaxTokensMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMMaxNumGenerationTokensMetricName, boundary, expectedCount)))
				}
			})

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

			resp, err := postFakeMetricsUpdate(client, reqBody)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			eventuallyMetrics(client, func(g Gomega, metricsData string) {
				buckets := metrics.BuildBuckets(1024, vllm.TokenBucketMantissas)
				var expectedCount int

				// After update: latency buckets [1, 0, 0, 1] -> counts: 1, 1, 1, 2, 2, ...
				for i, boundary := range common.RequestLatencyBucketsBoundaries {
					switch {
					case i < 3:
						expectedCount = 1
					default:
						expectedCount = 2
					}
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLME2EReqLatencyMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqQueueTimeMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqInferenceTimeMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMPrefillTimeMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMDecodeTimeMetricName, boundary, expectedCount)))
				}
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLME2EReqLatencyMetricName, math.Inf(1), 2)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqQueueTimeMetricName, math.Inf(1), 2)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMReqInferenceTimeMetricName, math.Inf(1), 2)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMPrefillTimeMetricName, math.Inf(1), 2)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMDecodeTimeMetricName, math.Inf(1), 2)))

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
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMParamMaxTokensMetricName, boundary, expectedCount)))
					g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMMaxNumGenerationTokensMetricName, boundary, expectedCount)))
				}
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMParamMaxTokensMetricName, math.Inf(1), 6)))
				g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllm.VLLMMaxNumGenerationTokensMetricName, math.Inf(1), 6)))
			})
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
			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="stop",model_name="%s"} 20`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="length",model_name="%s"} 5`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheHitsTotalMetricName, 500)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheQueriesTotalMetricName, 1000)))
			})

			// Update all three
			reqBody := `{
				"request-success-total":{"stop":100,"length":50,"tool_calls":10,"remote_decode":5},
				"prefix-cache-hits":750,
				"prefix-cache-queries":2000
			}`

			resp, err := postFakeMetricsUpdate(client, reqBody)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				// After update: values should be replaced, not accumulated
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="stop",model_name="%s"} 100`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="length",model_name="%s"} 50`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="tool_calls",model_name="%s"} 10`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`vllm:request_success_total{finish_reason="remote_decode",model_name="%s"} 5`, common.TestModelName)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheHitsTotalMetricName, 750)))
				g.Expect(metricsData).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllm.VLLMPrefixCacheQueriesTotalMetricName, 2000)))
			})
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
			eventuallyMetrics(client, func(g Gomega, metricsData string) {

				g.Expect(metricsData).To(ContainSubstring(`vllm:lora_requests_info{max_lora="1",running_lora_adapters="lora1,lora2",waiting_lora_adapters="lora3"} 1.000000001e+09`))
				g.Expect(metricsData).To(ContainSubstring(`vllm:lora_requests_info{max_lora="1",running_lora_adapters="lora1",waiting_lora_adapters=""} 1.000000002e+09`))
			})

			// Update lora metrics
			reqBody := `{
				"loras":[
					{"running":"lora4","waiting":"lora5,lora6","timestamp":2000000001}
				]
			}`

			resp, err := postFakeMetricsUpdate(client, reqBody)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			consistentlyMetrics(client, func(g Gomega, metricsData string) {

				// Old lora entries should be gone, only new one present
				g.Expect(metricsData).NotTo(ContainSubstring("lora1"))
				g.Expect(metricsData).NotTo(ContainSubstring("lora2"))
				g.Expect(metricsData).NotTo(ContainSubstring("lora3"))
				g.Expect(metricsData).To(ContainSubstring(`vllm:lora_requests_info{max_lora="1",running_lora_adapters="lora4",waiting_lora_adapters="lora5,lora6"} 2.000000001e+09`))
			})

			// Update lora metrics with an empty array
			reqBody = `{
				"loras":[]
			}`

			resp, err = postFakeMetricsUpdate(client, reqBody)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			consistentlyMetrics(client, func(g Gomega, metricsData string) {

				// Old lora entries should be gone, only a new empty one present
				g.Expect(metricsData).NotTo(ContainSubstring("lora4"))
				g.Expect(metricsData).NotTo(ContainSubstring("lora5"))
				g.Expect(metricsData).NotTo(ContainSubstring("lora6"))
				g.Expect(metricsData).To(ContainSubstring(`vllm:lora_requests_info{max_lora="1",running_lora_adapters="",waiting_lora_adapters=""}`))
			})
		})

		// This table tests the update logic for both prompt and generation token metrics.
		// Each entry exercises a 3-phase lifecycle: initial load -> first POST update -> second POST update.
		// For each phase, the test verifies the histogram buckets and total counter for both
		// request_prompt_tokens / prompt_tokens_total and request_generation_tokens / generation_tokens_total.
		//
		// Parameters per entry:
		//   initialMetrics   – JSON for --fake-metrics flag at startup
		//   initialPrompt/Gen – expected state after startup
		//   firstUpdate      – JSON for the first fake-metrics update
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
				eventuallyMetrics(client, func(g Gomega, metricsData string) {

					verifyTokenMetrics(g, metricsData, vllm.VLLMPromptTokensMetricName, vllm.VLLMPromptTokensTotalMetricName, initialPrompt)
					verifyTokenMetrics(g, metricsData, vllm.VLLMGenerationTokensMetricName, vllm.VLLMGenerationTokensTotalMetricName, initialGen)
				})

				// First update: POST new fake metrics and verify
				resp, err := postFakeMetricsUpdate(client, firstUpdate)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))

				eventuallyMetrics(client, func(g Gomega, metricsData string) {

					verifyTokenMetrics(g, metricsData, vllm.VLLMPromptTokensMetricName, vllm.VLLMPromptTokensTotalMetricName, firstPrompt)
					verifyTokenMetrics(g, metricsData, vllm.VLLMGenerationTokensMetricName, vllm.VLLMGenerationTokensTotalMetricName, firstGen)
				})

				// Second update: POST new fake metrics and verify
				resp, err = postFakeMetricsUpdate(client, secondUpdate)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.StatusCode).To(Equal(http.StatusOK))

				eventuallyMetrics(client, func(g Gomega, metricsData string) {

					verifyTokenMetrics(g, metricsData, vllm.VLLMPromptTokensMetricName, vllm.VLLMPromptTokensTotalMetricName, secondPrompt)
					verifyTokenMetrics(g, metricsData, vllm.VLLMGenerationTokensMetricName, vllm.VLLMGenerationTokensTotalMetricName, secondGen)
				})
			},
			// Prompt tokens: hist+total -> update hist -> update total
			// Generated tokens: only total -> add hist -> update total
			Entry("#1 Prompt tokens: hist+total, Generated tokens: only total; then update hists, then update totals",
				`{"request-prompt-tokens":[1,2,3], "total-prompt-tokens":12345, "total-generation-tokens":54321}`,
				tokenTestPhase{checkBuckets123, intPtr(12345)}, tokenTestPhase{nil, intPtr(54321)},
				`{"request-prompt-tokens":[10, 20], "request-generation-tokens":[10, 20]}`,
				tokenTestPhase{checkBuckets10_20, intPtr(50)}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"total-prompt-tokens":58, "total-generation-tokens":99}`,
				tokenTestPhase{checkBuckets10_20, intPtr(58)}, tokenTestPhase{checkBuckets10_20, intPtr(99)}),
			// Prompt tokens: only hist -> update hist -> update total
			// Generated tokens:    only hist -> update hist -> update total
			Entry("#2 Both: only hist; then update hists, then set totals",
				`{"request-prompt-tokens":[1,2,3], "request-generation-tokens":[10, 20]}`,
				tokenTestPhase{checkBuckets123, intPtr(20)}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"request-prompt-tokens":[10, 20], "request-generation-tokens":[1,2,3]}`,
				tokenTestPhase{checkBuckets10_20, intPtr(50)}, tokenTestPhase{checkBuckets123, intPtr(20)},
				`{"total-prompt-tokens":58, "total-generation-tokens":99}`,
				tokenTestPhase{checkBuckets10_20, intPtr(58)}, tokenTestPhase{checkBuckets123, intPtr(99)}),
			// Prompt tokens: empty -> set total -> add hist
			// Generated tokens: empty -> set total -> add hist
			Entry("#3 Both: empty; then set totals, then add hists",
				`{}`,
				tokenTestPhase{nil, nil}, tokenTestPhase{nil, nil},
				`{"total-prompt-tokens":58, "total-generation-tokens":77}`,
				tokenTestPhase{nil, intPtr(58)}, tokenTestPhase{nil, intPtr(77)},
				`{"request-prompt-tokens":[10, 20], "request-generation-tokens":[1,2,3]}`,
				tokenTestPhase{checkBuckets10_20, intPtr(50)}, tokenTestPhase{checkBuckets123, intPtr(20)}),
			// Prompt tokens: hist+total -> empty (no change) -> empty hist (remove)
			// Generated tokens: hist+total -> empty (no change) -> empty hist (remove)
			Entry("#4 Both: hist+total; then empty, then empty hists",
				`{"request-prompt-tokens":[1,2,3], "total-prompt-tokens":12345, "request-generation-tokens":[10,20], "total-generation-tokens":54321}`,
				tokenTestPhase{checkBuckets123, intPtr(12345)}, tokenTestPhase{checkBuckets10_20, intPtr(54321)},
				`{}`,
				tokenTestPhase{checkBuckets123, intPtr(12345)}, tokenTestPhase{checkBuckets10_20, intPtr(54321)},
				`{"request-prompt-tokens":[], "request-generation-tokens":[]}`,
				tokenTestPhase{nil, nil}, tokenTestPhase{nil, nil}),
			// Prompt tokens: only hist -> empty hist -> update hist
			// Generated tokens: only hist -> field absent (no-op) -> update hist
			Entry("#5 Both: only hist; then empty clears prompt, then re-add hists",
				`{"request-prompt-tokens":[1,2,3], "request-generation-tokens":[10,20]}`,
				tokenTestPhase{checkBuckets123, intPtr(20)}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"request-prompt-tokens":[]}`,
				tokenTestPhase{nil, nil}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"request-prompt-tokens":[10, 20], "request-generation-tokens":[1,2,3]}`,
				tokenTestPhase{checkBuckets10_20, intPtr(50)}, tokenTestPhase{checkBuckets123, intPtr(20)}),
			// Prompt tokens: empty -> add hist only -> update with hist+total simultaneously
			// Generated tokens: empty -> add hist only -> update with hist+total simultaneously
			Entry("#6 Both: empty; then add hists, then update with hist+total simultaneously",
				`{}`,
				tokenTestPhase{nil, nil}, tokenTestPhase{nil, nil},
				`{"request-prompt-tokens":[1,2,3], "request-generation-tokens":[10, 20]}`,
				tokenTestPhase{checkBuckets123, intPtr(20)}, tokenTestPhase{checkBuckets10_20, intPtr(50)},
				`{"request-prompt-tokens":[10,20], "total-prompt-tokens":999, "request-generation-tokens":[1,2,3], "total-generation-tokens":888}`,
				tokenTestPhase{checkBuckets10_20, intPtr(999)}, tokenTestPhase{checkBuckets123, intPtr(888)}),
			// Prompt tokens: only total -> update total -> add hist+total simultaneously
			// Generated tokens: only total -> update total -> add hist+total simultaneously
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

type checkBucketsFunc func(g Gomega, metricsData string, metricName string)

// tokenTestPhase describes the expected state of a token metric (prompt or generation)
// after a phase (initial load, first update, or second update).
type tokenTestPhase struct {
	checkBuckets checkBucketsFunc // nil means histogram should be absent
	total        *int             // nil means the total counter should be absent
}

func intPtr(v int) *int { return &v }

// verifyTokenMetrics checks the histogram and total counter for a single token type.
func verifyTokenMetrics(g Gomega, metricsData string, histMetricName string, totalMetricName string, phase tokenTestPhase) {
	if phase.total != nil {
		g.Expect(metricsData).To(ContainSubstring(fmt.Sprintf(`%s{model_name="%s"} %d`,
			totalMetricName, common.TestModelName, *phase.total)))
	} else {
		g.Expect(metricsData).NotTo(ContainSubstring(totalMetricName))
	}
	if phase.checkBuckets != nil {
		phase.checkBuckets(g, metricsData, histMetricName)
	} else {
		g.Expect(metricsData).NotTo(ContainSubstring(histMetricName))
	}
}

func checkBuckets123(g Gomega, metricsData string, metricName string) {
	buckets := metrics.BuildBuckets(1024, vllm.TokenBucketMantissas)
	for _, boundary := range buckets {
		switch {
		case boundary <= 1:
			g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 1)))
		case boundary <= 2:
			g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 3)))
		default:
			g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 6)))
		}
	}
	g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
		math.Inf(1), 6)))
}

func checkBuckets10_20(g Gomega, metricsData string, metricName string) {
	buckets := metrics.BuildBuckets(1024, vllm.TokenBucketMantissas)
	for _, boundary := range buckets {
		switch {
		case boundary <= 1:
			g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 10)))
		default:
			g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
				boundary, 30)))
		}
	}
	g.Expect(metricsData).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName,
		math.Inf(1), 30)))
}

var _ = Describe("total tokens", func() {
	It("should correctly calculate total tokens from bucket counts and boundaries", func() {
		tests := []struct {
			name        string
			counts      []int
			buckets     []float64
			expected    float64
			shouldBeNil bool
		}{
			{
				name:        "empty counts",
				counts:      []int{},
				buckets:     []float64{1, 2, 5},
				shouldBeNil: true,
				expected:    0,
			},
			{
				name:        "empty buckets",
				counts:      []int{10, 20},
				buckets:     []float64{},
				shouldBeNil: true,
				expected:    0,
			},
			{
				name:     "only first bucket has requests: [0,10]",
				counts:   []int{1},
				buckets:  []float64{10},
				expected: 10,
				// bucket0: [0,10] -> 1*10 = 10
				// total = 10
			},
			{
				name:     "first two buckets: [0,10], (10,20]",
				counts:   []int{2, 3},
				buckets:  []float64{10, 20},
				expected: 80,
				// bucket0: [0,10] ->  2*10 = 20
				// bucket1: (10,20] -> 3*20 = 60
				// total = 80
			},
			{
				name:     "three finite buckets + last (+Inf) bucket",
				counts:   []int{1, 1, 1, 1},
				buckets:  []float64{10, 20, 50},
				expected: 131,
				// bucket0: [0,10] -> 1*10 = 10
				// bucket1: (10,20] -> 1*20 = 20
				// bucket2: (20,50] -> 1*50 = 50
				// bucket3: (50,+Inf) -> 1*(50+1)=51
				// total = 131
			},
			{
				name:     "zero counts in some buckets",
				counts:   []int{0, 5, 0, 2},
				buckets:  []float64{1, 10, 100},
				expected: 252,
				// bucket1: (1,10] ->  5*10 = 50
				// bucket3: (100,+Inf) -> 2*(100+1) = 202
				// total = 252
			},
			{
				name:     "only last bucket has requests",
				counts:   []int{0, 0, 0, 4},
				buckets:  []float64{10, 100, 1000},
				expected: 4004,
				// bucket3: (1000,+Inf) -> 4*(1000+1) = 4004
			},
			{
				name:     "collaborator example: [10,20,30] with long buckets",
				counts:   []int{10, 20, 30},
				buckets:  []float64{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000},
				expected: 200,
				// bucket0: [0,1] -> 10*1 = 10
				// bucket1: (1,2] -> 20*2 = 40
				// bucket2: (2,5] -> 30*5 = 150
				// total = 200
			},
			{
				name:     "counts shorter than buckets (trailing zeros omitted)",
				counts:   []int{1, 1},
				buckets:  []float64{10, 100, 1000, 10000},
				expected: 110,
				// bucket0: [0,10] -> 1*10 = 10
				// bucket1: (10,100] -> 1*100 = 100
				// total = 110
			},
			{
				name:     "all zero counts",
				counts:   []int{0, 0, 0},
				buckets:  []float64{1, 10, 100},
				expected: 0,
				// all buckets have zero requests
			},
		}

		s := simulator.SimContext{}
		s.SetConfig(&common.Configuration{Model: "test", ServedModelNames: []string{"test"}})

		for _, test := range tests {
			hist := prometheus.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "dummy",
					Help:    "Test histogram",
					Buckets: test.buckets,
				}, []string{api.PromLabelModelName},
			)
			result := metrics.InitFakeHistogram(hist, s.Config().DisplayModelName, test.buckets, test.counts)
			if test.shouldBeNil {
				Expect(result).To(BeNil())
			} else {
				Expect(*result).To(Equal(test.expected), "test case: %s", test.name)
			}
		}
	})
})
