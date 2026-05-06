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
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

const (
	lora1 = "lora1"
	lora2 = "lora2"
)

var emptyArray = []string{}
var lora1Arr = []string{lora1}
var lora2Arr = []string{lora2}

var paramsLora1 openai.ChatCompletionNewParams = openai.ChatCompletionNewParams{
	Messages: []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(testUserMessage),
	},
	Model: "lora1",
}

var paramsLora2 openai.ChatCompletionNewParams = openai.ChatCompletionNewParams{
	Messages: []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(testUserMessage),
	},
	Model: "lora2",
}

var paramsLora3 openai.ChatCompletionNewParams = openai.ChatCompletionNewParams{
	Messages: []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(testUserMessage),
	},
	Model: "lora3",
}

var paramsLora4 openai.ChatCompletionNewParams = openai.ChatCompletionNewParams{
	Messages: []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(testUserMessage),
	},
	Model: "lora4",
}

var paramsLora5 openai.ChatCompletionNewParams = openai.ChatCompletionNewParams{
	Messages: []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(testUserMessage),
	},
	Model: "lora5",
}

var _ = Describe("Simulator metrics", Ordered, func() {
	It("should send correct running and waiting requests metrics", func() {
		// Three requests, only two can run in parallel, we expect
		// two running requests and one waiting request in the metrics
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000", "--max-num-seqs", "2"}

		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, false)

		for range 3 {
			go func() {
				defer GinkgoRecover()
				_, err := openaiclient.Chat.Completions.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
			}()
		}

		time.Sleep(300 * time.Millisecond)
		metricsResp, err := client.Get(metricsUrl)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := string(data)
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqRunningMetricName, 2)))
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqWaitingMetricName, 1)))
	})

	DescribeTable("should send correct running and waiting requests metrics with failures",
		func(stream bool) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--max-model-len", "2",
				"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}"}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, "lora1", testUserMessage, stream)

			_, err = openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).To(HaveOccurred())

			time.Sleep(300 * time.Millisecond)
			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			// There should be no running or waiting requests
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqRunningMetricName, 0)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqWaitingMetricName, 0)))

			// We sent one request (that failed), we expect to see (in this order)
			// 1. running: lora1, waiting: empty
			// 2. running: empty, waiting: empty
			metricsLines := strings.Split(metrics, "\n")
			Expect(isLoraMetricPresent(metricsLines, lora1Arr, emptyArray)).To(BeTrue())
			Expect(isLoraMetricPresent(metricsLines, emptyArray, emptyArray)).To(BeTrue())

			// Check the order
			timestamp1 := getLoraValidTimestamp(metricsLines, lora1Arr, emptyArray)
			timestamp2 := getLoraValidTimestamp(metricsLines, emptyArray, emptyArray)
			Expect(timestamp1 <= timestamp2).To(BeTrue())
		},
		Entry("no streaming", false),
		Entry("streaming", true),
	)

	It("Should record correct prompt and generation token counts", func() {
		ctx := context.TODO()

		prompt := strings.Repeat("hello ", 25)
		maxTokens := 25
		model := common.TestModelName
		expectedPromptTokensCnt := getChatPromptTokensCountForTestModel(prompt)

		args := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
			"--time-to-first-token", "100", "--max-num-seqs", "4"}

		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		params := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(prompt),
			},
			Model:       model,
			MaxTokens:   openai.Int(int64(maxTokens)),
			Temperature: openai.Float(0.0),
		}

		_, err = openaiclient.Chat.Completions.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())

		time.Sleep(500 * time.Millisecond)

		metricsResp, err := client.Get(metricsUrl)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := string(data)
		// request_prompt_tokens_bucket and request_params_max_tokens_bucket
		buckets := vllmsim.Build125Buckets(1024)

		for _, boundary := range buckets {
			if boundary <= 20 {
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(model, vllmsim.PromptTokensMetricName, boundary, 0)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(model, vllmsim.ParamMaxTokensMetricName, boundary, 0)))
			} else {
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(model, vllmsim.PromptTokensMetricName, boundary, 1)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(model, vllmsim.ParamMaxTokensMetricName, boundary, 1)))
			}
		}
		Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(model, vllmsim.PromptTokensMetricName, math.Inf(1), 1)))
		Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(model, vllmsim.ParamMaxTokensMetricName, math.Inf(1), 1)))

		Expect(metrics).To(MatchRegexp(fmt.Sprintf(`vllm:prompt_tokens_total{model_name="%s"} %d`, model, expectedPromptTokensCnt)))

		// request_generation_tokens
		// We do not verify the distribution of the number of tokens generated per request,
		// as the number of generated tokens is unpredictable in this test.
		// Therefore, we only verify the number of requests and the total number of generated tokens,
		// and skip the bucket distribution.
		Expect(metrics).To(ContainSubstring(getCountMetricLine(model, vllmsim.GenerationTokensMetricName+"_count", 1)))
		// request_success_total
		Expect(metrics).To(MatchRegexp(fmt.Sprintf(`vllm:request_success_total{finish_reason="(stop|length)",model_name="%s"} 1`, common.TestModelName)))
	})

	DescribeTable("should send correct lora metrics",
		func(stream bool) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
				"--time-to-first-token", "3000", "-v", "5",
				"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
				"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			if stream {
				stream1 := openaiclient.Chat.Completions.NewStreaming(ctx, paramsLora1)
				for stream1.Next() {
				}
				stream1.Close() //nolint:errcheck

				stream2 := openaiclient.Chat.Completions.NewStreaming(ctx, paramsLora2)
				for stream2.Next() {
				}
				stream2.Close() //nolint:errcheck
			} else {
				_, err = openaiclient.Chat.Completions.New(ctx, paramsLora1)
				Expect(err).NotTo(HaveOccurred())

				_, err = openaiclient.Chat.Completions.New(ctx, paramsLora2)
				Expect(err).NotTo(HaveOccurred())
			}

			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := strings.Split(string(data), "\n")

			// We sent two sequentual requests to two different LoRAs, we expect to see (in this order)
			// 1. running: lora1, waiting: empty
			// 2. running: lora2, waiting: empty
			// 3. running: empty, waiting: empty
			Expect(isLoraMetricPresent(metrics, lora1Arr, emptyArray)).To(BeTrue())
			Expect(isLoraMetricPresent(metrics, lora2Arr, emptyArray)).To(BeTrue())
			Expect(isLoraMetricPresent(metrics, emptyArray, emptyArray)).To(BeTrue())

			// Check the order
			timestamp1 := getLoraValidTimestamp(metrics, lora1Arr, emptyArray)
			timestamp2 := getLoraValidTimestamp(metrics, lora2Arr, emptyArray)
			timestamp3 := getLoraValidTimestamp(metrics, emptyArray, emptyArray)

			Expect(timestamp1 <= timestamp2).To(BeTrue())
			Expect(timestamp2 <= timestamp3).To(BeTrue())
		},
		Entry("no streaming", false),
		Entry("streaming", true),
	)

	It("Should send correct lora metrics for parallel requests with delay", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000",
			"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
			"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		var wg sync.WaitGroup
		wg.Add(2)

		// sends three requests with a delay of 0.5 second between them
		// request1 for lora1, request2 for lora2, and request 3 for lora1
		go func() {
			time.Sleep(500 * time.Millisecond)
			defer wg.Done()
			defer GinkgoRecover()
			_, err := openaiclient.Chat.Completions.New(ctx, paramsLora2)
			Expect(err).NotTo(HaveOccurred())
		}()
		go func() {
			time.Sleep(1 * time.Second)
			defer wg.Done()
			defer GinkgoRecover()
			_, err := openaiclient.Chat.Completions.New(ctx, paramsLora1)
			Expect(err).NotTo(HaveOccurred())
		}()

		_, err = openaiclient.Chat.Completions.New(ctx, paramsLora1)
		Expect(err).NotTo(HaveOccurred())

		wg.Wait()

		metricsResp, err := client.Get(metricsUrl)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := strings.Split(string(data), "\n")

		// max_loras is 1 by default
		// We sent 3 requests, we expect to see (in this order)
		// 1. running: lora1, waiting: empty
		// 2. running: lora1, waiting: lora2
		// 3. running: empty, waiting: lora2
		// 4. running: lora2, waiting: empty
		// 5. running: empty, waiting: empty
		// (Requests 1 and 3 can run in parallel)
		Expect(isLoraMetricPresent(metrics, lora1Arr, emptyArray)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, lora1Arr, lora2Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, emptyArray, lora2Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, lora2Arr, emptyArray)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, emptyArray, emptyArray)).To(BeTrue())

		// Check the order
		timestamp1 := getLoraValidTimestamp(metrics, lora1Arr, emptyArray)
		timestamp2 := getLoraValidTimestamp(metrics, lora1Arr, lora2Arr)
		timestamp3 := getLoraValidTimestamp(metrics, emptyArray, lora2Arr)
		timestamp4 := getLoraValidTimestamp(metrics, lora2Arr, emptyArray)
		timestamp5 := getLoraValidTimestamp(metrics, emptyArray, emptyArray)

		// in case of requests sent with delay the order is well-defined
		Expect(timestamp1 <= timestamp2).To(BeTrue())
		Expect(timestamp2 <= timestamp3).To(BeTrue())
		Expect(timestamp3 <= timestamp4).To(BeTrue())
		Expect(timestamp4 <= timestamp5).To(BeTrue())
	})

	It("Should send correct lora metrics for parallel requests without delay", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000",
			"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
			"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		var wg sync.WaitGroup
		wg.Add(1)

		// send two requests with lora1 and lora2 in parallel
		go func() {
			defer wg.Done()
			defer GinkgoRecover()
			_, err := openaiclient.Chat.Completions.New(ctx, paramsLora2)
			Expect(err).NotTo(HaveOccurred())
		}()

		_, err = openaiclient.Chat.Completions.New(ctx, paramsLora1)
		Expect(err).NotTo(HaveOccurred())

		wg.Wait()

		metricsResp, err := client.Get(metricsUrl)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := strings.Split(string(data), "\n")

		// We sent two parallel requests: first to lora1 and then to lora2,
		// we expect to see metrics in this order:
		// 1. running: one of the loras, waiting: another lora
		// 2. running: empty, waiting: another lora
		// 3. running: the second lora, waiting: empty
		// 4. running: empty, waiting: empty
		Expect(isLoraMetricPresent(metrics, lora1Arr, lora2Arr) || isLoraMetricPresent(metrics, lora2Arr, lora1Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, emptyArray, lora1Arr) || isLoraMetricPresent(metrics, emptyArray, lora2Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, lora1Arr, emptyArray) || isLoraMetricPresent(metrics, lora2Arr, emptyArray)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, emptyArray, emptyArray)).To(BeTrue())

		// Check the order:
		l1RunningL2Waiting, err := getLoraTimestamp(metrics, lora1Arr, lora2Arr)
		Expect(err).NotTo(HaveOccurred())
		l2RunningL1Waiting, err := getLoraTimestamp(metrics, lora2Arr, lora1Arr)
		Expect(err).NotTo(HaveOccurred())
		l1WatingEmptyRunning, err := getLoraTimestamp(metrics, emptyArray, lora1Arr)
		Expect(err).NotTo(HaveOccurred())
		l2WatingEmptyRunning, err := getLoraTimestamp(metrics, emptyArray, lora2Arr)
		Expect(err).NotTo(HaveOccurred())
		l1RunningEmptyWaiting, err := getLoraTimestamp(metrics, lora1Arr, emptyArray)
		Expect(err).NotTo(HaveOccurred())
		l2RunningEmptyWaiting, err := getLoraTimestamp(metrics, lora2Arr, emptyArray)
		Expect(err).NotTo(HaveOccurred())
		emptyTimestamp := getLoraValidTimestamp(metrics, emptyArray, emptyArray)

		if l1RunningL2Waiting != nil {
			Expect(l2RunningL1Waiting).To(BeNil())
			Expect(l2WatingEmptyRunning).NotTo(BeNil())
			Expect(l2RunningEmptyWaiting).NotTo(BeNil())
			Expect(*l1RunningL2Waiting <= *l2WatingEmptyRunning).To(BeTrue())
			Expect(*l2WatingEmptyRunning <= *l2RunningEmptyWaiting).To(BeTrue())
			Expect(*l2RunningEmptyWaiting <= emptyTimestamp).To(BeTrue())
		} else {
			Expect(l2RunningL1Waiting).NotTo(BeNil())
			Expect(l1WatingEmptyRunning).NotTo(BeNil())
			Expect(l1RunningEmptyWaiting).NotTo(BeNil())
			Expect(*l2RunningL1Waiting <= *l1WatingEmptyRunning).To(BeTrue())
			Expect(*l1WatingEmptyRunning <= *l1RunningEmptyWaiting).To(BeTrue())
			Expect(*l1RunningEmptyWaiting <= emptyTimestamp).To(BeTrue())
		}
	})

	It("Should send correct ttft, tpot and inter_token_latency metrics", func() {
		// Send one request, check that ttft, tpot, and inter_token_latency are as defined in the simulator command line params
		ctx := context.TODO()
		// use mode echo to be sure that response is more than one token - this makes sure that tpot is reported to prometheus
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
			"--time-to-first-token", "200", "--inter-token-latency", "100"}

		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, false)

		var reqWg, metricsWg sync.WaitGroup
		metricsWg.Add(1)
		reqWg.Add(1)

		go func() {
			defer reqWg.Done()
			defer GinkgoRecover()

			_, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
		}()

		go func() {
			defer metricsWg.Done()
			defer GinkgoRecover()

			reqWg.Wait()
			time.Sleep(300 * time.Millisecond)
			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)
			metricsLines := strings.Split(metrics, "\n")

			// ttft
			for _, boundary := range common.TTFTBucketsBoundaries {
				if boundary <= 0.1 {
					// buckets up to 0.1 should be empty
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, boundary, 0)))
				} else {
					// buckets higher than 0.1 should contain a single sample
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, boundary, 1)))
				}
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, math.Inf(1), 1)))

			// helper to validate a latency metric (used for both tpot and inter_token_latency)
			validateLatencyMetric := func(metricName string) {
				for _, boundary := range common.TPOTBucketsBoundaries {
					if boundary <= 0.075 {
						// ensure that values for buckets up to 0.075 have count 0
						Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, metricName, boundary, 0)))
					} else {
						// buckets higher than 0.75 should be greater than 0, we don't know the exact value since it depends on the random response length
						count := findIntMetric(metricsLines, getFloatBucketMetricPrefix(common.TestModelName, metricName, boundary))
						Expect(count).ToNot(BeNil())
						Expect(*count).To(BeNumerically(">", 0))
					}
				}
				count := findIntMetric(metricsLines, getFloatBucketMetricPrefix(common.TestModelName, metricName, math.Inf(1)))
				Expect(count).ToNot(BeNil())
				Expect(*count).To(BeNumerically(">", 0))
			}

			// validate legacy tpot metric
			validateLatencyMetric(vllmsim.TPOTMetricName)

			// validate new inter_token_latency metric
			validateLatencyMetric(vllmsim.InterTokenLatencyMetricName)
		}()

		metricsWg.Wait()
	})

	Context("kv cache metrics", func() {
		It("Should send correct kv cache usage metrics", func() {
			// Three requests, there are should be two blocks in the kv cache, because
			// the first and the second prompt share a block.
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--time-to-first-token", "5000"}

			client, err := startServerWithArgsAndEnv(ctx, common.ModeRandom, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			paramsArray := []openai.CompletionNewParams{
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in Haifa today? Is it cold?"),
					},
					Model: openai.CompletionNewParamsModel(common.QwenModelName),
				},
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in Haifa today?"),
					},
					Model: openai.CompletionNewParamsModel(common.QwenModelName),
				},
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in New York today?"),
					},
					Model: openai.CompletionNewParamsModel(common.QwenModelName),
				},
			}

			for _, params := range paramsArray {
				go func() {
					defer GinkgoRecover()
					_, err := openaiclient.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
				}()
			}

			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer GinkgoRecover()

				time.Sleep(4 * time.Second)
				metricsResp, err := client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

				data, err := io.ReadAll(metricsResp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics := string(data)
				// Expect three running requests and two blocks in the kv cache - usage 2/16=0.125
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqRunningMetricName, 3)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqWaitingMetricName, 0)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.KVCacheUsageMetricName, 0.125)))

				time.Sleep(4 * time.Second)
				metricsResp, err = client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

				data, err = io.ReadAll(metricsResp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics = string(data)
				// The requests finished running, expect 0 usage
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqRunningMetricName, 0)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqWaitingMetricName, 0)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.KVCacheUsageMetricName, 0)))
			}()
			wg.Wait()
		})

		It("Should send correct kv cache usage metrics for sequentual requests", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--time-to-first-token", "5000", "--max-num-seqs", "2"}

			client, err := startServerWithArgsAndEnv(ctx, common.ModeRandom, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			paramsArray := []openai.CompletionNewParams{
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in Haifa today? Is it cold?"),
					},
					Model: openai.CompletionNewParamsModel(common.QwenModelName),
				},
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in Haifa today?"),
					},
					Model: openai.CompletionNewParamsModel(common.QwenModelName),
				},
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in New York today?"),
					},
					Model: openai.CompletionNewParamsModel(common.QwenModelName),
				},
			}

			for i, params := range paramsArray {
				go func() {
					defer GinkgoRecover()
					time.Sleep(time.Duration(i*500) * time.Millisecond)
					_, err := openaiclient.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
				}()
			}

			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer GinkgoRecover()

				time.Sleep(3 * time.Second)
				metricsResp, err := client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

				data, err := io.ReadAll(metricsResp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics := string(data)
				// The requests were sent with 500 millisecond intervals, and the first two should be still running.
				// The third is waiting, and is still not in the kv-cache.
				// We expect one block in the kv-cache, usage 1/16=0.0625.
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqRunningMetricName, 2)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqWaitingMetricName, 1)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.KVCacheUsageMetricName, 0.0625)))
			}()
			wg.Wait()
		})

		It("Should increment prefix cache counters for requests with shared prefixes", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom,
				"--enable-kvcache", "true", "--kv-cache-size", "64", "--block-size", "8",
				"--time-to-first-token", "100"}

			client, err := startServerWithArgsAndEnv(ctx, common.ModeRandom, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			// Send requests sequentially so the cache is populated between requests
			prompts := []string{
				"What is the weather like in Haifa today?",
				"What is the weather like in Haifa today? Is it cold?",
			}
			for _, prompt := range prompts {
				_, err = openaiclient.Completions.New(ctx, openai.CompletionNewParams{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String(prompt),
					},
					Model: openai.CompletionNewParamsModel(common.QwenModelName),
				})
				Expect(err).NotTo(HaveOccurred())
			}

			time.Sleep(500 * time.Millisecond)
			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metricsLines := strings.Split(string(data), "\n")

			// prefix_cache_queries should reflect total prompt tokens across both requests
			queries := findIntMetric(metricsLines, getCountMetricPrefix(common.QwenModelName, vllmsim.PrefixCacheQueriesMetricName))
			Expect(queries).NotTo(BeNil())
			Expect(*queries).To(BeNumerically(">", 0))

			// The second request shares a prefix with the first, so hits should be non-zero
			hits := findIntMetric(metricsLines, getCountMetricPrefix(common.QwenModelName, vllmsim.PrefixCacheHitsMetricName))
			Expect(hits).NotTo(BeNil())
			Expect(*hits).To(BeNumerically(">", 0))

			// Hits cannot exceed queries
			Expect(*hits).To(BeNumerically("<=", *queries))
		})

		It("Should send correct kv cache usage metrics for parallel /responses requests", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--time-to-first-token", "2000"}

			client, err := startServerWithArgsAndEnv(ctx, common.ModeRandom, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			requests := []struct {
				input        string
				instructions string
			}{
				{"What is the weather like in Haifa today? Is it cold?", "Reply in French"},
				{"What is the weather like in Haifa today?", "Reply in French"},
				{"What is the weather like in Haifa today?", "Reply in English"},
				{"What is the weather like in New York today?", "Reply in English"},
			}

			for _, req := range requests {
				go func() {
					defer GinkgoRecover()
					time.Sleep(100 * time.Millisecond)
					openaiclient, params := getOpenAIClientAndResponsesParams(client, common.QwenModelName, req.input,
						req.instructions)
					_, err := openaiclient.Responses.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
				}()
			}

			var wg sync.WaitGroup
			wg.Go(func() {
				defer GinkgoRecover()

				time.Sleep(time.Second)
				metricsResp, err := client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

				data, err := io.ReadAll(metricsResp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics := string(data)
				// Expect four running requests
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqRunningMetricName, 4)))
				// There should be 2 blocks for the instructions.
				// The first two requests add 1 block. (The first request is not long enough for two blocks).
				// The third request adds 1 block, because it has a different parent from the first two requests.
				// The fourth request adds 1 block.
				// 5/16 = 0.3125
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.KVCacheUsageMetricName, 0.3125)))

				time.Sleep(2 * time.Second)
				metricsResp, err = client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

				data, err = io.ReadAll(metricsResp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics = string(data)
				// The requests finished running, expect 0 usage
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqRunningMetricName, 0)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.KVCacheUsageMetricName, 0)))
			})
			wg.Wait()
		})

		It("Should increment prefix cache counters for /responses requests with shared prefixes", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom,
				"--enable-kvcache", "true", "--kv-cache-size", "64", "--block-size", "8",
				"--time-to-first-token", "100"}

			client, err := startServerWithArgsAndEnv(ctx, common.ModeRandom, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			// Send requests sequentially so the cache is populated between requests
			inputs := []string{
				"What is the weather like in Haifa today?",
				"What is the weather like in Haifa today? Is it cold?",
			}
			for _, input := range inputs {
				openaiclient, params := getOpenAIClientAndResponsesParams(client, common.QwenModelName, input, "You are a helpful weather assistant.")
				_, err = openaiclient.Responses.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
			}

			time.Sleep(500 * time.Millisecond)
			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metricsLines := strings.Split(string(data), "\n")

			queries := findIntMetric(metricsLines, getCountMetricPrefix(common.QwenModelName, vllmsim.PrefixCacheQueriesMetricName))
			Expect(queries).NotTo(BeNil())
			Expect(*queries).To(BeNumerically(">", 0))

			// The second request shares a prefix with the first, so hits should be non-zero
			hits := findIntMetric(metricsLines, getCountMetricPrefix(common.QwenModelName, vllmsim.PrefixCacheHitsMetricName))
			Expect(hits).NotTo(BeNil())
			Expect(*hits).To(BeNumerically(">", 0))

			Expect(*hits).To(BeNumerically("<=", *queries))
		})

		It("Should send correct kv cache config metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeRandom,
				"--kv-cache-size", "16", "--block-size", "8"}

			client, err := startServerWithArgsAndEnv(ctx, common.ModeRandom, args, map[string]string{"POD_IP": "localhost"})
			Expect(err).NotTo(HaveOccurred())

			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)
			Expect(metrics).To(ContainSubstring("vllm:cache_config_info{block_size=\"8\",num_gpu_blocks=\"16\"} 1"))
		})
	})

	Context("single request latency metrics", func() {
		DescribeTable("should calculate all latency related metrics correctly for a single request",
			func(testNamePrefix string, ttft int, prefillTimePerToken int, interTokenLatency int,
				kvcacheTransferLatency int, kvCacheTransferTimePerToken int, doRemotePrefill bool) {

				_, tokens, err := tokenizerMngr.TestTokenizer().RenderText(testUserMessage)
				Expect(err).ShouldNot(HaveOccurred())
				numOfTokens := len(tokens)

				// send a single request with a prompt of 4 tokens and echo mode, so output tokens number of 4 too
				singleRequestLatencyTest(ttft, prefillTimePerToken, interTokenLatency, kvcacheTransferLatency,
					kvCacheTransferTimePerToken, false, numOfTokens, doRemotePrefill)
				singleRequestLatencyTest(ttft, prefillTimePerToken, interTokenLatency, kvcacheTransferLatency,
					kvCacheTransferTimePerToken, true, numOfTokens, doRemotePrefill)
			},
			func(testNamePrefix string, ttft int, prefillTimePerToken int, interTokenLatency int,
				kvcacheTransferLatency int, kvCacheTransferTimePerToken int, doRemotePrefill bool) string {
				return fmt.Sprintf("%s\nttft: %d, prefillTimePerToken: %d, interTokenLatency: %d, kvcacheTransferLatency: %d, kvCacheTransferTimePerToken: %d, doRemotePrefill: %t",
					testNamePrefix, ttft, prefillTimePerToken, interTokenLatency, kvcacheTransferLatency, kvCacheTransferTimePerToken, doRemotePrefill)
			},
			// Params order: testName, ttft, prefillTimePerToken, interTokenLatency, kvcacheTransferLatency, kvCacheTransferTimePerToken, doRemotePrefill)
			Entry(nil, "constant prefill + inter token time", 0, 0, 100, 0, 0, false),
			Entry(nil, "constant prefill + inter token time", 900, 0, 100, 0, 0, false),
			Entry(nil, "constant prefill + inter token time", 1000, 0, 100, 0, 0, false),
			Entry(nil, "prefill per token + inter token time", 0, 100, 100, 0, 0, false),
			Entry(nil, "remote prefill constant time", 0, 0, 0, 1000, 0, true),
			Entry(nil, "remote prefill constant time with non-remote times", 5000, 5000, 0, 1000, 0, true),
			Entry(nil, "remote prefill time per transferfed token", 0, 0, 0, 0, 100, true),
		)
	})

	Context("multiple requests latency metrics", func() {
		It("should calculate waiting and inference time correctly", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
				"--time-to-first-token", "1200", "--max-num-seqs", "1",
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, false)

			var reqWg sync.WaitGroup
			reqWg.Add(2)

			// send two requests
			for range 2 {
				go func() {
					defer reqWg.Done()
					defer GinkgoRecover()

					_, err := openaiclient.Chat.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
				}()
			}

			reqWg.Wait()
			time.Sleep(300 * time.Millisecond)
			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			for _, boundary := range common.RequestLatencyBucketsBoundaries {
				if boundary < 1.5 {
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqInferenceTimeMetricName, boundary, 0)))
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqQueueTimeMetricName, boundary, 0)))
				} else {
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqInferenceTimeMetricName, boundary, 2)))
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqQueueTimeMetricName, boundary, 1)))
				}
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqInferenceTimeMetricName, math.Inf(1), 2)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ReqQueueTimeMetricName, math.Inf(1), 1)))
		})
	})
})

var _ = Describe("build125Buckets", Ordered, func() {
	It("should create valid 125 buckets", func() {
		// tests the build125Buckets function with various inputs.
		tests := []struct {
			name     string
			maxValue int
			want     []float64
		}{
			{
				name:     "max_value zero",
				maxValue: 0,
				want:     []float64{}, // no bucket <= 0
			},
			{
				name:     "max_value one",
				maxValue: 1,
				want:     []float64{1},
			},
			{
				name:     "max_value five",
				maxValue: 5,
				want:     []float64{1, 2, 5},
			},
			{
				name:     "max_value ten",
				maxValue: 10,
				want:     []float64{1, 2, 5, 10},
			},
			{
				name:     "max_value 100",
				maxValue: 100,
				want:     []float64{1, 2, 5, 10, 20, 50, 100},
			},
			{
				name:     "max_value 999",
				maxValue: 999,
				want:     []float64{1, 2, 5, 10, 20, 50, 100, 200, 500},
			},
			{
				name:     "max_value 1024",
				maxValue: 1024,
				want:     []float64{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000},
			},
			{
				name:     "max_value 4096",
				maxValue: 4096,
				want:     []float64{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000},
			},
			{
				name:     "max_value 32768",
				maxValue: 32768,
				want:     []float64{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000},
			},
			{
				name:     "max_value just below power of 10",
				maxValue: 99,
				want:     []float64{1, 2, 5, 10, 20, 50},
			},
			{
				name:     "max_value negative",
				maxValue: -1,
				want:     []float64{}, // no positive bucket <= -1
			},
		}

		for _, test := range tests {
			got := vllmsim.Build125Buckets(test.maxValue)
			Expect(got).To(Equal(test.want))
		}
	})
})
