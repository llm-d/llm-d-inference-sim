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
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
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
		args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000", "--max-num-seqs", "2"}

		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)

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
		Expect(metrics).To(ContainSubstring(getCountMetricLine(testModel, reqRunningMetricName, 2)))
		Expect(metrics).To(ContainSubstring(getCountMetricLine(testModel, reqWaitingMetricName, 1)))
	})

	It("Should record correct prompt and generation token counts", func() {
		modelName := "testmodel"
		prompt := strings.Repeat("hello ", 25)
		maxTokens := 25

		ctx := context.TODO()
		args := []string{"cmd", "--model", modelName, "--mode", common.ModeRandom,
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
			Model:       modelName,
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
		buckets := build125Buckets(1024)

		for _, boundary := range buckets {
			if boundary <= 20 {
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, promptTokensMetricName, boundary, 0)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, paramMaxTokensMetricName, boundary, 0)))
			} else {
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, promptTokensMetricName, boundary, 1)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, paramMaxTokensMetricName, boundary, 1)))
			}
		}
		Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, promptTokensMetricName, math.Inf(1), 1)))
		Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, paramMaxTokensMetricName, math.Inf(1), 1)))

		// request_generation_tokens
		// We do not verify the distribution of the number of tokens generated per request,
		// as the number of generated tokens is unpredictable in this test.
		// Therefore, we only verify the number of requests and the total number of generated tokens,
		// and skip the bucket distribution.
		Expect(metrics).To(ContainSubstring(getCountMetricLine(testModel, generationTokensMetricName+"_count", 1)))
		// request_success_total
		Expect(metrics).To(MatchRegexp(`vllm:request_success_total{finish_reason="(stop|length)",model_name="testmodel"} 1`))
	})

	It("Should send correct lora metrics", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000",
			"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
			"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		_, err = openaiclient.Chat.Completions.New(ctx, paramsLora1)
		Expect(err).NotTo(HaveOccurred())

		_, err = openaiclient.Chat.Completions.New(ctx, paramsLora2)
		Expect(err).NotTo(HaveOccurred())

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
	})

	It("Should send correct lora metrics for parallel requests with delay", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom,
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
		args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom,
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

	It("Should send correct ttft and tpot metrics", func() {
		// Send one request, check that ttft and tpot are as defined in the simulator command line params
		ctx := context.TODO()
		// use mode echo to be sure that response is more than one token - this makes sure that tpot is reported to prometheus
		args := []string{"cmd", "--model", testModel, "--mode", common.ModeEcho,
			"--time-to-first-token", "200", "--inter-token-latency", "100"}

		client, err := startServerWithArgs(ctx, args)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)

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

			// ttft
			for _, boundary := range common.TTFTBucketsBoundaries {
				if boundary <= 0.1 {
					// buckets up to 0.1 should be empty
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, ttftMetricName, boundary, 0)))
				} else {
					// buckets higher than 0.1 should contain a single sample
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, ttftMetricName, boundary, 1)))
				}
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, ttftMetricName, math.Inf(1), 1)))

			// tpot
			metricsLines := strings.Split(metrics, "\n")
			var count *int

			for _, boundary := range common.TPOTBucketsBoundaries {
				if boundary <= 0.075 {
					// ensure that values for buckets up to 0.075 have count 0
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, tpotMetricName, boundary, 0)))
				} else {
					// buckets higher than 0.75 should be greater than 0, we don't know the exact value since it depends on the random response length
					count = findIntMetric(metricsLines, getFloatBucketMetricPrefix(testModel, tpotMetricName, 0.1))
					Expect(count).ToNot(BeNil())
					Expect(*count).To(BeNumerically(">", 0))
				}
			}
			count = findIntMetric(metricsLines, getFloatBucketMetricPrefix(testModel, tpotMetricName, math.Inf(1)))
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
		}()

		metricsWg.Wait()
	})

	Context("kv cache metrics", func() {
		tmpDir := "./tests-tmp/"
		AfterAll(func() {
			err := os.RemoveAll(tmpDir)
			Expect(err).NotTo(HaveOccurred())
		})
		It("Should send correct kv cache usage metrics", func() {
			// Three requests, there are should be two blocks in the kv cache, because
			// the first and the second prompt share a block.
			ctx := context.TODO()
			args := []string{"cmd", "--model", qwenModelName, "--mode", common.ModeRandom,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--time-to-first-token", "5000", "--tokenizers-cache-dir", tmpDir}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			paramsArray := []openai.CompletionNewParams{
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in Haifa today? Is it cold?"),
					},
					Model: openai.CompletionNewParamsModel(qwenModelName),
				},
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in Haifa today?"),
					},
					Model: openai.CompletionNewParamsModel(qwenModelName),
				},
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in New York today?"),
					},
					Model: openai.CompletionNewParamsModel(qwenModelName),
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
				Expect(metrics).To(ContainSubstring(getCountMetricLine(qwenModelName, reqRunningMetricName, 3)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(qwenModelName, reqWaitingMetricName, 0)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(qwenModelName, gpuCacheUsageMetricName, 0.125)))

				time.Sleep(4 * time.Second)
				metricsResp, err = client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

				data, err = io.ReadAll(metricsResp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics = string(data)
				// The requests finished running, expect 0 usage
				Expect(metrics).To(ContainSubstring(getCountMetricLine(qwenModelName, reqRunningMetricName, 0)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(qwenModelName, reqWaitingMetricName, 0)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(qwenModelName, gpuCacheUsageMetricName, 0)))
			}()
			wg.Wait()
		})

		It("Should send correct kv cache usage metrics for sequentual requests", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", qwenModelName, "--mode", common.ModeRandom,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--time-to-first-token", "5000", "--tokenizers-cache-dir", tmpDir, "--max-num-seqs", "2"}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			paramsArray := []openai.CompletionNewParams{
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in Haifa today? Is it cold?"),
					},
					Model: openai.CompletionNewParamsModel(qwenModelName),
				},
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in Haifa today?"),
					},
					Model: openai.CompletionNewParamsModel(qwenModelName),
				},
				{
					Prompt: openai.CompletionNewParamsPromptUnion{
						OfString: openai.String("What is the weather like in New York today?"),
					},
					Model: openai.CompletionNewParamsModel(qwenModelName),
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
				Expect(metrics).To(ContainSubstring(getCountMetricLine(qwenModelName, reqRunningMetricName, 2)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(qwenModelName, reqWaitingMetricName, 1)))
				Expect(metrics).To(ContainSubstring(getCountMetricLine(qwenModelName, gpuCacheUsageMetricName, 0.0625)))
			}()
			wg.Wait()
		})
	})

	Context("fake metrics", func() {
		It("Should respond with fake metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom,
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
					`"request-params-max-tokens":[10,20,30],` +
					`"ttft-buckets-values":[1,2,3],` +
					`"tpot-buckets-values":[0,0,1,2,3],` +
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
			Expect(metrics).To(ContainSubstring(getCountMetricLine(testModel, reqRunningMetricName, 10)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(testModel, reqWaitingMetricName, 30)))
			Expect(metrics).To(ContainSubstring(getCountMetricLine(testModel, gpuCacheUsageMetricName, 0.4)))
			Expect(metrics).To(ContainSubstring("vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora4,lora2\",waiting_lora_adapters=\"lora3\"} 1.257894567e+09"))
			Expect(metrics).To(ContainSubstring("vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora4,lora3\",waiting_lora_adapters=\"\"} 1.257894569e+09"))

			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, ttftMetricName, 0.001, 1)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, ttftMetricName, 0.005, 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, ttftMetricName, 0.01, 6)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, ttftMetricName, 0.02, 6)))

			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, tpotMetricName, 0.01, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, tpotMetricName, 0.025, 0)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, tpotMetricName, 0.05, 1)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, tpotMetricName, 0.075, 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, tpotMetricName, 0.1, 6)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, tpotMetricName, 0.15, 6)))

			buckets := build125Buckets(1024)
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

				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, generationTokensMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, promptTokensMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, paramMaxTokensMetricName, boundary, expectedCount)))

			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, generationTokensMetricName, math.Inf(1), expectedCount)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, promptTokensMetricName, math.Inf(1), expectedCount)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, paramMaxTokensMetricName, math.Inf(1), expectedCount)))

			Expect(metrics).To(ContainSubstring(`vllm:request_success_total{finish_reason="length",model_name="testmodel"} 0`))
			Expect(metrics).To(ContainSubstring(`vllm:request_success_total{finish_reason="remote_decode",model_name="testmodel"} 0`))
			Expect(metrics).To(ContainSubstring(`vllm:request_success_total{finish_reason="stop",model_name="testmodel"} 20`))
			Expect(metrics).To(ContainSubstring(`vllm:request_success_total{finish_reason="tool_calls",model_name="testmodel"} 0`))
		})
	})

	Context("fake ttft metrics", func() {
		It("Should respond with fake ttft metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeRandom,
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
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, ttftMetricName, boundary, 0)))
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, ttftMetricName, math.Inf(1), 1)))
		})
	})

	Context("fake latency metrics", func() {
		It("should respond with valid fake latency metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeEcho,
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

				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, e2eReqLatencyMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqInferenceTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqQueueTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, prefillTimeMetricName, boundary, expectedCount)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, decodeTimeMetricName, boundary, expectedCount)))
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, e2eReqLatencyMetricName, math.Inf(1), 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqInferenceTimeMetricName, math.Inf(1), 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqQueueTimeMetricName, math.Inf(1), 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, prefillTimeMetricName, math.Inf(1), 3)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, decodeTimeMetricName, math.Inf(1), 3)))
		})
	})

	Context("single request latency metrics", func() {
		numOfTokens := len(common.Tokenize(testUserMessage))

		DescribeTable("should calculate all latency related metrics correctly for a single request",
			func(testNamePrefix string, ttft int, prefillTimePerToken int, interTokenLatency int) {
				// send a single request with a prompt of 4 tokens and echo mode, so output tokens number of 4 too
				client := startServerAndSendRequest(testModel, testUserMessage, false, ttft, prefillTimePerToken, interTokenLatency)
				checkLatencyMertics(client, testModel, numOfTokens, numOfTokens, ttft, prefillTimePerToken, interTokenLatency)

				// same in streaming modeq
				client = startServerAndSendRequest(testModel, testUserMessage, true, ttft, prefillTimePerToken, interTokenLatency)
				checkLatencyMertics(client, testModel, numOfTokens, numOfTokens, ttft, prefillTimePerToken, interTokenLatency)
			},
			func(testNamePrefix string, ttft int, prefillTimePerToken int, interTokenLatency int) string {
				return fmt.Sprintf("%s\nttft: %d, prefillTimePerToken: %d, interTokenLatency: %d", testNamePrefix, ttft, prefillTimePerToken, interTokenLatency)
			},
			// Params order: testName, ttft, prefillTimePerToken, interTokenLatency
			Entry(nil, "constant prefill + inter token time", 0, 0, 100),
			Entry(nil, "constant prefill + inter token time", 900, 0, 100),
			Entry(nil, "constant prefill + inter token time", 1000, 0, 100),
			Entry(nil, "prefill per token + inter token time", 0, 100, 100),
		)
	})

	Context("multiple requests latency metrics", func() {
		It("should calculate waiting and inference time correctly", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeEcho,
				"--time-to-first-token", "1000", "--max-num-seqs", "1",
			}

			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, false)

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
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqInferenceTimeMetricName, boundary, 0)))
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqQueueTimeMetricName, boundary, 0)))
				} else {
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqInferenceTimeMetricName, boundary, 2)))
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqQueueTimeMetricName, boundary, 1)))
				}
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqInferenceTimeMetricName, math.Inf(1), 2)))
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(testModel, reqQueueTimeMetricName, math.Inf(1), 1)))
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
			got := build125Buckets(test.maxValue)
			Expect(got).To(Equal(test.want))
		}
	})
})
