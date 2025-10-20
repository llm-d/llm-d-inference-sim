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
	"io"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const (
	metricsUrl = "http://localhost/metrics"

	lora1 = "lora1"
	lora2 = "lora2"
)

var emptyArray = []string{}
var lora1Arr = []string{lora1}
var lora2Arr = []string{lora2}

var paramsLora1 openai.ChatCompletionNewParams = openai.ChatCompletionNewParams{
	Messages: []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(userMessage),
	},
	Model: "lora1",
}

var paramsLora2 openai.ChatCompletionNewParams = openai.ChatCompletionNewParams{
	Messages: []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(userMessage),
	},
	Model: "lora2",
}

var _ = Describe("Simulator metrics", Ordered, func() {
	It("Should send correct running and waiting requests metrics", func() {
		modelName := "testmodel"
		// Three requests, only two can run in parallel, we expect
		// two running requests and one waiting request in the metrics
		ctx := context.TODO()
		args := []string{"cmd", "--model", modelName, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000", "--max-num-seqs", "2"}

		client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClentAndChatParams(client, modelName, userMessage, false)

		var wg sync.WaitGroup
		wg.Add(1)

		for range 3 {
			go func() {
				defer GinkgoRecover()
				_, err := openaiclient.Chat.Completions.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
			}()
		}

		go func() {
			defer wg.Done()
			defer GinkgoRecover()

			time.Sleep(300 * time.Millisecond)
			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)
			Expect(metrics).To(ContainSubstring("vllm:num_requests_running{model_name=\"testmodel\"} 2"))
			Expect(metrics).To(ContainSubstring("vllm:num_requests_waiting{model_name=\"testmodel\"} 1"))
		}()

		wg.Wait()
	})

	It("Should send correct lora metrics", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000",
			"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
			"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

		client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
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
		// 1. running: empty, waiting: lora1
		// 2. running: lora1, waiting: empty
		// 3. running: empty, waiting: lora2
		// 4. running: lora2, waiting: empty
		// 5. running: empty, waiting: empty
		Expect(isLoraMetricPresent(metrics, emptyArray, lora1Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, lora1Arr, emptyArray)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, emptyArray, lora2Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, lora2Arr, emptyArray)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, emptyArray, emptyArray)).To(BeTrue())

		// Check the order
		timestamp1 := getLoraValidTimestamp(metrics, emptyArray, lora1Arr)
		timestamp2 := getLoraValidTimestamp(metrics, lora1Arr, emptyArray)
		timestamp3 := getLoraValidTimestamp(metrics, emptyArray, lora2Arr)
		timestamp4 := getLoraValidTimestamp(metrics, lora2Arr, emptyArray)
		timestamp5 := getLoraValidTimestamp(metrics, emptyArray, emptyArray)

		Expect(timestamp1 <= timestamp2).To(BeTrue())
		Expect(timestamp2 <= timestamp3).To(BeTrue())
		Expect(timestamp3 <= timestamp4).To(BeTrue())
		Expect(timestamp4 <= timestamp5).To(BeTrue())
	})

	It("Should send correct lora metrics for parallel requests with delay", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000",
			"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
			"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

		client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		var wg sync.WaitGroup
		wg.Add(1)

		// sends three requests with a delay of 0.5 second between them
		// request1 for lora1, request2 for lora2, and request 3 for lora1
		go func() {
			time.Sleep(500 * time.Millisecond)
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

		// We sent 3 requests, we expect to see (in this order)
		// 1. running: empty, waiting: lora1
		// 2. running: lora1, waiting: lora2
		// 3. running: lora1, lora2 (in any order), waiting: lora1
		// 4. running: lora1, lora2 (in any order), waiting: empty
		// 5. running: lora1, waiting: empty
		// 6. running: empty, waiting: empty
		Expect(isLoraMetricPresent(metrics, emptyArray, lora1Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, lora1Arr, lora2Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, []string{lora1, lora2}, lora1Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, []string{lora1, lora2}, emptyArray)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, lora1Arr, emptyArray)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, emptyArray, emptyArray)).To(BeTrue())

		// Check the order
		timestamp1 := getLoraValidTimestamp(metrics, emptyArray, lora1Arr)
		timestamp2 := getLoraValidTimestamp(metrics, lora1Arr, lora2Arr)
		timestamp3 := getLoraValidTimestamp(metrics, []string{lora1, lora2}, lora1Arr)
		timestamp4 := getLoraValidTimestamp(metrics, []string{lora1, lora2}, emptyArray)
		timestamp5 := getLoraValidTimestamp(metrics, lora1Arr, emptyArray)
		timestamp6 := getLoraValidTimestamp(metrics, emptyArray, emptyArray)

		// in case of requests sent with delay the order is well-defined
		Expect(timestamp1 <= timestamp2).To(BeTrue())
		Expect(timestamp2 <= timestamp3).To(BeTrue())
		Expect(timestamp3 <= timestamp4).To(BeTrue())
		Expect(timestamp4 <= timestamp5).To(BeTrue())
		Expect(timestamp5 <= timestamp6).To(BeTrue())
	})

	It("Should send correct lora metrics for parallel requests without delay", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000",
			"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
			"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

		client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
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
		// 1. running: empty, waiting: lora1 or lora2 (depends which request received first)
		// 2. running: one of the loras, waiting: another lora
		// 3. running: both lora2 and lora1 (the order of LoRAs doesn't matter here), waiting: empty
		// 4. running: empty, waiting: empty
		Expect(isLoraMetricPresent(metrics, emptyArray, lora1Arr) || isLoraMetricPresent(metrics, emptyArray, lora2Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, lora1Arr, lora2Arr) || isLoraMetricPresent(metrics, lora2Arr, lora1Arr)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, []string{lora1, lora2}, emptyArray)).To(BeTrue())
		Expect(isLoraMetricPresent(metrics, emptyArray, emptyArray)).To(BeTrue())

		// Check the order:
		// 1. one of the loras in the waiting list
		// 2. both loras in the running list
		// 3. empty
		l1WaitingTimestamp, err := getLoraTimestamp(metrics, emptyArray, lora1Arr)
		Expect(err).NotTo(HaveOccurred())
		l2WaitingTimestamp, err := getLoraTimestamp(metrics, emptyArray, lora2Arr)
		Expect(err).NotTo(HaveOccurred())
		Expect((l1WaitingTimestamp != nil)).ToNot(Equal((l2WaitingTimestamp != nil)))
		var singleWaitingTimestamp float64
		if l1WaitingTimestamp != nil {
			singleWaitingTimestamp = *l1WaitingTimestamp
		} else {
			singleWaitingTimestamp = *l2WaitingTimestamp
		}

		bothRunningTimestamp := getLoraValidTimestamp(metrics, []string{lora1, lora2}, emptyArray)
		emptyTimestamp := getLoraValidTimestamp(metrics, emptyArray, emptyArray)

		Expect(singleWaitingTimestamp <= bothRunningTimestamp).To(BeTrue())
		Expect(bothRunningTimestamp <= emptyTimestamp).To(BeTrue())
	})

	It("Should send correct ttft and tpot metrics", func() {
		modelName := "my_model"
		// Send one request, check that ttft and tpot are as defined in the simulator command line params
		ctx := context.TODO()
		args := []string{"cmd", "--model", modelName, "--mode", common.ModeRandom,
			"--time-to-first-token", "200", "--inter-token-latency", "100"}

		client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClentAndChatParams(client, modelName, userMessage, false)
		params.MaxTokens = openai.Int(5)

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
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.001\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.005\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.01\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.02\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.04\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.06\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.08\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.1\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.25\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.5\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.75\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"1\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"2.5\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"5\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"7.5\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"10\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"20\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"40\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"80\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"160\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"640\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"2560\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"+Inf\"} 1"))
			// tpot
			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.01\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.025\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.05\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.075\"} 0"))

			metricsLines := strings.Split(metrics, "\n")
			// the following values should be greater than 0, we don't know the exact value since it depends on the random response length
			count := findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.1\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.15\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.2\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.3\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.4\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.5\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.75\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"1\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"2.5\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"5\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"7.5\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"10\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"20\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"40\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"80\"}")
			Expect(count).ToNot(BeNil())
			Expect(*count).To(BeNumerically(">", 0))
			count = findIntMetric(metricsLines, "vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"+Inf\"}")
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

			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
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
				Expect(metrics).To(ContainSubstring("vllm:num_requests_running{model_name=\"Qwen/Qwen2-0.5B\"} 3"))
				Expect(metrics).To(ContainSubstring("vllm:num_requests_waiting{model_name=\"Qwen/Qwen2-0.5B\"} 0"))
				Expect(metrics).To(ContainSubstring("vllm:gpu_cache_usage_perc{model_name=\"Qwen/Qwen2-0.5B\"} 0.125"))

				time.Sleep(4 * time.Second)
				metricsResp, err = client.Get(metricsUrl)
				Expect(err).NotTo(HaveOccurred())
				Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

				data, err = io.ReadAll(metricsResp.Body)
				Expect(err).NotTo(HaveOccurred())
				metrics = string(data)
				// The requests finished running, expect 0 usage
				Expect(metrics).To(ContainSubstring("vllm:num_requests_running{model_name=\"Qwen/Qwen2-0.5B\"} 0"))
				Expect(metrics).To(ContainSubstring("vllm:num_requests_waiting{model_name=\"Qwen/Qwen2-0.5B\"} 0"))
				Expect(metrics).To(ContainSubstring("vllm:gpu_cache_usage_perc{model_name=\"Qwen/Qwen2-0.5B\"} 0"))
			}()
			wg.Wait()
		})

		It("Should send correct kv cache usage metrics for sequentual requests", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", qwenModelName, "--mode", common.ModeRandom,
				"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
				"--time-to-first-token", "5000", "--tokenizers-cache-dir", tmpDir, "--max-num-seqs", "2"}

			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
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
				Expect(metrics).To(ContainSubstring("vllm:num_requests_running{model_name=\"Qwen/Qwen2-0.5B\"} 2"))
				Expect(metrics).To(ContainSubstring("vllm:num_requests_waiting{model_name=\"Qwen/Qwen2-0.5B\"} 1"))
				Expect(metrics).To(ContainSubstring("vllm:gpu_cache_usage_perc{model_name=\"Qwen/Qwen2-0.5B\"} 0.0625"))
			}()
			wg.Wait()
		})
	})

	Context("fake metrics", func() {
		It("Should respond with fake metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
				"--fake-metrics",
				"{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":0.4,\"loras\":[{\"running\":\"lora4,lora2\",\"waiting\":\"lora3\",\"timestamp\":1257894567},{\"running\":\"lora4,lora3\",\"waiting\":\"\",\"timestamp\":1257894569}],\"ttft-buckets-values\":[1, 2, 3],\"tpot-buckets-values\": [0, 0, 1, 2, 3]}",
			}

			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)
			Expect(metrics).To(ContainSubstring("vllm:num_requests_running{model_name=\"my_model\"} 10"))
			Expect(metrics).To(ContainSubstring("vllm:num_requests_waiting{model_name=\"my_model\"} 30"))
			Expect(metrics).To(ContainSubstring("vllm:gpu_cache_usage_perc{model_name=\"my_model\"} 0.4"))
			Expect(metrics).To(ContainSubstring("vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora4,lora2\",waiting_lora_adapters=\"lora3\"} 1.257894567e+09"))
			Expect(metrics).To(ContainSubstring("vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora4,lora3\",waiting_lora_adapters=\"\"} 1.257894569e+09"))

			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.001\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.005\"} 3"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.01\"} 6"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.02\"} 6"))

			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.01\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.025\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.05\"} 1"))
			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.075\"} 3"))
			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.1\"} 6"))
			Expect(metrics).To(ContainSubstring("vllm:time_per_output_token_seconds_bucket{model_name=\"my_model\",le=\"0.15\"} 6"))
		})
	})

	Context("fake ttft metrics", func() {
		It("Should respond with fake ttft metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
				"--fake-metrics",
				"{\"ttft-buckets-values\":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}",
			}

			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
			Expect(err).NotTo(HaveOccurred())

			resp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.001\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.005\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.01\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.02\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.04\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.06\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.08\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.1\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.25\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.5\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"0.75\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"1\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"2.5\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"5\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"7.5\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"10\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"20\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"40\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"80\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"160\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"640\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"2560\"} 0"))
			Expect(metrics).To(ContainSubstring("vllm:time_to_first_token_seconds_bucket{model_name=\"my_model\",le=\"+Inf\"} 1"))
		})
	})
})

// isLoraMetricPresent checks if a matching metric exists
// metrics: the list of metrics
// running: list of loras in running_lora_adapters, the order does not matter
// waiting: list of loras in waiting_lora_adapters, the order does not matter
func isLoraMetricPresent(metrics []string, running, waiting []string) bool {
	return findLoraMetric(metrics, running, waiting) != ""
}

// getLoraTimestamp returns timestamp or nil, error
func getLoraTimestamp(metrics []string, running, waiting []string) (*float64, error) {
	mertic := findLoraMetric(metrics, running, waiting)
	if mertic == "" {
		return nil, nil // not found
	}
	// Extract timestamp: last part after space
	parts := strings.Split(mertic, " ")
	if len(parts) < 2 {
		return nil, errors.New("invalid metric format")
	}
	timestampStr := parts[len(parts)-1]
	timestamp, err := strconv.ParseFloat(timestampStr, 64)
	Expect(err).NotTo(HaveOccurred())

	return &timestamp, nil
}

func getLoraValidTimestamp(metrics []string, running, waiting []string) float64 {
	timestamp, err := getLoraTimestamp(metrics, running, waiting)
	Expect(err).NotTo(HaveOccurred())
	Expect(timestamp).ToNot(BeNil())
	return *timestamp
}

// findLoraMetric finds the relevant metric by comparing with the given loras sets (ignoring order)
// metrics: lines of metrics
// running: list of running loras to find
// waiting: list of waiting loras to find
// Looks for a line with the given running and waiting loras sets, the comparison is order agnostic.
// Return metric should match in both running and waiting sets.
// E.g. for input running=["l1", "l2", "l3"] and waiting=[] will return metric
// with running_lora_adapters=["l3", "l1", "l2"] and waiting_lora_adapters=[]
func findLoraMetric(metrics []string, running, waiting []string) string {
	// sort input arrays before compare, create string of all values, separated by comma
	sort.Strings(running)
	sort.Strings(waiting)
	runStr := strings.Join(running, ",")
	waitStr := strings.Join(waiting, ",")

	// regex to extract lora metrics and values
	re := regexp.MustCompile(`vllm:lora_requests_info\{.*running_lora_adapters="([^"]*)".*waiting_lora_adapters="([^"]*)".*\}\s+([0-9.e\+\-]+)`)
	for _, metric := range metrics {
		matches := re.FindStringSubmatch(metric)
		if len(matches) == 4 {
			// this line contains loraInfo metric, check running and waiting loras lists
			// split and sort metric's running and waiting loras lists for the comparison
			metricRun := splitString(matches[1])
			metricWait := splitString(matches[2])
			sort.Strings(metricRun)
			sort.Strings(metricWait)
			// if both lists are the same - return the metric
			if strings.Join(metricRun, ",") == runStr && strings.Join(metricWait, ",") == waitStr {
				return metric
			}
		} // if the metric is not in the required format - skip it
	}

	// required metric was not found
	return ""
}

// splits the given string to array of strings with separator = ","
func splitString(str string) []string {
	if str == "" {
		return []string{}
	}
	return strings.Split(str, ",")
}

func findMetric(metrics []string, metricPrefix string) string {
	// regex to extract metrics and values
	for _, metric := range metrics {
		if strings.Contains(metric, metricPrefix) {
			arr := strings.Split(metric, " ")
			if len(arr) == 2 {
				return arr[1]
			}
			break
		}
	}
	// required metric was not found
	return ""
}

func findIntMetric(metrics []string, metricPrefix string) *int {
	valueStr := findMetric(metrics, metricPrefix)
	if valueStr == "" {
		return nil
	}

	val, err := strconv.Atoi(valueStr)
	if err != nil {
		return nil
	}
	return &val
}
