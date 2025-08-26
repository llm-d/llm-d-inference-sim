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
	"io"
	"net/http"
	"regexp"
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

var _ = Describe("Simulator metrics", Ordered, func() {
	It("Should send correct running and waiting requests metrics", func() {
		modelName := "testmodel"
		// Three requests, only two can run in parallel, we expect
		// two running requests and one waiting request in the metrics
		ctx := context.TODO()
		args := []string{"cmd", "--model", modelName, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000", "--max-num-seqs", "2"}

		s, client, err := startServerWithArgsAndMetrics(ctx, common.ModeRandom, args, nil, true)
		Expect(err).NotTo(HaveOccurred())
		defer s.unregisterPrometheus()

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		params := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(userMessage),
			},
			Model: modelName,
		}

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
			metricsResp, err := client.Get("http://localhost/metrics")
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

		s, client, err := startServerWithArgsAndMetrics(ctx, common.ModeRandom, args, nil, true)
		Expect(err).NotTo(HaveOccurred())
		defer s.unregisterPrometheus()

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		params1 := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(userMessage),
			},
			Model: "lora1",
		}

		_, err = openaiclient.Chat.Completions.New(ctx, params1)
		Expect(err).NotTo(HaveOccurred())

		params2 := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(userMessage),
			},
			Model: "lora2",
		}

		_, err = openaiclient.Chat.Completions.New(ctx, params2)
		Expect(err).NotTo(HaveOccurred())

		metricsResp, err := client.Get("http://localhost/metrics")
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := string(data)

		// We sent two sequentual requests to two different LoRAs, we expect to see (in this order)
		// 1. running_lora_adapter = lora1
		// 2. running_lora_adapter = lora2
		// 3. running_lora_adapter = {}
		lora1 := "vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora1\",waiting_lora_adapters=\"\"}"
		lora2 := "vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora2\",waiting_lora_adapters=\"\"}"
		empty := "vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"\",waiting_lora_adapters=\"\"}"

		Expect(metrics).To(ContainSubstring(lora1))
		Expect(metrics).To(ContainSubstring(lora2))
		Expect(metrics).To(ContainSubstring(empty))

		// Check the order
		lora1Timestamp := extractTimestamp(metrics, lora1)
		lora2Timestamp := extractTimestamp(metrics, lora2)
		noLorasTimestamp := extractTimestamp(metrics, empty)

		Expect(lora1Timestamp < lora2Timestamp).To(BeTrue())
		Expect(lora2Timestamp < noLorasTimestamp).To(BeTrue())
	})

	It("Should send correct lora metrics for parallel requests", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
			"--time-to-first-token", "2000",
			"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
			"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

		s, client, err := startServerWithArgsAndMetrics(ctx, common.ModeRandom, args, nil, true)
		Expect(err).NotTo(HaveOccurred())

		defer s.unregisterPrometheus()

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client))

		params1 := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(userMessage),
			},
			Model: "lora1",
		}

		params2 := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(userMessage),
			},
			Model: "lora2",
		}

		var wg sync.WaitGroup
		wg.Add(1)

		go func() {
			time.Sleep(1 * time.Second)
			defer wg.Done()
			defer GinkgoRecover()
			_, err := openaiclient.Chat.Completions.New(ctx, params2)
			Expect(err).NotTo(HaveOccurred())
		}()

		_, err = openaiclient.Chat.Completions.New(ctx, params1)
		Expect(err).NotTo(HaveOccurred())

		wg.Wait()

		metricsResp, err := client.Get("http://localhost/metrics")
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := string(data)

		// We sent two parallel requests: first to lora1 and then to lora2 (with a delay), we expect
		// to see (in this order)
		// 1. running_lora_adapter = lora1
		// 2. running_lora_adapter = lora2,lora1 (the order of LoRAs doesn't matter here)
		// 3. running_lora_adapter = lora2
		// 4. running_lora_adapter = {}
		lora1 := "vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora1\",waiting_lora_adapters=\"\"}"
		lora12 := "vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora1,lora2\",waiting_lora_adapters=\"\"}"
		lora21 := "vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora2,lora1\",waiting_lora_adapters=\"\"}"
		lora2 := "vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"lora2\",waiting_lora_adapters=\"\"}"
		empty := "vllm:lora_requests_info{max_lora=\"1\",running_lora_adapters=\"\",waiting_lora_adapters=\"\"}"

		Expect(metrics).To(ContainSubstring(lora1))
		Expect(metrics).To(Or(ContainSubstring(lora12), ContainSubstring(lora21)))
		Expect(metrics).To(ContainSubstring(lora2))
		Expect(metrics).To(ContainSubstring(empty))

		// Check the order
		lora1Timestamp := extractTimestamp(metrics, lora1)
		lora2Timestamp := extractTimestamp(metrics, lora2)
		noLorasTimestamp := extractTimestamp(metrics, empty)
		var twoLorasTimestamp float64
		if strings.Contains(metrics, lora12) {
			twoLorasTimestamp = extractTimestamp(metrics, lora12)
		} else {
			twoLorasTimestamp = extractTimestamp(metrics, lora21)
		}
		Expect(lora1Timestamp < twoLorasTimestamp).To(BeTrue())
		Expect(twoLorasTimestamp < lora2Timestamp).To(BeTrue())
		Expect(lora2Timestamp < noLorasTimestamp).To(BeTrue())
	})

	Context("fake metrics", func() {
		It("Should respond with fake metrics to /metrics", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
				"--fake-metrics",
				"{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":0.4,\"loras\":[{\"running\":\"lora4,lora2\",\"waiting\":\"lora3\",\"timestamp\":1257894567},{\"running\":\"lora4,lora3\",\"waiting\":\"\",\"timestamp\":1257894569}]}",
			}

			s, client, err := startServerWithArgsAndMetrics(ctx, common.ModeRandom, args, nil, true)
			Expect(err).NotTo(HaveOccurred())

			defer s.unregisterPrometheus()

			resp, err := client.Get("http://localhost/metrics")
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
		})
	})
})

func extractTimestamp(metrics string, key string) float64 {
	re := regexp.MustCompile(key + ` (\S+)`)
	result := re.FindStringSubmatch(metrics)
	Expect(len(result)).To(BeNumerically(">", 1))
	f, err := strconv.ParseFloat(result[1], 64)
	Expect(err).NotTo(HaveOccurred())
	return f
}
