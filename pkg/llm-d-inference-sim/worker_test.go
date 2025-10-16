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
	"net/http"
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

var _ = Describe("Simulator requests scheduling", Ordered, func() {
	Context("Requests for already loaded loras should be handled first", func() {
		DescribeTable("Should process in correct order simultaneous requests to two loras", func(maxNumSeq string) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", common.ModeEcho,
				"--time-to-first-token", "500", "--max-num-seqs", maxNumSeq,
				"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
				"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

			client, err := startServerWithArgs(ctx, common.ModeEcho, args, nil)
			Expect(err).NotTo(HaveOccurred())
			openaiclient := openai.NewClient(option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			numberOfRequests := 4
			orderOfRequests := make([]int, 0)
			var wg sync.WaitGroup
			wg.Add(numberOfRequests)
			var mux sync.RWMutex

			// Send simultaneously half of the requests to lora1 and the second half to lora2
			for reqNum := range numberOfRequests {
				params := paramsLora2
				if reqNum%2 == 0 {
					params = paramsLora1
				}
				go sendReq(ctx, openaiclient, &wg, 0, params, reqNum, &mux, &orderOfRequests)
			}
			wg.Wait()

			// Check the order in which the requests are handled:
			// if the first handled request is even, all the first half of the requests should
			// be even (because they all use the same lora that is already loaded).
			firstReqIsEven := orderOfRequests[0]%2 == 0
			for i, reqNum := range orderOfRequests {
				if i < numberOfRequests/2 {
					// nolint
					Expect(reqNum%2 == 0).To(Equal(firstReqIsEven))
				} else {
					// nolint
					Expect(reqNum%2 == 0).NotTo(Equal(firstReqIsEven))
				}
			}
		},
			Entry("5 workers", "5"),
			Entry("1 worker", "1"),
		)

		DescribeTable("Should process in correct order delayed requests to two loras",
			func(maxNumSeq string, maxLoras string, checkOrder func([]int)) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", model, "--mode", common.ModeEcho,
					"--time-to-first-token", "1000",
					"--max-num-seqs", maxNumSeq, "--max-loras", maxLoras,
					"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
					"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

				client, err := startServerWithArgs(ctx, common.ModeEcho, args, nil)
				Expect(err).NotTo(HaveOccurred())

				openaiclient := openai.NewClient(option.WithBaseURL(baseURL),
					option.WithHTTPClient(client))

				numberOfRequests := 8
				orderOfRequests := make([]int, 0)
				var wg sync.WaitGroup
				wg.Add(numberOfRequests)
				var mux sync.RWMutex

				// Send three requests to lora1, after 100 milliseconds four requests to lora2,
				// and after 400 milliseconds a request to lora1 again.
				for reqNum := range 3 {
					go sendReq(ctx, openaiclient, &wg, 0, paramsLora1, reqNum, &mux, &orderOfRequests)
				}
				for reqNum := 4; reqNum < 8; reqNum++ {
					go sendReq(ctx, openaiclient, &wg, 100, paramsLora2, reqNum, &mux, &orderOfRequests)
				}
				go sendReq(ctx, openaiclient, &wg, 500, paramsLora1, 3, &mux, &orderOfRequests)

				wg.Wait()

				// Check the order in which the requests are handled
				checkOrder(orderOfRequests)
			},
			Entry("5 workers, max loras 1", "5", "1", checkOrderMaxLora1Workers5),
			Entry("2 workers, max loras 1", "2", "1", checkOrderMaxLora1Workers2),
			Entry("5 workers, max loras 5", "5", "5", checkOrderMaxLora5Workers5),
			Entry("1 worker, max loras 5", "1", "5", checkOrderMaxLora5Workers1),
		)

		It("Should keep the order of requests with one worker", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", common.ModeEcho,
				"--time-to-first-token", "500",
				"--max-num-seqs", "1", "--max-loras", "1",
				"--lora-modules",
				"{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
				"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
				"{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}",
				"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

			client, err := startServerWithArgs(ctx, common.ModeEcho, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			numberOfRequests := 9
			orderOfRequests := make([]int, 0)
			var wg sync.WaitGroup
			wg.Add(numberOfRequests)
			var mux sync.RWMutex

			// The order of the requests is:
			// 0-lora1 1-lora1 2-lora2 3-lora3 4-lora4 5-lora1 6-lora2 7-lora3 8-lora4
			go sendReq(ctx, openaiclient, &wg, 0, paramsLora1, 0, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 50, paramsLora1, 1, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 100, paramsLora2, 2, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 200, paramsLora3, 3, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 300, paramsLora4, 4, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 400, paramsLora1, 5, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 500, paramsLora2, 6, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 600, paramsLora3, 7, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 700, paramsLora4, 8, &mux, &orderOfRequests)
			wg.Wait()

			// Check the order in which the requests are handled
			checkOrderMaxLora1Workers1(orderOfRequests)
		})

		It("Should keep the order of requests with two workers", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", model, "--mode", common.ModeEcho,
				"--time-to-first-token", "500",
				"--max-num-seqs", "2", "--max-loras", "1",
				"--lora-modules",
				"{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
				"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
				"{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}",
				"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

			client, err := startServerWithArgs(ctx, common.ModeEcho, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			numberOfRequests := 8
			orderOfRequests := make([]int, 0)
			var wg sync.WaitGroup
			wg.Add(numberOfRequests)
			var mux sync.RWMutex

			// The order of the requests is:
			// 0-lora1 1-lora1 2-lora2 3-lora3 4-lora4 5-lora2 6-lora3 7-lora4
			go sendReq(ctx, openaiclient, &wg, 0, paramsLora1, 0, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 0, paramsLora1, 1, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 100, paramsLora2, 2, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 200, paramsLora3, 3, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 300, paramsLora4, 4, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 400, paramsLora2, 5, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 500, paramsLora3, 6, &mux, &orderOfRequests)
			go sendReq(ctx, openaiclient, &wg, 600, paramsLora4, 7, &mux, &orderOfRequests)
			wg.Wait()

			// Check the order in which the requests are handled
			checkOrderWorkers2(orderOfRequests)
		})

		DescribeTable("Should keep the order of requests with multiple workers and loras",
			func(maxNumSeq string, maxLoras string, checkOrder func([]int)) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", model, "--mode", common.ModeEcho,
					"--time-to-first-token", "1000",
					"--max-num-seqs", maxNumSeq, "--max-loras", maxLoras,
					"--lora-modules",
					"{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
					"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
					"{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}",
					"{\"name\":\"lora5\",\"path\":\"/path/to/lora5\"}",
					"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}"}

				client, err := startServerWithArgs(ctx, common.ModeEcho, args, nil)
				Expect(err).NotTo(HaveOccurred())

				openaiclient := openai.NewClient(option.WithBaseURL(baseURL),
					option.WithHTTPClient(client))

				numberOfRequests := 11
				orderOfRequests := make([]int, 0)
				var wg sync.WaitGroup
				wg.Add(numberOfRequests)
				var mux sync.RWMutex

				// The order of the requests is:
				// 0-lora1 1-lora1 2-lora2 3-lora3 4-lora4 5-lora5
				// 6-lora1 7-lora2 8-lora3 9-lora4 10-lora5
				go sendReq(ctx, openaiclient, &wg, 0, paramsLora1, 0, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 100, paramsLora1, 1, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 200, paramsLora2, 2, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 300, paramsLora3, 3, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 400, paramsLora4, 4, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 500, paramsLora5, 5, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 600, paramsLora1, 6, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 700, paramsLora2, 7, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 800, paramsLora3, 8, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 900, paramsLora4, 9, &mux, &orderOfRequests)
				go sendReq(ctx, openaiclient, &wg, 1000, paramsLora5, 10, &mux, &orderOfRequests)
				wg.Wait()

				// Check the order in which the requests are handled
				checkOrder(orderOfRequests)
			},
			Entry("4 workers, max loras 3", "4", "3", checkOrderMaxLora3),
			Entry("5 workers, max loras 3", "5", "3", checkOrderMaxLora3),
			Entry("5 workers, max loras 5", "5", "5", checkOrderMaxLora5),
		)

	})

	Context("Stress", func() {
		It("Should work correctly with many simultaneous requests", func() {
			modelName := "testmodel"
			ctx := context.TODO()
			args := []string{"cmd", "--model", modelName, "--mode", common.ModeRandom,
				"--time-to-first-token", "3000", "--max-num-seqs", "12", "--max-loras", "2",
				"--lora-modules",
				"{\"name\":\"lora0\",\"path\":\"/path/to/lora0\"}",
				"{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
				"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}",
				"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
				"{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}",
			}

			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			// Run 1000 requests for 5 loras simultaneously
			numberOfRequests := 1000
			for i := range numberOfRequests {
				go func() {
					defer GinkgoRecover()
					params := openai.ChatCompletionNewParams{
						Messages: []openai.ChatCompletionMessageParamUnion{
							openai.UserMessage(userMessage),
						},
						Model: fmt.Sprintf("lora%d", i%5),
					}
					_, err := openaiclient.Chat.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
				}()
			}

			time.Sleep(2000 * time.Millisecond)
			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)

			// max-num-seqs is 12, so number of running requests should be 12
			// and the number of waiting requests 1000-12=988
			Expect(metrics).To(ContainSubstring("vllm:num_requests_running{model_name=\"testmodel\"} 12"))
			Expect(metrics).To(ContainSubstring("vllm:num_requests_waiting{model_name=\"testmodel\"} 988"))

			// max-loras is 2, so the last lora metric should be:
			// running: two loras (doesn't matter which two)
			// waiting: all the five loras
			// (there can be more than one metric with the same timestamp, therefore we check all of them)
			lastLoraMetrics, err := getLastLoraMetrics(strings.Split(string(data), "\n"))
			Expect(err).NotTo(HaveOccurred())

			allLoras := []string{"lora1", "lora2", "lora3", "lora4", "lora0"}
			Expect(
				isLoraMetricPresent(lastLoraMetrics, []string{"lora1", "lora2"}, allLoras) ||
					isLoraMetricPresent(lastLoraMetrics, []string{"lora1", "lora3"}, allLoras) ||
					isLoraMetricPresent(lastLoraMetrics, []string{"lora1", "lora4"}, allLoras) ||
					isLoraMetricPresent(lastLoraMetrics, []string{"lora1", "lora0"}, allLoras) ||
					isLoraMetricPresent(lastLoraMetrics, []string{"lora3", "lora2"}, allLoras) ||
					isLoraMetricPresent(lastLoraMetrics, []string{"lora4", "lora2"}, allLoras) ||
					isLoraMetricPresent(lastLoraMetrics, []string{"lora0", "lora2"}, allLoras) ||
					isLoraMetricPresent(lastLoraMetrics, []string{"lora3", "lora4"}, allLoras) ||
					isLoraMetricPresent(lastLoraMetrics, []string{"lora3", "lora0"}, allLoras) ||
					isLoraMetricPresent(lastLoraMetrics, []string{"lora4", "lora0"}, allLoras)).
				To(BeTrue())
		})

		It("Should work correctly with many simultaneous requests with many workers", func() {
			modelName := "testmodel"
			runningMetric := "vllm:num_requests_running{model_name=\"testmodel\"}"
			waitingMetric := "vllm:num_requests_waiting{model_name=\"testmodel\"}"
			ctx := context.TODO()
			args := []string{"cmd", "--model", modelName, "--mode", common.ModeRandom,
				"--time-to-first-token", "2000", "--time-to-first-token-std-dev", "600",
				"--max-num-seqs", "1000", "--max-loras", "2", "--max-waiting-queue-length", "1500",
				"--lora-modules",
				"{\"name\":\"lora0\",\"path\":\"/path/to/lora0\"}",
				"{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
			}

			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			// Run 2000 requests for 5 loras simultaneously
			numberOfRequests := 2000
			for i := range numberOfRequests {
				go func() {
					defer GinkgoRecover()
					params := openai.ChatCompletionNewParams{
						Messages: []openai.ChatCompletionMessageParamUnion{
							openai.UserMessage(userMessage),
						},
						Model: fmt.Sprintf("lora%d", i%2),
					}
					_, err := openaiclient.Chat.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
				}()
			}

			time.Sleep(400 * time.Millisecond)
			metricsResp, err := client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := strings.Split(string(data), "\n")

			// max-num-seqs is 1000, so number of running requests should be 1000
			// and the number of waiting requests 2000-1000=2000
			runningStr := findMetric(metrics, runningMetric)
			Expect(runningStr).NotTo(Equal(""))
			running, err := strconv.Atoi(runningStr)
			Expect(err).NotTo(HaveOccurred())
			Expect(running).To(Equal(1000))
			waitingStr := findMetric(metrics, waitingMetric)
			waiting, err := strconv.Atoi(waitingStr)
			Expect(err).NotTo(HaveOccurred())
			Expect(waiting).To(Equal(1000))

			time.Sleep(1500 * time.Millisecond)

			// After about 2 secs (the mean ttft), send 500 more requests
			numberOfRequests = 500
			for i := range numberOfRequests {
				go func() {
					defer GinkgoRecover()
					params := openai.ChatCompletionNewParams{
						Messages: []openai.ChatCompletionMessageParamUnion{
							openai.UserMessage(userMessage),
						},
						Model: fmt.Sprintf("lora%d", i%2),
					}
					_, err := openaiclient.Chat.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
				}()
			}
			time.Sleep(400 * time.Millisecond)
			metricsResp, err = client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err = io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics = strings.Split(string(data), "\n")

			// We sent 2500 requests, after about 2.5 seconds
			// number of running requests should be 1000
			// and the number of waiting requests should be less than 1000
			runningStr = findMetric(metrics, runningMetric)
			Expect(runningStr).NotTo(Equal(""))
			running, err = strconv.Atoi(runningStr)
			Expect(err).NotTo(HaveOccurred())
			Expect(running).To(Equal(1000))
			waitingStr = findMetric(metrics, waitingMetric)
			waiting, err = strconv.Atoi(waitingStr)
			Expect(err).NotTo(HaveOccurred())
			Expect(waiting).To(BeNumerically("<", 1000))

			// Wait another second
			time.Sleep(1000 * time.Millisecond)
			metricsResp, err = client.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))
			data, err = io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics = strings.Split(string(data), "\n")

			// number of running requests should be 1000
			// and the number of waiting requests should be less than 1000
			runningStr = findMetric(metrics, runningMetric)
			Expect(runningStr).NotTo(Equal(""))
			running, err = strconv.Atoi(runningStr)
			Expect(err).NotTo(HaveOccurred())
			Expect(running).To(Equal(1000))
			waitingStr = findMetric(metrics, waitingMetric)
			waiting, err = strconv.Atoi(waitingStr)
			Expect(err).NotTo(HaveOccurred())
			Expect(waiting).To(BeNumerically("<", 1000))
		})
	})
})

func sendReq(ctx context.Context, openaiclient openai.Client, wg *sync.WaitGroup, delay int,
	params openai.ChatCompletionNewParams, reqNum int, mux *sync.RWMutex, orderOfRequests *[]int) {
	defer GinkgoRecover()
	defer wg.Done()
	time.Sleep(time.Duration(delay) * time.Millisecond)
	_, err := openaiclient.Chat.Completions.New(ctx, params)
	Expect(err).NotTo(HaveOccurred())
	mux.Lock()
	*orderOfRequests = append(*orderOfRequests, reqNum)
	mux.Unlock()
}

// Check the order of the delayed requests with max-loras=1 and five workers
// Three requests to lora1 (req numbers 0-2)
// after a delay four requests to lora2 (req numbers 4-7),
// after a delay one more request to lora1 (req number 3).
// All the requests to lora1 should be handled before the requests to lora2.
// The exact order of the first three requests to lora1 and the requests to lora 2
// is not important.
func checkOrderMaxLora1Workers5(orderOfRequests []int) {
	Expect(orderOfRequests).To(HaveLen(8))
	for i, reqNum := range orderOfRequests {
		switch {
		case i < 3:
			Expect(reqNum).To(BeNumerically("<", 4))
		case i == 3:
			Expect(reqNum).To(Equal(3))
		default:
			Expect(reqNum >= 4 && reqNum < 8).To(BeTrue())
		}
	}
}

// Check the order of the delayed requests with max-loras=1 and two workers
// Three requests to lora1 (req numbers 0-2)
// after a delay four requests to lora2 (req numbers 4-7),
// after a delay one more request to lora1 (req number 3).
// All the requests to lora1 should be handled before the requests to lora2.
// The first two requests have to be 0-2, the next two should be one of the requests
// from the first batch (0-2) and the last request to lora1 (req number 3), the
// next four should be requests to lora2 (4-7) in no particular order.
func checkOrderMaxLora1Workers2(orderOfRequests []int) {
	Expect(orderOfRequests).To(HaveLen(8))
	for i, reqNum := range orderOfRequests {
		switch {
		case i < 2:
			Expect(reqNum).To(BeNumerically("<", 3))
		case i < 4:
			Expect(reqNum).To(BeNumerically("<", 4))
		default:
			Expect(reqNum >= 4 && reqNum < 8).To(BeTrue())
		}
	}
}

// Check the order of the delayed requests with max-loras=5 and five workers
// Three requests to lora1 (req numbers 0-2)
// after a delay four requests to lora2 (req numbers 4-7),
// after a delay one more request to lora1 (req number 3).
// The requests should be handled in the order they are sent.
// The exact order of first three requests to lora1 and the four
// requests to lora2 is not important.
// The first three should be 0-2, the next two should be 4-7,
// the rest can be in any order.
func checkOrderMaxLora5Workers5(orderOfRequests []int) {
	for i, reqNum := range orderOfRequests {
		switch {
		case i < 3:
			Expect(reqNum).To(BeNumerically("<", 3))
		case i < 5:
			Expect(reqNum >= 4 && reqNum <= 7).To(BeTrue())
		default:
			Expect(reqNum).To(BeNumerically(">=", 3))
		}
	}
}

// Check the order of the delayed requests with max-loras=5 and one worker
// Three requests to lora1 (req numbers 0-2)
// after a delay four requests to lora2 (req numbers 4-7),
// after a delay one more request to lora1 (req number 3).
// The requests should be handled in the order they are sent.
// The exact order of first three requests to lora1 and the four
// requests to lora2 is not important.
// The first three should be 0-2, the next one should be 3,
// the rest 4-7.
func checkOrderMaxLora5Workers1(orderOfRequests []int) {
	for i, reqNum := range orderOfRequests {
		switch {
		case i < 3:
			Expect(reqNum).To(BeNumerically("<", 3))
		case i == 3:
			Expect(reqNum).To(Equal(3))
		default:
			Expect(reqNum).To(BeNumerically(">", 3))
		}
	}
}

// Check the order of requests sent in specific order with one worker
// The requests are sent with delays to make sure they enter the queue
// in the order they are sent.
// The order of the requests is:
// 0-lora1 1-lora1 2-lora2 3-lora3 4-lora4 5-lora1 6-lora2 7-lora3 8-lora4
// The expected order of processing:
// 015263748
func checkOrderMaxLora1Workers1(orderOfRequests []int) {
	expected := []int{0, 1, 5, 2, 6, 3, 7, 4, 8}
	Expect(orderOfRequests).To(HaveLen(9))
	for i, reqNum := range orderOfRequests {
		Expect(reqNum).To(Equal(expected[i]))
	}
}

// Check the order of requests sent in specific order with two workers
// The requests are sent with delays to make sure they enter the queue
// in the order they are sent.
// The order of the requests is:
// 0-lora1 1-lora1 2-lora2 3-lora3 4-lora4 5-lora2 6-lora3 7-lora4
// The expected order of processing:
// {01}{25}{36}{47} - the order inside the brackets doesn't matter
func checkOrderWorkers2(orderOfRequests []int) {
	expected1 := []int{0, 1, 2, 5, 3, 6, 4, 7}
	expected2 := []int{1, 0, 5, 2, 6, 3, 7, 4}
	Expect(orderOfRequests).To(HaveLen(8))
	for i, reqNum := range orderOfRequests {
		Expect(reqNum).To(Or(Equal(expected1[i]), Equal(expected2[i])))
	}
}

// Check the order of requests sent in specific order with max loras = 3
// The requests are sent with delays to make sure they enter the queue
// in the order they are sent.
// The order of the requests is:
// 0-lora1 1-lora1 2-lora2 3-lora3 4-lora4 5-lora5
// 6-lora1 7-lora2 8-lora3 9-lora4 10-lora5
// The expected order of processing:
// 0, 1, 2, 3, 6, 7, 8, {4, 9}, {5, 10} - the order inside the brackets doesn't matter
func checkOrderMaxLora3(orderOfRequests []int) {
	expected1 := []int{0, 1, 2, 3, 6, 7, 8, 4, 9, 5, 10}
	expected2 := []int{0, 1, 2, 3, 6, 7, 8, 4, 9, 10, 5}
	expected3 := []int{0, 1, 2, 3, 6, 7, 8, 9, 4, 5, 10}
	expected4 := []int{0, 1, 2, 3, 6, 7, 8, 9, 4, 10, 5}
	Expect(orderOfRequests).To(HaveLen(11))
	for i, reqNum := range orderOfRequests {
		Expect(reqNum).To(Or(Equal(expected1[i]), Equal(expected2[i]),
			Equal(expected3[i]), Equal(expected4[i])))
	}
}

// Check the order of requests sent in specific order with max loras = 5
// The requests are sent with delays to make sure they enter the queue
// in the order they are sent.
// The order of the requests is:
// 0-lora1 1-lora1 2-lora2 3-lora3 4-lora4 5-lora5
// 6-lora1 7-lora2 8-lora3 9-lora4 10-lora5
// The expected order of processing:
// 0, 1, 2, 3, 4, 6, 7, 8, 9, 5, 10
func checkOrderMaxLora5(orderOfRequests []int) {
	expected := []int{0, 1, 2, 3, 4, 6, 7, 8, 9, 5, 10}
	Expect(orderOfRequests).To(HaveLen(11))
	for i, reqNum := range orderOfRequests {
		Expect(reqNum).To(Equal(expected[i]))
	}
}
