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

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			numberOfRequests := 4
			var wg sync.WaitGroup
			wg.Add(numberOfRequests)

			orderOfRequests := make([]int, 0)
			var mux sync.RWMutex

			// Send simultaneously half of the requests to lora1 and the second half to lora2
			for reqNum := range numberOfRequests {
				go func() {
					defer GinkgoRecover()
					defer wg.Done()
					params := paramsLora2
					if reqNum%2 == 0 {
						params = paramsLora1
					}

					_, err := openaiclient.Chat.Completions.New(ctx, params)
					Expect(err).NotTo(HaveOccurred())
					mux.Lock()
					orderOfRequests = append(orderOfRequests, reqNum)
					mux.Unlock()
				}()
			}
			wg.Wait()

			// Check the order in which the requests are handled:
			// if the first handled request is even, all the first half of the requests should
			// be even (because they all use the same lora that is already loaded).
			firstReqIsEven := orderOfRequests[0]%2 == 0
			for i, reqNum := range orderOfRequests {
				if i < numberOfRequests/2 {
					Expect(reqNum%2 == 0).To(Equal(firstReqIsEven))
				} else {
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

				openaiclient := openai.NewClient(
					option.WithBaseURL(baseURL),
					option.WithHTTPClient(client))

				numberOfRequests := 8
				var wg sync.WaitGroup
				wg.Add(numberOfRequests)

				orderOfRequests := make([]int, 0)

				var mux sync.RWMutex

				for reqNum := range 3 {
					go func() {
						defer GinkgoRecover()
						defer wg.Done()
						_, err := openaiclient.Chat.Completions.New(ctx, paramsLora1)
						Expect(err).NotTo(HaveOccurred())
						mux.Lock()
						orderOfRequests = append(orderOfRequests, reqNum)
						mux.Unlock()
					}()
				}

				for reqNum := 4; reqNum < 8; reqNum++ {
					go func() {
						defer GinkgoRecover()
						defer wg.Done()
						time.Sleep(100 * time.Millisecond)
						_, err := openaiclient.Chat.Completions.New(ctx, paramsLora2)
						Expect(err).NotTo(HaveOccurred())
						mux.Lock()
						orderOfRequests = append(orderOfRequests, reqNum)
						mux.Unlock()
					}()
				}

				go func() {
					defer GinkgoRecover()
					defer wg.Done()
					time.Sleep(500 * time.Millisecond)
					_, err := openaiclient.Chat.Completions.New(ctx, paramsLora1)
					Expect(err).NotTo(HaveOccurred())
					mux.Lock()
					orderOfRequests = append(orderOfRequests, 3)
					mux.Unlock()
				}()

				wg.Wait()

				// Check the order in which the requests are handled
				checkOrder(orderOfRequests)
			},
			Entry("5 workers, max loras 1", "5", "1", checkOderMaxLora1Workers5),
			Entry("2 workers, max loras 1", "2", "1", checkOderMaxLora1Workers2),
			Entry("5 workers, max loras 5", "5", "5", checkOderMaxLora5Workers5),
			Entry("1 worker, max loras 5", "1", "5", checkOderMaxLora5Workers1),
		)
	})

	Context("Stress", func() {
		It("Should work correctly with many simultaneous requests", func() {
			modelName := "testmodel"
			// Three requests, only two can run in parallel, we expect
			// two running requests and one waiting request in the metrics
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

			// Run 999 requests for 5 loras simultaneously
			numberOfRequests := 999
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
			// and the number of waiting requests 999-12=987
			Expect(metrics).To(ContainSubstring("vllm:num_requests_running{model_name=\"testmodel\"} 12"))
			Expect(metrics).To(ContainSubstring("vllm:num_requests_waiting{model_name=\"testmodel\"} 987"))

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

	})
})

// Check the order of the delayed requests with max-loras=1 and five workers
// Three requests to lora1 (req numbers 0-2)
// after a delay four requests to lora2 (req numbers 4-7),
// after a delay one more request to lora1 (req number 3).
// All the requests to lora1 should be handled before the requests to lora2.
// The exact order of the first three requests to lora1 and the requests to lora 2
// is not important.
func checkOderMaxLora1Workers5(orderOfRequests []int) {
	fmt.Println(orderOfRequests)
	Expect(orderOfRequests).To(HaveLen(8))
	for i, reqNum := range orderOfRequests {
		if i < 3 {
			Expect(reqNum < 4).To(BeTrue())
		} else if i == 3 {
			Expect(reqNum).To(Equal(3))
		} else {
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
func checkOderMaxLora1Workers2(orderOfRequests []int) {
	fmt.Println(orderOfRequests)
	Expect(orderOfRequests).To(HaveLen(8))
	for i, reqNum := range orderOfRequests {
		if i < 2 {
			Expect(reqNum < 3).To(BeTrue())
		} else if i < 4 {
			Expect(reqNum < 4).To(BeTrue())
		} else {
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
func checkOderMaxLora5Workers5(orderOfRequests []int) {
	fmt.Println(orderOfRequests)
	for i, reqNum := range orderOfRequests {
		if i < 3 {
			Expect(reqNum < 3).To(BeTrue())
		} else if i < 5 {
			Expect(reqNum >= 4 && reqNum <= 7).To(BeTrue())
		} else {
			Expect(reqNum >= 3).To(BeTrue())
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
func checkOderMaxLora5Workers1(orderOfRequests []int) {
	fmt.Println(orderOfRequests)
	for i, reqNum := range orderOfRequests {
		if i < 3 {
			Expect(reqNum < 3).To(BeTrue())
		} else if i == 3 {
			Expect(reqNum).To(Equal(3))
		} else {
			Expect(reqNum > 3).To(BeTrue())
		}
	}
}

/*
stress:
1000 simultaneously
/metrics when they are still running
5 loras, max 2
check running and waiting, loras 2 running waiting 5 - last entry


ttft 1000 delay 100
max lora 1 workers 5
3 - lora1, delay 4 - lora2 , delay 2 - lora1
check first all lora1, then 2

same with workers 2
response from lora2 after 4 secs

---------

max lora 5 w 2
3 - lora1, delay 4 - lora2 , delay 2 - lora1
should be in this order - can't check the real order when they wait

same all with delays between all - still can't guarantee the exact order
check the order of reqs not just loras
----------

max lora 3 w 3
reqs to 5 loras
input: 11234512345
order of running: 11231234545

max lora 3 w 5
*/
