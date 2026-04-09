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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("gRPC Metrics", Ordered, func() {

	It("should send correct running and waiting requests metrics via gRPC", func() {
		// Three requests, only two can run in parallel, we expect
		// two running requests and one waiting request in the metrics
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--time-to-first-token", "3000", "--max-num-seqs", "2"}

		_, comm, httpClient, err := startServerHandle(ctx, common.ModeRandom, args, nil)
		Expect(err).NotTo(HaveOccurred())

		maxTokens := uint32(128)
		req := pb.GenerateRequest{
			RequestId: "grpc-metrics-test",
			Input: &pb.GenerateRequest_Tokenized{
				Tokenized: &pb.TokenizedInput{
					InputIds: []uint32{32, 45, 78, 13},
				},
			},
			SamplingParams: &pb.SamplingParams{
				MaxTokens: &maxTokens,
			},
		}

		for range 3 {
			go func() {
				defer GinkgoRecover()
				out := mockGenerateServer{start: time.Now(), ctx: context.Background()}
				err := comm.Generate(&req, &out)
				Expect(err).NotTo(HaveOccurred())
			}()
		}

		time.Sleep(300 * time.Millisecond)
		metricsResp, err := httpClient.Get(metricsUrl)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := string(data)
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqRunningMetricName, 2)))
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.ReqWaitingMetricName, 1)))
	})

	It("should send correct token metrics via gRPC", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}

		_, comm, httpClient, err := startServerHandle(ctx, common.ModeEcho, args, nil)
		Expect(err).NotTo(HaveOccurred())

		maxTokens := uint32(25)
		inputTokens := []uint32{32, 45, 78, 13, 99, 100, 101, 102, 103, 104,
			105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115}
		expectedPromptTokensCnt := len(inputTokens)

		req := pb.GenerateRequest{
			RequestId: "grpc-token-metrics",
			Input: &pb.GenerateRequest_Tokenized{
				Tokenized: &pb.TokenizedInput{
					InputIds: inputTokens,
				},
			},
			SamplingParams: &pb.SamplingParams{
				MaxTokens: &maxTokens,
			},
		}

		out := mockGenerateServer{start: time.Now(), ctx: context.Background()}
		err = comm.Generate(&req, &out)
		Expect(err).NotTo(HaveOccurred())

		time.Sleep(300 * time.Millisecond)
		metricsResp, err := httpClient.Get(metricsUrl)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := string(data)

		// Check prompt tokens and max tokens bucket distributions
		buckets := vllmsim.Build125Buckets(1024)
		for _, boundary := range buckets {
			if boundary <= 20 {
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PromptTokensMetricName, boundary, 0)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ParamMaxTokensMetricName, boundary, 0)))
			} else {
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PromptTokensMetricName, boundary, 1)))
				Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ParamMaxTokensMetricName, boundary, 1)))
			}
		}
		Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.PromptTokensMetricName, math.Inf(1), 1)))
		Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.ParamMaxTokensMetricName, math.Inf(1), 1)))

		Expect(metrics).To(MatchRegexp(fmt.Sprintf(`vllm:prompt_tokens_total{model_name="%s"} %d`, common.TestModelName, expectedPromptTokensCnt)))

		// Check generation tokens - in echo mode we get the same number of tokens back
		// We only check the count since the response length is deterministic in echo mode
		// and skip the bucket distribution.
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.TestModelName, vllmsim.GenerationTokensMetricName+"_count", 1)))
		// request_success_total
		Expect(metrics).To(MatchRegexp(fmt.Sprintf(`vllm:request_success_total{finish_reason="(stop|length)",model_name="%s"} 1`, common.TestModelName)))
	})

	It("should send correct ttft, tpot and inter_token_latency metrics via gRPC", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
			"--time-to-first-token", "100", "--inter-token-latency", "75"}

		_, comm, httpClient, err := startServerHandle(ctx, common.ModeEcho, args, nil)
		Expect(err).NotTo(HaveOccurred())

		maxTokens := uint32(128)
		req := pb.GenerateRequest{
			RequestId: "grpc-latency-metrics",
			Input: &pb.GenerateRequest_Tokenized{
				Tokenized: &pb.TokenizedInput{
					InputIds: []uint32{32, 45, 78, 13},
				},
			},
			SamplingParams: &pb.SamplingParams{
				MaxTokens: &maxTokens,
			},
			Stream: true,
		}

		var reqWg, metricsWg sync.WaitGroup
		metricsWg.Add(1)
		reqWg.Add(1)

		go func() {
			defer reqWg.Done()
			defer GinkgoRecover()
			out := mockGenerateServer{start: time.Now(), ctx: context.Background()}
			err := comm.Generate(&req, &out)
			Expect(err).NotTo(HaveOccurred())
		}()

		go func() {
			defer metricsWg.Done()
			defer GinkgoRecover()

			// Wait for request to complete
			reqWg.Wait()
			time.Sleep(300 * time.Millisecond)
			metricsResp, err := httpClient.Get(metricsUrl)
			Expect(err).NotTo(HaveOccurred())
			Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

			data, err := io.ReadAll(metricsResp.Body)
			Expect(err).NotTo(HaveOccurred())
			metrics := string(data)
			metricsLines := strings.Split(metrics, "\n")

			// Check TTFT buckets
			for _, boundary := range common.TTFTBucketsBoundaries {
				if boundary < 0.1 {
					// buckets up to 0.1 should be empty
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, boundary, 0)))
				} else {
					// buckets higher than 0.1 should contain a single sample
					Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, boundary, 1)))
				}
			}
			Expect(metrics).To(ContainSubstring(getFloatBucketMetricLine(common.TestModelName, vllmsim.TTFTMetricName, math.Inf(1), 1)))

			// Check TPOT and inter-token latency buckets
			for _, metricName := range []string{vllmsim.TPOTMetricName, vllmsim.InterTokenLatencyMetricName} {
				for _, boundary := range common.TPOTBucketsBoundaries {
					if boundary < 0.075 {
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
		}()

		metricsWg.Wait()
	})

	// Skip LoRA test for now as gRPC protobuf doesn't have LoraRequest field yet
	XIt("should send correct lora metrics via gRPC", func() {
		// This test is skipped because the gRPC protobuf definition doesn't include LoraRequest field
		// TODO: Add this test once the protobuf is updated to support LoRA requests
	})

	Context("latency metrics via gRPC", func() {
		DescribeTable("should calculate all latency related metrics correctly for a single gRPC request",
			func(testNamePrefix string, ttft int, prefillTimePerToken int, interTokenLatency int,
				kvcacheTransferLatency int, kvCacheTransferTimePerToken int, doRemotePrefill bool) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
					"--time-to-first-token", strconv.Itoa(ttft),
					"--prefill-time-per-token", strconv.Itoa(prefillTimePerToken),
					"--inter-token-latency", strconv.Itoa(interTokenLatency)}

				if doRemotePrefill {
					args = append(args, "--kv-cache-transfer-latency", strconv.Itoa(kvcacheTransferLatency))
					args = append(args, "--kv-cache-transfer-time-per-token", strconv.Itoa(kvCacheTransferTimePerToken))
				}

				_, comm, httpClient, err := startServerHandle(ctx, common.ModeEcho, args, nil)
				Expect(err).NotTo(HaveOccurred())

				maxTokens := uint32(128)
				numOfTokens := 10
				inputTokens := make([]uint32, numOfTokens)
				for i := range inputTokens {
					inputTokens[i] = uint32(100 + i)
				}

				req := pb.GenerateRequest{
					RequestId: "grpc-latency-single",
					Input: &pb.GenerateRequest_Tokenized{
						Tokenized: &pb.TokenizedInput{
							InputIds: inputTokens,
						},
					},
					SamplingParams: &pb.SamplingParams{
						MaxTokens: &maxTokens,
					},
				}

				out := mockGenerateServer{start: time.Now(), ctx: context.Background()}
				err = comm.Generate(&req, &out)
				Expect(err).NotTo(HaveOccurred())

				checkLatencyMetrics(httpClient, common.TestModelName, numOfTokens, numOfTokens, ttft, prefillTimePerToken,
					interTokenLatency, kvcacheTransferLatency, kvCacheTransferTimePerToken, doRemotePrefill)
			},
			func(testNamePrefix string, ttft int, prefillTimePerToken int, interTokenLatency int,
				kvcacheTransferLatency int, kvCacheTransferTimePerToken int, doRemotePrefill bool) string {
				return fmt.Sprintf("%s: ttft=%d, prefillTimePerToken=%d, interTokenLatency=%d, remotePrefill=%t",
					testNamePrefix, ttft, prefillTimePerToken, interTokenLatency, doRemotePrefill)
			},
			Entry(nil, "gRPC", 100, 10, 50, 0, 0, false),
		)
	})

	It("should send correct kv cache usage metrics via gRPC", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.QwenModelName, "--mode", common.ModeEcho,
			"--time-to-first-token", "2000", "--inter-token-latency", "2000",
			"--max-num-seqs", "3", "--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8"}

		_, comm, httpClient, err := startServerHandle(ctx, common.ModeEcho, args, map[string]string{"POD_IP": "localhost"})
		Expect(err).NotTo(HaveOccurred())

		maxTokens := uint32(128)

		// Send three requests with unique IDs
		for i := range 3 {
			go func(idx int) {
				defer GinkgoRecover()
				req := pb.GenerateRequest{
					RequestId: fmt.Sprintf("grpc-kvcache-%d", idx),
					Input: &pb.GenerateRequest_Tokenized{
						Tokenized: &pb.TokenizedInput{
							InputIds: []uint32{32, 45, 78, 13, 99, 100, 101, 102},
						},
					},
					SamplingParams: &pb.SamplingParams{
						MaxTokens: &maxTokens,
					},
				}
				out := mockGenerateServer{start: time.Now(), ctx: context.Background()}
				err := comm.Generate(&req, &out)
				Expect(err).NotTo(HaveOccurred())
			}(i)
		}

		// Wait for requests to start processing and KV cache to be populated
		time.Sleep(1 * time.Second)
		// Then wait for requests to be running (after TTFT)
		time.Sleep(3 * time.Second)
		metricsResp, err := httpClient.Get(metricsUrl)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := string(data)
		// Expect three running requests and one block in the kv cache (shared by all 3 requests) - usage 1/16=0.0625
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqRunningMetricName, 3)))
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqWaitingMetricName, 0)))
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.KVCacheUsageMetricName, 0.0625)))

		time.Sleep(15 * time.Second)
		metricsResp, err = httpClient.Get(metricsUrl)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err = io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics = string(data)
		// The requests finished running, expect 0 usage
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqRunningMetricName, 0)))
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.ReqWaitingMetricName, 0)))
		Expect(metrics).To(ContainSubstring(getCountMetricLine(common.QwenModelName, vllmsim.KVCacheUsageMetricName, 0)))
	})

	It("should calculate waiting and inference time correctly via gRPC", func() {
		ctx := context.TODO()
		args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
			"--time-to-first-token", "1200", "--max-num-seqs", "1"}

		_, comm, httpClient, err := startServerHandle(ctx, common.ModeEcho, args, nil)
		Expect(err).NotTo(HaveOccurred())

		maxTokens := uint32(128)
		req := pb.GenerateRequest{
			RequestId: "grpc-timing",
			Input: &pb.GenerateRequest_Tokenized{
				Tokenized: &pb.TokenizedInput{
					InputIds: []uint32{32, 45, 78, 13},
				},
			},
			SamplingParams: &pb.SamplingParams{
				MaxTokens: &maxTokens,
			},
		}

		var reqWg sync.WaitGroup
		reqWg.Add(2)

		// Send two requests - only one can run at a time, so one will wait
		for range 2 {
			go func() {
				defer reqWg.Done()
				defer GinkgoRecover()
				out := mockGenerateServer{start: time.Now(), ctx: context.Background()}
				err := comm.Generate(&req, &out)
				Expect(err).NotTo(HaveOccurred())
			}()
		}

		reqWg.Wait()
		time.Sleep(300 * time.Millisecond)
		metricsResp, err := httpClient.Get(metricsUrl)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsResp.StatusCode).To(Equal(http.StatusOK))

		data, err := io.ReadAll(metricsResp.Body)
		Expect(err).NotTo(HaveOccurred())
		metrics := string(data)

		// Check that inference time and queue time buckets are populated correctly
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
