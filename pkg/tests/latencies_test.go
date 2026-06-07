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
	"net/http"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Check latency calculator", Ordered, func() {
	// Check actual latencies for chat completion responses with various options
	DescribeTable("calculators",
		func(calculator string, ttft string, interToken string, cacheTransfer string, prefillOverhead string,
			prefillPerToken string, streaming bool, doRemotePrefill bool, comparator string, expectedTTFT int64,
			expectedTotal int64) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho, "--latency-calculator", calculator,
				"--time-to-first-token", ttft, "--kv-cache-transfer-latency", cacheTransfer,
				"--prefill-overhead", prefillOverhead, "--prefill-time-per-token", prefillPerToken,
				"--inter-token-latency", interToken,
			}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			ttftRes, totalTime := sendCompletionsRequestForLatencyTest(client, common.TestModelName, testUserMessage, streaming, doRemotePrefill)

			Expect(ttftRes.Milliseconds()).To(BeNumerically(comparator, expectedTTFT))
			Expect(totalTime.Milliseconds()).To(BeNumerically(comparator, expectedTotal))
		},
		func(calculator string, ttft string, interToken string, cacheTransfer string, prefillOverhead string,
			prefillPerToken string, streaming bool, doRemotePrefill bool, comparator string, expectedTTFT int64,
			expectedTotal int64) string {
			return fmt.Sprintf(
				`latency calculator: %s ttft: %s inter token: %s cache transfer: %s prefill overhead: %s per token: %s 
				streaming: %t remote prefill %t`,
				calculator, ttft, interToken, cacheTransfer, prefillOverhead, prefillPerToken, streaming, doRemotePrefill)
		},
		// Default, should use prefill-overhead and prefill-time-per-token, because ttft is 0
		Entry(nil, "", "0ms", "0ms", "0ms", "1s", "50ms", false, false, ">=", int64(1250), int64(1250)),
		// Constant, should use 0 ttft
		Entry(nil, "constant", "0ms", "0ms", "0ms", "500ms", "50ms", false, false, "<", int64(100), int64(100)),
		// Constant, remote prefill, should use cache transfer latency (50)
		Entry(nil, "constant", "1s", "0ms", "50ms", "500ms", "50ms", false, true, "<", int64(100), int64(100)),
		// Constant, remote prefill, streaming and inter token latency 50, should use cache transfer latency (50)
		Entry(nil, "constant", "0ms", "50ms", "50ms", "0ms", "0ms", true, true, ">=", int64(50), int64(250)),
		// Per token, should use prefill-overhead and prefill-time-per-token
		Entry(nil, "per-token", "50ms", "50ms", "20ms", "1s", "50ms", true, false, ">=", int64(1250), int64(1450)),
		// Per token, remote prefill, should use kv-cache-transfer-time-per-token, which is 0 here
		Entry(nil, "per-token", "50ms", "50ms", "20ms", "1s", "50ms", true, true, "<", int64(100), int64(300)),
	)
})

var _ = Describe("Latency admin config", func() {
	var (
		client *http.Client
		ctx    context.Context
	)

	AfterEach(func() {
		if ctx != nil {
			ctx.Done()
		}
	})

	Context("config validation and round-trip", Ordered, func() {
		BeforeAll(func() {
			ctx = context.Background()
			var err error
			client, err = startServerWithArgs(ctx, []string{
				"cmd", "--model", common.TestModelName,
				"--failure-injection-rate", "0",
			})
			Expect(err).ToNot(HaveOccurred())
		})

		It("updates latency fields and reflects them in GET /admin/config", func() {
			resp := postAdminConfig(client, `{"time-to-first-token":"250ms","time-factor-under-load":1.5}`)
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			resp, err := client.Get(adminConfigURL)
			Expect(err).ToNot(HaveOccurred())
			defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body, err := io.ReadAll(resp.Body)
			Expect(err).ToNot(HaveOccurred())
			Expect(string(body)).To(ContainSubstring(`"time-to-first-token":"250ms"`))
			Expect(string(body)).To(ContainSubstring(`"time-factor-under-load":1.5`))
		})

		It("rejects a latency StdDev that exceeds the 30% rule", func() {
			resp := postAdminConfig(client,
				`{"time-to-first-token":"1ms","time-to-first-token-std-dev":"0.5ms"}`)
			defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
		})

		It("accepts a latency-calculator update", func() {
			resp := postAdminConfig(client, `{"latency-calculator":"constant"}`)
			defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
		})

		It("rejects an unknown latency-calculator value", func() {
			resp := postAdminConfig(client, `{"latency-calculator":"bogus"}`)
			defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
		})
	})

	// These tests prove that rebuildLatencyCalculator() actually ran by
	// observing measured latency change after an /admin/config update.
	Context("calculator rebuild on latency update", func() {
		BeforeEach(func() {
			ctx = context.Background()
			var err error
			client, err = startServerWithArgs(ctx, []string{
				"cmd", "--model", common.TestModelName,
				"--mode", common.ModeEcho,
				"--latency-calculator", common.ConstantLatencyCalculator,
				"--time-to-first-token", "0ms",
			})
			Expect(err).ToNot(HaveOccurred())
		})

		It("applies a new time-to-first-token to subsequent requests", func() {
			// Before: TTFT=0ms, so the request should complete quickly.
			ttft, _ := sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, false, false)
			Expect(ttft.Milliseconds()).To(BeNumerically("<", 100))

			// Bump TTFT to 1s. Constant calc returns TTFT verbatim for
			// non-remote requests, so the next request should take >= 1s.
			resp := postAdminConfig(client, `{"time-to-first-token":"1s"}`)
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			ttft, _ = sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, false, false)
			Expect(ttft.Milliseconds()).To(BeNumerically(">=", 1000))
		})

		It("applies a new latency-calculator value to subsequent requests", func() {
			// Before: constant calc with TTFT=0ms, so the request should complete quickly.
			ttft, _ := sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, false, false)
			Expect(ttft.Milliseconds()).To(BeNumerically("<", 100))

			// Switch from constant latency calculator (TTFT 0) to default with a 1s
			// prefill overhead. Default uses prefill-overhead when TTFT is 0.
			resp := postAdminConfig(client,
				`{"latency-calculator":"","prefill-overhead":"1s"}`)
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			ttft, _ = sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, false, false)
			Expect(ttft.Milliseconds()).To(BeNumerically(">=", 1000))
		})

		It("applies a new inter-token-latency to subsequent streaming requests", func() {
			// Before: ITL=0ms, so decode time should be negligible.
			ttft, total := sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, true, false)
			Expect((total - ttft).Milliseconds()).To(BeNumerically("<", 100))

			// Bump ITL to 200ms.
			// Echo of "This is a test." produces > 4 tokens, so totalTime - ttft must
			// clearly exceed 600ms.
			resp := postAdminConfig(client, `{"inter-token-latency":"200ms"}`)
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			ttft, total = sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, true, false)
			Expect((total - ttft).Milliseconds()).To(BeNumerically(">=", 600))
		})
	})

	Context("calculator rebuild for KV transfer on remote prefill", func() {
		BeforeEach(func() {
			ctx = context.Background()
			var err error
			client, err = startServerWithArgs(ctx, []string{
				"cmd", "--model", common.TestModelName,
				"--mode", common.ModeEcho,
				"--latency-calculator", common.ConstantLatencyCalculator,
				"--time-to-first-token", "0ms",
				"--kv-cache-transfer-latency", "0ms",
			})
			Expect(err).ToNot(HaveOccurred())
		})

		It("applies a new kv-cache-transfer-latency to subsequent remote-prefill requests", func() {
			// Before: KV transfer latency=0ms, so the request should complete quickly.
			ttft, _ := sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, false, true)
			Expect(ttft.Milliseconds()).To(BeNumerically("<", 100))

			// Bump KV transfer latency to 1s. Constant calc uses
			// kv-cache-transfer-latency for remote-prefill requests.
			resp := postAdminConfig(client, `{"kv-cache-transfer-latency":"1s"}`)
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			ttft, _ = sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, false, true)
			Expect(ttft.Milliseconds()).To(BeNumerically(">=", 1000))
		})
	})

	// A rejected admin update must not rebuild the calculator: the old
	// values have to keep applying. This guards against a future refactor
	// that swaps the rebuild order with validation.
	Context("rejected update preserves the calculator", func() {
		BeforeEach(func() {
			ctx = context.Background()
			var err error
			client, err = startServerWithArgs(ctx, []string{
				"cmd", "--model", common.TestModelName,
				"--mode", common.ModeEcho,
				"--latency-calculator", common.ConstantLatencyCalculator,
				"--time-to-first-token", "1s",
			})
			Expect(err).ToNot(HaveOccurred())
		})

		It("keeps the original TTFT after a validation-failing update", func() {
			// std-dev > 30% of TTFT trips validation. Update must be rejected
			// AND the in-memory calculator must keep its 1s TTFT.
			resp := postAdminConfig(client,
				`{"time-to-first-token":"50ms","time-to-first-token-std-dev":"40ms"}`)
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

			ttft, _ := sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, false, false)
			Expect(ttft.Milliseconds()).To(BeNumerically(">=", 1000))
		})
	})

	// Sanity check that a non-latency update leaves the calculator alone:
	// even though our optimization skips the rebuild on this path, the
	// observed behavior must still match the previously-configured latency.
	Context("non-latency update leaves the calculator intact", func() {
		BeforeEach(func() {
			ctx = context.Background()
			var err error
			client, err = startServerWithArgs(ctx, []string{
				"cmd", "--model", common.TestModelName,
				"--mode", common.ModeEcho,
				"--latency-calculator", common.ConstantLatencyCalculator,
				"--time-to-first-token", "1s",
				"--failure-injection-rate", "0",
			})
			Expect(err).ToNot(HaveOccurred())
		})

		It("preserves TTFT after a failure-injection-rate update", func() {
			resp := postAdminConfig(client, `{"failure-injection-rate":0}`)
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			ttft, _ := sendCompletionsRequestForLatencyTest(
				client, common.TestModelName, testUserMessage, false, false)
			Expect(ttft.Milliseconds()).To(BeNumerically(">=", 1000))
		})
	})
})
