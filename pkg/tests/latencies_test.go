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

			ttftRes, totalTime := sendCompletionRequestForLatencyTest(client, common.TestModelName, testUserMessage, streaming, doRemotePrefill)

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
		Entry(nil, "", "0", "0", "0", "1000", "50", false, false, ">=", int64(1250), int64(1250)),
		// Constant, should use 0 ttft
		Entry(nil, "constant", "0", "0", "0", "500", "50", false, false, "<", int64(100), int64(100)),
		// Constant, remote prefill, should use cache transfer latency (50)
		Entry(nil, "constant", "1000", "0", "50", "500", "50", false, true, "<", int64(100), int64(100)),
		// Constant, remote prefill, streaming and inter token latency 50, should use cache transfer latency (50)
		Entry(nil, "constant", "0", "50", "50", "0", "0", true, true, ">=", int64(50), int64(250)),
		// Per token, should use prefill-overhead and prefill-time-per-token
		Entry(nil, "per-token", "50", "50", "20", "1000", "50", true, false, ">=", int64(1250), int64(1450)),
		// Per token, remote prefill, should use kv-cache-transfer-time-per-token, which is 0 here
		Entry(nil, "per-token", "50", "50", "20", "1000", "50", true, true, "<", int64(100), int64(300)),
	)
})
