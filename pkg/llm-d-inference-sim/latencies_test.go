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
	"fmt"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

var _ = Describe("Check random latencies", Ordered, func() {
	var simulator *VllmSimulator

	BeforeAll(func() {
		var err error
		simulator, err = New(klog.Background())
		Expect(err).NotTo(HaveOccurred())

		simulator.config = &common.Configuration{
			TimeToFirstToken:             2048,
			TimeToFirstTokenStdDev:       2048,
			KVCacheTransferLatency:       2048,
			KVCacheTransferLatencyStdDev: 2048,
		}

		common.InitRandom(time.Now().UnixNano())
	})

	DescribeTable("should calculate inter token latency correctly",
		func(interTokenLatency int, stddev int) {
			simulator.config.InterTokenLatency = interTokenLatency
			simulator.config.InterTokenLatencyStdDev = stddev
			interToken := simulator.getInterTokenLatency()
			Expect(interToken).To(BeNumerically(">=", int(float32(interTokenLatency)*0.3)))
			Expect(interToken).To(BeNumerically("<=", int(float32(interTokenLatency)*1.7)))
		},
		func(interTokenLatency int, stddev int) string {
			return fmt.Sprintf("interTokenLatency: %d stddev: %d", interTokenLatency, stddev)
		},
		Entry(nil, 1000, 300),
		Entry(nil, 1000, 800), // invalid std dev, used for testing purposes
		Entry(nil, 1000, 900), // invalid std dev, used for testing purposes
		Entry(nil, 1000, 0),
	)

	DescribeTable("should calculate total inter token latency correctly",
		func(interTokenLatency int, stddev int, numberOfTokens int) {
			simulator.config.InterTokenLatency = interTokenLatency
			simulator.config.InterTokenLatencyStdDev = stddev
			simulator.config.MaxNumSeqs = 1
			simulator.config.TimeFactorUnderLoad = 1.0

			latency := 0
			for range numberOfTokens - 1 {
				latency += simulator.getInterTokenLatency()
			}

			Expect(latency).To(BeNumerically(">=", int(float32(interTokenLatency)*0.3*float32(numberOfTokens))))
			Expect(latency).To(BeNumerically("<=", int(float32(interTokenLatency)*1.7*float32(numberOfTokens))))
		},
		func(interTokenLatency int, stddev int, numberOfTokens int) string {
			return fmt.Sprintf("interTokenLatency: %d stddev: %d, numberOfTokens: %d", interTokenLatency,
				stddev, numberOfTokens)
		},
		Entry(nil, 1000, 30, 100),
		Entry(nil, 1000, 800, 20), // invalid std dev, used for testing purposes
		Entry(nil, 1000, 900, 5),  // invalid std dev, used for testing purposes
		Entry(nil, 1000, 0, 50),
	)

	DescribeTable("should calculate time to first token correctly",
		func(timeToFirstToken int, timeToFirstTokenStdDev int,
			kvCacheLatency int, kvCacheLatencyStdDev int, doREmotePrefill bool) {
			simulator.config.TimeToFirstToken = timeToFirstToken
			simulator.config.TimeToFirstTokenStdDev = timeToFirstTokenStdDev
			simulator.config.KVCacheTransferLatency = kvCacheLatency
			simulator.config.KVCacheTransferLatencyStdDev = kvCacheLatencyStdDev
			timeToFirst := simulator.getWaitTimeToFirstToken(1, 0, doREmotePrefill)
			if doREmotePrefill {
				Expect(timeToFirst).To(BeNumerically(">=", int(float32(kvCacheLatency)*0.3)))
				Expect(timeToFirst).To(BeNumerically("<=", int(float32(kvCacheLatency)*1.7)))
			} else {
				Expect(timeToFirst).To(BeNumerically(">=", int(float32(timeToFirstToken)*0.3)))
				Expect(timeToFirst).To(BeNumerically("<=", int(float32(timeToFirstToken)*1.7)))
			}
		},
		func(timeToFirstToken int, timeToFirstTokenStdDev int,
			kvCacheLatency int, kvCacheLatencyStdDev int, doREmotePrefill bool) string {
			return fmt.Sprintf("timeToFirstToken: %d stddev: %d kvCacheLatency: %d stddev: %d doREmotePrefill: %t",
				timeToFirstToken, timeToFirstTokenStdDev, kvCacheLatency, kvCacheLatencyStdDev, doREmotePrefill)
		},
		Entry(nil, 10000, 300, 1000, 200, true),
		Entry(nil, 10000, 300, 1000, 200, false),
		Entry(nil, 10000, 9000, 1000, 800, true),  // invalid std dev, used for testing purposes
		Entry(nil, 10000, 8000, 1000, 900, false), // invalid std dev, used for testing purposes
		Entry(nil, 10000, 0, 1000, 0, true),
		Entry(nil, 10000, 0, 1000, 0, false),
	)

	It("when <time-to-first-token> is not 0, ignore <prefill-overhead>", func() {
		timeToFirstToken := 1000
		simulator.config.TimeToFirstToken = timeToFirstToken
		simulator.config.TimeToFirstTokenStdDev = 0

		simulator.config.PrefillOverhead = 100
		simulator.config.PrefillTimePerToken = 200
		simulator.config.PrefillTimeStdDev = 80

		ttft := simulator.getWaitTimeToFirstToken(128, 0, false)

		Expect(ttft).To(BeNumerically("==", timeToFirstToken))
	})

	It("when <time-to-first-token> is 0, and <prefill-overhead> is not 0, use <prefill-overhead>", func() {
		simulator.config.TimeToFirstToken = 0
		simulator.config.TimeToFirstTokenStdDev = 0

		simulator.config.PrefillOverhead = 100
		simulator.config.PrefillTimePerToken = 200
		simulator.config.PrefillTimeStdDev = 80

		ttft := simulator.getWaitTimeToFirstToken(128, 0, false)
		Expect(ttft).NotTo(BeNumerically("==", 0))
	})

	DescribeTable("time to first token is against number of prompt tokens with std",
		func(prefillOverhead int, prefillTimePerToken int, stdDev int, nTokens int, nCachedTokens int) {
			simulator.config.TimeToFirstToken = 0
			simulator.config.PrefillOverhead = prefillOverhead
			simulator.config.PrefillTimePerToken = prefillTimePerToken
			simulator.config.PrefillTimeStdDev = stdDev

			ttft := simulator.getWaitTimeToFirstToken(nTokens, nCachedTokens, false)

			expectedTTFT := prefillOverhead + prefillTimePerToken*(nTokens-nCachedTokens)
			Expect(ttft).To(BeNumerically(">=", int(float64(expectedTTFT)*0.3)))
			Expect(ttft).To(BeNumerically("<=", int(float64(expectedTTFT)*1.7)))
		},
		func(prefillOverhead int, prefillTimePerToken, stdDev int, nTokens int, nCachedTokens int) string {
			return fmt.Sprintf("prefillOverhead: %d, prefillTimePerToken: %d, stdDev: %d, nTokens: %d nCachedTokens: %d",
				prefillOverhead, prefillTimePerToken, stdDev, nTokens, nCachedTokens)
		},
		Entry("single token", 100, 50, 10, 1, 0),
		Entry("single token big std", 100, 50, 70, 1, 0),
		Entry("stddev is 0", 100, 50, 0, 1, 0),
		Entry("medium overhead, 512 tokens", 200, 1000, 150, 512, 0),
		Entry("large overhead, 1024 tokens", 2000, 3000, 800, 1024, 0),
		Entry("very long prompt", 150, 200, 70, 20000, 0),
		Entry("medium overhead, 512 tokens, 256 cached", 200, 1000, 150, 512, 256),
		Entry("large overhead, 1024 tokens, 1008 cached", 2000, 3000, 800, 1024, 1008),
		Entry("very long prompt, 1024 cached", 150, 200, 70, 20000, 1024),
	)

	DescribeTable("time to first token is against number of prompt tokens",
		func(prefillOverhead int, prefillTimePerToken int, nTokens int, nCachedTokens int) {
			simulator.config.TimeToFirstToken = 0
			simulator.config.PrefillOverhead = prefillOverhead
			simulator.config.PrefillTimePerToken = prefillTimePerToken
			simulator.config.PrefillTimeStdDev = 0

			ttft := simulator.getWaitTimeToFirstToken(nTokens, nCachedTokens, false)
			expectedTTFT := prefillOverhead + prefillTimePerToken*(nTokens-nCachedTokens)
			Expect(ttft).To(Equal(expectedTTFT))
		},
		func(prefillOverhead int, prefillTimePerToken, nTokens int, nCachedTokens int) string {
			return fmt.Sprintf("prefillOverhead: %d, prefillTimePerToken: %d, nTokens: %d nCachedTokens: %d",
				prefillOverhead, prefillTimePerToken, nTokens, nCachedTokens)
		},
		Entry("single token", 100, 50, 1, 0),
		Entry("medium overhead, 512 tokens", 200, 1000, 512, 0),
		Entry("large overhead, 1024 tokens", 2000, 3000, 1024, 0),
		Entry("very long prompt", 150, 200, 20000, 0),
		Entry("medium overhead, 512 tokens, 256 cached", 200, 1000, 512, 256),
		Entry("large overhead, 1024 tokens, 128 cached", 2000, 3000, 1024, 128),
		Entry("very long prompt, 1024 cached", 150, 200, 20000, 1024),
	)

	It("when <kv-cache-transfer-latency> not 0, ignore <kv-cache-transfer-overhead>", func() {
		simulator.config.KVCacheTransferLatency = 200
		simulator.config.KVCacheTransferLatencyStdDev = 0

		simulator.config.KVCacheTransferTimePerToken = 100
		simulator.config.KVCacheTransferTimeStdDev = 0

		ttft := simulator.getWaitTimeToFirstToken(128, 0, true)
		Expect(ttft).To(BeNumerically("==", 200))
	})

	It("when <kv-cache-transfer-latency> is 0, and <kv-cache-transfer-overhead> is not 0, use <kv-cache-transfer-overhead>", func() {
		simulator.config.KVCacheTransferLatency = 0
		simulator.config.KVCacheTransferLatencyStdDev = 0

		simulator.config.KVCacheTransferTimePerToken = 100
		simulator.config.KVCacheTransferTimeStdDev = 0

		ttft := simulator.getWaitTimeToFirstToken(128, 0, true)
		Expect(ttft).To(BeNumerically("==", 12800))
	})

	DescribeTable("kv cache transfer time against number of prompt tokens",
		func(kvCacheTransTPT int, stddev int, nTokens int) {
			simulator.config.TimeToFirstToken = 0
			simulator.config.PrefillOverhead = 1
			simulator.config.KVCacheTransferTimePerToken = kvCacheTransTPT
			simulator.config.KVCacheTransferTimeStdDev = stddev

			ttft := simulator.getWaitTimeToFirstToken(nTokens, 0, true)

			expectedTTFT := kvCacheTransTPT * nTokens
			Expect(ttft).To(BeNumerically(">=", int(float64(expectedTTFT)*0.3)))
			Expect(ttft).To(BeNumerically("<=", int(float64(expectedTTFT)*1.7)))

		},
		func(kvCacheTransferTimePerToken int, stddev int, nTokens int) string {
			return fmt.Sprintf("kvCacheTransferTimePerToken: %d stddev: %d nTokens: %d",
				kvCacheTransferTimePerToken, stddev, nTokens)
		},
		Entry("single token", 100, 70, 1),
		Entry("stddev is 0", 100, 0, 1),
		Entry("medium overhead, 512 tokens", 200, 150, 512),
		Entry("large overhead, 1024 tokens", 2000, 1800, 1024),
		Entry("very long prompt", 150, 100, 20000),
	)

	It("when time-factor-under-load is 1, the time to first token should be equal to time-to-first-token", func() {
		simulator.config.TimeToFirstToken = 42
		simulator.config.TimeToFirstTokenStdDev = 0
		simulator.config.TimeFactorUnderLoad = 1.0

		simulator.runReqChan <- 100

		ttft := simulator.getWaitTimeToFirstToken(128, 0, false)
		Expect(ttft).To(Equal(42))
	})

	It("when time-factor-under-load is > 1, but max-num-seqs is 1, the factor will not take effect", func() {
		simulator.config.TimeToFirstToken = 42
		simulator.config.TimeToFirstTokenStdDev = 0
		simulator.config.TimeFactorUnderLoad = 100.0
		simulator.config.MaxNumSeqs = 1

		for len(simulator.runReqChan) > 0 {
			<-simulator.runReqChan
		}

		simulator.runReqChan <- 1

		ttft := simulator.getWaitTimeToFirstToken(128, 0, false)
		Expect(ttft).To(Equal(42))
	})

	DescribeTable("when time-factor-under-load is > 1, and the sim is fully loaded, the time to first token should be time-factor-under-load * time-to-first-token",
		func(timeFactorUnderLoad float64, maxNumOfReq int) {
			simulator.config.TimeToFirstToken = 42
			simulator.config.TimeToFirstTokenStdDev = 0
			simulator.config.TimeFactorUnderLoad = timeFactorUnderLoad
			simulator.config.MaxNumSeqs = maxNumOfReq
			simulator.nRunningReqs = int64(maxNumOfReq)

			ttft := simulator.getWaitTimeToFirstToken(128, 0, false)
			Expect(ttft).To(Equal(int(float64(42) * timeFactorUnderLoad)))

		},
		func(timeFactorUnderLoad float64, maxNumOfReq int64) string {
			return fmt.Sprintf("timeFactorUnderLoad: %f maxNumOfReq: %d",
				timeFactorUnderLoad, maxNumOfReq)
		},

		Entry("factor: 1.5", 1.5, 70),
		Entry("factor: 2.0", 2.0, 2),
		Entry("factor: 100.0", 100.0, 150),
		Entry("factor: 20000.0", 20000.0, 310),
	)

	DescribeTable("when time-factor-under-load is > 1, and the sim is partially loaded, the time to first token should be linear interpolation between time-to-first-token and time-factor-under-load * time-to-first-token",
		func(timeFactorUnderLoad float64, maxNumOfReq int, nCurrNumOfReq int) {
			simulator.config.TimeToFirstToken = 42
			simulator.config.TimeToFirstTokenStdDev = 0
			simulator.config.TimeFactorUnderLoad = timeFactorUnderLoad
			simulator.config.MaxNumSeqs = maxNumOfReq
			simulator.nRunningReqs = int64(nCurrNumOfReq)

			ttft := simulator.getWaitTimeToFirstToken(128, 0, false)
			max := timeFactorUnderLoad * float64(42)
			Expect(ttft).To(BeNumerically(">=", 42))
			Expect(ttft).To(BeNumerically("<=", max))

		},
		func(timeFactorUnderLoad float64, maxNumOfReq int, nCurrNumOfReq int) string {
			return fmt.Sprintf("timeFactorUnderLoad: %f maxNumOfReq: %d nCurrNumOfReq: %d",
				timeFactorUnderLoad, maxNumOfReq, nCurrNumOfReq)
		},

		Entry("factor: 1.5", 1.5, 70, 35),
		Entry("factor: 2.0", 2.0, 2, 1),
		Entry("factor: 100.0", 100.0, 150, 75),
		Entry("factor: 20000.0", 20000.0, 310, 155),
	)

	It("when TimeFactorUnderLoad is 1.0, calcLoadFactor should give 1", func() {
		simulator.config.TimeFactorUnderLoad = 1.0
		simulator.config.MaxNumSeqs = 11
		simulator.nRunningReqs = 3

		factor := simulator.getCurrLoadFactor()
		Expect(factor).To(BeNumerically("==", 1.0))
	})

	It("when TimeFactorUnderLoad is > 1.0, and sim is fully loaded, calcLoadFactor should give TimeFactorUnderLoad", func() {
		simulator.config.TimeFactorUnderLoad = 2.0
		simulator.config.MaxNumSeqs = 11
		simulator.nRunningReqs = 11

		factor := simulator.getCurrLoadFactor()
		Expect(factor).To(BeNumerically("==", simulator.config.TimeFactorUnderLoad))

	})

	It("when TimeFactorUnderLoad is > 1.0, and sim is partially loaded, calcLoadFactor should give a value between 1 and TimeFactorUnderLoad", func() {
		simulator.config.TimeFactorUnderLoad = 2.0
		simulator.config.MaxNumSeqs = 11
		simulator.nRunningReqs = 6

		factor := simulator.getCurrLoadFactor()
		Expect(factor).To(BeNumerically(">", 1.0))
		Expect(factor).To(BeNumerically("<", simulator.config.TimeFactorUnderLoad))
	})
})
