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

func milliseconds(m int) common.Duration {
	return common.Duration(time.Duration(m) * time.Millisecond)
}

var _ = Describe("Check random latencies", Ordered, func() {
	var simulator *VllmSimulator

	BeforeAll(func() {
		var err error
		simulator, err = New(klog.Background())
		Expect(err).NotTo(HaveOccurred())

		simulator.config = &common.Configuration{
			TimeToFirstToken:             milliseconds(2048),
			TimeToFirstTokenStdDev:       milliseconds(2048),
			KVCacheTransferLatency:       milliseconds(2048),
			KVCacheTransferLatencyStdDev: milliseconds(2048),
		}

		simulator.metrics.runReqChan = make(chan int64, 100)

		simulator.random = common.NewRandom(time.Now().UnixNano(), 8080)
	})

	DescribeTable("should calculate inter token latency correctly",
		func(interTokenLatency common.Duration, stddev common.Duration) {
			simulator.config.InterTokenLatency = interTokenLatency
			simulator.config.InterTokenLatencyStdDev = stddev
			interToken := simulator.getInterTokenLatency()

			Expect(interToken).To(BeNumerically(">=", int(float32(interTokenLatency)*0.3)))
			Expect(interToken).To(BeNumerically("<=", int(float32(interTokenLatency)*1.7)))
		},
		func(interTokenLatency common.Duration, stddev common.Duration) string {
			return fmt.Sprintf("interTokenLatency: %d stddev: %d", interTokenLatency, stddev)
		},
		Entry(nil, milliseconds(1000), milliseconds(300)),
		Entry(nil, milliseconds(1000), milliseconds(800)), // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(900)), // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(0)),
	)

	DescribeTable("should calculate total inter token latency correctly",
		func(interTokenLatency common.Duration, stddev common.Duration, numberOfTokens int) {
			simulator.config.InterTokenLatency = interTokenLatency
			simulator.config.InterTokenLatencyStdDev = stddev
			simulator.config.MaxNumSeqs = 1
			simulator.config.TimeFactorUnderLoad = 1.0

			var latency time.Duration
			for range numberOfTokens - 1 {
				latency += simulator.getInterTokenLatency()
			}

			Expect(latency).To(BeNumerically(">=", int(float32(interTokenLatency)*0.3*float32(numberOfTokens-1))))
			Expect(latency).To(BeNumerically("<=", int(float32(interTokenLatency)*1.7*float32(numberOfTokens-1))))
		},
		func(interTokenLatency common.Duration, stddev common.Duration, numberOfTokens int) string {
			return fmt.Sprintf("interTokenLatency: %d stddev: %d, numberOfTokens: %d", interTokenLatency,
				stddev, numberOfTokens)
		},
		Entry(nil, milliseconds(1000), milliseconds(30), 100),
		Entry(nil, milliseconds(1000), milliseconds(800), 20), // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(900), 5),  // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(0), 50),
	)

	DescribeTable("should calculate time to first token correctly",
		func(timeToFirstToken common.Duration, timeToFirstTokenStdDev common.Duration,
			kvCacheLatency common.Duration, kvCacheLatencyStdDev common.Duration, doREmotePrefill bool) {
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
		func(timeToFirstToken common.Duration, timeToFirstTokenStdDev common.Duration,
			kvCacheLatency common.Duration, kvCacheLatencyStdDev common.Duration, doREmotePrefill bool) string {
			return fmt.Sprintf("timeToFirstToken: %d stddev: %d kvCacheLatency: %d stddev: %d doREmotePrefill: %t",
				timeToFirstToken, timeToFirstTokenStdDev,
				kvCacheLatency, kvCacheLatencyStdDev, doREmotePrefill)
		},
		Entry(nil, milliseconds(1000), milliseconds(300), milliseconds(1000), milliseconds(200), true),
		Entry(nil, milliseconds(1000), milliseconds(300), milliseconds(1000), milliseconds(200), false),
		Entry(nil, milliseconds(1000), milliseconds(9000), milliseconds(1000), milliseconds(800), true),  // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(8000), milliseconds(1000), milliseconds(900), false), // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(0), milliseconds(1000), milliseconds(0), true),
		Entry(nil, milliseconds(1000), milliseconds(0), milliseconds(1000), milliseconds(0), false),
	)

	It("when <time-to-first-token> is not 0, ignore <prefill-overhead>", func() {
		timeToFirstToken := milliseconds(1000)
		simulator.config.TimeToFirstToken = timeToFirstToken
		simulator.config.TimeToFirstTokenStdDev = 0

		simulator.config.PrefillOverhead = milliseconds(100)
		simulator.config.PrefillTimePerToken = milliseconds(200)
		simulator.config.PrefillTimeStdDev = milliseconds(80)

		ttft := simulator.getWaitTimeToFirstToken(128, 0, false)

		Expect(ttft).To(BeNumerically("==", timeToFirstToken))
	})

	It("when <time-to-first-token> is 0, and <prefill-overhead> is not 0, use <prefill-overhead>", func() {
		simulator.config.TimeToFirstToken = 0
		simulator.config.TimeToFirstTokenStdDev = 0

		simulator.config.PrefillOverhead = milliseconds(100)
		simulator.config.PrefillTimePerToken = milliseconds(200)
		simulator.config.PrefillTimeStdDev = milliseconds(80)

		ttft := simulator.getWaitTimeToFirstToken(128, 0, false)
		Expect(ttft).NotTo(BeNumerically("==", 0))
	})

	DescribeTable("time to first token is against number of prompt tokens with std",
		func(prefillOverhead common.Duration, prefillTimePerToken common.Duration, stdDev common.Duration, nTokens int, nCachedTokens int) {
			simulator.config.TimeToFirstToken = 0
			simulator.config.PrefillOverhead = prefillOverhead
			simulator.config.PrefillTimePerToken = prefillTimePerToken
			simulator.config.PrefillTimeStdDev = stdDev

			ttft := simulator.getWaitTimeToFirstToken(nTokens, nCachedTokens, false)

			expectedTTFT := prefillOverhead + prefillTimePerToken*common.Duration(nTokens-nCachedTokens)
			Expect(ttft).To(BeNumerically(">=", int(float64(expectedTTFT)*0.3)))
			Expect(ttft).To(BeNumerically("<=", int(float64(expectedTTFT)*1.7)))
		},
		func(prefillOverhead common.Duration, prefillTimePerToken, stdDev common.Duration, nTokens int, nCachedTokens int) string {
			return fmt.Sprintf("prefillOverhead: %d, prefillTimePerToken: %d, stdDev: %d, nTokens: %d nCachedTokens: %d",
				prefillOverhead, prefillTimePerToken, stdDev, nTokens, nCachedTokens)
		},
		Entry("single token", milliseconds(100), milliseconds(50), milliseconds(10), 1, 0),
		Entry("single token big std", milliseconds(100), milliseconds(50), milliseconds(70), 1, 0),
		Entry("stddev is 0", milliseconds(100), milliseconds(50), milliseconds(0), 1, 0),
		Entry("medium overhead, 512 tokens", milliseconds(200), milliseconds(1000), milliseconds(150), 512, 0),
		Entry("large overhead, 1024 tokens", milliseconds(2000), milliseconds(3000), milliseconds(800), 1024, 0),
		Entry("very long prompt", milliseconds(150), milliseconds(200), milliseconds(70), 20000, 0),
		Entry("medium overhead, 512 tokens, 256 cached", milliseconds(200), milliseconds(1000), milliseconds(150), 512, 256),
		Entry("large overhead, 1024 tokens, 1008 cached", milliseconds(2000), milliseconds(3000), milliseconds(800), 1024, 1008),
		Entry("very long prompt, 1024 cached", milliseconds(150), milliseconds(200), milliseconds(70), 20000, 1024),
	)

	DescribeTable("time to first token is against number of prompt tokens",
		func(prefillOverhead common.Duration, prefillTimePerToken common.Duration, nTokens int, nCachedTokens int) {
			simulator.config.TimeToFirstToken = 0
			simulator.config.PrefillOverhead = prefillOverhead
			simulator.config.PrefillTimePerToken = prefillTimePerToken
			simulator.config.PrefillTimeStdDev = 0

			ttft := simulator.getWaitTimeToFirstToken(nTokens, nCachedTokens, false)
			expectedTTFT := prefillOverhead + prefillTimePerToken*common.Duration(nTokens-nCachedTokens)
			Expect(ttft).To(Equal(expectedTTFT))
		},
		func(prefillOverhead common.Duration, prefillTimePerToken common.Duration, nTokens int, nCachedTokens int) string {
			return fmt.Sprintf("prefillOverhead: %d, prefillTimePerToken: %d, nTokens: %d nCachedTokens: %d",
				prefillOverhead, prefillTimePerToken, nTokens, nCachedTokens)
		},
		Entry("single token", milliseconds(100), milliseconds(50), 1, 0),
		Entry("medium overhead, 512 tokens", milliseconds(200), milliseconds(1000), 512, 0),
		Entry("large overhead, 1024 tokens", milliseconds(2000), milliseconds(3000), 1024, 0),
		Entry("very long prompt", milliseconds(150), milliseconds(200), 20000, 0),
		Entry("medium overhead, 512 tokens, 256 cached", milliseconds(200), milliseconds(1000), 512, 256),
		Entry("large overhead, 1024 tokens, 128 cached", milliseconds(2000), milliseconds(3000), 1024, 128),
		Entry("very long prompt, 1024 cached", milliseconds(150), milliseconds(200), 20000, 1024),
	)

	It("when <kv-cache-transfer-latency> not 0, ignore <kv-cache-transfer-overhead>", func() {
		simulator.config.KVCacheTransferLatency = milliseconds(200)
		simulator.config.KVCacheTransferLatencyStdDev = 0

		simulator.config.KVCacheTransferTimePerToken = milliseconds(100)
		simulator.config.KVCacheTransferTimeStdDev = 0

		ttft := simulator.getWaitTimeToFirstToken(128, 0, true)
		Expect(ttft).To(BeNumerically("==", 200))
	})

	It("when <kv-cache-transfer-latency> is 0, and <kv-cache-transfer-overhead> is not 0, use <kv-cache-transfer-overhead>", func() {
		simulator.config.KVCacheTransferLatency = 0
		simulator.config.KVCacheTransferLatencyStdDev = 0

		simulator.config.KVCacheTransferTimePerToken = milliseconds(200)
		simulator.config.KVCacheTransferTimeStdDev = 0

		ttft := simulator.getWaitTimeToFirstToken(128, 0, true)
		Expect(ttft).To(BeNumerically("==", 12800))
	})

	DescribeTable("kv cache transfer time against number of prompt tokens",
		func(kvCacheTransTPT common.Duration, stddev common.Duration, nTokens int) {
			simulator.config.TimeToFirstToken = 0
			simulator.config.PrefillOverhead = milliseconds(1)
			simulator.config.KVCacheTransferTimePerToken = kvCacheTransTPT
			simulator.config.KVCacheTransferTimeStdDev = stddev

			ttft := simulator.getWaitTimeToFirstToken(nTokens, 0, true)

			expectedTTFT := kvCacheTransTPT * common.Duration(nTokens)
			Expect(ttft).To(BeNumerically(">=", int(float64(expectedTTFT)*0.3)))
			Expect(ttft).To(BeNumerically("<=", int(float64(expectedTTFT)*1.7)))

		},
		func(kvCacheTransferTimePerToken common.Duration, stddev common.Duration, nTokens int) string {
			return fmt.Sprintf("kvCacheTransferTimePerToken: %d stddev: %d nTokens: %d",
				kvCacheTransferTimePerToken, stddev, nTokens)
		},
		Entry("single token", milliseconds(100), milliseconds(70), 1),
		Entry("stddev is 0", milliseconds(100), milliseconds(0), 1),
		Entry("medium overhead, 512 tokens", milliseconds(200), milliseconds(150), 512),
		Entry("large overhead, 1024 tokens", milliseconds(2000), milliseconds(1800), 1024),
		Entry("very long prompt", milliseconds(150), milliseconds(100), 20000),
	)

	It("when time-factor-under-load is 1, the time to first token should be equal to time-to-first-token", func() {
		simulator.config.TimeToFirstToken = milliseconds(42)
		simulator.config.TimeToFirstTokenStdDev = 0
		simulator.config.TimeFactorUnderLoad = 1.0

		simulator.metrics.runReqChan <- 100

		ttft := simulator.getWaitTimeToFirstToken(128, 0, false)
		Expect(ttft).To(Equal(42))
	})

	It("when time-factor-under-load is > 1, but max-num-seqs is 1, the factor will not take effect", func() {
		simulator.config.TimeToFirstToken = milliseconds(42)
		simulator.config.TimeToFirstTokenStdDev = 0
		simulator.config.TimeFactorUnderLoad = 100.0
		simulator.config.MaxNumSeqs = 1

		for len(simulator.metrics.runReqChan) > 0 {
			<-simulator.metrics.runReqChan
		}

		simulator.metrics.runReqChan <- 1

		ttft := simulator.getWaitTimeToFirstToken(128, 0, false)
		Expect(ttft).To(Equal(42))
	})

	DescribeTable("when time-factor-under-load is > 1, and the sim is fully loaded, the time to first token should be time-factor-under-load * time-to-first-token",
		func(timeFactorUnderLoad float64, maxNumOfReq int) {
			simulator.config.TimeToFirstToken = milliseconds(42)
			simulator.config.TimeToFirstTokenStdDev = 0
			simulator.config.TimeFactorUnderLoad = timeFactorUnderLoad
			simulator.config.MaxNumSeqs = maxNumOfReq
			simulator.metrics.nRunningReqs = int64(maxNumOfReq)

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
			simulator.config.TimeToFirstToken = milliseconds(42)
			simulator.config.TimeToFirstTokenStdDev = 0
			simulator.config.TimeFactorUnderLoad = timeFactorUnderLoad
			simulator.config.MaxNumSeqs = maxNumOfReq
			simulator.metrics.nRunningReqs = int64(nCurrNumOfReq)

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
		simulator.metrics.nRunningReqs = 3

		factor := simulator.getCurrLoadFactor()
		Expect(factor).To(BeNumerically("==", 1.0))
	})

	It("when TimeFactorUnderLoad is > 1.0, and sim is fully loaded, calcLoadFactor should give TimeFactorUnderLoad", func() {
		simulator.config.TimeFactorUnderLoad = 2.0
		simulator.config.MaxNumSeqs = 11
		simulator.metrics.nRunningReqs = 11

		factor := simulator.getCurrLoadFactor()
		Expect(factor).To(BeNumerically("==", simulator.config.TimeFactorUnderLoad))

	})

	It("when TimeFactorUnderLoad is > 1.0, and sim is partially loaded, calcLoadFactor should give a value between 1 and TimeFactorUnderLoad", func() {
		simulator.config.TimeFactorUnderLoad = 2.0
		simulator.config.MaxNumSeqs = 11
		simulator.metrics.nRunningReqs = 6

		factor := simulator.getCurrLoadFactor()
		Expect(factor).To(BeNumerically(">", 1.0))
		Expect(factor).To(BeNumerically("<", simulator.config.TimeFactorUnderLoad))
	})
})
