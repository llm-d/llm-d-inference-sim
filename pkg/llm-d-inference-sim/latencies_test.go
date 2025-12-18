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
)

var _ = Describe("Check random latencies", Ordered, func() {
	var config *common.Configuration
	var random *common.Random

	BeforeAll(func() {
		config = &common.Configuration{
			TimeToFirstToken:             2048,
			TimeToFirstTokenStdDev:       2048,
			KVCacheTransferLatency:       2048,
			KVCacheTransferLatencyStdDev: 2048,
		}

		// latencyCalculator.metrics.runReqChan = make(chan int64, 100)

		random = common.NewRandom(time.Now().UnixNano(), 8080)
	})

	DescribeTable("should calculate inter token latency correctly",
		func(interTokenLatency int, stddev int) {
			config.InterTokenLatency = interTokenLatency
			config.InterTokenLatencyStdDev = stddev
			latencyCalculator := newDefaultCalculator(config, random)
			interToken := latencyCalculator.GetInterTokenLatency(&InterTokenParams{}).Milliseconds()
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
			config.InterTokenLatency = interTokenLatency
			config.InterTokenLatencyStdDev = stddev
			config.MaxNumSeqs = 1
			config.TimeFactorUnderLoad = 1.0
			latencyCalculator := newDefaultCalculator(config, random)

			var latency int64
			for range numberOfTokens - 1 {
				latency += latencyCalculator.GetInterTokenLatency(&InterTokenParams{}).Milliseconds()
			}

			Expect(latency).To(BeNumerically(">=", int(float32(interTokenLatency)*0.3*float32(numberOfTokens-1))))
			Expect(latency).To(BeNumerically("<=", int(float32(interTokenLatency)*1.7*float32(numberOfTokens-1))))
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
			config.TimeToFirstToken = timeToFirstToken
			config.TimeToFirstTokenStdDev = timeToFirstTokenStdDev
			config.KVCacheTransferLatency = kvCacheLatency
			config.KVCacheTransferLatencyStdDev = kvCacheLatencyStdDev
			latencyCalculator := newDefaultCalculator(config, random)
			params := TTFTParams{
				PromptTokens:    1,
				DoRemotePrefill: doREmotePrefill,
			}
			timeToFirst := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
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
		config.TimeToFirstToken = timeToFirstToken
		config.TimeToFirstTokenStdDev = 0

		config.PrefillOverhead = 100
		config.PrefillTimePerToken = 200
		config.PrefillTimeStdDev = 80

		latencyCalculator := newDefaultCalculator(config, random)
		params := TTFTParams{
			PromptTokens: 128,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()

		Expect(ttft).To(BeNumerically("==", timeToFirstToken))
	})

	It("when <time-to-first-token> is 0, and <prefill-overhead> is not 0, use <prefill-overhead>", func() {
		config.TimeToFirstToken = 0
		config.TimeToFirstTokenStdDev = 0

		config.PrefillOverhead = 100
		config.PrefillTimePerToken = 200
		config.PrefillTimeStdDev = 80

		latencyCalculator := newDefaultCalculator(config, random)
		params := TTFTParams{
			PromptTokens: 128,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params)
		Expect(ttft).NotTo(BeNumerically("==", 0))
	})

	DescribeTable("time to first token is against number of prompt tokens with std",
		func(prefillOverhead int, prefillTimePerToken int, stdDev int, nTokens int, nCachedTokens int) {
			config.TimeToFirstToken = 0
			config.PrefillOverhead = prefillOverhead
			config.PrefillTimePerToken = prefillTimePerToken
			config.PrefillTimeStdDev = stdDev

			latencyCalculator := newDefaultCalculator(config, random)
			params := TTFTParams{
				PromptTokens:       nTokens,
				CachedPromptTokens: nCachedTokens,
			}
			ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()

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
			config.TimeToFirstToken = 0
			config.PrefillOverhead = prefillOverhead
			config.PrefillTimePerToken = prefillTimePerToken
			config.PrefillTimeStdDev = 0

			latencyCalculator := newDefaultCalculator(config, random)
			params := TTFTParams{
				PromptTokens:       nTokens,
				CachedPromptTokens: nCachedTokens,
			}
			ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
			expectedTTFT := prefillOverhead + prefillTimePerToken*(nTokens-nCachedTokens)
			Expect(ttft).To(BeNumerically("==", expectedTTFT))
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
		config.KVCacheTransferLatency = 200
		config.KVCacheTransferLatencyStdDev = 0

		config.KVCacheTransferTimePerToken = 100
		config.KVCacheTransferTimeStdDev = 0

		latencyCalculator := newDefaultCalculator(config, random)
		params := TTFTParams{
			PromptTokens:    128,
			DoRemotePrefill: true,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
		Expect(ttft).To(BeNumerically("==", 200))
	})

	It("when <kv-cache-transfer-latency> is 0, and <kv-cache-transfer-overhead> is not 0, use <kv-cache-transfer-overhead>", func() {
		config.KVCacheTransferLatency = 0
		config.KVCacheTransferLatencyStdDev = 0

		config.KVCacheTransferTimePerToken = 100
		config.KVCacheTransferTimeStdDev = 0

		latencyCalculator := newDefaultCalculator(config, random)
		params := TTFTParams{
			PromptTokens:    128,
			DoRemotePrefill: true,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
		Expect(ttft).To(BeNumerically("==", 12800))
	})

	DescribeTable("kv cache transfer time against number of prompt tokens",
		func(kvCacheTransTPT int, stddev int, nTokens int) {
			config.TimeToFirstToken = 0
			config.PrefillOverhead = 1
			config.KVCacheTransferTimePerToken = kvCacheTransTPT
			config.KVCacheTransferTimeStdDev = stddev

			latencyCalculator := newDefaultCalculator(config, random)
			params := TTFTParams{
				PromptTokens:    nTokens,
				DoRemotePrefill: true,
			}
			ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()

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
		config.TimeToFirstToken = 42
		config.TimeToFirstTokenStdDev = 0
		config.TimeFactorUnderLoad = 1.0

		latencyCalculator := newDefaultCalculator(config, random)
		params := TTFTParams{
			PromptTokens: 128,
			RunningReqs:  100,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
		Expect(ttft).To(BeNumerically("==", 42))
	})

	It("when time-factor-under-load is > 1, but max-num-seqs is 1, the factor will not take effect", func() {
		config.TimeToFirstToken = 42
		config.TimeToFirstTokenStdDev = 0
		config.TimeFactorUnderLoad = 100.0
		config.MaxNumSeqs = 1
		latencyCalculator := newDefaultCalculator(config, random)

		params := TTFTParams{
			PromptTokens: 128,
			RunningReqs:  1,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
		Expect(ttft).To(BeNumerically("==", 42))
	})

	DescribeTable("when time-factor-under-load is > 1, and the sim is fully loaded, the time to first token should be time-factor-under-load * time-to-first-token",
		func(timeFactorUnderLoad float64, maxNumOfReq int) {
			config.TimeToFirstToken = 42
			config.TimeToFirstTokenStdDev = 0
			config.TimeFactorUnderLoad = timeFactorUnderLoad
			config.MaxNumSeqs = maxNumOfReq
			latencyCalculator := newDefaultCalculator(config, random)

			params := TTFTParams{
				PromptTokens: 128,
				RunningReqs:  int64(maxNumOfReq),
			}
			ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
			Expect(ttft).To(Equal(int64(float64(42) * timeFactorUnderLoad)))

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
			config.TimeToFirstToken = 42
			config.TimeToFirstTokenStdDev = 0
			config.TimeFactorUnderLoad = timeFactorUnderLoad
			config.MaxNumSeqs = maxNumOfReq
			latencyCalculator := newDefaultCalculator(config, random)

			params := TTFTParams{
				PromptTokens: 128,
				RunningReqs:  int64(nCurrNumOfReq),
			}
			ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
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
		config.TimeFactorUnderLoad = 1.0
		config.MaxNumSeqs = 11
		latencyCalculator := newDefaultCalculator(config, random)

		factor := latencyCalculator.getCurrLoadFactor(3)
		Expect(factor).To(BeNumerically("==", 1.0))
	})

	It("when TimeFactorUnderLoad is > 1.0, and sim is fully loaded, calcLoadFactor should give TimeFactorUnderLoad", func() {
		config.TimeFactorUnderLoad = 2.0
		config.MaxNumSeqs = 11
		latencyCalculator := newDefaultCalculator(config, random)

		factor := latencyCalculator.getCurrLoadFactor(11)
		Expect(factor).To(BeNumerically("==", config.TimeFactorUnderLoad))

	})

	It("when TimeFactorUnderLoad is > 1.0, and sim is partially loaded, calcLoadFactor should give a value between 1 and TimeFactorUnderLoad", func() {
		config.TimeFactorUnderLoad = 2.0
		config.MaxNumSeqs = 11
		latencyCalculator := newDefaultCalculator(config, random)

		factor := latencyCalculator.getCurrLoadFactor(6)
		Expect(factor).To(BeNumerically(">", 1.0))
		Expect(factor).To(BeNumerically("<", config.TimeFactorUnderLoad))
	})
})
