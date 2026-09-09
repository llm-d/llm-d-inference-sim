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

package simulator

import (
	"fmt"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func milliseconds(m int) time.Duration {
	return time.Duration(m) * time.Millisecond
}

var _ = Describe("Check random latencies", Ordered, func() {
	var config *common.Configuration
	var random *common.Random

	BeforeAll(func() {
		config = &common.Configuration{
			Latencies: common.LatenciesConfig{
				TimeToFirstToken:             milliseconds(2048),
				TimeToFirstTokenStdDev:       milliseconds(2048),
				KVCacheTransferLatency:       milliseconds(2048),
				KVCacheTransferLatencyStdDev: milliseconds(2048),
			},
		}

		random = common.NewRandom(time.Now().UnixNano(), 8080)
	})

	DescribeTable("should calculate inter token latency correctly",
		func(interTokenLatency time.Duration, stddev time.Duration) {
			config.Latencies.InterTokenLatency = interTokenLatency
			config.Latencies.InterTokenLatencyStdDev = stddev
			latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
			interToken := latencyCalculator.GetInterTokenLatency(&InterTokenParams{})
			Expect(interToken).To(BeNumerically(">=", float32(interTokenLatency)*0.3))
			Expect(interToken).To(BeNumerically("<=", float32(interTokenLatency)*1.7))
		},
		func(interTokenLatency time.Duration, stddev time.Duration) string {
			return fmt.Sprintf("interTokenLatency: %d stddev: %d", interTokenLatency, stddev)
		},
		Entry(nil, milliseconds(1000), milliseconds(300)),
		Entry(nil, milliseconds(1000), milliseconds(800)), // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(900)), // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(0)),
	)

	DescribeTable("should calculate total inter token latency correctly",
		func(interTokenLatency time.Duration, stddev time.Duration, numberOfTokens int) {
			config.Latencies.InterTokenLatency = interTokenLatency
			config.Latencies.InterTokenLatencyStdDev = stddev
			config.MaxNumSeqs = 1
			config.Latencies.TimeFactorUnderLoad = 1.0
			latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)

			var latency time.Duration
			for range numberOfTokens - 1 {
				latency += latencyCalculator.GetInterTokenLatency(&InterTokenParams{})
			}

			Expect(latency).To(BeNumerically(">=", time.Duration(float32(interTokenLatency)*0.3*float32(numberOfTokens-1))))
			Expect(latency).To(BeNumerically("<=", time.Duration(float32(interTokenLatency)*1.7*float32(numberOfTokens-1))))
		},
		func(interTokenLatency time.Duration, stddev time.Duration, numberOfTokens int) string {
			return fmt.Sprintf("interTokenLatency: %d stddev: %d, numberOfTokens: %d", interTokenLatency,
				stddev, numberOfTokens)
		},
		Entry(nil, milliseconds(1000), milliseconds(30), 100),
		Entry(nil, milliseconds(1000), milliseconds(800), 20), // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(900), 5),  // invalid std dev, used for testing purposes
		Entry(nil, milliseconds(1000), milliseconds(0), 50),
	)

	DescribeTable("should calculate time to first token correctly",
		func(timeToFirstToken time.Duration, timeToFirstTokenStdDev time.Duration,
			kvCacheLatency time.Duration, kvCacheLatencyStdDev time.Duration, doREmotePrefill bool) {
			config.Latencies.TimeToFirstToken = timeToFirstToken
			config.Latencies.TimeToFirstTokenStdDev = timeToFirstTokenStdDev
			config.Latencies.KVCacheTransferLatency = kvCacheLatency
			config.Latencies.KVCacheTransferLatencyStdDev = kvCacheLatencyStdDev
			latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
			params := TTFTParams{
				PromptTokens:    1,
				DoRemotePrefill: doREmotePrefill,
			}
			timeToFirst := latencyCalculator.GetTimeToFirstToken(&params)
			if doREmotePrefill {
				Expect(timeToFirst).To(BeNumerically(">=", float32(kvCacheLatency)*0.3))
				Expect(timeToFirst).To(BeNumerically("<=", float32(kvCacheLatency)*1.7))
			} else {
				Expect(timeToFirst).To(BeNumerically(">=", float32(timeToFirstToken)*0.3))
				Expect(timeToFirst).To(BeNumerically("<=", float32(timeToFirstToken)*1.7))
			}
		},
		func(timeToFirstToken time.Duration, timeToFirstTokenStdDev time.Duration,
			kvCacheLatency time.Duration, kvCacheLatencyStdDev time.Duration, doREmotePrefill bool) string {
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
		config.Latencies.TimeToFirstToken = timeToFirstToken
		config.Latencies.TimeToFirstTokenStdDev = 0

		config.Latencies.PrefillOverhead = milliseconds(100)
		config.Latencies.PrefillTimePerToken = milliseconds(200)
		config.Latencies.PrefillTimeStdDev = milliseconds(80)

		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
		params := TTFTParams{
			PromptTokens: 128,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params)

		Expect(ttft).To(BeNumerically("==", timeToFirstToken))
	})

	It("when <time-to-first-token> is 0, and <prefill-overhead> is not 0, use <prefill-overhead>", func() {
		config.Latencies.TimeToFirstToken = 0
		config.Latencies.TimeToFirstTokenStdDev = 0

		config.Latencies.PrefillOverhead = milliseconds(100)
		config.Latencies.PrefillTimePerToken = milliseconds(200)
		config.Latencies.PrefillTimeStdDev = milliseconds(80)

		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
		params := TTFTParams{
			PromptTokens: 128,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params)
		Expect(ttft).NotTo(BeNumerically("==", 0))
	})

	DescribeTable("time to first token is against number of prompt tokens with std",
		func(prefillOverhead time.Duration, prefillTimePerToken time.Duration, stdDev time.Duration, nTokens int, nCachedTokens int) {
			config.Latencies.TimeToFirstToken = 0
			config.Latencies.PrefillOverhead = prefillOverhead
			config.Latencies.PrefillTimePerToken = prefillTimePerToken
			config.Latencies.PrefillTimeStdDev = stdDev

			latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
			params := TTFTParams{
				PromptTokens:       nTokens,
				CachedPromptTokens: nCachedTokens,
			}
			ttft := latencyCalculator.GetTimeToFirstToken(&params)

			expectedTTFT := prefillOverhead + prefillTimePerToken*time.Duration(nTokens-nCachedTokens)
			Expect(ttft).To(BeNumerically(">=", float64(expectedTTFT)*0.3))
			Expect(ttft).To(BeNumerically("<=", float64(expectedTTFT)*1.7))
		},
		func(prefillOverhead time.Duration, prefillTimePerToken, stdDev time.Duration, nTokens int, nCachedTokens int) string {
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
		func(prefillOverhead time.Duration, prefillTimePerToken time.Duration, nTokens int, nCachedTokens int) {
			config.Latencies.TimeToFirstToken = 0
			config.Latencies.PrefillOverhead = prefillOverhead
			config.Latencies.PrefillTimePerToken = prefillTimePerToken
			config.Latencies.PrefillTimeStdDev = 0

			latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
			params := TTFTParams{
				PromptTokens:       nTokens,
				CachedPromptTokens: nCachedTokens,
			}
			ttft := latencyCalculator.GetTimeToFirstToken(&params)
			expectedTTFT := prefillOverhead + prefillTimePerToken*time.Duration(nTokens-nCachedTokens)
			Expect(ttft).To(BeNumerically("==", expectedTTFT))
		},
		func(prefillOverhead time.Duration, prefillTimePerToken time.Duration, nTokens int, nCachedTokens int) string {
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
		config.Latencies.KVCacheTransferLatency = milliseconds(200)
		config.Latencies.KVCacheTransferLatencyStdDev = 0

		config.Latencies.KVCacheTransferTimePerToken = milliseconds(100)
		config.Latencies.KVCacheTransferTimeStdDev = 0

		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
		params := TTFTParams{
			PromptTokens:    128,
			DoRemotePrefill: true,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
		Expect(ttft).To(BeNumerically("==", 200))
	})

	It("when <kv-cache-transfer-latency> is 0, and <kv-cache-transfer-overhead> is not 0, use <kv-cache-transfer-overhead>", func() {
		config.Latencies.KVCacheTransferLatency = 0
		config.Latencies.KVCacheTransferLatencyStdDev = 0

		config.Latencies.KVCacheTransferTimePerToken = milliseconds(100)
		config.Latencies.KVCacheTransferTimeStdDev = 0

		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
		params := TTFTParams{
			PromptTokens:    128,
			DoRemotePrefill: true,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
		Expect(ttft).To(BeNumerically("==", 12800))
	})

	DescribeTable("kv cache transfer time against number of prompt tokens",
		func(kvCacheTransTPT time.Duration, stddev time.Duration, nTokens int) {
			config.Latencies.TimeToFirstToken = 0
			config.Latencies.PrefillOverhead = milliseconds(1)
			config.Latencies.KVCacheTransferTimePerToken = kvCacheTransTPT
			config.Latencies.KVCacheTransferTimeStdDev = stddev

			latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
			params := TTFTParams{
				PromptTokens:    nTokens,
				DoRemotePrefill: true,
			}
			ttft := latencyCalculator.GetTimeToFirstToken(&params)

			expectedTTFT := kvCacheTransTPT * time.Duration(nTokens)
			Expect(ttft).To(BeNumerically(">=", float64(expectedTTFT)*0.3))
			Expect(ttft).To(BeNumerically("<=", float64(expectedTTFT)*1.7))

		},
		func(kvCacheTransferTimePerToken time.Duration, stddev time.Duration, nTokens int) string {
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
		config.Latencies.TimeToFirstToken = milliseconds(42)
		config.Latencies.TimeToFirstTokenStdDev = 0
		config.Latencies.TimeFactorUnderLoad = 1.0

		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)
		params := TTFTParams{
			PromptTokens: 128,
			RunningReqs:  100,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
		Expect(ttft).To(BeNumerically("==", 42))
	})

	It("when time-factor-under-load is > 1, but max-num-seqs is 1, the factor will not take effect", func() {
		config.Latencies.TimeToFirstToken = milliseconds(42)
		config.Latencies.TimeToFirstTokenStdDev = 0
		config.Latencies.TimeFactorUnderLoad = 100.0
		config.MaxNumSeqs = 1
		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)

		params := TTFTParams{
			PromptTokens: 128,
			RunningReqs:  1,
		}
		ttft := latencyCalculator.GetTimeToFirstToken(&params).Milliseconds()
		Expect(ttft).To(BeNumerically("==", 42))
	})

	DescribeTable("when time-factor-under-load is > 1, and the sim is fully loaded, the time to first token should be time-factor-under-load * time-to-first-token",
		func(timeFactorUnderLoad float64, maxNumOfReq int) {
			config.Latencies.TimeToFirstToken = milliseconds(42)
			config.Latencies.TimeToFirstTokenStdDev = 0
			config.Latencies.TimeFactorUnderLoad = timeFactorUnderLoad
			config.MaxNumSeqs = maxNumOfReq
			latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)

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
			config.Latencies.TimeToFirstToken = milliseconds(42)
			config.Latencies.TimeToFirstTokenStdDev = 0
			config.Latencies.TimeFactorUnderLoad = timeFactorUnderLoad
			config.MaxNumSeqs = maxNumOfReq
			latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)

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
		config.Latencies.TimeFactorUnderLoad = 1.0
		config.MaxNumSeqs = 11
		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)

		factor := latencyCalculator.getCurrLoadFactor(3)
		Expect(factor).To(BeNumerically("==", 1.0))
	})

	It("when TimeFactorUnderLoad is > 1.0, and sim is fully loaded, calcLoadFactor should give TimeFactorUnderLoad", func() {
		config.Latencies.TimeFactorUnderLoad = 2.0
		config.MaxNumSeqs = 11
		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)

		factor := latencyCalculator.getCurrLoadFactor(11)
		Expect(factor).To(BeNumerically("==", config.Latencies.TimeFactorUnderLoad))

	})

	It("when TimeFactorUnderLoad is > 1.0, and sim is partially loaded, calcLoadFactor should give a value between 1 and TimeFactorUnderLoad", func() {
		config.Latencies.TimeFactorUnderLoad = 2.0
		config.MaxNumSeqs = 11
		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)

		factor := latencyCalculator.getCurrLoadFactor(6)
		Expect(factor).To(BeNumerically(">", 1.0))
		Expect(factor).To(BeNumerically("<", config.Latencies.TimeFactorUnderLoad))
	})

	It("returns zero image generation latency when time-to-generate-image is unset", func() {
		config.Latencies.TimeToGenerateImage = 0
		config.Latencies.TimeToGenerateImageStdDev = 0
		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)

		Expect(latencyCalculator.GetImageGenerationLatency()).To(BeNumerically("==", 0))
	})

	It("calculates image generation latency within the expected range", func() {
		config.Latencies.TimeToGenerateImage = milliseconds(500)
		config.Latencies.TimeToGenerateImageStdDev = milliseconds(100)
		latencyCalculator := newDefaultCalculator(&config.Latencies, config.MaxNumSeqs, random)

		latency := latencyCalculator.GetImageGenerationLatency()
		Expect(latency).To(BeNumerically(">=", float32(milliseconds(500))*0.3))
		Expect(latency).To(BeNumerically("<=", float32(milliseconds(500))*1.7))
	})
})
