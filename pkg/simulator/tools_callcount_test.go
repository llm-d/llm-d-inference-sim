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
	"encoding/json"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// makeTools builds n tools by unmarshalling minimal JSON definitions, the same
// path production code uses. Each tool has one required string parameter so
// generateToolArguments can produce values for the chosen calls.
func makeTools(n int) []api.Tool {
	tools := make([]api.Tool, n)
	for i := range n {
		name := "tool_" + string(rune('a'+i))
		toolJSON := `{
			"type": "function",
			"function": {
				"name": "` + name + `",
				"description": "test tool",
				"parameters": {
					"type": "object",
					"properties": {
						"location": {"type": "string"}
					},
					"required": ["location"]
				}
			}
		}`
		Expect(json.Unmarshal([]byte(toolJSON), &tools[i])).To(Succeed())
	}
	return tools
}

var _ = Describe("createToolCalls call-count distribution", func() {
	var (
		tk         tokenizer.Tokenizer
		idPrefix   = "call_"
		numTools   = 5
		requiredTC = api.NewToolChoiceRequired()
	)

	BeforeEach(func() {
		tk = tokenizer.NewSimpleTokenizer()
	})

	DescribeTable("number of calls follows the ToolCallExtraCallProbability",
		func(probability, expectedCalls int) {
			config := &common.Configuration{
				Model:                        "test",
				ServedModelNames:             []string{"test"},
				ToolCallExtraCallProbability: probability,
			}

			// A single Random is reused across trials so its RNG state advances,
			// mirroring how production shares one Random across requests. With p
			// fixed at an edge the count is constant regardless of the rolls.
			random := common.NewRandom(0, 0)
			for range 200 {
				calls, _, err := createToolCalls(
					makeTools(numTools), requiredTC, config, random, tk, idPrefix)
				Expect(err).NotTo(HaveOccurred())
				Expect(calls).To(HaveLen(expectedCalls))
			}
		},
		// p=0 with tool_choice="required" (minCalls=1): never adds extra calls.
		Entry("probability 0 always produces minCalls", 0, 1),
		// p=100 always rolls "yes": loops until reaching len(availableTools).
		Entry("probability 100 always produces maxCalls", 100, numTools),
	)

	It("with default probability 45, mostly produces minCalls but can reach maxCalls", func() {
		config := &common.Configuration{
			Model:                        "test",
			ServedModelNames:             []string{"test"},
			ToolCallExtraCallProbability: 45,
		}

		const trials = 5000
		counts := make(map[int]int)
		random := common.NewRandom(0, 0)
		for range trials {
			calls, _, err := createToolCalls(
				makeTools(numTools), requiredTC, config, random, tk, idPrefix)
			Expect(err).NotTo(HaveOccurred())
			counts[len(calls)]++
		}

		// minCalls (1) should be the dominant outcome: P(1) = 1 - 0.45 = 55%.
		Expect(counts[1]).To(BeNumerically(">", trials*45/100))
		// The max (5) should be reachable: (0.45)^4 ~= 4.1% of trials.
		Expect(counts[numTools]).To(BeNumerically(">", 0))
		// And every count in between should be possible.
		for i := 2; i < numTools; i++ {
			Expect(counts[i]).To(BeNumerically(">", 0))
		}
	})

	It("with tool_choice auto (minCalls=0), p=0 produces no tool calls", func() {
		autoTC := api.ToolChoice{}
		config := &common.Configuration{
			Model:                        "test",
			ServedModelNames:             []string{"test"},
			ToolCallExtraCallProbability: 0,
		}

		random := common.NewRandom(0, 0)
		for range 50 {
			calls, _, err := createToolCalls(
				makeTools(numTools), autoTC, config, random, tk, idPrefix)
			Expect(err).NotTo(HaveOccurred())
			Expect(calls).To(BeEmpty())
		}
	})
})
