/*
Copyright 2026 The llm-d-inference-sim Authors.

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

package communication

import (
	"encoding/json"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func eventTypesFromChunk(chunk sseChunk) []string {
	Expect(chunk).NotTo(BeNil())
	named, ok := chunk.(*namedEventChunk)
	Expect(ok).To(BeTrue())
	return named.names
}

func eventPayloadsFromChunk(chunk sseChunk) []json.RawMessage {
	named := chunk.(*namedEventChunk)
	out := make([]json.RawMessage, len(named.data))
	for i, d := range named.data {
		b, err := json.Marshal(d)
		Expect(err).NotTo(HaveOccurred())
		out[i] = b
	}
	return out
}

var _ = Describe("Responses HTTP tool streaming", func() {
	It("builds function_call output items with call_ id derived from fc_ id", func() {
		name := "get_weather"
		item := functionCallOutputItem(api.ToolCall{
			ID: "fc_abc123",
			Function: api.FunctionCall{
				Name:      &name,
				Arguments: `{"city":"Paris"}`,
			},
		}, api.ResponsesStatusCompleted)
		Expect(item.ID).To(Equal("fc_abc123"))
		Expect(item.CallID).To(Equal("call_abc123"))
		Expect(item.Name).To(Equal("get_weather"))
		Expect(item.Arguments).To(Equal(`{"city":"Paris"}`))
		Expect(item.Status).To(Equal(api.ResponsesStatusCompleted))
	})

	It("emits function_call SSE events for argument tokens", func() {
		b := &responsesHTTPRespBuilder{inToolMode: true}
		name := "get_weather"

		first := &api.ToolCall{
			ID:   "fc_abc123",
			Type: "function",
			Function: api.FunctionCall{
				Name:      &name,
				Arguments: `{"city":`,
			},
		}
		chunk1 := b.createToolChunk(first)
		Expect(eventTypesFromChunk(chunk1)).To(Equal([]string{
			api.ResponsesEventOutputItemAdded,
			api.ResponsesEventFunctionCallArgumentsDelta,
		}))
		payloads := eventPayloadsFromChunk(chunk1)
		Expect(string(payloads[0])).To(ContainSubstring(`"type":"function_call"`))
		Expect(string(payloads[0])).To(ContainSubstring(`"name":"get_weather"`))
		Expect(string(payloads[0])).To(ContainSubstring(`"status":"in_progress"`))
		Expect(string(payloads[0])).To(ContainSubstring(`"call_id":"call_abc123"`))

		second := &api.ToolCall{
			ID:   "fc_abc123",
			Type: "function",
			Function: api.FunctionCall{
				Arguments: `"Paris"}`,
			},
		}
		chunk2 := b.createToolChunk(second)
		Expect(eventTypesFromChunk(chunk2)).To(Equal([]string{
			api.ResponsesEventFunctionCallArgumentsDelta,
		}))

		last := b.createLastChunk(nil, "tool_calls", 0)
		Expect(eventTypesFromChunk(last)).To(Equal([]string{
			api.ResponsesEventFunctionCallArgumentsDone,
			api.ResponsesEventOutputItemDone,
		}))
		lastPayloads := eventPayloadsFromChunk(last)
		var argsDone api.ResponsesItemEvent
		Expect(json.Unmarshal(lastPayloads[0], &argsDone)).To(Succeed())
		Expect(argsDone.Name).To(Equal("get_weather"))
		Expect(argsDone.Arguments).To(Equal(`{"city":"Paris"}`))
		Expect(argsDone.ItemID).To(Equal("fc_abc123"))
		Expect(string(lastPayloads[1])).To(ContainSubstring(`"status":"completed"`))
	})
})
