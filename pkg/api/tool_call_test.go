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

package api

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// marshalDeltaToolCall marshals a tool call the way a chat completions chunk carries it.
func marshalDeltaToolCall(tc *ToolCall) map[string]any {
	b, err := json.Marshal(Message{ToolCalls: []ToolCall{*tc}})
	Expect(err).NotTo(HaveOccurred())
	var delta struct {
		ToolCalls []map[string]any `json:"tool_calls"`
	}
	Expect(json.Unmarshal(b, &delta)).To(Succeed())
	Expect(delta.ToolCalls).To(HaveLen(1))
	return delta.ToolCalls[0]
}

var _ = Describe("ToolCall stream deltas", func() {
	It("carry id, type and name only in the first delta of each tool call", func() {
		weather, temperature := "get_weather", "get_temperature"
		calls := []*ToolCall{
			{ID: "chatcmpl-tool-1", Type: "function", Index: 0, Function: FunctionCall{Name: &weather}},
			{ID: "chatcmpl-tool-2", Type: "function", Index: 1, Function: FunctionCall{Name: &temperature}},
		}
		argTokens := []string{`{"city":`, `"Paris"`, `}`}

		for _, tc := range calls {
			for i, token := range argTokens {
				delta := marshalDeltaToolCall(NewToolCallStreamDelta(tc, token, i == 0))
				Expect(delta).To(HaveKeyWithValue("index", BeNumerically("==", tc.Index)))
				Expect(delta["function"]).To(HaveKeyWithValue("arguments", token))
				if i == 0 {
					Expect(delta).To(HaveKeyWithValue("id", tc.ID))
					Expect(delta).To(HaveKeyWithValue("type", "function"))
					Expect(delta["function"]).To(HaveKeyWithValue("name", *tc.Function.Name))
					continue
				}
				Expect(delta).NotTo(HaveKey("id"))
				Expect(delta).NotTo(HaveKey("type"))
				Expect(delta["function"]).NotTo(HaveKey("name"))
			}
		}
	})

	It("leaves an empty id in a tool call that is not a stream delta", func() {
		name := "get_weather"
		tc := marshalDeltaToolCall(&ToolCall{Function: FunctionCall{Name: &name, Arguments: "{}"}})
		Expect(tc).To(HaveKeyWithValue("id", ""))
		Expect(tc).To(HaveKeyWithValue("type", ""))
	})
})
