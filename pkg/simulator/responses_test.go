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

package simulator

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func mustTool(name string) api.Tool {
	body := `{"model":"m","input":"x","tools":[{
		"type": "function",
		"name": "` + name + `",
		"description": "test tool ` + name + `",
		"parameters": {
			"type": "object",
			"properties": {"city": {"type": "string"}},
			"required": ["city"]
		}
	}]}`
	var req api.ResponsesRequest
	Expect(json.Unmarshal([]byte(body), &req)).To(Succeed())
	Expect(req.Tools).To(HaveLen(1))
	return req.Tools[0]
}

func newResponsesToolTestCtx(tools []api.Tool, choice api.ToolChoice, input []api.InputItem) *responsesReqCtx {
	cfg := &common.Configuration{
		MaxToolCallIntegerParam:                   100,
		MinToolCallIntegerParam:                   0,
		MaxToolCallNumberParam:                    100,
		MinToolCallNumberParam:                    0,
		MaxToolCallArrayParamLength:               5,
		MinToolCallArrayParamLength:               1,
		ToolCallNotRequiredParamProbability:       50,
		ObjectToolCallNotRequiredParamProbability: 50,
	}
	sim := &SimContext{}
	sim.SetConfig(cfg)
	sim.Random = common.NewRandom(time.Now().UnixNano(), 8080)
	sim.Tokenizer = tokenizer.NewSimpleTokenizer()

	req := &ResponsesRequest{}
	req.Tools = tools
	req.ToolChoice = choice
	req.Input = input
	req.RequestID = "req-test"

	return &responsesReqCtx{
		baseRequestContext: baseRequestContext{sim: sim},
		req:                req,
		toolIDPrefix:       api.ResponsesFunctionCallIDPrefix,
	}
}

var _ = Describe("Responses createToolCalls policy", func() {
	It("emits exactly one tool call when tools are present", func() {
		ctx := newResponsesToolTestCtx(
			[]api.Tool{mustTool("get_weather"), mustTool("get_temperature")},
			api.ToolChoice{},
			[]api.InputItem{&api.InputMessage{
				Type:    "message",
				Role:    api.RoleUser,
				Content: []api.InputContent{{Type: api.ResponsesInputText, Text: "weather?"}},
			}},
		)
		calls, tokens, finish, err := ctx.createToolCalls()
		Expect(err).NotTo(HaveOccurred())
		Expect(calls).To(HaveLen(1))
		Expect(tokens).To(BeNumerically(">", 0))
		Expect(finish).To(Equal(common.ToolsFinishReason))
		Expect(calls[0].ID).To(HavePrefix(api.ResponsesFunctionCallIDPrefix))
		Expect(calls[0].Function.Name).NotTo(BeNil())
		Expect(*calls[0].Function.Name).To(BeElementOf("get_weather", "get_temperature"))
	})

	It("returns nil when function_call_output is already in input", func() {
		ctx := newResponsesToolTestCtx(
			[]api.Tool{mustTool("get_weather")},
			api.ToolChoice{},
			[]api.InputItem{
				&api.InputMessage{
					Type:    "message",
					Role:    api.RoleUser,
					Content: []api.InputContent{{Type: api.ResponsesInputText, Text: "weather?"}},
				},
				&api.InputFunctionCall{
					Type: "function_call", CallID: "call_1", Name: "get_weather", Arguments: `{"city":"Paris"}`,
				},
				&api.InputFunctionCallOutput{Type: "function_call_output", CallID: "call_1", Output: "sunny"},
			},
		)
		calls, tokens, finish, err := ctx.createToolCalls()
		Expect(err).NotTo(HaveOccurred())
		Expect(calls).To(BeNil())
		Expect(tokens).To(Equal(0))
		Expect(finish).To(Equal(""))
	})

	It("returns nil when tool_choice is none", func() {
		ctx := newResponsesToolTestCtx(
			[]api.Tool{mustTool("get_weather")},
			api.NewToolChoiceNone(),
			[]api.InputItem{&api.InputMessage{
				Type:    "message",
				Role:    api.RoleUser,
				Content: []api.InputContent{{Type: api.ResponsesInputText, Text: "weather?"}},
			}},
		)
		calls, _, finish, err := ctx.createToolCalls()
		Expect(err).NotTo(HaveOccurred())
		Expect(calls).To(BeNil())
		Expect(finish).To(Equal(""))
	})

	It("honors forced function tool_choice", func() {
		ctx := newResponsesToolTestCtx(
			[]api.Tool{mustTool("get_weather"), mustTool("get_temperature")},
			api.NewToolChoiceFunction("get_temperature"),
			[]api.InputItem{&api.InputMessage{
				Type:    "message",
				Role:    api.RoleUser,
				Content: []api.InputContent{{Type: api.ResponsesInputText, Text: "need data"}},
			}},
		)
		calls, _, finish, err := ctx.createToolCalls()
		Expect(err).NotTo(HaveOccurred())
		Expect(calls).To(HaveLen(1))
		Expect(finish).To(Equal(common.ToolsFinishReason))
		Expect(*calls[0].Function.Name).To(Equal("get_temperature"))
	})

	It("keeps tools disabled after function_call_output even with a later user message", func() {
		ctx := newResponsesToolTestCtx(
			[]api.Tool{mustTool("get_weather")},
			api.ToolChoice{},
			[]api.InputItem{
				&api.InputMessage{
					Type:    "message",
					Role:    api.RoleUser,
					Content: []api.InputContent{{Type: api.ResponsesInputText, Text: "weather?"}},
				},
				&api.InputFunctionCall{
					Type: "function_call", CallID: "call_1", Name: "get_weather", Arguments: `{"city":"Paris"}`,
				},
				&api.InputFunctionCallOutput{Type: "function_call_output", CallID: "call_1", Output: "sunny"},
				&api.InputMessage{
					Type:    "message",
					Role:    api.RoleUser,
					Content: []api.InputContent{{Type: api.ResponsesInputText, Text: "and the stock price?"}},
				},
			},
		)
		calls, tokens, finish, err := ctx.createToolCalls()
		Expect(err).NotTo(HaveOccurred())
		Expect(calls).To(BeNil())
		Expect(tokens).To(Equal(0))
		Expect(finish).To(Equal(""))
	})

	It("does not force-required for allowed_tools tool_choice", func() {
		var choice api.ToolChoice
		Expect(json.Unmarshal([]byte(`{
			"type": "allowed_tools",
			"allowed_tools": {
				"mode": "auto",
				"tools": [{"type": "function", "name": "get_weather"}]
			}
		}`), &choice)).To(Succeed())
		Expect(choice.OfAllowedTools).NotTo(BeNil())
		Expect(shouldForceRequiredToolChoice(choice)).To(BeFalse())
		Expect(shouldForceRequiredToolChoice(api.ToolChoice{})).To(BeTrue())
		Expect(shouldForceRequiredToolChoice(api.NewToolChoiceFunction("get_weather"))).To(BeFalse())
	})

	It("does not force-required for custom tool_choice", func() {
		var choice api.ToolChoice
		Expect(json.Unmarshal([]byte(`{"type":"custom","custom":{"name":"my_custom"}}`), &choice)).To(Succeed())
		Expect(choice.OfCustomToolChoice).NotTo(BeNil())
		Expect(shouldForceRequiredToolChoice(choice)).To(BeFalse())
	})
})

var _ = Describe("Responses convertInputToMessages", func() {
	It("includes function_call and function_call_output in tokenized prompt", func() {
		tok := tokenizer.NewSimpleTokenizer()
		msgOnly := convertInputToMessages([]api.InputItem{
			&api.InputMessage{
				Type:    "message",
				Role:    api.RoleUser,
				Content: []api.InputContent{{Type: api.ResponsesInputText, Text: "weather?"}},
			},
		})
		withTools := convertInputToMessages([]api.InputItem{
			&api.InputMessage{
				Type:    "message",
				Role:    api.RoleUser,
				Content: []api.InputContent{{Type: api.ResponsesInputText, Text: "weather?"}},
			},
			&api.InputFunctionCall{
				Type: "function_call", ID: "fc_1", CallID: "call_1",
				Name: "get_weather", Arguments: `{"city":"Paris"}`,
			},
			&api.InputFunctionCallOutput{
				Type: "function_call_output", CallID: "call_1", Output: "sunny, 22C",
			},
		})
		Expect(withTools).To(HaveLen(3))
		Expect(withTools[1].Role).To(Equal(api.RoleAssistant))
		Expect(withTools[1].ToolCalls).To(HaveLen(1))
		Expect(withTools[2].Role).To(Equal("tool"))
		Expect(withTools[2].Content.Raw).To(Equal("sunny, 22C"))

		tokensOnly, _, _, err := tok.RenderMessages(msgOnly)
		Expect(err).NotTo(HaveOccurred())
		tokensWith, _, _, err := tok.RenderMessages(withTools)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(tokensWith)).To(BeNumerically(">", len(tokensOnly)))
	})
})

var _ = Describe("Responses tokenizedPromptForEcho", func() {
	It("echoes function_call_output text when it is the last input item", func() {
		ctx := newResponsesToolTestCtx(
			[]api.Tool{mustTool("get_weather")},
			api.ToolChoice{},
			[]api.InputItem{
				&api.InputMessage{
					Type:    "message",
					Role:    api.RoleUser,
					Content: []api.InputContent{{Type: api.ResponsesInputText, Text: "weather?"}},
				},
				&api.InputFunctionCallOutput{
					Type: "function_call_output", CallID: "call_1", Output: "sunny, 22C",
				},
			},
		)
		tokenized, err := ctx.tokenizedPromptForEcho()
		Expect(err).NotTo(HaveOccurred())
		Expect(strings.Join(tokenized.Strings, "")).To(Equal("sunny, 22C"))
	})

	It("echoes function_call name and arguments when it is the last input item", func() {
		ctx := newResponsesToolTestCtx(
			nil,
			api.ToolChoice{},
			[]api.InputItem{
				&api.InputFunctionCall{
					Type: "function_call", CallID: "call_1",
					Name: "get_weather", Arguments: `{"city":"Paris"}`,
				},
			},
		)
		tokenized, err := ctx.tokenizedPromptForEcho()
		Expect(err).NotTo(HaveOccurred())
		Expect(strings.Join(tokenized.Strings, "")).To(Equal(`get_weather({"city":"Paris"})`))
	})
})

var _ = Describe("createSingleToolCall", func() {
	It("never returns more than one call", func() {
		cfg := &common.Configuration{
			MaxToolCallIntegerParam:                   100,
			MinToolCallIntegerParam:                   0,
			MaxToolCallNumberParam:                    100,
			MinToolCallNumberParam:                    0,
			MaxToolCallArrayParamLength:               5,
			MinToolCallArrayParamLength:               1,
			ToolCallNotRequiredParamProbability:       50,
			ObjectToolCallNotRequiredParamProbability: 50,
		}
		random := common.NewRandom(1, 8080)
		tok := tokenizer.NewSimpleTokenizer()
		tools := []api.Tool{mustTool("get_weather"), mustTool("get_temperature")}
		for range 20 {
			calls, _, err := createSingleToolCall(
				tools,
				api.NewToolChoiceRequired(),
				cfg,
				random,
				tok,
				api.ResponsesFunctionCallIDPrefix,
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(calls).To(HaveLen(1))
		}
	})
})
