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

package tests

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

func responsesWeatherTool() responses.ToolUnionParam {
	tool := responses.ToolParamOfFunction(
		"get_weather",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []any{"city"},
		},
		false,
	)
	tool.OfFunction.Description = param.NewOpt("Get current weather for a city")
	return tool
}

func responsesTemperatureTool() responses.ToolUnionParam {
	tool := responses.ToolParamOfFunction(
		"get_temperature",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []any{"city"},
		},
		false,
	)
	tool.OfFunction.Description = param.NewOpt("Get temperature for a city")
	return tool
}

var _ = Describe("Responses API tools", func() {
	It("emits exactly one function_call when tools are present", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, "What is the weather in Paris?")
		params.Tools = []responses.ToolUnionParam{responsesWeatherTool()}

		resp, err := openaiclient.Responses.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Status).To(Equal(responses.ResponseStatusCompleted))
		Expect(resp.Output).To(HaveLen(1))
		Expect(resp.Output[0].Type).To(Equal("function_call"))

		fc := resp.Output[0].AsFunctionCall()
		Expect(fc.Name).To(Equal("get_weather"))
		Expect(fc.Status).To(Equal(responses.ResponseFunctionToolCallStatusCompleted))
		Expect(fc.ID).To(HavePrefix(api.ResponsesFunctionCallIDPrefix))
		Expect(fc.CallID).To(HavePrefix(api.ResponsesCallIDPrefix))
		var args map[string]any
		Expect(json.Unmarshal([]byte(fc.Arguments), &args)).To(Succeed())
		Expect(args).To(HaveKey("city"))
	})

	It("returns a message after function_call_output is present in input", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client),
			option.WithMaxRetries(0))

		params := responses.ResponseNewParams{
			Model: common.TestModelName,
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: responses.ResponseInputParam{
					responses.ResponseInputItemParamOfMessage("What is the weather?", responses.EasyInputMessageRoleUser),
					responses.ResponseInputItemParamOfFunctionCall(`{"city":"Paris"}`, "call_1", "get_weather"),
					responses.ResponseInputItemParamOfFunctionCallOutput("call_1", "sunny, 22C"),
				},
			},
			Tools: []responses.ToolUnionParam{responsesWeatherTool()},
		}

		resp, err := openaiclient.Responses.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Output).NotTo(BeEmpty())
		Expect(resp.Output[0].Type).To(Equal("message"))
		Expect(resp.OutputText()).NotTo(BeEmpty())
	})

	It("returns a message when tool_choice is none", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, "What is the weather?")
		params.Tools = []responses.ToolUnionParam{responsesWeatherTool()}
		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
		}

		resp, err := openaiclient.Responses.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Output).NotTo(BeEmpty())
		Expect(resp.Output[0].Type).To(Equal("message"))
	})

	It("emits exactly one function_call when multiple tools are defined", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, "Need weather and temperature")
		params.Tools = []responses.ToolUnionParam{responsesWeatherTool(), responsesTemperatureTool()}

		resp, err := openaiclient.Responses.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Output).To(HaveLen(1))
		Expect(resp.Output[0].Type).To(Equal("function_call"))
		fc := resp.Output[0].AsFunctionCall()
		Expect(fc.Name).To(BeElementOf("get_weather", "get_temperature"))
	})

	It("honors forced function tool_choice", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, "Need data")
		params.Tools = []responses.ToolUnionParam{responsesWeatherTool(), responsesTemperatureTool()}
		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfFunctionTool: &responses.ToolChoiceFunctionParam{Name: "get_temperature"},
		}

		resp, err := openaiclient.Responses.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Output).To(HaveLen(1))
		Expect(resp.Output[0].Type).To(Equal("function_call"))
		Expect(resp.Output[0].AsFunctionCall().Name).To(Equal("get_temperature"))
	})

	It("rejects invalid tool schemas with 400", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		reqBody := `{
			"model": "` + common.TestModelName + `",
			"input": "hello",
			"tools": [{
				"type": "function",
				"name": "bad_tool",
				"description": "missing parameters type"
			}]
		}`
		resp, err := client.Post("http://localhost/v1/responses", "application/json", strings.NewReader(reqBody))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(ContainSubstring("Tool validation failed"))
	})
})
