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

	It("streams function_call events when tools are present", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, "What is the weather in Paris?")
		params.Tools = []responses.ToolUnionParam{responsesWeatherTool()}

		stream := openaiclient.Responses.NewStreaming(ctx, params)
		defer func() {
			Expect(stream.Close()).NotTo(HaveOccurred())
		}()

		var eventTypes []string
		var argDeltas []string
		var addedName string
		var doneArgs string
		var doneName string

		for stream.Next() {
			event := stream.Current()
			eventTypes = append(eventTypes, event.Type)
			switch event.Type {
			case api.ResponsesEventOutputItemAdded:
				added := event.AsResponseOutputItemAdded()
				Expect(added.Item.Type).To(Equal("function_call"))
				fc := added.Item.AsFunctionCall()
				addedName = fc.Name
				Expect(fc.ID).To(HavePrefix(api.ResponsesFunctionCallIDPrefix))
				Expect(fc.CallID).To(HavePrefix(api.ResponsesCallIDPrefix))
			case api.ResponsesEventFunctionCallArgumentsDelta:
				delta := event.AsResponseFunctionCallArgumentsDelta()
				argDeltas = append(argDeltas, delta.Delta)
				Expect(delta.ItemID).To(HavePrefix(api.ResponsesFunctionCallIDPrefix))
			case api.ResponsesEventFunctionCallArgumentsDone:
				done := event.AsResponseFunctionCallArgumentsDone()
				doneArgs = done.Arguments
				doneName = done.Name
				Expect(done.ItemID).To(HavePrefix(api.ResponsesFunctionCallIDPrefix))
			case api.ResponsesEventCompleted:
				completed := event.AsResponseCompleted()
				Expect(string(completed.Response.Status)).To(Equal(api.ResponsesStatusCompleted))
				Expect(completed.Response.Output).To(HaveLen(1))
				Expect(completed.Response.Output[0].Type).To(Equal("function_call"))
				fc := completed.Response.Output[0].AsFunctionCall()
				Expect(fc.Name).To(Equal(addedName))
				Expect(fc.Arguments).To(Equal(doneArgs))
			}
		}
		Expect(stream.Err()).NotTo(HaveOccurred())

		Expect(eventTypes[0]).To(Equal(api.ResponsesEventCreated))
		Expect(eventTypes[1]).To(Equal(api.ResponsesEventInProgress))
		Expect(eventTypes).To(ContainElement(api.ResponsesEventOutputItemAdded))
		Expect(eventTypes).To(ContainElement(api.ResponsesEventFunctionCallArgumentsDelta))
		Expect(eventTypes).To(ContainElement(api.ResponsesEventFunctionCallArgumentsDone))
		Expect(eventTypes).To(ContainElement(api.ResponsesEventOutputItemDone))
		Expect(eventTypes[len(eventTypes)-1]).To(Equal(api.ResponsesEventCompleted))
		Expect(eventTypes).NotTo(ContainElement(api.ResponsesEventTextDelta))
		Expect(eventTypes).NotTo(ContainElement(api.ResponsesEventContentPartAdded))

		Expect(addedName).To(Equal("get_weather"))
		Expect(doneName).To(Equal("get_weather"))
		Expect(doneArgs).To(Equal(strings.Join(argDeltas, "")))
		var args map[string]any
		Expect(json.Unmarshal([]byte(doneArgs), &args)).To(Succeed())
		Expect(args).To(HaveKey("city"))

		functionCallAddedCount := 0
		for _, t := range eventTypes {
			if t == api.ResponsesEventOutputItemAdded {
				functionCallAddedCount++
			}
		}
		Expect(functionCallAddedCount).To(Equal(1))
	})

	It("streams a message after function_call_output on re-entry", func() {
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

		stream := openaiclient.Responses.NewStreaming(ctx, params)
		defer func() {
			Expect(stream.Close()).NotTo(HaveOccurred())
		}()

		var eventTypes []string
		for stream.Next() {
			event := stream.Current()
			eventTypes = append(eventTypes, event.Type)
			if event.Type == api.ResponsesEventCompleted {
				completed := event.AsResponseCompleted()
				Expect(completed.Response.Output).NotTo(BeEmpty())
				Expect(completed.Response.Output[0].Type).To(Equal("message"))
			}
		}
		Expect(stream.Err()).NotTo(HaveOccurred())
		Expect(eventTypes).To(ContainElement(api.ResponsesEventTextDelta))
		Expect(eventTypes).NotTo(ContainElement(api.ResponsesEventFunctionCallArgumentsDelta))
		Expect(eventTypes[len(eventTypes)-1]).To(Equal(api.ResponsesEventCompleted))
	})

	It("streams forced function tool_choice name on output_item.added", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, "Need data")
		params.Tools = []responses.ToolUnionParam{responsesWeatherTool(), responsesTemperatureTool()}
		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfFunctionTool: &responses.ToolChoiceFunctionParam{Name: "get_temperature"},
		}

		stream := openaiclient.Responses.NewStreaming(ctx, params)
		defer func() {
			Expect(stream.Close()).NotTo(HaveOccurred())
		}()

		var addedName string
		for stream.Next() {
			event := stream.Current()
			switch event.Type {
			case api.ResponsesEventOutputItemAdded:
				fc := event.AsResponseOutputItemAdded().Item.AsFunctionCall()
				addedName = fc.Name
			case api.ResponsesEventCompleted:
				completed := event.AsResponseCompleted()
				Expect(completed.Response.Output).To(HaveLen(1))
				Expect(completed.Response.Output[0].AsFunctionCall().Name).To(Equal("get_temperature"))
			}
		}
		Expect(stream.Err()).NotTo(HaveOccurred())
		Expect(addedName).To(Equal("get_temperature"))
	})

	It("errors when forced tool_choice names a missing function", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, postErr := client.Post("http://localhost/v1/responses", "application/json", strings.NewReader(`{
			"model": "`+common.TestModelName+`",
			"input": "Need data",
			"tools": [{
				"type": "function",
				"name": "get_weather",
				"description": "Get weather",
				"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
			}],
			"tool_choice": {"type": "function", "name": "does_not_exist"}
		}`))
		Expect(postErr).NotTo(HaveOccurred())
		defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(ContainSubstring("not found in the tools list"))
	})

	It("echoes function_call_output in echo mode", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeEcho)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client),
			option.WithMaxRetries(0))

		const toolOutput = "sunny, 22C"
		params := responses.ResponseNewParams{
			Model: common.TestModelName,
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: responses.ResponseInputParam{
					responses.ResponseInputItemParamOfMessage("What is the weather?", responses.EasyInputMessageRoleUser),
					responses.ResponseInputItemParamOfFunctionCall(`{"city":"Paris"}`, "call_1", "get_weather"),
					responses.ResponseInputItemParamOfFunctionCallOutput("call_1", toolOutput),
				},
			},
			Tools: []responses.ToolUnionParam{responsesWeatherTool()},
		}

		resp, err := openaiclient.Responses.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Output).NotTo(BeEmpty())
		Expect(resp.Output[0].Type).To(Equal("message"))
		Expect(resp.OutputText()).To(Equal(toolOutput))
	})

	It("keeps tools disabled on a later user turn after function_call_output", func() {
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
					responses.ResponseInputItemParamOfMessage("What is the stock price of ACME?", responses.EasyInputMessageRoleUser),
				},
			},
			Tools: []responses.ToolUnionParam{responsesWeatherTool()},
		}

		resp, err := openaiclient.Responses.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Output).NotTo(BeEmpty())
		Expect(resp.Output[0].Type).To(Equal("message"))
	})

	It("counts function_call input items toward usage.input_tokens", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client),
			option.WithMaxRetries(0))

		msgOnly, err := openaiclient.Responses.New(ctx, responses.ResponseNewParams{
			Model: common.TestModelName,
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: responses.ResponseInputParam{
					responses.ResponseInputItemParamOfMessage("What is the weather?", responses.EasyInputMessageRoleUser),
				},
			},
			ToolChoice: responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
			},
			Tools: []responses.ToolUnionParam{responsesWeatherTool()},
		})
		Expect(err).NotTo(HaveOccurred())

		withHistory, err := openaiclient.Responses.New(ctx, responses.ResponseNewParams{
			Model: common.TestModelName,
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: responses.ResponseInputParam{
					responses.ResponseInputItemParamOfMessage("What is the weather?", responses.EasyInputMessageRoleUser),
					responses.ResponseInputItemParamOfFunctionCall(`{"city":"Paris"}`, "call_1", "get_weather"),
					responses.ResponseInputItemParamOfFunctionCallOutput("call_1", "sunny, 22C"),
				},
			},
			Tools: []responses.ToolUnionParam{responsesWeatherTool()},
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(withHistory.Usage.InputTokens).To(BeNumerically(">", msgOnly.Usage.InputTokens))
	})
})
