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
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
)

var tools = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "get_weather",
			Description: openai.String("Get weather at the given location"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]string{
						"type": "string",
					},
				},
				"required": []string{"location"},
			},
		},
	},
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "get_temperature",
			Description: openai.String("Get temperature at the given location"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"city": map[string]string{
						"type": "string",
					},
					"unit": map[string]interface{}{
						"type": "string",
						"enum": []string{"C", "F"},
					},
				},
				"required": []string{"city", "unit"},
			},
		},
	},
}

var invalidTools = [][]openai.ChatCompletionToolParam{
	{
		{
			Function: openai.FunctionDefinitionParam{
				Name:        "get_weather",
				Description: openai.String("Get weather at the given location"),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]string{
							"type": "string",
						},
					},
					"required": []string{"location"},
				},
			},
		},
		{
			Function: openai.FunctionDefinitionParam{
				Name:        "get_temperature",
				Description: openai.String("Get temperature at the given location"),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]string{
							"type": "string",
						},
						"unit": map[string]interface{}{
							"type": "string",
							"enum": []int{5, 6},
						},
					},
					"required": []string{"city", "unit"},
				},
			},
		},
	},

	{
		{
			Function: openai.FunctionDefinitionParam{
				Name:        "get_weather",
				Description: openai.String("Get weather at the given location"),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]string{
							"type": "stringstring",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	},

	{
		{
			Function: openai.FunctionDefinitionParam{
				Name:        "get_weather",
				Description: openai.String("Get weather at the given location"),
			},
		},
	},
}

var toolWithArray = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "multiply_numbers",
			Description: openai.String("Multiply an array of numbers"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"numbers": map[string]interface{}{
						"type":        "array",
						"items":       map[string]string{"type": "number"},
						"description": "List of numbers to multiply",
					},
				},
				"required": []string{"numbers"},
			},
		},
	},
}

var toolWith3DArray = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "process_tensor",
			Description: openai.String("Process a 3D tensor of strings"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"tensor": map[string]interface{}{
						"type":     "array",
						"minItems": 2,
						"items": map[string]any{
							"type":     "array",
							"minItems": 0,
							"maxItems": 1,
							"items": map[string]any{
								"type":     "array",
								"items":    map[string]string{"type": "string"},
								"maxItems": 3,
							},
						},
						"description": "List of strings",
					},
				},
				"required": []string{"tensor"},
			},
		},
	},
}

var toolWithWrongMinMax = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "multiply_numbers",
			Description: openai.String("Multiply an array of numbers"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"numbers": map[string]interface{}{
						"type":        "array",
						"items":       map[string]string{"type": "number"},
						"description": "List of numbers to multiply",
						"minItems":    3,
						"maxItems":    1,
					},
				},
				"required": []string{"numbers"},
			},
		},
	},
}

var toolWithObjects = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "process_order",
			Description: openai.String("Process a customer order"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"order_info": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"item": map[string]interface{}{
								"type": "string",
							},
							"quantity": map[string]string{
								"type": "integer",
							},
							"address": map[string]interface{}{
								"type": "object",
								"properties": map[string]interface{}{
									"street": map[string]interface{}{
										"type": "string",
									},
									"number": map[string]interface{}{
										"type": "integer",
									},
									"home": map[string]interface{}{
										"type": "boolean",
									},
								},
								"required": []string{"street", "number", "home"},
							},
						},
						"required": []string{"item", "quantity", "address"},
					},
					"name": map[string]interface{}{
						"type": "string",
					},
				},
				"required": []string{"order_info", "name"},
			},
		},
	},
}

var toolWithObjectAndArray = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "submit_survey",
			Description: openai.String("Submit a survey with user information."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"user_info": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"name": map[string]interface{}{
								"type":        "string",
								"description": "The user's name",
							},
							"age": map[string]string{
								"type":        "integer",
								"description": "The user's age",
							},
							"hobbies": map[string]interface{}{
								"type":        "array",
								"items":       map[string]string{"type": "string"},
								"description": "A list of the user's hobbies",
							},
						},
						"required": []string{"name", "age", "hobbies"},
					},
				},
				"required": []string{"user_info"},
			},
		},
	},
}

var toolWithoutRequiredParams = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "get_temperature",
			Description: openai.String("Get temperature at the given location"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"city": map[string]string{
						"type": "string",
					},
					"country": map[string]string{
						"type": "string",
					},
					"unit": map[string]interface{}{
						"type": "string",
						"enum": []string{"C", "F"},
					},
				},
			},
		},
	},
}

var toolWithObjectWithoutRequiredParams = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name: "process_order",
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"order_info": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"item": map[string]interface{}{
								"type": "string",
							},
							"quantity": map[string]string{
								"type": "integer",
							},
							"address": map[string]interface{}{
								"type": "string",
							},
						},
					},
				},
				"required": []string{"order_info"},
			},
		},
	},
}

var _ = Describe("Simulator for request with tools", func() {

	DescribeTable("streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndChatParams(client, model, userMessage, true)
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
			params.Tools = tools

			stream := openaiclient.Chat.Completions.NewStreaming(ctx, params)
			defer func() {
				err := stream.Close()
				Expect(err).NotTo(HaveOccurred())
			}()
			args := make(map[string][]string)
			role := ""
			var chunk openai.ChatCompletionChunk
			numberOfChunksWithUsage := 0
			lastIndex := -1
			var functionName string
			for stream.Next() {
				chunk = stream.Current()
				for _, choice := range chunk.Choices {
					if choice.Delta.Role != "" {
						role = choice.Delta.Role
					} else if choice.FinishReason == "" || choice.FinishReason == dataset.ToolsFinishReason {
						toolCalls := choice.Delta.ToolCalls
						Expect(toolCalls).To(HaveLen(1))
						tc := toolCalls[0]
						Expect(tc.Index).To(Or(BeNumerically("==", lastIndex), BeNumerically("==", lastIndex+1)))
						if tc.Index > int64(lastIndex) {
							Expect(tc.Function.Name).To(Or(Equal("get_weather"), Equal("get_temperature")))
							lastIndex++
							args[tc.Function.Name] = []string{tc.Function.Arguments}
							functionName = tc.Function.Name
						} else {
							Expect(tc.Function.Name).To(BeEmpty())
							args[functionName] = append(args[functionName], tc.Function.Arguments)
						}
						Expect(tc.ID).NotTo(BeEmpty())
						Expect(tc.Type).To(Equal("function"))
					}
				}
				if chunk.Usage.CompletionTokens != 0 || chunk.Usage.PromptTokens != 0 || chunk.Usage.TotalTokens != 0 {
					numberOfChunksWithUsage++
				}
				Expect(string(chunk.Object)).To(Equal(chatCompletionChunkObject))
			}

			Expect(numberOfChunksWithUsage).To(Equal(1))
			Expect(chunk.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(chunk.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(chunk.Usage.TotalTokens).To(Equal(chunk.Usage.PromptTokens + chunk.Usage.CompletionTokens))

			Expect(role).Should(Equal("assistant"))

			for functionName, callArgs := range args {
				joinedArgs := strings.Join(callArgs, "")
				argsMap := make(map[string]string)
				err := json.Unmarshal([]byte(joinedArgs), &argsMap)
				Expect(err).NotTo(HaveOccurred())

				if functionName == "get_weather" {
					Expect(joinedArgs).To(ContainSubstring("location"))
				} else {
					Expect(joinedArgs).To(ContainSubstring("city"))
					Expect(joinedArgs).To(ContainSubstring("unit"))
					Expect(argsMap["unit"]).To(Or(Equal("C"), Equal("F")))
				}
			}
		},
		func(mode string) string {
			return "mode: " + mode
		},
		// Call several times because the tools and arguments are chosen randomly
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeRandom),
	)

	DescribeTable("no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
			params.Tools = tools

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			content := resp.Choices[0].Message.Content
			Expect(content).Should(BeEmpty())

			toolCalls := resp.Choices[0].Message.ToolCalls
			Expect(toolCalls).ToNot(BeEmpty())
			for _, tc := range toolCalls {
				Expect(tc.Function.Name).To(Or(Equal("get_weather"), Equal("get_temperature")))
				Expect(tc.ID).NotTo(BeEmpty())
				Expect(string(tc.Type)).To(Equal("function"))
				args := make(map[string]string)
				err := json.Unmarshal([]byte(tc.Function.Arguments), &args)
				Expect(err).NotTo(HaveOccurred())

				if tc.Function.Name == "get_weather" {
					Expect(tc.Function.Arguments).To(ContainSubstring("location"))
				} else {
					Expect(tc.Function.Arguments).To(ContainSubstring("city"))
					Expect(tc.Function.Arguments).To(ContainSubstring("unit"))
					Expect(args["unit"]).To(Or(Equal("C"), Equal("F")))
				}
			}
		},
		func(mode string) string {
			return "mode: " + mode
		},
		// Call several times because the tools and arguments are chosen randomly
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeRandom),
	)

	DescribeTable("check validator",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			for _, invalidTool := range invalidTools {
				params := openai.ChatCompletionNewParams{
					Messages:   []openai.ChatCompletionMessageParamUnion{openai.UserMessage(userMessage)},
					Model:      model,
					ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")},
					Tools:      invalidTool,
				}

				_, err := openaiclient.Chat.Completions.New(ctx, params)
				Expect(err).To(HaveOccurred())
			}
		},
		func(mode string) string {
			return "mode: " + mode
		},
		Entry(nil, common.ModeRandom),
	)

	DescribeTable("array parameter, no streaming",
		func(mode string, minLength int, maxLength int, min float64, max float64) {
			ctx := context.TODO()
			serverArgs := []string{"cmd", "--model", model, "--mode", mode,
				"--min-tool-call-array-param-length", strconv.Itoa(minLength),
				"--max-tool-call-array-param-length", strconv.Itoa(maxLength),
				"--min-tool-call-number-param", fmt.Sprint(min),
				"--max-tool-call-number-param", fmt.Sprint(max),
			}
			client, err := startServerWithArgs(ctx, common.ModeEcho, serverArgs, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
			params.Tools = toolWithArray

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			content := resp.Choices[0].Message.Content
			Expect(content).Should(BeEmpty())

			toolCalls := resp.Choices[0].Message.ToolCalls
			Expect(toolCalls).To(HaveLen(1))
			tc := toolCalls[0]
			Expect(tc.Function.Name).To(Equal("multiply_numbers"))
			Expect(tc.ID).NotTo(BeEmpty())
			Expect(string(tc.Type)).To(Equal("function"))
			args := make(map[string][]float64)
			err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			Expect(err).NotTo(HaveOccurred())
			Expect(args["numbers"]).ToNot(BeEmpty())
			Expect(len(args["numbers"])).To(BeNumerically(">=", minLength))
			Expect(len(args["numbers"])).To(BeNumerically("<=", maxLength))
			for _, number := range args["numbers"] {
				Expect(number).To(BeNumerically(">=", min))
				Expect(number).To(BeNumerically("<=", max))
			}
		},
		func(mode string, minLength int, maxLength int, min float64, max float64) string {
			return fmt.Sprintf("mode: %s, min array length: %d, max array length: %d, min number: %f max number %f ",
				mode, minLength, maxLength, min, max)
		},
		Entry(nil, common.ModeRandom, 3, 7, -100.2, -5.75),
		Entry(nil, common.ModeRandom, 2, 10, 0.0, 34.5),
		Entry(nil, common.ModeRandom, 2, 2, -100.0, 100.0),
		Entry(nil, common.ModeRandom, 4, 5, 222.222, 333.333),
	)

	DescribeTable("3D array parameter, no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
			params.Tools = toolWith3DArray

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			content := resp.Choices[0].Message.Content
			Expect(content).Should(BeEmpty())

			toolCalls := resp.Choices[0].Message.ToolCalls
			Expect(toolCalls).To(HaveLen(1))
			tc := toolCalls[0]
			Expect(tc.Function.Name).To(Equal("process_tensor"))
			Expect(tc.ID).NotTo(BeEmpty())
			Expect(string(tc.Type)).To(Equal("function"))

			args := make(map[string][][][]string)
			err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			Expect(err).NotTo(HaveOccurred())
			Expect(args["tensor"]).ToNot(BeEmpty())
			tensor := args["tensor"]
			Expect(len(tensor)).To(BeNumerically(">=", 2))
			Expect(len(tensor)).To(BeNumerically("<=", 5)) // Default configuration
			for _, elem := range tensor {
				Expect(len(elem)).To(Or(Equal(0), Equal(1)))
				for _, inner := range elem {
					Expect(len(inner)).To(Or(Equal(1), Equal(2), Equal(3)))
				}
			}
		},
		func(mode string) string {
			return "mode: " + mode
		},
		// Call several times because the tools and arguments are chosen randomly
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeRandom),
		Entry(nil, common.ModeRandom),
	)

	DescribeTable("array parameter with wrong min and max items, no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
			params.Tools = toolWithWrongMinMax

			_, err = openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).To(HaveOccurred())
		},
		func(mode string) string {
			return "mode: " + mode
		},
		Entry(nil, common.ModeRandom),
	)

	DescribeTable("objects, no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
			params.Tools = toolWithObjects

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			content := resp.Choices[0].Message.Content
			Expect(content).Should(BeEmpty())

			toolCalls := resp.Choices[0].Message.ToolCalls
			Expect(toolCalls).To(HaveLen(1))
			tc := toolCalls[0]
			Expect(tc.Function.Name).To(Equal("process_order"))
			Expect(tc.ID).NotTo(BeEmpty())
			Expect(string(tc.Type)).To(Equal("function"))

			args := make(map[string]any)
			err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			Expect(err).NotTo(HaveOccurred())
			Expect(args["name"]).ToNot(BeEmpty())
			Expect(args["order_info"]).ToNot(BeEmpty())
			orderInfo, ok := args["order_info"].(map[string]any)
			Expect(ok).To(BeTrue())
			Expect(orderInfo["item"]).ToNot(BeEmpty())
			Expect(orderInfo).To(HaveKey("quantity"))
			Expect(orderInfo["address"]).ToNot(BeEmpty())
			address, ok := orderInfo["address"].(map[string]any)
			Expect(ok).To(BeTrue())
			Expect(address["street"]).ToNot(BeEmpty())
			_, ok = address["street"].(string)
			Expect(ok).To(BeTrue())
			_, ok = address["number"].(float64)
			Expect(ok).To(BeTrue())
			_, ok = address["home"].(bool)
			Expect(ok).To(BeTrue())
		},
		func(mode string) string {
			return "mode: " + mode
		},
		Entry(nil, common.ModeRandom),
	)

	DescribeTable("objects with array field, no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
			params.Tools = toolWithObjectAndArray

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(userMsgTokens))
			Expect(resp.Usage.CompletionTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens + resp.Usage.CompletionTokens))

			content := resp.Choices[0].Message.Content
			Expect(content).Should(BeEmpty())

			toolCalls := resp.Choices[0].Message.ToolCalls
			Expect(toolCalls).To(HaveLen(1))
			tc := toolCalls[0]
			Expect(tc.Function.Name).To(Equal("submit_survey"))
			Expect(tc.ID).NotTo(BeEmpty())
			Expect(string(tc.Type)).To(Equal("function"))

			args := make(map[string]any)
			err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			Expect(err).NotTo(HaveOccurred())
			Expect(args["user_info"]).ToNot(BeEmpty())

			userInfo, ok := args["user_info"].(map[string]any)
			Expect(ok).To(BeTrue())
			Expect(userInfo).To(HaveKey("age"))
			Expect(userInfo["name"]).ToNot(BeEmpty())
			Expect(userInfo["hobbies"]).ToNot(BeEmpty())
			_, ok = userInfo["hobbies"].([]any)
			Expect(ok).To(BeTrue())
		},
		func(mode string) string {
			return "mode: " + mode
		},
		Entry(nil, common.ModeRandom),
	)

	DescribeTable("tool with not required params",
		func(probability int, numberOfParams int) {
			ctx := context.TODO()
			serverArgs := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
				"--tool-call-not-required-param-probability", strconv.Itoa(probability),
			}
			client, err := startServerWithArgs(ctx, common.ModeEcho, serverArgs, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
			params.Tools = toolWithoutRequiredParams

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			toolCalls := resp.Choices[0].Message.ToolCalls
			Expect(toolCalls).To(HaveLen(1))
			tc := toolCalls[0]
			Expect(tc.Function.Name).To(Equal("get_temperature"))
			Expect(tc.ID).NotTo(BeEmpty())
			Expect(string(tc.Type)).To(Equal("function"))
			args := make(map[string]string)
			err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			Expect(err).NotTo(HaveOccurred())
			Expect(args).To(HaveLen(numberOfParams))
		},
		func(probability int, numberOfParams int) string {
			return fmt.Sprintf("probability: %d", probability)
		},
		Entry(nil, 0, 0),
		Entry(nil, 100, 3),
	)

	DescribeTable("tool with object with not required params",
		func(probability int, numberOfParams int, min int, max int) {
			ctx := context.TODO()
			serverArgs := []string{"cmd", "--model", model, "--mode", common.ModeRandom,
				"--object-tool-call-not-required-field-probability", strconv.Itoa(probability),
				"--min-tool-call-integer-param", strconv.Itoa(min),
				"--max-tool-call-integer-param", strconv.Itoa(max),
			}
			client, err := startServerWithArgs(ctx, common.ModeEcho, serverArgs, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndChatParams(client, model, userMessage, false)
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
			params.Tools = toolWithObjectWithoutRequiredParams

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			toolCalls := resp.Choices[0].Message.ToolCalls
			Expect(toolCalls).To(HaveLen(1))
			tc := toolCalls[0]
			Expect(tc.Function.Name).To(Equal("process_order"))
			Expect(tc.ID).NotTo(BeEmpty())
			Expect(string(tc.Type)).To(Equal("function"))

			args := make(map[string]any)
			err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			Expect(err).NotTo(HaveOccurred())
			Expect(args["order_info"]).ToNot(BeNil())
			orderInfo, ok := args["order_info"].(map[string]any)
			Expect(ok).To(BeTrue())
			Expect(orderInfo).To(HaveLen(numberOfParams))
			if numberOfParams > 0 {
				Expect(orderInfo).To(HaveKey("quantity"))
				quantityFloat, ok := orderInfo["quantity"].(float64)
				Expect(ok).To(BeTrue())
				quantity := int(quantityFloat)
				Expect(quantity).To(BeNumerically(">=", min))
				Expect(quantity).To(BeNumerically("<=", max))
			}
		},
		func(probability int, numberOfParams int, min int, max int) string {
			return fmt.Sprintf("probability: %d min: %d, max %d", probability, min, max)
		},
		Entry(nil, 0, 0, 0, 0),
		Entry(nil, 100, 3, 0, 0),
		Entry(nil, 100, 3, 5, 150),
		Entry(nil, 100, 3, 150, 2500),
	)
})
