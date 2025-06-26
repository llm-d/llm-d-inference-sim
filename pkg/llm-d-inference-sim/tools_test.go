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
	"strings"

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
						"type": "array",
						"items": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type":  "array",
								"items": map[string]string{"type": "string"},
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

var _ = Describe("Simulator for request with tools", func() {

	DescribeTable("streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage(userMessage),
				},
				Model:         model,
				StreamOptions: openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)},
				ToolChoice:    openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")},
				Tools:         tools,
			}
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
					} else if choice.FinishReason == "" || choice.FinishReason == toolsFinishReason {
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
			Expect(chunk.Usage.PromptTokens).To(Equal(int64(4)))
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
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
	)

	DescribeTable("no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.ChatCompletionNewParams{
				Messages:   []openai.ChatCompletionMessageParamUnion{openai.UserMessage(userMessage)},
				Model:      model,
				ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")},
				Tools:      tools,
			}

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(int64(4)))
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
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
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
		Entry(nil, modeRandom),
	)

	DescribeTable("array parameter, no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.ChatCompletionNewParams{
				Messages:   []openai.ChatCompletionMessageParamUnion{openai.UserMessage(userMessage)},
				Model:      model,
				ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")},
				Tools:      toolWithArray,
			}

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(int64(4)))
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
		},
		func(mode string) string {
			return "mode: " + mode
		},
		// Call several times because the tools and arguments are chosen randomly
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
	)

	DescribeTable("3D array parameter, no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.ChatCompletionNewParams{
				Messages:   []openai.ChatCompletionMessageParamUnion{openai.UserMessage(userMessage)},
				Model:      model,
				ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")},
				Tools:      toolWith3DArray,
			}

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(int64(4)))
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
		},
		func(mode string) string {
			return "mode: " + mode
		},
		// Call several times because the tools and arguments are chosen randomly
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
		Entry(nil, modeRandom),
	)

	DescribeTable("objects, no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.ChatCompletionNewParams{
				Messages:   []openai.ChatCompletionMessageParamUnion{openai.UserMessage(userMessage)},
				Model:      model,
				ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")},
				Tools:      toolWithObjects,
			}

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(int64(4)))
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
		Entry(nil, modeRandom),
	)

	DescribeTable("objects with array field, no streaming",
		func(mode string) {
			ctx := context.TODO()
			client, err := startServer(ctx, mode)
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))

			params := openai.ChatCompletionNewParams{
				Messages:   []openai.ChatCompletionMessageParamUnion{openai.UserMessage(userMessage)},
				Model:      model,
				ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")},
				Tools:      toolWithObjectAndArray,
			}

			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(chatCompletionObject))

			Expect(resp.Usage.PromptTokens).To(Equal(int64(4)))
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
		Entry(nil, modeRandom),
	)
})
