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
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

var _ = Describe("Simulator for request with response_format", forEachEngine(func() {
	jsonObject := openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
	}
	jsonSchema := openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name: "city",
				Schema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name":       map[string]any{"type": "string"},
						"population": map[string]any{"type": "integer"},
					},
					"required": []any{"name", "population"},
				},
			},
		},
	}

	// chat sends params and returns the message content and finish reason,
	// joining the deltas when streaming.
	chat := func(mode string, format openai.ChatCompletionNewParamsResponseFormatUnion, maxTokens int64,
		streaming bool) (string, string, int64) {
		ctx := context.TODO()
		client, err := startServer(ctx, mode)
		Expect(err).NotTo(HaveOccurred())

		openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName, testUserMessage, streaming)
		params.ResponseFormat = format
		if maxTokens > 0 {
			params.MaxTokens = param.NewOpt(maxTokens)
		}

		if !streaming {
			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).To(HaveLen(1))
			return resp.Choices[0].Message.Content, resp.Choices[0].FinishReason, resp.Usage.CompletionTokens
		}

		stream := openaiclient.Chat.Completions.NewStreaming(ctx, params)
		defer func() {
			Expect(stream.Close()).To(Succeed())
		}()
		var content strings.Builder
		var finishReason string
		var completionTokens int64
		for stream.Next() {
			chunk := stream.Current()
			for _, choice := range chunk.Choices {
				content.WriteString(choice.Delta.Content)
				if choice.FinishReason != "" {
					finishReason = choice.FinishReason
				}
			}
			if chunk.Usage.CompletionTokens > 0 {
				completionTokens = chunk.Usage.CompletionTokens
			}
		}
		Expect(stream.Err()).NotTo(HaveOccurred())
		return content.String(), finishReason, completionTokens
	}

	DescribeTable("returns a JSON object for json_object",
		func(streaming bool) {
			content, finishReason, _ := chat(common.ModeRandom, jsonObject, 0, streaming)
			var object map[string]any
			Expect(json.Unmarshal([]byte(content), &object)).To(Succeed(), content)
			Expect(finishReason).To(Equal(common.StopFinishReason))
		},
		Entry("non-streaming", false),
		Entry("streaming", true),
	)

	DescribeTable("returns JSON that matches the schema for json_schema",
		func(streaming bool) {
			content, finishReason, _ := chat(common.ModeRandom, jsonSchema, 0, streaming)
			var city struct {
				Name       *string `json:"name"`
				Population *int    `json:"population"`
			}
			decoder := json.NewDecoder(strings.NewReader(content))
			decoder.DisallowUnknownFields()
			Expect(decoder.Decode(&city)).To(Succeed(), content)
			Expect(city.Name).NotTo(BeNil(), content)
			Expect(city.Population).NotTo(BeNil(), content)
			Expect(finishReason).To(Equal(common.StopFinishReason))
		},
		Entry("non-streaming", false),
		Entry("streaming", true),
	)

	It("cuts the JSON at max_tokens", func() {
		content, finishReason, completionTokens := chat(common.ModeRandom, jsonSchema, 3, false)
		Expect(content).To(HavePrefix("{"))
		Expect(json.Valid([]byte(content))).To(BeFalse(), content)
		Expect(finishReason).To(Equal(common.LengthFinishReason))
		Expect(completionTokens).To(Equal(int64(3)))
	})

	It("returns random text for the text type", func() {
		text := openai.ChatCompletionNewParamsResponseFormatUnion{OfText: &shared.ResponseFormatTextParam{}}
		content, _, _ := chat(common.ModeRandom, text, 0, false)
		Expect(content).NotTo(BeEmpty())
		Expect(json.Valid([]byte(content))).To(BeFalse(), content)
	})

	It("echoes the prompt in echo mode", func() {
		content, _, _ := chat(common.ModeEcho, jsonObject, 0, false)
		Expect(content).To(Equal(testUserMessage))
	})
}))
