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

package openaiserverapi

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// Helper function to create string pointer
func stringPtr(s string) *string {
	return &s
}

const (
	model  = "test_model"
	prompt = "Hello, world!"
)

var _ = Describe("TextCompletionsRenderRequest", func() {
	Describe("NewTextCompletionsRenderRequest", func() {
		It("creates a new text completions render request with correct fields", func() {
			req := NewTextCompletionsRenderRequest(model, prompt)

			Expect(req.GetModel()).To(Equal(model))
			Expect(req.Prompt).To(Equal(prompt))
			Expect(req.GetEndpoint()).To(Equal("/v1/completions"))
			Expect(req.IsMultiModal()).To(BeFalse())
		})
	})

	Describe("MarshalForRenderer", func() {
		It("marshals basic text completion request with model and prompt", func() {
			req := NewTextCompletionsRenderRequest(model, prompt)

			payload, err := req.MarshalForRenderer()
			Expect(err).NotTo(HaveOccurred())
			Expect(payload).NotTo(BeNil())

			var result map[string]interface{}
			err = json.Unmarshal(payload, &result)
			Expect(err).NotTo(HaveOccurred())

			Expect(result["model"]).To(Equal(model))
			Expect(result["prompt"]).To(Equal(prompt))
			// Ensure only model and prompt are present
			Expect(result).To(HaveLen(2))
		})

		It("marshals text completion with empty prompt", func() {
			req := NewTextCompletionsRenderRequest(model, "")

			payload, err := req.MarshalForRenderer()
			Expect(err).NotTo(HaveOccurred())
			Expect(payload).NotTo(BeNil())

			var result map[string]interface{}
			err = json.Unmarshal(payload, &result)
			Expect(err).NotTo(HaveOccurred())

			Expect(result["model"]).To(Equal(model))
			Expect(result["prompt"]).To(Equal(""))
			Expect(result).To(HaveLen(2))
		})

		It("marshals text completion with multiline prompt", func() {
			multilinePrompt := "Line 1\nLine 2\nLine 3"
			req := NewTextCompletionsRenderRequest(model, multilinePrompt)

			payload, err := req.MarshalForRenderer()
			Expect(err).NotTo(HaveOccurred())
			Expect(payload).NotTo(BeNil())

			var result map[string]interface{}
			err = json.Unmarshal(payload, &result)
			Expect(err).NotTo(HaveOccurred())

			Expect(result["model"]).To(Equal(model))
			Expect(result["prompt"]).To(Equal(multilinePrompt))
			Expect(result).To(HaveLen(2))
		})
	})

	Describe("GetEndpoint", func() {
		It("returns the correct endpoint", func() {
			req := NewTextCompletionsRenderRequest(model, prompt)
			Expect(req.GetEndpoint()).To(Equal("/v1/completions"))
		})
	})

	Describe("IsMultiModal", func() {
		It("returns false for text completions", func() {
			req := NewTextCompletionsRenderRequest(model, prompt)
			Expect(req.IsMultiModal()).To(BeFalse())
		})
	})
})

var _ = Describe("ChatCompletionsRenderRequest", func() {
	Describe("NewChatCompletionsRenderRequest", func() {
		It("creates a new chat completions render request with correct fields", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Raw: "Hello!",
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)

			Expect(req.GetModel()).To(Equal(model))
			Expect(req.Messages).To(HaveLen(1))
			Expect(req.GetEndpoint()).To(Equal("/v1/chat/completions"))
		})
	})

	Describe("MarshalForRenderer", func() {
		It("marshals basic chat completion with simple messages", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Raw: "Hello!",
					},
				},
				{
					Role: RoleAssistant,
					Content: ChatComplContent{
						Raw: "Hi there!",
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)

			payload, err := req.MarshalForRenderer()
			Expect(err).NotTo(HaveOccurred())
			Expect(payload).NotTo(BeNil())

			var result map[string]interface{}
			err = json.Unmarshal(payload, &result)
			Expect(err).NotTo(HaveOccurred())

			Expect(result["model"]).To(Equal(model))
			Expect(result).To(HaveKey("messages"))

			msgs, ok := result["messages"].([]interface{})
			Expect(ok).To(BeTrue())
			Expect(msgs).To(HaveLen(2))

			// Ensure only model and messages are present
			Expect(result).To(HaveLen(2))
		})

		It("marshals chat completion with structured content including images", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Structured: []ChatComplContentBlock{
							{
								Type: "text",
								Text: "What's in this image?",
							},
							{
								Type: "image_url",
								ImageURL: ChatComplImageBlock{
									Url: "https://example.com/image.jpg",
								},
							},
						},
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)

			payload, err := req.MarshalForRenderer()
			Expect(err).NotTo(HaveOccurred())
			Expect(payload).NotTo(BeNil())

			var result map[string]interface{}
			err = json.Unmarshal(payload, &result)
			Expect(err).NotTo(HaveOccurred())

			Expect(result["model"]).To(Equal(model))
			Expect(result).To(HaveKey("messages"))

			msgs, ok := result["messages"].([]interface{})
			Expect(ok).To(BeTrue())
			Expect(msgs).To(HaveLen(1))

			msg := msgs[0].(map[string]interface{})
			Expect(msg["role"]).To(Equal(RoleUser))

			content, ok := msg["content"].([]interface{})
			Expect(ok).To(BeTrue())
			Expect(content).To(HaveLen(2))

			// Check text block
			textBlock := content[0].(map[string]interface{})
			Expect(textBlock["type"]).To(Equal("text"))
			Expect(textBlock["text"]).To(Equal("What's in this image?"))

			// Check image block
			imageBlock := content[1].(map[string]interface{})
			Expect(imageBlock["type"]).To(Equal("image_url"))
			imageURL := imageBlock["image_url"].(map[string]interface{})
			Expect(imageURL["url"]).To(Equal("https://example.com/image.jpg"))

			// Ensure only model and messages are present
			Expect(result).To(HaveLen(2))
		})

		It("marshals chat completion with tool calls", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Raw: "What's the weather?",
					},
				},
				{
					Role: RoleAssistant,
					Content: ChatComplContent{
						Raw: "",
					},
					ToolCalls: []ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: FunctionCall{
								Name:      stringPtr("get_weather"),
								Arguments: `{"location": "San Francisco"}`,
							},
						},
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)

			payload, err := req.MarshalForRenderer()
			Expect(err).NotTo(HaveOccurred())
			Expect(payload).NotTo(BeNil())

			var result map[string]interface{}
			err = json.Unmarshal(payload, &result)
			Expect(err).NotTo(HaveOccurred())

			Expect(result["model"]).To(Equal(model))
			Expect(result).To(HaveKey("messages"))

			msgs, ok := result["messages"].([]interface{})
			Expect(ok).To(BeTrue())
			Expect(msgs).To(HaveLen(2))

			// Check assistant message with tool calls
			assistantMsg := msgs[1].(map[string]interface{})
			Expect(assistantMsg["role"]).To(Equal(RoleAssistant))

			toolCalls, ok := assistantMsg["tool_calls"].([]interface{})
			Expect(ok).To(BeTrue())
			Expect(toolCalls).To(HaveLen(1))

			toolCall := toolCalls[0].(map[string]interface{})
			Expect(toolCall["id"]).To(Equal("call_123"))
			Expect(toolCall["type"]).To(Equal("function"))

			function := toolCall["function"].(map[string]interface{})
			Expect(function["name"]).To(Equal("get_weather"))
			Expect(function["arguments"]).To(Equal(`{"location": "San Francisco"}`))

			// Ensure only model and messages are present
			Expect(result).To(HaveLen(2))
		})

		It("marshals chat completion with empty messages array", func() {
			messages := []ChatComplMessage{}
			req := NewChatCompletionsRenderRequest(model, messages)

			payload, err := req.MarshalForRenderer()
			Expect(err).NotTo(HaveOccurred())
			Expect(payload).NotTo(BeNil())

			var result map[string]interface{}
			err = json.Unmarshal(payload, &result)
			Expect(err).NotTo(HaveOccurred())

			Expect(result["model"]).To(Equal(model))
			Expect(result).To(HaveKey("messages"))

			msgs, ok := result["messages"].([]interface{})
			Expect(ok).To(BeTrue())
			Expect(msgs).To(BeEmpty())

			// Ensure only model and messages are present
			Expect(result).To(HaveLen(2))
		})

		It("marshals chat completion with multiple content blocks", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Structured: []ChatComplContentBlock{
							{
								Type: "text",
								Text: "First text block",
							},
							{
								Type: "text",
								Text: "Second text block",
							},
							{
								Type: "image_url",
								ImageURL: ChatComplImageBlock{
									Url: "https://example.com/image1.jpg",
								},
							},
							{
								Type: "image_url",
								ImageURL: ChatComplImageBlock{
									Url: "https://example.com/image2.jpg",
								},
							},
						},
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)

			payload, err := req.MarshalForRenderer()
			Expect(err).NotTo(HaveOccurred())
			Expect(payload).NotTo(BeNil())

			var result map[string]interface{}
			err = json.Unmarshal(payload, &result)
			Expect(err).NotTo(HaveOccurred())

			Expect(result["model"]).To(Equal(model))
			msgs, ok := result["messages"].([]interface{})
			Expect(ok).To(BeTrue())
			Expect(msgs).To(HaveLen(1))

			msg := msgs[0].(map[string]interface{})
			content, ok := msg["content"].([]interface{})
			Expect(ok).To(BeTrue())
			Expect(content).To(HaveLen(4))
		})
	})

	Describe("GetEndpoint", func() {
		It("returns the correct endpoint", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Raw: "Hello!",
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)
			Expect(req.GetEndpoint()).To(Equal("/v1/chat/completions"))
		})
	})

	Describe("IsMultiModal", func() {
		It("returns false for text-only messages", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Raw: "Hello!",
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)
			Expect(req.IsMultiModal()).To(BeFalse())
		})

		It("returns false for structured text-only content", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Structured: []ChatComplContentBlock{
							{
								Type: "text",
								Text: "Hello!",
							},
						},
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)
			Expect(req.IsMultiModal()).To(BeFalse())
		})

		It("returns true when messages contain image_url", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Structured: []ChatComplContentBlock{
							{
								Type: "text",
								Text: "What's in this image?",
							},
							{
								Type: "image_url",
								ImageURL: ChatComplImageBlock{
									Url: "https://example.com/image.jpg",
								},
							},
						},
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)
			Expect(req.IsMultiModal()).To(BeTrue())
		})

		It("returns true when any message contains image_url", func() {
			messages := []ChatComplMessage{
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Raw: "First message",
					},
				},
				{
					Role: RoleAssistant,
					Content: ChatComplContent{
						Raw: "Response",
					},
				},
				{
					Role: RoleUser,
					Content: ChatComplContent{
						Structured: []ChatComplContentBlock{
							{
								Type: "image_url",
								ImageURL: ChatComplImageBlock{
									Url: "https://example.com/image.jpg",
								},
							},
						},
					},
				},
			}
			req := NewChatCompletionsRenderRequest(model, messages)
			Expect(req.IsMultiModal()).To(BeTrue())
		})

		It("returns false for empty messages", func() {
			messages := []ChatComplMessage{}
			req := NewChatCompletionsRenderRequest(model, messages)
			Expect(req.IsMultiModal()).To(BeFalse())
		})
	})
})

// Made with Bob
