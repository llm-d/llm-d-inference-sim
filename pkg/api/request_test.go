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

const (
	model  = "test_model"
	prompt = "Hello, world!"
)

var _ = Describe("render requests", func() {
	It("creates a new text completions render request with correct fields", func() {
		req := NewTextCompletionsRenderRequest(model, prompt)

		Expect(req.Model()).To(Equal(model))
		Expect(req.Prompt).To(Equal(prompt))
		Expect(req.Endpoint()).To(Equal("/v1/completions"))
		Expect(req.IsMultiModal()).To(BeFalse())
	})

	It("creates a new chat completions render request with correct fields", func() {
		messages := []Message{
			{
				Role: RoleUser,
				Content: ChatComplContent{
					Raw: prompt,
				},
			},
		}
		req := NewChatCompletionsRenderRequest(model, messages)

		Expect(req.Model()).To(Equal(model))
		Expect(req.Messages).To(HaveLen(1))
		Expect(req.Endpoint()).To(Equal("/v1/chat/completions"))
	})
})

var _ = Describe("GetN", func() {
	It("returns 1 when N is nil", func() {
		req := &baseCompletionsRequest{}
		Expect(req.GetN()).To(Equal(1))
	})

	It("returns 1 when N is zero", func() {
		n := 0
		req := &baseCompletionsRequest{N: &n}
		Expect(req.GetN()).To(Equal(1))
	})

	It("returns 1 when N is negative", func() {
		n := -5
		req := &baseCompletionsRequest{N: &n}
		Expect(req.GetN()).To(Equal(1))
	})

	It("returns the value when N is positive", func() {
		n := 3
		req := &baseCompletionsRequest{N: &n}
		Expect(req.GetN()).To(Equal(3))
	})

	It("unmarshals n from JSON", func() {
		jsonData := []byte(`{"n": 5, "prompt": "test"}`)
		var req TextCompletionsParsedRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		Expect(req.GetN()).To(Equal(5))
	})

	It("unmarshals n from chat completions JSON", func() {
		jsonData := []byte(`{"n": 3, "model": "test", "messages": [{"role": "user", "content": "hi"}]}`)
		var req ChatCompletionsRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		Expect(req.GetN()).To(Equal(3))
	})

	It("defaults to 1 when n is absent from JSON", func() {
		jsonData := []byte(`{"model": "test", "messages": [{"role": "user", "content": "hi"}]}`)
		var req ChatCompletionsRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		Expect(req.GetN()).To(Equal(1))
	})
})

var _ = Describe("Message PlainText", func() {
	Context("role:tool message with ToolCallID", func() {
		It("includes tool_call_id in the role prefix when includeRole is true", func() {
			msg := Message{
				Role:       "tool",
				ToolCallID: "call_abc123",
				Content:    ChatComplContent{Raw: "sunny"},
			}
			Expect(msg.PlainText(true)).To(Equal("tool(call_abc123): sunny"))
		})

		It("omits the role prefix when includeRole is false", func() {
			msg := Message{
				Role:       "tool",
				ToolCallID: "call_abc123",
				Content:    ChatComplContent{Raw: "sunny"},
			}
			Expect(msg.PlainText(false)).To(Equal("sunny"))
		})
	})

	Context("assistant message with ToolCalls", func() {
		It("appends each tool call as [name(args)] when includeRole is true", func() {
			funcName := "get_weather"
			msg := Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{
						ID:   "call_xyz",
						Type: "function",
						Function: FunctionCall{
							Name:      &funcName,
							Arguments: `{"location":"NYC"}`,
						},
					},
				},
			}
			Expect(msg.PlainText(true)).To(Equal(`assistant: [get_weather({"location":"NYC"})]`))
		})
	})

	Context("JSON roundtrip", func() {
		It("marshals and unmarshals tool_call_id", func() {
			msg := Message{
				Role:       "tool",
				ToolCallID: "call_abc123",
				Content:    ChatComplContent{Raw: "result"},
			}
			data, err := json.Marshal(msg)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(data)).To(ContainSubstring(`"tool_call_id":"call_abc123"`))

			var got Message
			Expect(json.Unmarshal(data, &got)).To(Succeed())
			Expect(got.ToolCallID).To(Equal("call_abc123"))
		})

		It("deserializes tool_call_id from a chat completions request message", func() {
			jsonData := []byte(`{"role":"tool","tool_call_id":"call_abc123","content":"result"}`)
			var msg Message
			Expect(json.Unmarshal(jsonData, &msg)).To(Succeed())
			Expect(msg.ToolCallID).To(Equal("call_abc123"))
			Expect(msg.Content.Raw).To(Equal("result"))
		})
	})
})

var _ = Describe("TextCompletionsParsedRequest prompt", func() {
	Context("UnmarshalJSON", func() {
		It("should unmarshal a string prompt as a one-element slice", func() {
			jsonData := []byte(`{"prompt": "Hello, world!"}`)
			var req TextCompletionsParsedRequest
			err := json.Unmarshal(jsonData, &req)
			Expect(err).NotTo(HaveOccurred())
			Expect(req.Prompt).To(Equal([]PromptInput{{Text: "Hello, world!"}}))
		})

		It("should unmarshal an array prompt", func() {
			jsonData := []byte(`{"prompt": ["Hello", "world"]}`)
			var req TextCompletionsParsedRequest
			err := json.Unmarshal(jsonData, &req)
			Expect(err).NotTo(HaveOccurred())
			Expect(req.Prompt).To(Equal([]PromptInput{{Text: "Hello"}, {Text: "world"}}))
		})

		It("should return error for invalid prompt type", func() {
			jsonData := []byte(`{"prompt": 123}`)
			var req TextCompletionsParsedRequest
			err := json.Unmarshal(jsonData, &req)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("prompt must be a string, an array of strings, an array of token ids, or an array of arrays of token ids"))
		})

		It("should leave Prompt nil when the field is absent or null", func() {
			var req TextCompletionsParsedRequest
			Expect(json.Unmarshal([]byte(`{}`), &req)).To(Succeed())
			Expect(req.Prompt).To(BeNil())

			req = TextCompletionsParsedRequest{}
			Expect(json.Unmarshal([]byte(`{"prompt": null}`), &req)).To(Succeed())
			Expect(req.Prompt).To(BeNil())
		})
	})
})

var _ = Describe("InputContent UnmarshalJSON", func() {
	It("should unmarshal input_text content", func() {
		jsonData := []byte(`{"type": "input_text", "text": "hello"}`)
		var content InputContent
		err := json.Unmarshal(jsonData, &content)
		Expect(err).NotTo(HaveOccurred())
		Expect(content.Type).To(Equal(ResponsesInputText))
		Expect(content.Text).To(Equal("hello"))
	})

	It("should unmarshal input_image content", func() {
		jsonData := []byte(`{"type": "input_image", "image_url": "https://example.com/img.png"}`)
		var content InputContent
		err := json.Unmarshal(jsonData, &content)
		Expect(err).NotTo(HaveOccurred())
		Expect(content.Type).To(Equal(ResponsesInputImage))
		Expect(content.ImageURL).To(Equal("https://example.com/img.png"))
	})

	It("should unmarshal input_audio content", func() {
		jsonData := []byte(`{"type": "input_audio", "data": "base64audiodata", "format": "wav"}`)
		var content InputContent
		err := json.Unmarshal(jsonData, &content)
		Expect(err).NotTo(HaveOccurred())
		Expect(content.Type).To(Equal(ResponsesInputAudio))
		Expect(content.AudioData).To(Equal("base64audiodata"))
		Expect(content.AudioFormat).To(Equal("wav"))
	})

	It("should reject unsupported content type", func() {
		jsonData := []byte(`{"type": "input_video", "url": "http://example.com/v.mp4"}`)
		var content InputContent
		err := json.Unmarshal(jsonData, &content)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("unsupported input content type"))
	})

	It("should default to input_text when type is empty", func() {
		jsonData := []byte(`{"text": "hello"}`)
		var content InputContent
		err := json.Unmarshal(jsonData, &content)
		Expect(err).NotTo(HaveOccurred())
		Expect(content.Type).To(Equal(ResponsesInputText))
		Expect(content.Text).To(Equal("hello"))
	})
})

var _ = Describe("ResponsesRequest UnmarshalJSON with content types", func() {
	It("should unmarshal request with text input", func() {
		jsonData := []byte(`{"model":"m","input":"hello"}`)
		var req ResponsesRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		Expect(req.Input).To(HaveLen(1))
		msg := req.Input[0].(*InputMessage)
		Expect(msg.Content).To(HaveLen(1))
		Expect(msg.Content[0].Type).To(Equal(ResponsesInputText))
		Expect(msg.Content[0].Text).To(Equal("hello"))
	})

	It("should unmarshal request with mixed image and text content", func() {
		jsonData := []byte(`{
			"model": "m",
			"input": [{
				"role": "user",
				"content": [
					{"type": "input_text", "text": "Describe this image"},
					{"type": "input_image", "image_url": "https://example.com/cat.jpg"}
				]
			}]
		}`)
		var req ResponsesRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		Expect(req.Input).To(HaveLen(1))
		msg := req.Input[0].(*InputMessage)
		Expect(msg.Content).To(HaveLen(2))
		Expect(msg.Content[0].Type).To(Equal(ResponsesInputText))
		Expect(msg.Content[0].Text).To(Equal("Describe this image"))
		Expect(msg.Content[1].Type).To(Equal(ResponsesInputImage))
		Expect(msg.Content[1].ImageURL).To(Equal("https://example.com/cat.jpg"))
	})

	It("should unmarshal request with audio content", func() {
		jsonData := []byte(`{
			"model": "m",
			"input": [{
				"role": "user",
				"content": [
					{"type": "input_text", "text": "Transcribe this audio"},
					{"type": "input_audio", "data": "base64encodedaudio", "format": "mp3"}
				]
			}]
		}`)
		var req ResponsesRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		Expect(req.Input).To(HaveLen(1))
		msg := req.Input[0].(*InputMessage)
		Expect(msg.Content).To(HaveLen(2))
		Expect(msg.Content[0].Type).To(Equal(ResponsesInputText))
		Expect(msg.Content[1].Type).To(Equal(ResponsesInputAudio))
		Expect(msg.Content[1].AudioData).To(Equal("base64encodedaudio"))
		Expect(msg.Content[1].AudioFormat).To(Equal("mp3"))
	})

	It("should unmarshal request with image, audio, and text content combined", func() {
		jsonData := []byte(`{
			"model": "m",
			"input": [{
				"role": "user",
				"content": [
					{"type": "input_text", "text": "What do you see and hear?"},
					{"type": "input_image", "image_url": "https://example.com/photo.png"},
					{"type": "input_audio", "data": "audiobase64", "format": "wav"}
				]
			}]
		}`)
		var req ResponsesRequest
		err := json.Unmarshal(jsonData, &req)
		Expect(err).NotTo(HaveOccurred())
		msg := req.Input[0].(*InputMessage)
		Expect(msg.Content).To(HaveLen(3))
		Expect(msg.Content[0].Type).To(Equal(ResponsesInputText))
		Expect(msg.Content[1].Type).To(Equal(ResponsesInputImage))
		Expect(msg.Content[2].Type).To(Equal(ResponsesInputAudio))
	})
})

var _ = Describe("InputMessage PlainText with content types", func() {
	It("should include image URL in plain text", func() {
		msg := &InputMessage{
			Role: RoleUser,
			Content: []InputContent{
				{Type: ResponsesInputText, Text: "Look at this"},
				{Type: ResponsesInputImage, ImageURL: "https://example.com/cat.jpg"},
			},
		}
		text := msg.PlainText(false)
		Expect(text).To(Equal("Look at this\nimage: https://example.com/cat.jpg"))
	})

	It("should include audio format in plain text", func() {
		msg := &InputMessage{
			Role: RoleUser,
			Content: []InputContent{
				{Type: ResponsesInputText, Text: "Listen to this"},
				{Type: ResponsesInputAudio, AudioData: "base64data", AudioFormat: "wav"},
			},
		}
		text := msg.PlainText(false)
		Expect(text).To(Equal("Listen to this\naudio: wav"))
	})

	It("should include role when requested", func() {
		msg := &InputMessage{
			Role: RoleUser,
			Content: []InputContent{
				{Type: ResponsesInputText, Text: "hello"},
				{Type: ResponsesInputImage, ImageURL: "https://example.com/img.png"},
			},
		}
		text := msg.PlainText(true)
		Expect(text).To(Equal("user: hello\nimage: https://example.com/img.png"))
	})
})
