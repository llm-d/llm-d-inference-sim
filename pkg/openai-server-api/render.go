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

// Contains structures and functions related to requests for all supported APIs
package openaiserverapi

import (
	"encoding/json"
)

type RenderRequest interface {
	GetModel() string
	GetEndpoint() string
	IsMultiModal() bool
	MarshalForRenderer() ([]byte, error)
}

func NewTextCompletionsRenderRequest(model, prompt string) TextCompletionsRenderRequest {
	return TextCompletionsRenderRequest{
		baseRenderRequest: baseRenderRequest{
			Model:    model,
			Endpoint: "/v1/completions",
		},
		Prompt: prompt,
	}
}

func NewChatCompletionsRenderRequest(model string, messages []ChatComplMessage) ChatCompletionsRenderRequest {
	return ChatCompletionsRenderRequest{
		baseRenderRequest: baseRenderRequest{
			Model:    model,
			Endpoint: "/v1/chat/completions",
		},
		Messages: messages,
	}
}

type baseRenderRequest struct {
	Model    string `json:"model"`
	Endpoint string `json:"-"`
}

func (b *baseRenderRequest) GetModel() string {
	return b.Model
}

func (b *baseRenderRequest) GetEndpoint() string {
	return b.Endpoint
}

func (b *baseRenderRequest) IsMultiModal() bool {
	return false
}

// TextCompletionsRenderRequest contains text completions render request related information
type TextCompletionsRenderRequest struct {
	baseRenderRequest

	// Prompt defines request's content
	Prompt string `json:"prompt"`
}

// MarshalForRenderer creates a minimal JSON payload for the renderer
// containing only model and prompt fields
func (t *TextCompletionsRenderRequest) MarshalForRenderer() ([]byte, error) {
	return json.Marshal(t)
}

// ChatCompletionsRenderRequest contains chat completions render request related information
type ChatCompletionsRenderRequest struct {
	baseRenderRequest

	// Messages list of request's Messages
	Messages []ChatComplMessage `json:"messages"`
}

func (c *ChatCompletionsRenderRequest) IsMultiModal() bool {
	for _, msg := range c.Messages {
		for _, block := range msg.Content.Structured {
			if block.Type == "image_url" {
				return true
			}
		}
	}
	return false
}

// MarshalForRenderer creates a minimal JSON payload for the renderer
// containing only model and messages fields with all their inner fields
func (c *ChatCompletionsRenderRequest) MarshalForRenderer() ([]byte, error) {
	return json.Marshal(c)
}
