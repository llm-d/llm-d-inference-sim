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

// Contains structures and functions related to requests for all supported APIs
package openaiserverapi

type RenderRequest interface {
	Model() string
	Endpoint() string
	IsMultiModal() bool
}

func NewTextCompletionsRenderRequest(model, prompt string) TextCompletionsRenderRequest {
	return TextCompletionsRenderRequest{
		baseRenderRequest: baseRenderRequest{
			ModelName: model,
			endpoint:  "/v1/completions",
		},
		Prompt: prompt,
	}
}

func NewChatCompletionsRenderRequest(model string, messages []Message) ChatCompletionsRenderRequest {
	return ChatCompletionsRenderRequest{
		baseRenderRequest: baseRenderRequest{
			ModelName: model,
			endpoint:  "/v1/chat/completions",
		},
		Messages: messages,
	}
}

type baseRenderRequest struct {
	ModelName string `json:"model"`
	endpoint  string `json:"-"`
}

func (b *baseRenderRequest) Model() string {
	return b.ModelName
}

func (b *baseRenderRequest) Endpoint() string {
	return b.endpoint
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

// ChatCompletionsRenderRequest contains chat completions render request related information
type ChatCompletionsRenderRequest struct {
	baseRenderRequest

	// Messages list of request's Messages
	Messages []Message `json:"messages"`
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

type RenderResponse struct {
	TokenIDs []uint32          `json:"token_ids"`
	Features *RenderMMFeatures `json:"features,omitempty"`
}

type RenderMMFeatures struct {
	MMHashes       map[string][]string            `json:"mm_hashes"`
	MMPlaceholders map[string][]RenderPlaceholder `json:"mm_placeholders"`
}

type RenderPlaceholder struct {
	Offset int `json:"offset"`
	Length int `json:"length"`
}
