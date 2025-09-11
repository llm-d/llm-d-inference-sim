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

package vllmapi

import (
	"strings"

	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

// TokenizeRequest is a request to /tokenize endpoint.
// Should contain either a prompt or messages, not both.
type TokenizeRequest struct {
	// Model is the model for tokenization
	Model string `json:"model"`
	// Prompt is the text to tokenize
	Prompt string `json:"prompt"`
	// Messages is an array of messages to tokenize
	Messages []openaiserverapi.Message `json:"messages"`
}

// GetPrompt returns the text to tokenize, either the text prompt
// or the concatenation of the messages (we reject requests with both
// prompt and messages set).
func (t *TokenizeRequest) GetPrompt() string {
	if t.Prompt != "" {
		return t.Prompt
	}

	messages := make([]string, 0)
	for _, message := range t.Messages {
		messages = append(messages, message.Content.PlainText())
	}
	return strings.Join(messages, " ")
}

// TokenizeResponse is a response for tokenize request
type TokenizeResponse struct {
	// MaxModelLen is max model length as dfined in the configuration
	MaxModelLen int `json:"max_model_len"`
	// Count is the number of returned tokens
	Count int `json:"count"`
	// Tokens are an array of tokens - the result of request tokenization
	Tokens []uint32 `json:"tokens"`
	// TokenStrs is currently unsupported, will allways be null
	TokenStrs []int `json:"token_strs"`
}
