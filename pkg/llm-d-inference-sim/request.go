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
package llmdinferencesim

import (
	"strings"
	"sync"

	"github.com/valyala/fasthttp"
)

// completionRequest interface representing both completion request types (text and chat)
type completionRequest interface {
	// createResponseText creates and returns response payload based on this request,
	// and the finish reason
	createResponseText(mode string) (string, string, error)
	// createToolCalls creates and returns response payload based on this request
	// (tool calls or nothing in case we randomly choose not to generate calls), the finish reason,
	// and the number of generated completion tokens
	createToolCalls() ([]toolCall, string, int, error)
	// isStream returns boolean that defines is response should be streamed
	isStream() bool
	// getModel returns model name as defined in the request
	getModel() string
	// includeUsage returns true if usage statistics should be include in the response
	includeUsage() bool
	// getNumberOfPromptTokens returns the number of tokens in the prompt
	getNumberOfPromptTokens() int
	// getTools() returns tools to use (in chat completion)
	getTools() []tool
	// getToolChoice() returns tool choice (in chat completion)
	getToolChoice() string
}

// baseCompletionRequest contains base completion request related information
type baseCompletionRequest struct {
	// Stream is a boolean value, defines whether response should be sent as a Stream
	Stream bool `json:"stream"`
	// StreamOptions defines streaming options in case Stream is set to true
	StreamOptions streamOptions `json:"stream_options"`
	// Model defines Model name to use for "inference", could be base Model name or one of available LoRA adapters
	Model string `json:"model"`
}

// StreamOptions defines streaming options for streaming requests
type streamOptions struct {
	// IncludeUsage is a boolean value, defines whether response contain usage statistics
	IncludeUsage bool `json:"include_usage"`
}

func (b *baseCompletionRequest) isStream() bool {
	return b.Stream
}

func (b *baseCompletionRequest) getModel() string {
	return b.Model
}

func (b *baseCompletionRequest) includeUsage() bool {
	return !b.Stream || b.StreamOptions.IncludeUsage
}

// completionReqCtx is a context passed in the simulator's flow, it contains the request data needed
// to generate the simulator's response
type completionReqCtx struct {
	completionReq    completionRequest
	httpReqCtx       *fasthttp.RequestCtx
	isChatCompletion bool
	wg               *sync.WaitGroup
}

// chatCompletionRequest defines structure of /chat/completion request
type chatCompletionRequest struct {
	baseCompletionRequest
	// Messages list of request's Messages
	Messages []message `json:"messages"`

	// The maximum number of tokens that can be generated in the chat
	// completion. This value can be used to control costs for text
	// generated via API.
	// This value is now deprecated in favor of max_completion_tokens
	// and is not compatible with o1 series models.
	MaxTokens *int64 `json:"max_tokens"`

	// An upper bound for the number of tokens that can be
	// generated for a completion, including visible output
	// tokens and reasoning tokens.
	MaxCompletionTokens *int64 `json:"max_completion_tokens"`

	// Tools is a list of tools the model may call.
	Tools []tool `json:"tools,omitempty"`

	// ToolChoice controls which (if any) tool is called by the model,
	// possible values: none, auto, required.
	// Sending an object with a specific tool, is currently not supported.
	ToolChoice string `json:"tool_choice,omitempty"`
}

// function defines a tool
type function struct {
	// Name is the function's name
	Name string `json:"name"`
	// Parameters are the parameters the function accepts
	Parameters map[string]any `json:"parameters,omitempty"`
	// Description is the function's description
	Description string `json:"description"`
}

// tool defines a tool to use in chat completion
type tool struct {
	// Function describes the tool
	Function function `json:"function"`
	// Type defines the type of the tool, currently only functions are
	// supported by vLLM
	Type string `json:"type"`
}

func (c *chatCompletionRequest) getNumberOfPromptTokens() int {
	var messages string
	for _, message := range c.Messages {
		messages += message.Content.PlainText() + " "
	}
	return len(strings.Fields(messages))
}

func (c *chatCompletionRequest) getTools() []tool {
	return c.Tools
}

func (c *chatCompletionRequest) getToolChoice() string {
	return c.ToolChoice
}

// getLastUserMsg returns last message from this request's messages with user role,
// if does not exist - returns an empty string
func (req *chatCompletionRequest) getLastUserMsg() string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == roleUser {
			return req.Messages[i].Content.PlainText()
		}
	}

	return ""
}

// v1/completion
// textCompletionRequest defines structure of /completion request
type textCompletionRequest struct {
	baseCompletionRequest
	// Prompt defines request's content
	Prompt string `json:"prompt"`

	// The maximum number of [tokens](/tokenizer) that can be generated in the
	// completion.
	//
	// The token count of your prompt plus `max_tokens` cannot exceed the model's
	// context length.
	MaxTokens *int64 `json:"max_tokens"`
}

func (t *textCompletionRequest) getNumberOfPromptTokens() int {
	return len(strings.Fields(t.Prompt))
}

func (c *textCompletionRequest) getTools() []tool {
	return nil
}

func (c *textCompletionRequest) getToolChoice() string {
	return ""
}
