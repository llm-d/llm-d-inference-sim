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
	"sync"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/valyala/fasthttp"
)

const (
	RoleAssistant = "assistant"
	RoleUser      = "user"
)

// CompletionRequest interface representing both completion request types (text and chat)
type CompletionRequest interface {
	// GetRequestID returns the unique request id
	GetRequestID() string
	// IsStream returns boolean that defines is response should be streamed
	IsStream() bool
	// GetModel returns model name as defined in the request
	GetModel() string
	// IncludeUsage returns true if usage statistics should be include in the response
	IncludeUsage() bool
	// GetNumberOfPromptTokens returns the number of tokens in the prompt
	GetNumberOfPromptTokens() int
	// GetNumberOfCachedPromptTokens returns the number of tokens in the prompt that are
	// in the local KV Cache
	GetNumberOfCachedPromptTokens() int
	// SetNumberOfCachedPromptTokens sets the number of tokens in the prompt that are
	// in the local KV Cache
	SetNumberOfCachedPromptTokens(cachedPromptTokens int)
	// GetPrompt returns the prompt
	GetPrompt() string
	// GetTools() returns tools to use (in chat completion)
	GetTools() []Tool
	// GetToolChoice() returns tool choice (in chat completion)
	GetToolChoice() string
	// GetMaxCompletionTokens returns the maximum completion tokens requested
	GetMaxCompletionTokens() *int64
	// GetIgnoreEOS returns true if the end-of-sequence tokens will be ignored
	GetIgnoreEOS() bool
	// IsDoRemoteDecode() returns true if do_remote_decode field is true in the request,
	// when the field is true, the decode phase should be done on remote pod,
	// whereas prefill phase is done on local pod, thus this is a prefill request
	IsDoRemoteDecode() bool
	// IsDoRemotePrefill() returns true if do_remote_prefill field is true in the request,
	// when the field is true, the prefill phase should be done on remote pod,
	// whereas decode phase is done on local pod, thus this is a decode request
	IsDoRemotePrefill() bool
	// GetFullPrompt returns the full prompt including system and user prompts
	GetFullPrompt() string
}

// BaseCompletionRequest contains base completion request related information
type BaseCompletionRequest struct {
	// RequestID is the unique id of this request
	RequestID string
	// Stream is a boolean value, defines whether response should be sent as a Stream
	Stream bool `json:"stream"`
	// StreamOptions defines streaming options in case Stream is set to true
	StreamOptions StreamOptions `json:"stream_options"`
	// Model defines Model name to use for "inference", could be base Model name or one of available LoRA adapters
	Model string `json:"model"`
	// DoRemoteDecode boolean value, true when request's decode will be done on remote pod
	DoRemoteDecode bool `json:"do_remote_decode"`
	// DoRemotePrefill boolean value, true when request's prefill was done on remote pod
	DoRemotePrefill bool `json:"do_remote_prefill"`
	// RemoteBlockIds is a list of block identifiers to process remotely for distributed decoding
	RemoteBlockIds []string `json:"remote_block_ids"`
	// RemoteEngineId is an identifier of the remote inference engine or backend to use for processing requests
	RemoteEngineId string `json:"remote_engine_id"`
	// RemoteHost is a hostname or IP address of the remote server handling prefill
	RemoteHost string `json:"remote_host"`
	// RemotePort is a port of the remote server handling prefill
	RemotePort int `json:"remote_port"`
	// The number of tokens in the prompt that are in the local KV Cache
	cachedPromptTokens int
	// IgnoreEOS is a boolean value, true when the model should ignore end-of-sequence tokens
	IgnoreEOS bool `json:"ignore_eos"`
}

// StreamOptions defines streaming options for streaming requests
type StreamOptions struct {
	// IncludeUsage is a boolean value, defines whether response contain usage statistics
	IncludeUsage bool `json:"include_usage"`
}

func (b *BaseCompletionRequest) GetRequestID() string {
	return b.RequestID
}

func (b *BaseCompletionRequest) IsStream() bool {
	return b.Stream
}

func (b *BaseCompletionRequest) GetModel() string {
	return b.Model
}

func (b *BaseCompletionRequest) IncludeUsage() bool {
	return !b.Stream || b.StreamOptions.IncludeUsage
}

func (b *BaseCompletionRequest) IsDoRemoteDecode() bool {
	return b.DoRemoteDecode
}

func (b *BaseCompletionRequest) IsDoRemotePrefill() bool {
	return b.DoRemotePrefill
}

// GetNumberOfCachedPromptTokens returns the number of tokens in the prompt that are
// in the local KV Cache
func (b *BaseCompletionRequest) GetNumberOfCachedPromptTokens() int {
	return b.cachedPromptTokens
}

// GetIgnoreEOS returns the value of IgnoreEOS
func (b *BaseCompletionRequest) GetIgnoreEOS() bool {
	return b.IgnoreEOS
}

// SetNumberOfCachedPromptTokens sets the number of tokens in the prompt that are
// in the local KV Cache
func (b *BaseCompletionRequest) SetNumberOfCachedPromptTokens(cachedPromptTokens int) {
	b.cachedPromptTokens = cachedPromptTokens
}

// CompletionReqCtx is a context passed in the simulator's flow, it contains the request data needed
// to generate the simulator's response
type CompletionReqCtx struct {
	CompletionReq    CompletionRequest
	HTTPReqCtx       *fasthttp.RequestCtx
	IsChatCompletion bool
	Wg               *sync.WaitGroup
}

// ChatCompletionRequest defines structure of /chat/completion request
type ChatCompletionRequest struct {
	BaseCompletionRequest
	// Messages list of request's Messages
	Messages []Message `json:"messages"`

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
	Tools []Tool `json:"tools,omitempty"`

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

// Tool defines a Tool to use in chat completion
type Tool struct {
	// Function describes the tool
	Function function `json:"function"`
	// Type defines the type of the tool, currently only functions are
	// supported by vLLM
	Type string `json:"type"`
}

func (c *ChatCompletionRequest) GetPrompt() string {
	var messages string
	for _, message := range c.Messages {
		messages += message.Content.PlainText() + " "
	}
	return messages
}

func (c *ChatCompletionRequest) GetNumberOfPromptTokens() int {
	return len(common.Tokenize(c.GetPrompt()))
}

func (c *ChatCompletionRequest) GetTools() []Tool {
	return c.Tools
}

func (c *ChatCompletionRequest) GetToolChoice() string {
	return c.ToolChoice
}

func (c *ChatCompletionRequest) GetMaxCompletionTokens() *int64 {
	if c.MaxCompletionTokens != nil {
		return c.MaxCompletionTokens
	}
	return c.MaxTokens
}

// getLastUserMsg returns last message from this request's messages with user role,
// if does not exist - returns an empty string
func (req *ChatCompletionRequest) GetLastUserMsg() string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == RoleUser {
			return req.Messages[i].Content.PlainText()
		}
	}

	return ""
}

func (req *ChatCompletionRequest) GetFullPrompt() string {
	prompt := ""
	for _, msg := range req.Messages {
		switch msg.Role {
		case RoleUser:
			prompt += "### user:\n" + msg.Content.Raw + "\n"
		case RoleAssistant:
			prompt += "### assistant:\n" + msg.Content.Raw + "\n"
		default:
			prompt += "### unknown:\n" + msg.Content.Raw + "\n"
		}
	}
	return prompt
}

// v1/completion
// TextCompletionRequest defines structure of /completion request
type TextCompletionRequest struct {
	BaseCompletionRequest
	// Prompt defines request's content
	Prompt string `json:"prompt"`

	// The maximum number of [tokens](/tokenizer) that can be generated in the
	// completion.
	//
	// The token count of your prompt plus `max_tokens` cannot exceed the model's
	// context length.
	MaxTokens *int64 `json:"max_tokens"`
}

func (t *TextCompletionRequest) GetPrompt() string {
	return t.Prompt
}

func (t *TextCompletionRequest) GetNumberOfPromptTokens() int {
	return len(common.Tokenize(t.GetPrompt()))
}

func (c *TextCompletionRequest) GetTools() []Tool {
	return nil
}

func (c *TextCompletionRequest) GetToolChoice() string {
	return ""
}

func (c *TextCompletionRequest) GetMaxCompletionTokens() *int64 {
	return c.MaxTokens
}

func (t *TextCompletionRequest) GetFullPrompt() string {
	return "### user:\n" + t.Prompt + "\n"
}
