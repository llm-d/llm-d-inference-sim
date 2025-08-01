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

// Contains structures and functions related to responses for all supported APIs
package openaiserverapi

import (
	"encoding/json"
	"errors"
	"strings"
)

// CompletionResponse interface representing both completion response types (text and chat)
type CompletionResponse interface{}

// BaseCompletionResponse contains base completion response related information
type BaseCompletionResponse struct {
	// ID defines the response ID
	ID string `json:"id"`
	// Created defines the response creation timestamp
	Created int64 `json:"created"`
	// Model defines the Model name for current request
	Model string `json:"model"`
	// Usage contains the token usage statistics for the request
	Usage *Usage `json:"usage"`
	// Object is the Object type, "text_completion", "chat.completion", or "chat.completion.chunk"
	Object string `json:"object"`
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
}

// Usage contains token Usage statistics
type Usage struct {
	// PromptTokens is the number of tokens in the prompt
	PromptTokens int `json:"prompt_tokens"`
	// CompletionTokens is the number of tokens generated by the model as the response
	CompletionTokens int `json:"completion_tokens"`
	// TotalTokens is the total number of tokens processed for the request (the sum of the two values above)
	TotalTokens int `json:"total_tokens"`
}

// ChatCompletionResponse defines structure of /chat/completion response
type ChatCompletionResponse struct {
	BaseCompletionResponse
	// Choices list of Choices of the response, according of OpenAI API
	Choices []ChatRespChoice `json:"choices"`
}

// BaseResponseChoice contains base completion response's choice related information
type BaseResponseChoice struct {
	// Index defines completion response choise Index
	Index int `json:"index"`
	// FinishReason defines finish reason for response or for chunks, for not last chinks is defined as null
	FinishReason *string `json:"finish_reason"`
}

// v1/chat/completion
// Message defines vLLM chat completion Message
type Message struct {
	// Role is the message Role, optional values are 'user', 'assistant', ...
	Role string `json:"role,omitempty"`
	// Content defines text of this message
	Content Content `json:"content,omitempty"`
	// ToolCalls are the tool calls created by the model
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type Content struct {
	Raw        string
	Structured []ContentBlock
}

type ContentBlock struct {
	Type     string     `json:"type"`
	Text     string     `json:"text,omitempty"`
	ImageURL ImageBlock `json:"image_url,omitempty"`
}

type ImageBlock struct {
	Url string `json:"url,omitempty"`
}

// UnmarshalJSON allow use both format
func (mc *Content) UnmarshalJSON(data []byte) error {
	// Raw format
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		mc.Raw = str
		return nil
	}

	// Block format
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		mc.Structured = blocks
		return nil
	}

	return errors.New("content format not supported")
}

func (mc Content) MarshalJSON() ([]byte, error) {
	if mc.Raw != "" {
		return json.Marshal(mc.Raw)
	}
	if mc.Structured != nil {
		return json.Marshal(mc.Structured)
	}
	return json.Marshal("")
}

func (mc Content) PlainText() string {
	if mc.Raw != "" {
		return mc.Raw
	}
	var sb strings.Builder
	for _, block := range mc.Structured {
		if block.Type == "text" {
			sb.WriteString(block.Text)
			sb.WriteString(" ")
		}
	}
	return sb.String()
}

// FunctionCall defines a tool call generated by the model including its arguments
type FunctionCall struct {
	// Name is the function's name, can be null in streaming in not the first chunk
	Name *string `json:"name"`
	// Arguments are the arguments of the function call
	Arguments string `json:"arguments,omitempty"`
	// TokenizedArguments is an array of tokenized arguments
	TokenizedArguments []string
}

// ToolCall defines a tool call generated by the model
type ToolCall struct {
	// Function is a tool call generated by the model
	Function FunctionCall `json:"function"`
	// ID is the ID of the tool call
	ID string `json:"id"`
	// Type is the type of the tool, only functions are supported
	Type string `json:"type"`
	// Index is the index of the tool in the sequence of tools generated by the model
	Index int `json:"index"`
}

// ChatRespChoice represents a single chat completion response choise
type ChatRespChoice struct {
	BaseResponseChoice
	// Message contains choice's Message
	Message Message `json:"message"`
}

// TextCompletionResponse defines structure of /completion response
type TextCompletionResponse struct {
	BaseCompletionResponse
	// Choices list of Choices of the response, according of OpenAI API
	Choices []TextRespChoice `json:"choices"`
}

// TextRespChoice represents a single text completion response choise
type TextRespChoice struct {
	BaseResponseChoice
	// Text defines request's content
	Text string `json:"text"`
}

// CompletionRespChunk is an interface that defines a single response chunk
type CompletionRespChunk interface{}

// ChatCompletionRespChunk is a single chat completion response chunk
type ChatCompletionRespChunk struct {
	BaseCompletionResponse
	// Choices list of Choices of the response, according of OpenAI API
	Choices []ChatRespChunkChoice `json:"choices"`
}

// ChatRespChunkChoice represents a single chat completion response choise in case of streaming
type ChatRespChunkChoice struct {
	BaseResponseChoice
	// Delta is a content of the chunk
	Delta Message `json:"delta"`
}

// CompletionError defines the simulator's response in case of an error
type CompletionError struct {
	// Object is a type of this Object, "error"
	Object string `json:"object"`
	// Message is an error Message
	Message string `json:"message"`
	// Type is a type of the error
	Type string `json:"type"`
	// Params is the error's parameters
	Param *string `json:"param"`
	// Code is http status Code
	Code int `json:"code"`
}
