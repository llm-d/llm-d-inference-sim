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
	"errors"
	"fmt"
	"strings"

	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
)

const (
	RoleAssistant      = "assistant"
	RoleUser           = "user"
	inputItemMessage   = "message"
	ResponsesInputText = "input_text"
)

// Request defines an interface for request information retrieval
type Request interface {
	// GetRequestID returns the unique request id
	GetRequestID() string
	// SetRequestID sets the request id
	SetRequestID(id string)
	// IsStream returns boolean that defines is response should be streamed
	IsStream() bool
	// GetModel returns model name as defined in the request
	GetModel() string
	// GetDisplayedModel returns model name to be used in the processing
	// in case served model names were defined on the simulator load, and this request
	// contains one of aliases of the base model - the first alias is used as the model name
	// in all other cases - the model name is used as is from the request
	GetDisplayedModel() string
	// SetDisplayedModel sets the displayed model name for the request
	SetDisplayedModel(model string)
	// GetLoraName returns the LoRA name or nil if model is the base model
	GetLoraName() *string
	GetLoraID() *int
	// SetModelLoraID sets the LoRA ID for the request, ID == 0 means the base model
	SetModelLoraID(id int)
	// IncludeUsage returns true if usage statistics should be include in the response
	IncludeUsage() bool
	// GetNumberOfCachedPromptTokens returns the number of tokens in the prompt that are
	// in the local KV Cache
	GetNumberOfCachedPromptTokens() int
	// SetNumberOfCachedPromptTokens sets the number of tokens in the prompt that are
	// in the local KV Cache
	SetNumberOfCachedPromptTokens(cachedPromptTokens int)
	// GetTools returns tools to use (in chat completions)
	GetTools() []Tool
	// GetToolChoice returns tool choice (in chat completions)
	GetToolChoice() ToolChoice
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
	// ExtractMaxTokens extracts the max tokens from the request:
	// for chat completions - max_completion_tokens field is used
	// for text completions - max_tokens field is used
	ExtractMaxTokens() *int64
	// GetLogprobs returns nil if no logprobs needed, or pointer to number of logprob options to include
	GetLogprobs() *int
	// GetCacheHitThreshold returns the cache hit threshold (0-1) or nil if not set
	GetCacheHitThreshold() *float64
	// TokenizedPrompt returns the tokenized prompt
	TokenizedPrompt() *Tokenized
	// SetTokenizedPrompt sets the tokenized prompt
	SetTokenizedPrompt(tokenized *Tokenized)
	// TokenizedPromptForEcho returns the tokenized response in echo mode
	TokenizedPromptForEcho() *Tokenized
	// SetTokenizedPromptForEcho sets the tokenized response in echo mode
	SetTokenizedPromptForEcho(tokenized *Tokenized)
	// MMFeatures returns the multimodal features
	MMFeatures() *tokenization.MultiModalFeatures
	// SetMMFeatures sets the multimodal features
	SetMMFeatures(mmFeatures *tokenization.MultiModalFeatures)

	// CacheThresholdFinishReason returns cacheThresholdFinishReason,  when true,
	// forces a cache_threshold finish reason
	CacheThresholdFinishReason() bool
	// SetCacheThresholdFinishReason sets cacheThresholdFinishReason
	SetCacheThresholdFinishReason(bool)
}

// baseRequest contains base completions request related information
type baseRequest struct {
	// RequestID is the unique id of this request
	RequestID string
	// Model defines Model name to use for "inference",
	// could be base Model name or one of available LoRA adapters
	Model string `json:"model"`
	// DisplayedModel is the model name to be used in the request processing and in the response,
	// in case served model names were defined on the simulator load, and this request contains one of aliases of the base model - the first alias is used as the DisplayedModel name
	// in all other cases - the Model name is used as is from the request
	DisplayedModel string
	// ID of the LoRA adapter if the model is a LoRA, 0 if the model is the base model
	loraID int
	// Stream is a boolean value, defines whether response should be sent as a Stream
	Stream bool `json:"stream"`
	// KVParams kv transfer related fields
	KVParams *KVTransferParams `json:"kv_transfer_params"`
	// The number of tokens in the prompt that are in the local KV Cache
	cachedPromptTokens int
	// IgnoreEOS is a boolean value, true when the model should ignore end-of-sequence tokens
	IgnoreEOS bool `json:"ignore_eos"`
	// tokenizedPrompt is the tokenized prompt
	tokenizedPrompt *Tokenized
	// tokenizedPromptForEcho is the tokenized part of the prompt to be used in echo mode, exists only in echo mode
	tokenizedPromptForEcho *Tokenized
	// mmFeatures holds multimodal metadata produced by the tokenizer, exists only for multimodal requests
	mmFeatures *tokenization.MultiModalFeatures
}

// baseCompletionsRequest contains base completions request related information
type baseCompletionsRequest struct {
	baseRequest
	// StreamOptions defines streaming options in case Stream is set to true
	StreamOptions StreamOptions `json:"stream_options"`
	// CacheHitThreshold is a value between 0 and 1 that specifies the minimum cache hit rate required
	// to proceed with request processing. If the actual cache hit rate is below this threshold,
	// the request will return with cache_threshold finish reason.
	CacheHitThreshold *float64 `json:"cache_hit_threshold,omitempty"`
	// cacheThresholdFinishReason is a boolean value extracted from the request's HTTP header,
	//  when true, forces a cache_threshold finish reason
	cacheThresholdFinishReason bool
}

type KVTransferParams struct {
	// DoRemoteDecode boolean value, true when request's decode will be done on remote pod
	DoRemoteDecode bool `json:"do_remote_decode"`
	// DoRemotePrefill boolean value, true when request's prefill was done on remote pod
	DoRemotePrefill bool `json:"do_remote_prefill"`
	// RemoteEngineId is an identifier of the remote inference engine or backend to use for processing requests
	RemoteEngineId string `json:"remote_engine_id"`
	// RemoteBlockIds is a list of block identifiers to process remotely for distributed decoding
	RemoteBlockIds []string `json:"remote_block_ids"`
	// RemoteHost is a hostname or IP address of the remote server handling prefill
	RemoteHost string `json:"remote_host"`
	// RemotePort is a port of the remote server handling prefill
	RemotePort int `json:"remote_port"`
	// TPSize is the tensor parallelism size for KV cache transfer
	TPSize int `json:"tp_size" default:"1"`
}

// StreamOptions defines streaming options for streaming requests
type StreamOptions struct {
	// IncludeUsage is a boolean value, defines whether response contain usage statistics
	IncludeUsage bool `json:"include_usage"`
}

// Tokenized is the tokenized representation with numerical and string tokens
type Tokenized struct {
	Tokens  []uint32
	Strings []string
}

// Length returns the number of tokens of the Tokenized
func (t *Tokenized) Length() int {
	if len(t.Tokens) != 0 {
		return len(t.Tokens)
	}
	return len(t.Strings)
}

func (t *Tokenized) Trim(maxLen int) {
	if len(t.Tokens) > maxLen {
		t.Tokens = t.Tokens[:maxLen]
	}
	if len(t.Strings) > maxLen {
		t.Strings = t.Strings[:maxLen]
	}
}

func (t *Tokenized) Append(other Tokenized) {
	t.Strings = append(t.Strings, other.Strings...)
	t.Tokens = append(t.Tokens, other.Tokens...)
}

func (b *baseRequest) GetRequestID() string {
	return b.RequestID
}

func (b *baseRequest) SetRequestID(id string) {
	b.RequestID = id
}

func (b *baseRequest) IsStream() bool {
	return b.Stream
}

func (b *baseRequest) GetModel() string {
	return b.Model
}

func (b *baseRequest) GetDisplayedModel() string {
	return b.DisplayedModel
}

func (b *baseRequest) SetDisplayedModel(model string) {
	b.DisplayedModel = model
}

func (b *baseRequest) GetLoraName() *string {
	if b.loraID > 0 {
		name := b.Model
		return &name
	}
	return nil
}

func (b *baseRequest) GetLoraID() *int {
	if b.loraID > 0 {
		id := b.loraID
		return &id
	}
	return nil
}

func (b *baseRequest) SetModelLoraID(id int) {
	b.loraID = id
}

func (b *baseRequest) IncludeUsage() bool {
	return true
}

func (b *baseRequest) IsDoRemoteDecode() bool {
	return b.KVParams != nil && b.KVParams.DoRemoteDecode
}

func (b *baseRequest) IsDoRemotePrefill() bool {
	return b.KVParams != nil && b.KVParams.DoRemotePrefill
}

// GetNumberOfCachedPromptTokens returns the number of tokens in the prompt that are
// in the local KV Cache
func (b *baseRequest) GetNumberOfCachedPromptTokens() int {
	return b.cachedPromptTokens
}

// GetIgnoreEOS returns the value of IgnoreEOS
func (b *baseRequest) GetIgnoreEOS() bool {
	return b.IgnoreEOS
}

// SetIgnoreEOS sets the value of IgnoreEOS
func (b *baseRequest) SetIgnoreEOS(ignorEOS bool) {
	b.IgnoreEOS = ignorEOS
}

// SetNumberOfCachedPromptTokens sets the number of tokens in the prompt that are
// in the local KV Cache
func (b *baseRequest) SetNumberOfCachedPromptTokens(cachedPromptTokens int) {
	b.cachedPromptTokens = cachedPromptTokens
}

// GetCacheHitThreshold returns the cache hit threshold value
func (b *baseRequest) GetCacheHitThreshold() *float64 {
	return nil
}

// CacheThresholdFinishReason returns cacheThresholdFinishReason,  when true,
// forces a cache_threshold finish reason
func (b *baseRequest) CacheThresholdFinishReason() bool {
	return false
}

// SetCacheThresholdFinishReason sets cacheThresholdFinishReason
func (b *baseRequest) SetCacheThresholdFinishReason(value bool) {
}

// TokenizedPrompt returns the tokenized prompt
func (b *baseRequest) TokenizedPrompt() *Tokenized {
	return b.tokenizedPrompt
}

// SetTokenizedPrompt sets the tokenized prompt
func (b *baseRequest) SetTokenizedPrompt(tokenized *Tokenized) {
	b.tokenizedPrompt = tokenized
}

// TokenizedPromptForEcho returns the tokenized response in echo mode
func (b *baseRequest) TokenizedPromptForEcho() *Tokenized {
	return b.tokenizedPromptForEcho
}

// SetTokenizedPromptForEcho sets the tokenized response in echo mode
func (b *baseRequest) SetTokenizedPromptForEcho(tokenized *Tokenized) {
	b.tokenizedPromptForEcho = tokenized
}

// TokenizedPrompt returns the tokenized prompt
func (b *baseRequest) MMFeatures() *tokenization.MultiModalFeatures {
	return b.mmFeatures
}

// SetMMFeatures sets the multimodal features
func (b *baseRequest) SetMMFeatures(mmFeatures *tokenization.MultiModalFeatures) {
	b.mmFeatures = mmFeatures
}

func (b *baseCompletionsRequest) IncludeUsage() bool {
	return !b.Stream || b.StreamOptions.IncludeUsage
}

// GetCacheHitThreshold returns the cache hit threshold value
func (b *baseCompletionsRequest) GetCacheHitThreshold() *float64 {
	return b.CacheHitThreshold
}

// CacheThresholdFinishReason returns cacheThresholdFinishReason,  when true,
// forces a cache_threshold finish reason
func (b *baseCompletionsRequest) CacheThresholdFinishReason() bool {
	return b.cacheThresholdFinishReason
}

// SetCacheThresholdFinishReason sets cacheThresholdFinishReason
func (b *baseCompletionsRequest) SetCacheThresholdFinishReason(value bool) {
	b.cacheThresholdFinishReason = value
}

// ChatCompletionsRequest defines structure of /chat/completions request
type ChatCompletionsRequest struct {
	baseCompletionsRequest
	// Messages list of request's Messages
	Messages []ChatComplMessage `json:"messages"`

	// The maximum number of tokens that can be generated in the chat
	// completions. This value can be used to control costs for text
	// generated via API.
	// This value is now deprecated in favor of max_completion_tokens
	// and is not compatible with o1 series models.
	MaxTokens *int64 `json:"max_tokens"`

	// An upper bound for the number of tokens that can be
	// generated for a completions, including visible output
	// tokens and reasoning tokens.
	MaxCompletionTokens *int64 `json:"max_completion_tokens"`

	// Tools is a list of tools the model may call.
	Tools []Tool `json:"tools,omitempty"`

	// ToolChoice controls which (if any) tool is called by the model.
	// It can be a string ("none", "auto", "required") or an object specifying the function.
	ToolChoice ToolChoice `json:"tool_choice,omitzero"`

	// Logprobs controls whether log probabilities are included in the response
	Logprobs bool `json:"logprobs,omitempty"`

	// TopLogprobs controls how many alternative tokens to include in the logprobs
	TopLogprobs *int `json:"top_logprobs,omitempty"`
}

var _ Request = (*ChatCompletionsRequest)(nil)

// function defines a tool
type function struct {
	// Name is the function's name
	Name string `json:"name"`
	// Parameters are the parameters the function accepts
	Parameters map[string]any `json:"parameters,omitempty"`
	// Description is the function's description
	Description string `json:"description"`
}

// Tool defines a Tool to use in chat completions
type Tool struct {
	// Function describes the tool
	Function function `json:"function"`
	// Type defines the type of the tool, currently only functions are
	// supported by vLLM
	Type string `json:"type"`
}

func (c *ChatCompletionsRequest) GetTools() []Tool {
	return c.Tools
}

func (c *ChatCompletionsRequest) GetToolChoice() ToolChoice {
	return c.ToolChoice
}

func (c *ChatCompletionsRequest) GetMaxCompletionTokens() *int64 {
	if c.MaxCompletionTokens != nil {
		return c.MaxCompletionTokens
	}
	return c.MaxTokens
}

// ExtractMaxTokens extracts the max tokens from the request
// for chat completions - max_completion_tokens field is used
func (req *ChatCompletionsRequest) ExtractMaxTokens() *int64 {
	return req.GetMaxCompletionTokens()
}

func (c *ChatCompletionsRequest) GetLogprobs() *int {
	if !c.Logprobs {
		return nil // No logprobs requested
	}
	if c.TopLogprobs != nil {
		return c.TopLogprobs // Return the top_logprobs value
	}
	// Default to 1 if logprobs=true but no top_logprobs specified
	defaultVal := 1
	return &defaultVal
}

// v1/completions
// TextCompletionsRequest defines structure of /completions request
type TextCompletionsRequest struct {
	baseCompletionsRequest
	// Prompt defines request's content
	Prompt string `json:"prompt"`

	// The maximum number of [tokens](/tokenizer) that can be generated in the
	// completions.
	//
	// The token count of your prompt plus `max_tokens` cannot exceed the model's
	// context length.
	MaxTokens *int64 `json:"max_tokens"`

	// Logprobs includes the log probabilities on the logprobs most likely tokens,
	// as well the chosen tokens. For example, if logprobs is 5, the API will return
	// a list of the 5 most likely tokens. The API will always return the logprob
	// of the sampled token, so there may be up to logprobs+1 elements in the response.
	Logprobs *int `json:"logprobs,omitempty"`
}

var _ Request = (*TextCompletionsRequest)(nil)

func (c *TextCompletionsRequest) GetTools() []Tool {
	return nil
}

func (c *TextCompletionsRequest) GetToolChoice() ToolChoice {
	return ToolChoice{}
}

func (c *TextCompletionsRequest) GetMaxCompletionTokens() *int64 {
	return c.MaxTokens
}

// ExtractMaxTokens extracts the max tokens from the request
// for text completions - max_tokens field is used
func (req *TextCompletionsRequest) ExtractMaxTokens() *int64 {
	return req.MaxTokens
}

func (t *TextCompletionsRequest) GetLogprobs() *int {
	return t.Logprobs
}

// GenerationRequest defines structure of generation request
type GenerationRequest struct {
	baseRequest
	// Prompt defines request's content
	Prompt string

	// The maximum number of [tokens](/tokenizer) that can be generated in the
	// completions.
	//
	// The token count of your prompt plus `max_tokens` cannot exceed the model's
	// context length.
	MaxTokens *int64
}

var _ Request = (*GenerationRequest)(nil)

func (t *GenerationRequest) GetPrompt() string {
	return t.Prompt
}

func (c *GenerationRequest) GetTools() []Tool {
	return nil
}

func (c *GenerationRequest) GetToolChoice() ToolChoice {
	return ToolChoice{}
}

func (c *GenerationRequest) GetMaxCompletionTokens() *int64 {
	return c.MaxTokens
}

func (t *GenerationRequest) GetFullPrompt() string {
	return t.Prompt
}

// ExtractMaxTokens extracts the max tokens from the request
// for text completions - max_tokens field is used
func (req *GenerationRequest) ExtractMaxTokens() *int64 {
	return req.MaxTokens
}

func (t *GenerationRequest) GetLogprobs() *int {
	return nil
}

func NewGenerationRequest(requestID string, stream bool, model string, maxTokens *int64) *GenerationRequest {
	return &GenerationRequest{
		baseRequest: baseRequest{
			RequestID: requestID,
			Stream:    stream,
			Model:     model,
		},
		MaxTokens: maxTokens,
	}
}

// Responses

type ResponsesRequest struct {
	baseRequest
	Input           []InputItem `json:"input,omitempty"`
	Instructions    string      `json:"instructions,omitempty"`
	MaxOutputTokens *int64      `json:"max_output_tokens,omitempty"`
	// Ignored for now, always text
	Text *TextSettings `json:"text,omitempty"`
}

var _ Request = (*ResponsesRequest)(nil)

type TextSettings struct {
	Format *TextFormat `json:"format,omitempty"`
}

type TextFormat struct {
	Type       string          `json:"type"` // text, json_object, json_schema
	JsonSchema *JSONSchemaSpec `json:"json_schema,omitempty"`
}

type JSONSchemaSpec struct {
	Name   string         `json:"name"`
	Strict *bool          `json:"strict,omitempty"`
	Schema map[string]any `json:"schema"`
}

type InputItem interface {
	isInputItem()
	json.Unmarshaler
}

type InputMessage struct {
	Type    string         `json:"type"` // always "message"
	Role    string         `json:"role"` // user, system, developer
	Status  string         `json:"status,omitempty"`
	Content []InputContent `json:"content"`
}

func (InputMessage) isInputItem() {}

func (m *InputMessage) UnmarshalJSON(data []byte) error {
	var raw struct {
		Type    string          `json:"type"`
		Role    string          `json:"role"`
		Status  string          `json:"status"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if raw.Type != "" && raw.Type != inputItemMessage {
		return fmt.Errorf("unsupported input item type %q", raw.Type)
	}
	m.Type = inputItemMessage
	m.Role = raw.Role
	m.Status = raw.Status

	if len(raw.Content) == 0 || string(raw.Content) == "null" {
		return nil
	}
	// content can be a plain string or an array of InputContent objects
	var str string
	if err := json.Unmarshal(raw.Content, &str); err == nil {
		m.Content = []InputContent{{Type: ResponsesInputText, Text: str}}
		return nil
	}
	return json.Unmarshal(raw.Content, &m.Content)
}

func (m *InputMessage) ReadableText() string {
	var parts []string
	for _, c := range m.Content {
		switch c.Type { // nolint
		case ResponsesInputText:
			parts = append(parts, c.Text)
		}
	}
	return strings.Join(parts, "\n")
}

type InputContent struct {
	Type string `json:"type"` // input_text
	Text string `json:"text"`
}

func (c *InputContent) UnmarshalJSON(data []byte) error {
	var raw struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if raw.Type != "" && raw.Type != ResponsesInputText {
		return fmt.Errorf("unsupported input content type %q", raw.Type)
	}
	c.Type = ResponsesInputText
	c.Text = raw.Text
	return nil
}

// At the moment UnmarshalJSON handles only two forms of the `input` field:
// - a plain string: wrapped into a single user InputMessage
// - an array of message objects: each element is unmarshaled as *InputMessage
func (req *ResponsesRequest) UnmarshalJSON(data []byte) error {
	// Use an alias to unmarshal all fields except Input normally.
	type alias struct {
		baseRequest
		Input           json.RawMessage `json:"input,omitempty"`
		Instructions    string          `json:"instructions,omitempty"`
		MaxOutputTokens *int64          `json:"max_output_tokens,omitempty"`
		Text            *TextSettings   `json:"text,omitempty"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	req.baseRequest = a.baseRequest
	req.Instructions = a.Instructions
	req.MaxOutputTokens = a.MaxOutputTokens
	req.Text = a.Text

	if len(a.Input) == 0 || string(a.Input) == "null" {
		return errors.New("input is required")
	}

	// string input: wrap as a single user message
	var str string
	if err := json.Unmarshal(a.Input, &str); err == nil {
		req.Input = []InputItem{&InputMessage{
			Type:    inputItemMessage,
			Role:    RoleUser,
			Content: []InputContent{{Type: ResponsesInputText, Text: str}},
		}}
		return nil
	}

	// array input: unmarshal each element as *InputMessage
	var raw []json.RawMessage
	if err := json.Unmarshal(a.Input, &raw); err != nil {
		return fmt.Errorf("input must be a string or array: %w", err)
	}
	req.Input = make([]InputItem, 0, len(raw))
	for _, r := range raw {
		msg := &InputMessage{}
		if err := json.Unmarshal(r, msg); err != nil {
			return err
		}
		req.Input = append(req.Input, msg)
	}
	return nil
}

func (req *ResponsesRequest) GetTools() []Tool {
	return nil
}

func (req *ResponsesRequest) GetToolChoice() ToolChoice {
	return ToolChoice{}
}

func (req *ResponsesRequest) GetMaxCompletionTokens() *int64 {
	return req.ExtractMaxTokens()
}

func (req *ResponsesRequest) GetLogprobs() *int {
	return nil
}

func (req *ResponsesRequest) ExtractMaxTokens() *int64 {
	return req.MaxOutputTokens
}
