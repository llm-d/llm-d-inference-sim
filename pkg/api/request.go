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
package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

const (
	RoleAssistant         = "assistant"
	RoleUser              = "user"
	inputItemMessage      = "message"
	ResponsesInputText    = "input_text"
	ResponsesInputImage   = "input_image"
	ResponsesInputAudio   = "input_audio"
	StartMessageSeparator = "### "
	EndMessageSeparator   = "\n"
	nullString            = "null"
	// ResponsesIncludeLogprobs is the include value that enables logprobs in the Responses API
	ResponsesIncludeLogprobs = "message.output_text.logprobs"
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
	// GetN returns the number of completion choices to generate, defaulting to 1
	GetN() int
	// GetRawN returns the raw n pointer from the request, nil when the field was
	// omitted. Used by validation to reject explicit n <= 0 while still allowing
	// the absent case to default to 1.
	GetRawN() *int
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
	MMFeatures() *RenderMMFeatures
	// SetMMFeatures sets the multimodal features
	SetMMFeatures(mmFeatures *RenderMMFeatures)

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
	KVParams *KVTransferParams `json:"kv_transfer_params,omitempty"`
	// The number of tokens in the prompt that are in the local KV Cache
	cachedPromptTokens int
	// IgnoreEOS is a boolean value, true when the model should ignore end-of-sequence tokens
	IgnoreEOS bool `json:"ignore_eos"`
	// tokenizedPrompt is the tokenized prompt
	tokenizedPrompt *Tokenized
	// tokenizedPromptForEcho is the tokenized part of the prompt to be used in echo mode, exists only in echo mode
	tokenizedPromptForEcho *Tokenized
	// mmFeatures holds multimodal metadata produced by the tokenizer, exists only for multimodal requests
	mmFeatures *RenderMMFeatures
}

// baseCompletionsRequest contains base completions request related information
type baseCompletionsRequest struct {
	baseRequest
	// StreamOptions defines streaming options in case Stream is set to true
	StreamOptions *StreamOptions `json:"stream_options,omitempty"`
	// CacheHitThreshold is a value between 0 and 1 that specifies the minimum cache hit rate required
	// to proceed with request processing. If the actual cache hit rate is below this threshold,
	// the request will return with cache_threshold finish reason.
	CacheHitThreshold *float64 `json:"cache_hit_threshold,omitempty"`
	// cacheThresholdFinishReason is a boolean value extracted from the request's HTTP header,
	//  when true, forces a cache_threshold finish reason
	cacheThresholdFinishReason bool
	// N is the number of completion choices to generate for each prompt.
	// Optional and defaults to 1.
	N *int `json:"n,omitempty"`
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

// PromptInput is a single prompt as it arrived on the wire. Exactly one of
// Text or Tokens is populated: Text for string-form prompts, Tokens for
// token-id-array-form prompts. /completions accepts both forms (and arrays of
// either), and downstream code branches on IsTokens to decide whether the
// prompt still needs to be tokenized.
type PromptInput struct {
	Text   string
	Tokens []uint32
}

// IsTokens reports whether this prompt is already tokenized.
func (p PromptInput) IsTokens() bool {
	return p.Tokens != nil
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

// GetN returns 1 for non-completions requests that don't support the n parameter.
func (b *baseRequest) GetN() int {
	return 1
}

// GetRawN returns nil for non-completions requests that don't support the n parameter.
func (b *baseRequest) GetRawN() *int {
	return nil
}

// SetIgnoreEOS sets the value of IgnoreEOS
func (b *baseRequest) SetIgnoreEOS(ignoreEOS bool) {
	b.IgnoreEOS = ignoreEOS
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
func (b *baseRequest) MMFeatures() *RenderMMFeatures {
	return b.mmFeatures
}

// SetMMFeatures sets the multimodal features
func (b *baseRequest) SetMMFeatures(mmFeatures *RenderMMFeatures) {
	b.mmFeatures = mmFeatures
}

func (b *baseCompletionsRequest) IncludeUsage() bool {
	return !b.Stream || (b.StreamOptions != nil && b.StreamOptions.IncludeUsage)
}

// GetN returns the number of completion choices to generate, defaulting to 1.
func (b *baseCompletionsRequest) GetN() int {
	if b.N == nil || *b.N <= 0 {
		return 1
	}
	return *b.N
}

// GetRawN returns the raw n pointer, nil when the field was omitted.
func (b *baseCompletionsRequest) GetRawN() *int {
	return b.N
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
	Messages []Message `json:"messages"`

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

// baseTextCompletionsRequest is the shared envelope for both forms of a
// /completions request: TextCompletionsParsedRequest (wire form, prompt may be
// a single string or an array) and TextCompletionsRequest (post-split
// processing form, single prompt). Only the Prompt field differs between the
// two; everything else lives here. The type is unexported because callers in
// other packages should hold a TextCompletionsParsedRequest or a
// TextCompletionsRequest and let the envelope details ride along by embedding.
type baseTextCompletionsRequest struct {
	baseCompletionsRequest

	// The maximum number of [tokens](/tokenizer) that can be generated in the
	// completions.
	//
	// The token count of your prompt plus `max_tokens` cannot exceed the
	// model's context length.
	MaxTokens *int64 `json:"max_tokens"`

	// Logprobs includes the log probabilities on the logprobs most likely
	// tokens, as well as the chosen tokens. For example, if logprobs is 5,
	// the API will return a list of the 5 most likely tokens. The API will
	// always return the logprob of the sampled token, so there may be up to
	// logprobs+1 elements in the response.
	Logprobs *int `json:"logprobs,omitempty"`
}

// TextCompletionsParsedRequest is the wire form of a /completions request.
// On the wire `prompt` may be a single string or an array of strings; the
// custom UnmarshalJSON below normalizes both forms into Prompt — a plain
// string becomes a one-element slice. Used only between JSON unmarshalling
// and the simulator's split step; workers never see this type.
type TextCompletionsParsedRequest struct {
	baseTextCompletionsRequest
	// Prompt holds one or more prompts. Always non-empty for valid requests.
	Prompt []PromptInput `json:"prompt"`
}

// TextCompletionsRequest is the processing form of a /completions request:
// it always carries a single prompt.
type TextCompletionsRequest struct {
	baseTextCompletionsRequest
	// Prompt is the single prompt this request will generate against.
	Prompt PromptInput
}

// UnmarshalJSON accepts any of the four wire forms allowed for the `prompt`
// field — a string, an array of strings, an array of token ids, or an array
// of token-id arrays — and normalizes them into a []PromptInput.
func (t *TextCompletionsParsedRequest) UnmarshalJSON(data []byte) error {
	type alias struct {
		baseTextCompletionsRequest
		Prompt json.RawMessage `json:"prompt"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	t.baseTextCompletionsRequest = a.baseTextCompletionsRequest

	if len(a.Prompt) == 0 || string(a.Prompt) == nullString {
		t.Prompt = nil
		return nil
	}

	// prompt is a string
	var str string
	if err := json.Unmarshal(a.Prompt, &str); err == nil {
		t.Prompt = []PromptInput{{Text: str}}
		return nil
	}
	// prompt is an array of strings
	var strs []string
	if err := json.Unmarshal(a.Prompt, &strs); err == nil {
		t.Prompt = make([]PromptInput, len(strs))
		for i, s := range strs {
			t.Prompt[i] = PromptInput{Text: s}
		}
		return nil
	}
	// prompt is an array of token ids
	var tokens []uint32
	if err := json.Unmarshal(a.Prompt, &tokens); err == nil {
		t.Prompt = []PromptInput{{Tokens: tokens}}
		return nil
	}
	// prompt is an array of arrays of token ids
	var tokenLists [][]uint32
	if err := json.Unmarshal(a.Prompt, &tokenLists); err != nil {
		return fmt.Errorf("prompt must be a string, an array of strings, an array of token ids, or an array of arrays of token ids: %w", err)
	}
	t.Prompt = make([]PromptInput, len(tokenLists))
	for i, ids := range tokenLists {
		t.Prompt[i] = PromptInput{Tokens: ids}
	}
	return nil
}

// MarshalJSON emits the prompt back in the wire form that mirrors the input
// shape preserved on each PromptInput: a single text prompt as a string, a
// single token-id prompt as a numeric array, multiple text prompts as an
// array of strings, and multiple token-id prompts as an array of numeric
// arrays. nil/empty Prompt is emitted as JSON null.
// Used in the tests.
func (t *TextCompletionsParsedRequest) MarshalJSON() ([]byte, error) {
	type alias struct {
		baseTextCompletionsRequest
		Prompt any `json:"prompt"`
	}
	a := alias{baseTextCompletionsRequest: t.baseTextCompletionsRequest}

	switch {
	case len(t.Prompt) == 0:
		a.Prompt = nil
	case len(t.Prompt) == 1:
		if t.Prompt[0].IsTokens() {
			a.Prompt = t.Prompt[0].Tokens
		} else {
			a.Prompt = t.Prompt[0].Text
		}
	default:
		// vLLM only accepts homogeneous prompt arrays — all strings or all
		// token-id arrays. Reject mixed inputs rather than silently dropping
		// the minority shape.
		firstIsTokens := t.Prompt[0].IsTokens()
		for i, p := range t.Prompt[1:] {
			if p.IsTokens() != firstIsTokens {
				return nil, fmt.Errorf("prompt array is not homogeneous: entry 0 and entry %d have different types", i+1)
			}
		}
		if firstIsTokens {
			arrs := make([][]uint32, len(t.Prompt))
			for i, p := range t.Prompt {
				arrs[i] = p.Tokens
			}
			a.Prompt = arrs
		} else {
			strs := make([]string, len(t.Prompt))
			for i, p := range t.Prompt {
				strs[i] = p.Text
			}
			a.Prompt = strs
		}
	}
	return json.Marshal(a)
}

// AsSingle returns a single-prompt TextCompletionsRequest for t.Prompt[index],
// sharing this request's envelope (model, lora, max_tokens, KV params, …). The
// sub-request's RequestID is stamped as "<requestID>-<index>" so each sub
// carries a unique, deterministic id derived from the parent.
//
// This helper exists so the splitting logic — which lives in another package —
// can produce sub-requests without needing access to the unexported
// baseTextCompletionsRequest field.
func (t *TextCompletionsParsedRequest) AsSingle(index int) TextCompletionsRequest {
	sub := TextCompletionsRequest{
		baseTextCompletionsRequest: t.baseTextCompletionsRequest,
		Prompt:                     t.Prompt[index],
	}
	sub.RequestID = fmt.Sprintf("%s-%d", t.RequestID, index)
	if sub.Prompt.IsTokens() {
		// prompt arrived already tokenized; pre-populate TokenizedPrompt so the
		// worker can skip the encode() round-trip. Strings is left nil and
		// rebuilt on demand by the request's tokenizedPromptForEcho.
		sub.tokenizedPrompt = &Tokenized{Tokens: sub.Prompt.Tokens}
	}
	return sub
}

var _ Request = (*TextCompletionsRequest)(nil)
var _ Request = (*TextCompletionsParsedRequest)(nil)

// Methods below are shared between TextCompletionsRequest and
// TextCompletionsParsedRequest via embedding of baseTextCompletionsRequest.

func (b *baseTextCompletionsRequest) GetTools() []Tool {
	return nil
}

func (b *baseTextCompletionsRequest) GetToolChoice() ToolChoice {
	return ToolChoice{}
}

func (b *baseTextCompletionsRequest) GetMaxCompletionTokens() *int64 {
	return b.MaxTokens
}

// ExtractMaxTokens extracts the max tokens from the request
// for text completions - max_tokens field is used.
func (b *baseTextCompletionsRequest) ExtractMaxTokens() *int64 {
	return b.MaxTokens
}

func (b *baseTextCompletionsRequest) GetLogprobs() *int {
	return b.Logprobs
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

func (c *GenerationRequest) GetTools() []Tool {
	return nil
}

func (c *GenerationRequest) GetToolChoice() ToolChoice {
	return ToolChoice{}
}

func (c *GenerationRequest) GetMaxCompletionTokens() *int64 {
	return c.MaxTokens
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
	// Include specifies additional output data to include. Use "message.output_text.logprobs" to include logprobs.
	Include []string `json:"include,omitempty"`
	// TopLogprobs specifies the number of most likely tokens to return at each position with log probabilities.
	TopLogprobs *int `json:"top_logprobs,omitempty"`
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

	if len(raw.Content) == 0 || string(raw.Content) == nullString {
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

func (m *InputMessage) PlainText(includeRole bool) string {
	var builder strings.Builder

	if includeRole {
		builder.WriteString(m.Role)
		builder.WriteString(": ")
	}

	var parts []string
	for _, c := range m.Content {
		switch c.Type {
		case ResponsesInputText:
			parts = append(parts, c.Text)
		case ResponsesInputImage:
			parts = append(parts, "image: "+c.ImageURL)
		case ResponsesInputAudio:
			parts = append(parts, "audio: "+c.AudioFormat)
		}
	}
	builder.WriteString(strings.Join(parts, "\n"))

	return builder.String()
}

type InputContent struct {
	Type     string `json:"type"` // input_text, input_image, input_audio
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"` // URL for input_image
	// Fields for input_audio
	AudioData   string `json:"data,omitempty"`   // base64-encoded audio data
	AudioFormat string `json:"format,omitempty"` // audio format (e.g. "wav", "mp3")
}

func (c *InputContent) UnmarshalJSON(data []byte) error {
	var raw struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL string `json:"image_url"`
		// Audio fields
		Data   string `json:"data"`
		Format string `json:"format"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	switch raw.Type {
	case "", ResponsesInputText, ResponsesOutputText:
		c.Type = ResponsesInputText
		c.Text = raw.Text
	case ResponsesInputImage:
		c.Type = ResponsesInputImage
		c.ImageURL = raw.ImageURL
	case ResponsesInputAudio:
		c.Type = ResponsesInputAudio
		c.AudioData = raw.Data
		c.AudioFormat = raw.Format
	default:
		return fmt.Errorf("unsupported input content type %q", raw.Type)
	}
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
		Include         []string        `json:"include,omitempty"`
		TopLogprobs     *int            `json:"top_logprobs,omitempty"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	req.baseRequest = a.baseRequest
	req.Instructions = a.Instructions
	req.MaxOutputTokens = a.MaxOutputTokens
	req.Text = a.Text
	req.Include = a.Include
	req.TopLogprobs = a.TopLogprobs

	if len(a.Input) == 0 || string(a.Input) == nullString {
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
	// include logprobs only if "message.output_text.logprobs" presents in the Include list
	if slices.Contains(req.Include, ResponsesIncludeLogprobs) {
		if req.TopLogprobs != nil {
			return req.TopLogprobs
		}
		zero := 0
		return &zero
	}
	return nil
}

func (req *ResponsesRequest) ExtractMaxTokens() *int64 {
	return req.MaxOutputTokens
}

// GenerateRequest defines structure of generate request
type GenerateRequest struct {
	baseRequest
	TokenIDs       []uint32          `json:"token_ids"`
	SamplingParams *SamplingParams   `json:"sampling_params"`
	Features       *EncodeMMFeatures `json:"features"`
	StreamOptions  *StreamOptions    `json:"stream_options,omitempty"`
}

func (g *GenerateRequest) IncludeUsage() bool {
	return !g.Stream || (g.StreamOptions != nil && g.StreamOptions.IncludeUsage)
}

type SamplingParams struct {
	MaxTokens *int64             `json:"max_tokens"`
	ExtraArgs *SamplingExtraArgs `json:"extra_args"`
}

// SamplingExtraArgs holds extra fields nested inside sampling_params.extra_args.
// Real vLLM has a bug where kv_transfer_params is sometimes passed here instead of
// at the root of the request, so we accept it in either location.
type SamplingExtraArgs struct {
	KVTransferParams *KVTransferParams `json:"kv_transfer_params"`
}

type EncodeMMFeatures struct {
	MMHashes map[string][]string `json:"mm_hashes"`
}

var _ Request = (*GenerateRequest)(nil)

func (g *GenerateRequest) GetTools() []Tool {
	return nil
}

func (g *GenerateRequest) GetToolChoice() ToolChoice {
	return ToolChoice{}
}

func (g *GenerateRequest) GetMaxCompletionTokens() *int64 {
	return g.SamplingParams.MaxTokens
}

func (g *GenerateRequest) ExtractMaxTokens() *int64 {
	return g.SamplingParams.MaxTokens
}

func (g *GenerateRequest) GetLogprobs() *int {
	return nil
}

// Anthropic Messages API

// AnthropicImageSource is the source payload for an "image" content block.
type AnthropicImageSource struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // e.g. "image/jpeg", present for base64 sources
	Data      string `json:"data,omitempty"`       // base64-encoded bytes, present for base64 sources
	URL       string `json:"url,omitempty"`        // present for url sources
}

// AnthropicToolResultContent is the content field inside a "tool_result" block.
// It can be a plain string or an array of text blocks.
type AnthropicToolResultContent struct {
	Raw    string
	Blocks []AnthropicContentBlock
}

func (c *AnthropicToolResultContent) UnmarshalJSON(data []byte) error {
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		c.Raw = str
		return nil
	}
	var blocks []AnthropicContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		c.Blocks = blocks
		return nil
	}
	return errors.New("tool_result content format not supported")
}

// PlainText returns the concatenated plain text from the tool result.
func (c *AnthropicToolResultContent) PlainText() string {
	if c.Raw != "" {
		return c.Raw
	}
	var parts []string
	for _, b := range c.Blocks {
		if b.Type == ContentTypeText {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// AnthropicContentBlock is a single content block in an Anthropic message.
// Fields are populated according to Type:
//   - "text":        Text
//   - "image":       Source
//   - "tool_use":    ID, Name, Input
//   - "tool_result": ToolUseID, Content
type AnthropicContentBlock struct {
	Type string `json:"type"`
	// text block
	Text string `json:"text,omitempty"`
	// image block
	Source *AnthropicImageSource `json:"source,omitempty"`
	// tool_use block
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name,omitempty"`
	Input map[string]any `json:"input,omitempty"`
	// tool_result block
	ToolUseID string                      `json:"tool_use_id,omitempty"`
	Content   *AnthropicToolResultContent `json:"content,omitempty"`
}

// AnthropicMessageContent holds the content of an Anthropic message.
// It can be a plain string or an array of content blocks.
type AnthropicMessageContent struct {
	Raw    string
	Blocks []AnthropicContentBlock
}

func (c *AnthropicMessageContent) UnmarshalJSON(data []byte) error {
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		c.Raw = str
		return nil
	}
	var blocks []AnthropicContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		c.Blocks = blocks
		return nil
	}
	return errors.New("content format not supported")
}

func (c AnthropicMessageContent) MarshalJSON() ([]byte, error) {
	if c.Raw != "" {
		return json.Marshal(c.Raw)
	}
	if c.Blocks != nil {
		return json.Marshal(c.Blocks)
	}
	return json.Marshal("")
}

// PlainText returns the concatenated text from all text blocks.
func (c *AnthropicMessageContent) PlainText() string {
	if c.Raw != "" {
		return c.Raw
	}
	var parts []string
	for _, block := range c.Blocks {
		if block.Type == ContentTypeText {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// AnthropicMessage is a single message in the Anthropic Messages API.
type AnthropicMessage struct {
	// Role is "user" or "assistant"
	Role    string                  `json:"role"`
	Content AnthropicMessageContent `json:"content"`
}

// PlainText returns the plain text representation of the message.
func (m *AnthropicMessage) PlainText(includeRole bool) string {
	var builder strings.Builder
	if includeRole {
		builder.WriteString(m.Role)
		builder.WriteString(": ")
	}
	builder.WriteString(m.Content.PlainText())
	return builder.String()
}

// AnthropicTool defines a tool in the Anthropic Messages API format.
type AnthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// AnthropicToolChoice controls which tool the model calls.
// Type is "auto" (default), "any" (force a tool call), or "tool" (specific tool via Name).
type AnthropicToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"` // only for type "tool"
}

// MessagesRequest defines the structure of an Anthropic /v1/messages request.
type MessagesRequest struct {
	baseRequest
	// Messages is the list of input messages (required)
	Messages []AnthropicMessage `json:"messages"`
	// System is the optional system prompt
	System string `json:"system,omitempty"`
	// MaxTokens is the maximum number of output tokens (required by the API)
	MaxTokens *int64 `json:"max_tokens"`
	// Tools is the list of tools available to the model
	Tools []AnthropicTool `json:"tools,omitempty"`
	// ToolChoice controls which tool (if any) is called
	ToolChoice *AnthropicToolChoice `json:"tool_choice,omitempty"`
}

var _ Request = (*MessagesRequest)(nil)

func (m *MessagesRequest) GetTools() []Tool {
	return nil
}

func (m *MessagesRequest) GetToolChoice() ToolChoice {
	return ToolChoice{}
}

func (m *MessagesRequest) GetMaxCompletionTokens() *int64 {
	return m.MaxTokens
}

func (m *MessagesRequest) ExtractMaxTokens() *int64 {
	return m.MaxTokens
}

func (m *MessagesRequest) GetLogprobs() *int {
	return nil
}

// ToChatCompletionsRequest converts this Anthropic Messages request to an
// equivalent ChatCompletionsRequest for processing by the existing pipeline.
//
// Mapping rules:
//   - system prompt → leading Message{Role: "system"}
//   - "text" block  → ChatComplContentBlock{Type: "text"}
//   - "image" block → ChatComplContentBlock{Type: "image_url"} (base64 sources become data URLs)
//   - "tool_use" block (assistant) → Message.ToolCalls entry
//   - "tool_result" block (user) → separate Message{Role: "tool", ToolCallID: …}
//   - AnthropicTool → Tool with function.parameters = input_schema
func (m *MessagesRequest) ToChatCompletionsRequest() *ChatCompletionsRequest {
	msgs := make([]Message, 0, len(m.Messages)+1)
	if m.System != "" {
		msgs = append(msgs, Message{
			Role:    "system",
			Content: ChatComplContent{Raw: m.System},
		})
	}

	for _, am := range m.Messages {
		// String-form content: emit directly.
		if am.Content.Raw != "" {
			msgs = append(msgs, Message{
				Role:    am.Role,
				Content: ChatComplContent{Raw: am.Content.Raw},
			})
			continue
		}

		var contentBlocks []ChatComplContentBlock
		var toolCalls []ToolCall
		var toolResults []AnthropicContentBlock

		for _, b := range am.Content.Blocks {
			switch b.Type {
			case ContentTypeText:
				contentBlocks = append(contentBlocks, ChatComplContentBlock{
					Type: ContentTypeText,
					Text: b.Text,
				})
			case "image":
				if b.Source != nil {
					var url string
					if b.Source.Type == "base64" {
						url = "data:" + b.Source.MediaType + ";base64," + b.Source.Data
					} else {
						url = b.Source.URL
					}
					contentBlocks = append(contentBlocks, ChatComplContentBlock{
						Type:     "image_url",
						ImageURL: ChatComplImageBlock{Url: url},
					})
				}
			case "tool_use":
				var args []byte
				if b.Input != nil {
					args, _ = json.Marshal(b.Input)
				}
				if len(args) == 0 {
					args = []byte("{}")
				}
				name := b.Name
				toolCalls = append(toolCalls, ToolCall{
					ID:   b.ID,
					Type: "function",
					Function: FunctionCall{
						Name:      &name,
						Arguments: string(args),
					},
				})
			case "tool_result":
				toolResults = append(toolResults, b)
			default:
				// unknown block types (e.g. "document", "thinking") are skipped to
				// allow forward compatibility with new Anthropic block types
			}
		}

		// Emit the message itself when it carries content or tool calls.
		if len(contentBlocks) > 0 || len(toolCalls) > 0 {
			msg := Message{Role: am.Role, ToolCalls: toolCalls}
			if len(contentBlocks) > 0 {
				msg.Content = ChatComplContent{Structured: contentBlocks}
			}
			msgs = append(msgs, msg)
		}

		// Each tool_result block becomes a separate "tool" role message.
		for _, tr := range toolResults {
			var content string
			if tr.Content != nil {
				content = tr.Content.PlainText()
			}
			msgs = append(msgs, Message{
				Role:       "tool",
				Content:    ChatComplContent{Raw: content},
				ToolCallID: tr.ToolUseID,
			})
		}
	}

	// Convert Anthropic tools to OpenAI format (input_schema → parameters).
	var tools []Tool
	if len(m.Tools) > 0 {
		tools = make([]Tool, 0, len(m.Tools))
	}
	for _, at := range m.Tools {
		tools = append(tools, Tool{
			Type: "function",
			Function: function{
				Name:        at.Name,
				Description: at.Description,
				Parameters:  at.InputSchema,
			},
		})
	}

	var toolChoice ToolChoice
	if m.ToolChoice != nil {
		switch m.ToolChoice.Type {
		case "any":
			toolChoice = NewToolChoiceRequired()
		case "none":
			toolChoice = NewToolChoiceNone()
		case "tool":
			if m.ToolChoice.Name != "" {
				toolChoice = NewToolChoiceFunction(m.ToolChoice.Name)
			}
		}
	}

	return &ChatCompletionsRequest{
		baseCompletionsRequest: baseCompletionsRequest{
			baseRequest: m.baseRequest,
		},
		Messages:            msgs,
		MaxCompletionTokens: m.MaxTokens,
		Tools:               tools,
		ToolChoice:          toolChoice,
	}
}
