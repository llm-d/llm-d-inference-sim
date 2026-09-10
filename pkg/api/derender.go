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

// Contains structures and constructors for the derender endpoints, the
// inverse of the render endpoints: they turn a GenerateResponse carrying raw
// token ids into a client-facing OpenAI response.
package api

import "encoding/json"

// DerenderChatRequest is the body of POST /v1/chat/completions/derender.
type DerenderChatRequest struct {
	// Stream is vLLM's request-shape discriminator; streaming derender is
	// not supported and stream=true is rejected
	Stream bool `json:"stream"`
	// ModelName is the requested model; empty means the served model
	ModelName string `json:"model"`
	// GenerateResponse is the generation result to detokenize
	GenerateResponse *GenerateResponse `json:"generate_response"`
	// PromptTokens is the prompt token count reported in usage
	PromptTokens int `json:"prompt_tokens"`
	// ChatRequest is the original request context. vLLM uses it for
	// tool-call and reasoning parsing; the simulator accepts and ignores it
	ChatRequest json.RawMessage `json:"chat_request,omitempty"`
}

// DerenderCompletionRequest is the body of POST /v1/completions/derender.
type DerenderCompletionRequest struct {
	// Stream is vLLM's request-shape discriminator; streaming derender is
	// not supported and stream=true is rejected
	Stream bool `json:"stream"`
	// ModelName is the requested model; empty means the served model
	ModelName string `json:"model"`
	// GenerateResponses are the generation results to detokenize, one per prompt
	GenerateResponses []GenerateResponse `json:"generate_responses"`
	// PromptTokens are per-response prompt token counts; when present its
	// length must equal the length of GenerateResponses
	PromptTokens []int `json:"prompt_tokens"`
	// CompletionRequest is the original request context, accepted and ignored
	CompletionRequest json.RawMessage `json:"completion_request,omitempty"`
}

// NewDetokenizeDerenderRequest builds the minimal single-response derender
// payload used to decode token ids through a render service.
func NewDetokenizeDerenderRequest(model string, tokenIDs []uint32) *DerenderCompletionRequest {
	return &DerenderCompletionRequest{
		ModelName: model,
		GenerateResponses: []GenerateResponse{{
			Choices: []GenerateRespChoice{{TokenIDs: tokenIDs}},
		}},
	}
}

// CreateDerenderChatCompletionsResponse assembles a chat completions response
// for the derender endpoint. Unlike CreateChatCompletionsResponse, the id is
// taken verbatim from the generate response's request id and kv_transfer_params
// are passed through, matching vLLM's derender behavior.
func CreateDerenderChatCompletionsResponse(id, model string, created int64, choices []ChatRespChoice,
	usage *Usage, kvParams *KVTransferParams) *ChatCompletionsResponse {
	return &ChatCompletionsResponse{
		baseCompletionsResponse: baseCompletionsResponse{
			baseResponse: baseResponse{
				ID:       id,
				Model:    model,
				Object:   ChatCompletionObject,
				KVParams: kvParams,
			},
			Created: created,
			Usage:   usage,
		},
		Choices: choices,
	}
}

// CreateDerenderTextCompletionsResponse assembles a text completions response
// for the derender endpoint; see CreateDerenderChatCompletionsResponse.
func CreateDerenderTextCompletionsResponse(id, model string, created int64, choices []TextRespChoice,
	usage *Usage, kvParams *KVTransferParams) *TextCompletionsResponse {
	return &TextCompletionsResponse{
		baseCompletionsResponse: baseCompletionsResponse{
			baseResponse: baseResponse{
				ID:       id,
				Model:    model,
				Object:   TextCompletionObject,
				KVParams: kvParams,
			},
			Created: created,
			Usage:   usage,
		},
		Choices: choices,
	}
}
