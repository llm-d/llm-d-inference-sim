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

package endpoint

import (
	"encoding/json"
	"fmt"
	"reflect"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	"github.com/valyala/fasthttp"
)

// DerenderableRequest is implemented by the request types of the
// /v1/{chat/,}completions/derender endpoints. Like RenderableRequest, it lets
// the HTTP layer parse and answer without going through the worker pipeline.
type DerenderableRequest interface {
	Unmarshal(data []byte) error
	// ValidateBody checks that the unmarshalled body matches the endpoint's
	// expected shape.
	ValidateBody() *api.Error
	// GetModel returns the requested model name; empty means the served model.
	GetModel() string
	// Derender detokenizes the carried generation result and assembles the
	// OpenAI response. displayModel is used when the request carries no model.
	Derender(tk tokenizer.Tokenizer, displayModel string, logger logr.Logger) (any, *api.Error)
}

// Implementation of request for chat completions derender requests
type DerenderChatRequest struct {
	api.DerenderChatRequest
}

func (d *DerenderChatRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, d)
}

func (d *DerenderChatRequest) GetModel() string {
	return d.ModelName
}

func (d *DerenderChatRequest) ValidateBody() *api.Error {
	if err := validateNoStreaming(d.Stream); err != nil {
		return err
	}
	if d.GenerateResponse == nil {
		serverErr := api.NewError("generate_response is required", fasthttp.StatusBadRequest, nil)
		return &serverErr
	}
	return validateGenerateChoices(d.GenerateResponse)
}

func (d *DerenderChatRequest) Derender(tk tokenizer.Tokenizer, displayModel string, _ logr.Logger) (any, *api.Error) {
	gen := d.GenerateResponse
	choices := make([]api.ChatRespChoice, len(gen.Choices))
	completionTokens := 0
	for i, ch := range gen.Choices {
		text, err := tk.Detokenize(ch.TokenIDs)
		if err != nil {
			return nil, detokenizeError(err)
		}
		completionTokens += len(ch.TokenIDs)
		message := api.Message{Role: api.RoleAssistant, Content: api.ChatComplContent{Raw: text}}
		choices[i] = api.CreateChatRespChoice(api.CreateBaseResponseChoice(ch.Index, ch.FinishReason), message)
	}

	usage := &api.Usage{
		PromptTokens:     d.PromptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      d.PromptTokens + completionTokens,
	}
	return api.CreateDerenderChatCompletionsResponse(gen.GenRequestID, derenderModel(d.ModelName, displayModel),
		time.Now().Unix(), choices, usage, gen.KVParams), nil
}

// Implementation of request for text completions derender requests
type DerenderCompletionRequest struct {
	api.DerenderCompletionRequest
}

func (d *DerenderCompletionRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, d)
}

func (d *DerenderCompletionRequest) GetModel() string {
	return d.ModelName
}

func (d *DerenderCompletionRequest) ValidateBody() *api.Error {
	if err := validateNoStreaming(d.Stream); err != nil {
		return err
	}
	if len(d.GenerateResponses) == 0 {
		serverErr := api.NewError("generate_responses must not be empty", fasthttp.StatusBadRequest, nil)
		return &serverErr
	}
	if d.PromptTokens != nil && len(d.PromptTokens) != len(d.GenerateResponses) {
		serverErr := api.NewError(
			fmt.Sprintf("prompt_tokens length (%d) must equal generate_responses length (%d)",
				len(d.PromptTokens), len(d.GenerateResponses)),
			fasthttp.StatusBadRequest, nil)
		return &serverErr
	}
	for i := range d.GenerateResponses {
		if err := validateGenerateChoices(&d.GenerateResponses[i]); err != nil {
			return err
		}
	}
	return nil
}

func (d *DerenderCompletionRequest) Derender(tk tokenizer.Tokenizer, displayModel string,
	logger logr.Logger) (any, *api.Error) {
	var choices []api.TextRespChoice
	promptTokens := 0
	completionTokens := 0
	for i := range d.GenerateResponses {
		gen := &d.GenerateResponses[i]
		if d.PromptTokens != nil {
			promptTokens += d.PromptTokens[i]
		}
		for _, ch := range gen.Choices {
			text, err := tk.Detokenize(ch.TokenIDs)
			if err != nil {
				return nil, detokenizeError(err)
			}
			completionTokens += len(ch.TokenIDs)
			// choices carry a flat running index across all generate
			// responses, matching vLLM
			choices = append(choices,
				api.CreateTextRespChoice(api.CreateBaseResponseChoice(len(choices), ch.FinishReason), text))
		}
	}

	// kv_transfer_params are taken from the first response; differing values
	// across responses cannot be represented and are dropped
	kvParams := d.GenerateResponses[0].KVParams
	for i := range d.GenerateResponses[1:] {
		if !reflect.DeepEqual(d.GenerateResponses[i+1].KVParams, kvParams) {
			logger.V(logging.WARN).Info("generate_responses carry differing kv_transfer_params, omitting them from the response")
			kvParams = nil
			break
		}
	}

	usage := &api.Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	}
	return api.CreateDerenderTextCompletionsResponse(d.GenerateResponses[0].GenRequestID,
		derenderModel(d.ModelName, displayModel), time.Now().Unix(), choices, usage, kvParams), nil
}

func validateNoStreaming(stream bool) *api.Error {
	if stream {
		serverErr := api.NewError("streaming derender is not supported", fasthttp.StatusBadRequest, nil)
		return &serverErr
	}
	return nil
}

func validateGenerateChoices(gen *api.GenerateResponse) *api.Error {
	for _, ch := range gen.Choices {
		if len(ch.TokenIDs) == 0 {
			serverErr := api.NewError(fmt.Sprintf("choice %d has empty or null token_ids", ch.Index),
				fasthttp.StatusBadRequest, nil)
			return &serverErr
		}
	}
	return nil
}

func derenderModel(requested, displayModel string) string {
	if requested != "" {
		return requested
	}
	return displayModel
}

func detokenizeError(err error) *api.Error {
	serverErr := api.NewError("Detokenization failed, "+err.Error(), fasthttp.StatusInternalServerError, nil)
	return &serverErr
}
