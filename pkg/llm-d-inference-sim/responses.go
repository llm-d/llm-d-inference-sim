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

package llmdinferencesim

import (
	"encoding/json"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
)

// Implementation of request for /responses requests
type ResponsesCreateRequest struct {
	openaiserverapi.ResponsesCreateRequest
}

// reads and parses data from the body of the given request
func (r *ResponsesCreateRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, r)
}

func (r *ResponsesCreateRequest) validate(toolsValidator *toolsValidator) (string, int) {
	return validateRequest(r)
}

func (r *ResponsesCreateRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo]) requestContext {
	reqCtx := &responsesCreateReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel),
		req:                r,
	}
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (r *ResponsesCreateRequest) AsString() string {
	return "responses create request (req id " + r.RequestID + ")"
}

func (r *ResponsesCreateRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *openaiserverapi.Tokenized, finishReason *string, usageData *openaiserverapi.Usage, sendUsageData bool,
	logprobs *int, toolCalls []openaiserverapi.ToolCall) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, r.GetRequestID(), r.IsDoRemotePrefill(), r.IsDoRemoteDecode(), r.GetNumberOfCachedPromptTokens())
	return &responsesCreateResponseCtx{
		baseResponseContext: base,
	}
}

var _ Request = (*ResponsesCreateRequest)(nil)

// Implementation of requestContext for /responses requests
type responsesCreateReqCtx struct {
	baseRequestContext
	req *ResponsesCreateRequest
}

func (r *responsesCreateReqCtx) request() Request {
	return r.req
}

func (r *responsesCreateReqCtx) encode() ([]uint32, []string, *tokenization.MultiModalFeatures, error) {
	var messages []openaiserverapi.Message

	if r.req.Instructions != "" {
		messages = append(messages, openaiserverapi.Message{
			Role:    "system",
			Content: openaiserverapi.Content{Raw: r.req.Instructions},
		})
	}

	for _, item := range r.req.Input {
		if msg, ok := item.(*openaiserverapi.InputMessage); ok {
			var text strings.Builder
			for i, c := range msg.Content {
				if i > 0 {
					text.WriteString(" ")
				}
				text.WriteString(c.Text)
			}
			messages = append(messages, openaiserverapi.Message{
				Role:    msg.Role,
				Content: openaiserverapi.Content{Raw: text.String()},
			})
		}
	}

	tokens, strTokens, _, err := r.sim.Tokenizer.RenderChatCompletion(messages)
	return tokens, strTokens, nil, err
}

func (r *responsesCreateReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

func (r *responsesCreateReqCtx) tokenizedPromptForEcho() (*openaiserverapi.Tokenized, error) {
	// echo the text of the last input message, matching chat completion behavior
	text := ""
	for i := len(r.req.Input) - 1; i >= 0; i-- {
		if msg, ok := r.req.Input[i].(*openaiserverapi.InputMessage); ok {
			var sb strings.Builder
			for j, c := range msg.Content {
				if j > 0 {
					sb.WriteString(" ")
				}
				sb.WriteString(c.Text)
			}
			text = sb.String()
			break
		}
	}
	tokens, strTokens, err := r.sim.Tokenizer.RenderText(text)
	if err != nil {
		return nil, err
	}
	return &openaiserverapi.Tokenized{Tokens: tokens, Strings: strTokens}, nil
}

var _ requestContext = (*responsesCreateReqCtx)(nil)

// Implementation of responseContext for /responses requests
type responsesCreateResponseCtx struct {
	baseResponseContext
}

func (respCtx *responsesCreateResponseCtx) Instructions() *string {
	if s := respCtx.reqCtx.request().(*ResponsesCreateRequest).Instructions; s != "" {
		return &s
	}
	return nil
}

func (respCtx *responsesCreateResponseCtx) ToolCalls() []openaiserverapi.ToolCall {
	return nil
}

var _ ResponseContext = (*responsesCreateResponseCtx)(nil)
