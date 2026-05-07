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

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

// Implementation of request for /responses requests
type ResponsesRequest struct {
	openaiserverapi.ResponsesRequest
}

// reads and parses data from the body of the given request
func (r *ResponsesRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, r)
}

func (r *ResponsesRequest) validate(toolsValidator *toolsValidator) (string, int) {
	return validateRequest(r)
}

func (r *ResponsesRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo]) requestContext {
	reqCtx := &responsesReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel),
		req:                r,
	}
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (r *ResponsesRequest) AsString() string {
	return "responses create request (req id " + r.RequestID + ")"
}

func (r *ResponsesRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *openaiserverapi.Tokenized, finishReason *string, usageData *openaiserverapi.Usage, sendUsageData bool,
	logprobs *int, toolCalls []openaiserverapi.ToolCall) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, r.GetRequestID(), r.IsDoRemotePrefill(), r.IsDoRemoteDecode(), r.GetNumberOfCachedPromptTokens())
	return &responsesResponseCtx{
		baseResponseContext: base,
	}
}

var _ Request = (*ResponsesRequest)(nil)

// Implementation of requestContext for /responses requests
type responsesReqCtx struct {
	baseRequestContext
	req *ResponsesRequest
}

func (r *responsesReqCtx) request() Request {
	return r.req
}

// convertInputToMessages converts ResponsesRequest Input to Messages
func convertInputToMessages(input []openaiserverapi.InputItem) []openaiserverapi.Message {
	messages := make([]openaiserverapi.Message, 0, len(input))
	for _, item := range input {
		if inputMsg, ok := item.(*openaiserverapi.InputMessage); ok {
			msg := openaiserverapi.Message{
				Role: inputMsg.Role,
			}

			// Convert InputContent to ChatComplContent
			if len(inputMsg.Content) == 1 && inputMsg.Content[0].Type == openaiserverapi.ResponsesInputText {
				// Simple text content
				msg.Content.Raw = inputMsg.Content[0].Text
			} else {
				// Structured content
				blocks := make([]openaiserverapi.ChatComplContentBlock, 0, len(inputMsg.Content))
				for _, content := range inputMsg.Content {
					if content.Type == openaiserverapi.ResponsesInputText {
						blocks = append(blocks, openaiserverapi.ChatComplContentBlock{
							Type: "text",
							Text: content.Text,
						})
					}
				}
				msg.Content.Structured = blocks
			}

			messages = append(messages, msg)
		}
	}
	return messages
}

func (r *responsesReqCtx) encode() ([]uint32, []string, *openaiserverapi.RenderMMFeatures, error) {
	messages := convertInputToMessages(r.req.Input)
	return r.sim.Tokenizer.RenderMessages(messages)
}

func (r *responsesReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

func (r *responsesReqCtx) tokenizedPromptForEcho() (*openaiserverapi.Tokenized, error) {
	// echo the text of the last input message, matching chat completion behavior
	lastMsg := ""
	if len(r.req.Input) > 0 {
		// in echo mode return the last message without role
		if msg, ok := r.req.Input[len(r.req.Input)-1].(*openaiserverapi.InputMessage); ok {
			lastMsg = msg.PlainText(false)
		}
	}
	tokens, strTokens, err := r.sim.Tokenizer.RenderText(lastMsg)
	if err != nil {
		return nil, err
	}
	return &openaiserverapi.Tokenized{Tokens: tokens, Strings: strTokens}, nil
}

var _ requestContext = (*responsesReqCtx)(nil)

// Implementation of responseContext for /responses requests
type responsesResponseCtx struct {
	baseResponseContext
}

func (respCtx *responsesResponseCtx) Instructions() *string {
	if s := respCtx.reqCtx.request().(*ResponsesRequest).Instructions; s != "" {
		return &s
	}
	return nil
}

func (respCtx *responsesResponseCtx) ToolCalls() []openaiserverapi.ToolCall {
	return nil
}

var _ ResponseContext = (*responsesResponseCtx)(nil)
