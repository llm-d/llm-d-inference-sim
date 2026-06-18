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

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// Implementation of request for /responses requests
type ResponsesRequest struct {
	api.ResponsesRequest
}

// reads and parses data from the body of the given request
func (r *ResponsesRequest) Unmarshal(data []byte) error {
	return json.Unmarshal(data, r)
}

func (r *ResponsesRequest) validate(toolsValidator *toolsValidator) *api.Error {
	return validateRequest(r)
}

func (r *ResponsesRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo],
	choiceIdx int, doneFn func()) requestContext {
	reqCtx := &responsesReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel, choiceIdx, doneFn),
		req:                r,
	}
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (r *ResponsesRequest) AsString() string {
	return "responses create request (req id " + r.RequestID + ")"
}

func (r *ResponsesRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *api.Tokenized, finishReason *string, usageData *api.Usage, sendUsageData bool,
	logprobs *int, toolCalls []api.ToolCall, _ bool) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, r.GetRequestID(), r.IsDoRemotePrefill(), r.IsDoRemoteDecode(), r.GetNumberOfCachedPromptTokens())
	return &responsesResponseCtx{
		baseResponseContext: base,
	}
}

// split is a no-op: responses requests always carry a single prompt.
func (r *ResponsesRequest) split() []Request {
	return []Request{r}
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
func convertInputToMessages(input []api.InputItem) []api.Message {
	messages := make([]api.Message, 0, len(input))
	for _, item := range input {
		if inputMsg, ok := item.(*api.InputMessage); ok {
			msg := api.Message{
				Role: inputMsg.Role,
			}

			// Convert InputContent to ChatComplContent
			if len(inputMsg.Content) == 1 && inputMsg.Content[0].Type == api.ResponsesInputText {
				// Simple text content
				msg.Content.Raw = inputMsg.Content[0].Text
			} else {
				// Structured content
				blocks := make([]api.ChatComplContentBlock, 0, len(inputMsg.Content))
				for _, content := range inputMsg.Content {
					if content.Type == api.ResponsesInputText {
						blocks = append(blocks, api.ChatComplContentBlock{
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

func (r *responsesReqCtx) encode() ([]uint32, []string, *api.RenderMMFeatures, error) {
	messages := convertInputToMessages(r.req.Input)
	return r.sim.Tokenizer.RenderMessages(messages)
}

func (r *responsesReqCtx) createToolCalls() ([]api.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

func (r *responsesReqCtx) tokenizedPromptForEcho() (*api.Tokenized, error) {
	// echo the text of the last input message, matching chat completion behavior
	lastMsg := ""
	if len(r.req.Input) > 0 {
		// in echo mode return the last message without role
		if msg, ok := r.req.Input[len(r.req.Input)-1].(*api.InputMessage); ok {
			lastMsg = msg.PlainText(false)
		}
	}
	tokens, strTokens, err := r.sim.Tokenizer.RenderText(lastMsg)
	if err != nil {
		return nil, err
	}
	return &api.Tokenized{Tokens: tokens, Strings: strTokens}, nil
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

func (respCtx *responsesResponseCtx) ToolCalls() []api.ToolCall {
	return nil
}

var _ ResponseContext = (*responsesResponseCtx)(nil)
