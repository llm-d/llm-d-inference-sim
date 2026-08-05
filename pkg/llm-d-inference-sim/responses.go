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

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/valyala/fasthttp"

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
	for _, tool := range r.Tools {
		toolJson, err := json.Marshal(tool.Function)
		if err != nil {
			serverErr := api.NewError("Failed to marshal request tools: "+err.Error(),
				fasthttp.StatusBadRequest, nil)
			return &serverErr
		}
		err = toolsValidator.validateTool(toolJson)
		if err != nil {
			serverErr := api.NewError("Tool validation failed: "+err.Error(),
				fasthttp.StatusBadRequest, nil)
			return &serverErr
		}
	}
	return validateRequest(r)
}

func (r *ResponsesRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo],
	choiceIdx int, doneFn func()) requestContext {
	reqCtx := &responsesReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel, choiceIdx, doneFn),
		req:                r,
		toolIDPrefix:       api.ResponsesFunctionCallIDPrefix,
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
		toolsCalls:          toolCalls,
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
	req          *ResponsesRequest
	toolIDPrefix string
}

func (r *responsesReqCtx) request() Request {
	return r.req
}

// convertInputToMessages converts ResponsesRequest Input to Messages for tokenization.
// function_call / function_call_output items are mapped to chat-style assistant/tool
// messages so they contribute to usage.input_tokens (and prefix/cache logic).
func convertInputToMessages(input []api.InputItem) []api.Message {
	messages := make([]api.Message, 0, len(input))
	for _, item := range input {
		switch v := item.(type) {
		case *api.InputMessage:
			msg := api.Message{
				Role: v.Role,
			}

			if len(v.Content) == 1 && v.Content[0].Type == api.ResponsesInputText {
				// Simple text content
				msg.Content.Raw = v.Content[0].Text
			} else {
				// Structured content (text, images, audio)
				blocks := make([]api.ChatComplContentBlock, 0, len(v.Content))
				for _, content := range v.Content {
					switch content.Type {
					case api.ResponsesInputText:
						blocks = append(blocks, api.ChatComplContentBlock{
							Type: "text",
							Text: content.Text,
						})
					case api.ResponsesInputImage:
						blocks = append(blocks, api.ChatComplContentBlock{
							Type:     "image_url",
							ImageURL: &api.ChatComplImageBlock{Url: content.ImageURL},
						})
					case api.ResponsesInputAudio:
						blocks = append(blocks, api.ChatComplContentBlock{
							Type:       "input_audio",
							InputAudio: &api.ChatComplAudioBlock{Data: content.AudioData, Format: content.AudioFormat},
						})
					}
				}
				msg.Content.Structured = blocks
			}

			messages = append(messages, msg)
		case *api.InputFunctionCall:
			name := v.Name
			tcID := v.ID
			if tcID == "" {
				tcID = v.CallID
			}
			messages = append(messages, api.Message{
				Role: api.RoleAssistant,
				ToolCalls: []api.ToolCall{{
					ID:   tcID,
					Type: "function",
					Function: api.FunctionCall{
						Name:      &name,
						Arguments: v.Arguments,
					},
				}},
			})
		case *api.InputFunctionCallOutput:
			messages = append(messages, api.Message{
				Role:       "tool",
				ToolCallID: v.CallID,
				Content:    api.ChatComplContent{Raw: v.Output},
			})
		}
	}
	return messages
}

func (r *responsesReqCtx) encode() ([]uint32, []string, *api.RenderMMFeatures, error) {
	messages := convertInputToMessages(r.req.Input)
	return r.sim.Tokenizer.RenderMessages(messages)
}

func (r *responsesReqCtx) createToolCalls() ([]api.ToolCall, int, string, error) {
	req := r.req
	if isToolChoiceNone(req.GetToolChoice()) || len(req.GetTools()) == 0 {
		return nil, 0, "", nil
	}
	// After tool results are present, finish with a normal message so Praxis can exit the loop.
	if api.HasFunctionCallOutput(req.Input) {
		return nil, 0, "", nil
	}

	toolChoice := req.GetToolChoice()
	// Deterministic first tool turn: treat omitted/auto as required unless a function is forced.
	// Do not upgrade allowed_tools/custom — those are unenforced and must keep auto-like min=0.
	if shouldForceRequiredToolChoice(toolChoice) {
		toolChoice = api.NewToolChoiceRequired()
	}

	toolCalls, completionTokens, err := createSingleToolCall(
		req.GetTools(), toolChoice, r.sim.Config(), r.sim.Random, r.sim.Tokenizer, r.toolIDPrefix)
	if err != nil {
		return nil, 0, "", err
	}
	if len(toolCalls) == 0 {
		return nil, 0, "", nil
	}
	return toolCalls, completionTokens, common.ToolsFinishReason, nil
}

// shouldForceRequiredToolChoice reports whether Responses should rewrite tool_choice
// to required for a deterministic first tool turn.
func shouldForceRequiredToolChoice(toolChoice api.ToolChoice) bool {
	if toolChoice.OfAllowedTools != nil || toolChoice.OfCustomToolChoice != nil {
		return false
	}
	if toolChoice.GetFunction() != nil {
		return false
	}
	return param.IsOmitted(toolChoice.OfAuto) || toolChoice.OfAuto.Or("") == toolChoiceAuto
}

func (r *responsesReqCtx) tokenizedPromptForEcho() (*api.Tokenized, error) {
	// echo the text of the last input item, matching chat completion behavior for messages
	lastMsg := ""
	if len(r.req.Input) > 0 {
		switch last := r.req.Input[len(r.req.Input)-1].(type) {
		case *api.InputMessage:
			lastMsg = last.PlainText(false)
		case *api.InputFunctionCallOutput:
			lastMsg = last.Output
		case *api.InputFunctionCall:
			lastMsg = last.Name + "(" + last.Arguments + ")"
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
	toolsCalls []api.ToolCall
}

func (respCtx *responsesResponseCtx) Instructions() *string {
	if s := respCtx.reqCtx.request().(*ResponsesRequest).Instructions; s != "" {
		return &s
	}
	return nil
}

func (respCtx *responsesResponseCtx) ToolCalls() []api.ToolCall {
	return respCtx.toolsCalls
}

var _ ResponseContext = (*responsesResponseCtx)(nil)
