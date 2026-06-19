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

	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// MessagesRequest handles /v1/messages (Anthropic Messages API).
// Unmarshal parses the Anthropic wire format and converts it to the OpenAI
// chat completions format; all processing is then delegated to the embedded
// ChatCompletionsRequest via the existing chat completions pipeline.
type MessagesRequest struct {
	ChatCompletionsRequest
	orig api.MessagesRequest // retains original Anthropic fields for validation
}

// Unmarshal parses the Anthropic Messages API body and converts it to a
// ChatCompletionsRequest for the existing processing pipeline.
func (m *MessagesRequest) Unmarshal(data []byte) error {
	if err := json.Unmarshal(data, &m.orig); err != nil {
		return err
	}
	m.ChatCompletionsRequest.ChatCompletionsRequest = *m.orig.ToChatCompletionsRequest()
	return nil
}

// validateBlocks checks Anthropic-specific constraints that are lost after conversion:
// image blocks require a source, and tool_choice of type "tool" requires a name.
func (m *MessagesRequest) validateBlocks() *api.Error {
	for _, msg := range m.orig.Messages {
		for _, block := range msg.Content.Blocks {
			if block.Type == "image" && block.Source == nil {
				err := api.NewError("image content block is missing required 'source' field",
					fasthttp.StatusBadRequest, nil)
				return &err
			}
		}
	}
	if m.orig.ToolChoice != nil && m.orig.ToolChoice.Type == "tool" && m.orig.ToolChoice.Name == "" {
		err := api.NewError("tool_choice of type 'tool' requires a non-empty 'name'",
			fasthttp.StatusBadRequest, nil)
		return &err
	}
	return nil
}

func (m *MessagesRequest) validate(tv *toolsValidator) *api.Error {
	if err := m.validateBlocks(); err != nil {
		return err
	}
	if len(m.Messages) == 0 {
		err := api.NewError("messages must not be empty", fasthttp.StatusBadRequest, nil)
		return &err
	}
	if m.GetMaxCompletionTokens() == nil {
		err := api.NewError("max_tokens is required", fasthttp.StatusBadRequest, nil)
		return &err
	}
	return m.ChatCompletionsRequest.validate(tv)
}

func (m *MessagesRequest) AsString() string {
	return "messages request (req id " + m.RequestID + ")"
}

func (m *MessagesRequest) split() []Request {
	n := m.GetN()
	out := make([]Request, n)
	for i := range n {
		cp := *m
		out[i] = &cp
	}
	return out
}

func (m *MessagesRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo],
	choiceIdx int, doneFn func()) requestContext {
	reqCtx := &chatCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel, choiceIdx, doneFn),
		req:                &m.ChatCompletionsRequest,
		toolIDPrefix:       common.MessagesToolIDPrefix,
	}
	reqCtx.requestContext = reqCtx
	return reqCtx
}

var _ Request = (*MessagesRequest)(nil)
