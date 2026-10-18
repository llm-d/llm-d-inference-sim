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
	"bytes"
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
	renderTools, err := messagesRenderTools(data, m.orig.ToolChoice)
	if err != nil {
		return err
	}
	m.renderTools = renderTools
	return nil
}

// messagesRenderTools converts the request's tools and tool_choice to the
// chat-completions form the render service takes, as vLLM converts them.
// input_schema is kept as raw JSON: the chat template renders it, and
// re-encoding the parsed map would reorder its keys. As in vLLM, a schema
// without a type gets type "object", added after its other keys.
func messagesRenderTools(data []byte, toolChoice *api.AnthropicToolChoice) (api.RenderTools, error) {
	var raw struct {
		Tools []struct {
			Name        string          `json:"name"`
			Description *string         `json:"description"`
			InputSchema json.RawMessage `json:"input_schema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return api.RenderTools{}, err
	}

	type chatFunction struct {
		Name        string          `json:"name"`
		Description *string         `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	}
	type chatTool struct {
		Type     string       `json:"type"`
		Function chatFunction `json:"function"`
	}

	var renderTools api.RenderTools
	if len(raw.Tools) > 0 {
		tools := make([]chatTool, len(raw.Tools))
		for i, t := range raw.Tools {
			tools[i] = chatTool{Type: "function",
				Function: chatFunction{Name: t.Name, Description: t.Description,
					Parameters: withObjectType(present(t.InputSchema))}}
		}
		b, err := json.Marshal(tools)
		if err != nil {
			return api.RenderTools{}, err
		}
		renderTools.Tools = b
	}

	if toolChoice != nil {
		var choice api.ToolChoice
		known := true
		switch toolChoice.Type {
		case "auto":
			choice = api.NewToolChoiceAuto()
		case "any":
			choice = api.NewToolChoiceRequired()
		case "none":
			choice = api.NewToolChoiceNone()
		case "tool":
			choice = api.NewToolChoiceFunction(toolChoice.Name)
		default:
			known = false
		}
		if known {
			b, err := json.Marshal(choice)
			if err != nil {
				return api.RenderTools{}, err
			}
			renderTools.ToolChoice = b
		}
	}
	return renderTools, nil
}

// withObjectType returns schema with "type":"object" appended when it is a JSON
// object that has no top-level type key, and schema unchanged otherwise.
func withObjectType(schema json.RawMessage) json.RawMessage {
	var keys map[string]json.RawMessage
	if json.Unmarshal(schema, &keys) != nil {
		return schema
	}
	if _, found := keys["type"]; found {
		return schema
	}
	if len(keys) == 0 {
		return json.RawMessage(`{"type":"object"}`)
	}
	body := bytes.TrimRight(schema, " \t\r\n")
	return json.RawMessage(string(body[:len(body)-1]) + `,"type":"object"}`) // body ends with the closing brace
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

// ValidateBody checks the body shape shared by /v1/messages, /v1/messages/render,
// and /v1/messages/count_tokens; only generation requires max_tokens.
func (m *MessagesRequest) ValidateBody() *api.Error {
	if err := m.validateBlocks(); err != nil {
		return err
	}
	return m.ChatCompletionsRequest.ValidateBody()
}

// validateMaxTokens checks max_tokens, which vLLM requires to be a positive
// integer wherever it parses a Messages request.
func (m *MessagesRequest) validateMaxTokens() *api.Error {
	maxTokens := m.GetMaxCompletionTokens()
	if maxTokens == nil {
		err := api.NewError("max_tokens is required", fasthttp.StatusBadRequest, nil)
		return &err
	}
	if *maxTokens <= 0 {
		err := api.NewError(common.InvalidMaxTokensErrMsg, fasthttp.StatusBadRequest, nil)
		return &err
	}
	return nil
}

func (m *MessagesRequest) Validate(tv *ToolsValidator) *api.Error {
	if err := m.ValidateBody(); err != nil {
		return err
	}
	if err := m.validateMaxTokens(); err != nil {
		return err
	}
	return m.ChatCompletionsRequest.Validate(tv)
}

func (m *MessagesRequest) AsString() string {
	return "messages request (req id " + m.RequestID + ")"
}

func (m *MessagesRequest) Split() []Request {
	n := m.GetN()
	out := make([]Request, n)
	for i := range n {
		cp := *m
		out[i] = &cp
	}
	return out
}

func (m *MessagesRequest) BuildRequestContext(runtime Runtime, channel common.Channel[*ResponseInfo],
	choiceIdx int, doneFn func()) RequestContext {
	reqCtx := &chatCompletionReqCtx{
		baseRequestContext: newBaseRequestContext(runtime, channel, choiceIdx, doneFn),
		req:                &m.ChatCompletionsRequest,
		toolIDPrefix:       common.MessagesToolIDPrefix,
	}
	reqCtx.RequestContext = reqCtx
	return reqCtx
}

var _ RenderableRequest = (*MessagesRequest)(nil)

// MessagesRenderRequest is the request type for /v1/messages/render. vLLM parses
// its body as a /v1/messages request, so max_tokens is required here, unlike on
// /v1/messages/count_tokens, which takes a MessagesRequest as is.
type MessagesRenderRequest struct {
	MessagesRequest
}

// ValidateBody adds the max_tokens requirement to the shared body checks.
func (r *MessagesRenderRequest) ValidateBody() *api.Error {
	if err := r.MessagesRequest.ValidateBody(); err != nil {
		return err
	}
	return r.validateMaxTokens()
}

var _ RenderableRequest = (*MessagesRenderRequest)(nil)
