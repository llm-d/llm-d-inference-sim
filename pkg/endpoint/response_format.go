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

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const (
	responseFormatJSONObject = "json_object"
	responseFormatJSONSchema = "json_schema"
)

// jsonObjectSchema shapes the output for json_object, which carries no schema.
var jsonObjectSchema = map[string]any{
	"type":       "object",
	"properties": map[string]any{"text": map[string]any{"type": "string"}},
	"required":   []any{"text"},
}

type responseFormat struct {
	Type       string `json:"type"`
	JSONSchema struct {
		Schema map[string]any `json:"schema"`
	} `json:"json_schema"`
}

// createStructuredOutput returns JSON content for a response_format of type json_object or
// json_schema, in place of the predefined random text. As in vLLM, the content stops at the
// token budget even when that cuts the JSON short. It returns nil tokens when the request
// does not ask for JSON, or when responses come from echo mode or a dataset.
func (c *chatCompletionReqCtx) createStructuredOutput() (*api.Tokenized, string, error) {
	config := c.runtime.Config()
	if c.req.responseFormat == nil || config.Mode != common.ModeRandom || config.MMEncoderOnly ||
		config.Dataset.DatasetPath != "" || config.Dataset.DatasetURL != "" {
		return nil, "", nil
	}

	var format responseFormat
	if json.Unmarshal(c.req.responseFormat, &format) != nil {
		return nil, "", nil
	}
	var schema map[string]any
	switch format.Type {
	case responseFormatJSONObject:
		schema = jsonObjectSchema
	case responseFormatJSONSchema:
		schema = format.JSONSchema.Schema
	default:
		return nil, "", nil
	}

	value, err := createArgument(schema, &config.ToolCalls, c.runtime.GetRandom())
	if err != nil {
		return nil, "", err
	}
	content, err := json.Marshal(value)
	if err != nil {
		return nil, "", err
	}
	tokens, strs, err := c.runtime.GetTokenizer().RenderText(string(content))
	if err != nil {
		return nil, "", err
	}

	if limit := c.maxResponseTokens(); len(tokens) > limit {
		return &api.Tokenized{Tokens: tokens[:limit], Strings: strs[:limit]}, common.LengthFinishReason, nil
	}
	return &api.Tokenized{Tokens: tokens, Strings: strs}, common.StopFinishReason, nil
}
