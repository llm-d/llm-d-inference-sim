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

	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const (
	responseFormatText          = "text"
	responseFormatJSONObject    = "json_object"
	responseFormatJSONSchema    = "json_schema"
	responseFormatStructuralTag = "structural_tag"
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

// validateResponseFormat rejects what vLLM rejects: a response_format that is not an object, an
// unknown type, and a json_schema type without json_schema. It also rejects a schema that
// createArgument cannot generate a value for, which is one outside the tool parameter subset.
func validateResponseFormat(raw json.RawMessage, toolsValidator *ToolsValidator) *api.Error {
	if raw == nil {
		return nil
	}
	badRequest := func(message string) *api.Error {
		param := "response_format"
		err := api.NewError(message, fasthttp.StatusBadRequest, &param)
		return &err
	}

	var format struct {
		Type       string          `json:"type"`
		JSONSchema json.RawMessage `json:"json_schema"`
	}
	if err := json.Unmarshal(raw, &format); err != nil {
		return badRequest("response_format must be an object with a string type: " + err.Error())
	}
	switch format.Type {
	case responseFormatText, responseFormatJSONObject, responseFormatStructuralTag:
		return nil
	case responseFormatJSONSchema:
	default:
		return badRequest(fmt.Sprintf("response_format type must be one of %s, %s, %s or %s, got %q",
			responseFormatText, responseFormatJSONObject, responseFormatJSONSchema, responseFormatStructuralTag,
			format.Type))
	}

	var jsonSchema struct {
		Schema json.RawMessage `json:"schema"`
	}
	if present(format.JSONSchema) == nil || json.Unmarshal(format.JSONSchema, &jsonSchema) != nil {
		return badRequest("When response_format type is 'json_schema', the 'json_schema' field must be an object.")
	}
	if present(jsonSchema.Schema) == nil {
		return badRequest("response_format json_schema.schema is required")
	}
	var schema any
	if err := json.Unmarshal(jsonSchema.Schema, &schema); err != nil {
		return badRequest("response_format json_schema.schema is not valid JSON: " + err.Error())
	}
	if err := toolsValidator.validateParameters(schema); err != nil {
		return badRequest("response_format json_schema.schema is not supported: " + err.Error())
	}
	return nil
}

// createStructuredOutput returns JSON content for a response_format of type json_object or
// json_schema, in place of the predefined random text. As in vLLM, the content stops at the
// token budget even when that cuts the JSON short. It returns nil tokens when the request
// does not ask for JSON, or when responses come from echo mode or a dataset.
func (c *chatCompletionReqCtx) createStructuredOutput() (*api.Tokenized, string, error) {
	if c.req.responseFormat == nil || !c.runtime.RandomResponses() {
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

	value, err := createArgument(schema, &c.runtime.Config().ToolCalls, c.runtime.GetRandom())
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
