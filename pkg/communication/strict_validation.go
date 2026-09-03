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

package communication

import (
	"encoding/json"
	"fmt"
	"mime"
	"strconv"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/valyala/fasthttp"
)

func validateStrictContentType(contentType string) *api.Error {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err == nil && mediaType == "application/json" {
		return nil
	}

	return &api.Error{
		Message: "1 validation error:\n  Unsupported Media Type: Only 'application/json' is allowed " +
			"[\"Unsupported Media Type: Only 'application/json' is allowed\"]",
		Type: "Bad Request",
		Code: fasthttp.StatusBadRequest,
	}
}

// validateStrictCompletionBody mirrors validation performed by vLLM after
// request-schema decoding. Malformed JSON and field-type errors are left to
// the endpoint's normal decoder so lenient mode keeps its existing behavior.
func validateStrictCompletionBody(body []byte, path string) *api.Error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil || fields == nil {
		return nil
	}

	if n, ok := integerField(fields, "n"); ok && n < 1 {
		return badRequest(fmt.Sprintf("n must be at least 1, got %d.", n), nil)
	}

	maxTokens, hasMaxTokens := integerField(fields, "max_tokens")
	maxCompletionTokens, hasMaxCompletionTokens := integerField(fields, "max_completion_tokens")
	effectiveMaxTokens, hasEffectiveMaxTokens := maxTokens, hasMaxTokens
	if hasMaxCompletionTokens {
		effectiveMaxTokens, hasEffectiveMaxTokens = maxCompletionTokens, true
	}
	if hasEffectiveMaxTokens && effectiveMaxTokens < 1 {
		return parameterError("max_tokens", effectiveMaxTokens,
			fmt.Sprintf("max_tokens must be at least 1, got %d.", effectiveMaxTokens))
	}

	if value, ok := numberField(fields, "temperature"); ok && value < 0 {
		formatted := formatFloat(value)
		return parameterError("temperature", formatted,
			fmt.Sprintf("temperature must be non-negative, got %s.", formatted))
	}
	if value, ok := numberField(fields, "top_p"); ok && (value <= 0 || value > 1) {
		formatted := formatFloat(value)
		return parameterError("top_p", formatted,
			fmt.Sprintf("top_p must be in (0, 1], got %s.", formatted))
	}
	if value, ok := numberField(fields, "presence_penalty"); ok && (value < -2 || value > 2) {
		return badRequest(fmt.Sprintf("presence_penalty must be in [-2, 2], got %s.", formatFloat(value)), nil)
	}
	if value, ok := numberField(fields, "frequency_penalty"); ok && (value < -2 || value > 2) {
		return badRequest(fmt.Sprintf("frequency_penalty must be in [-2, 2], got %s.", formatFloat(value)), nil)
	}
	if value, ok := numberField(fields, "repetition_penalty"); ok && value <= 0 {
		return badRequest(fmt.Sprintf("repetition_penalty must be greater than zero, got %s.", formatFloat(value)), nil)
	}
	if value, ok := numberField(fields, "min_p"); ok && (value < 0 || value > 1) {
		return badRequest(fmt.Sprintf("min_p must be in [0, 1], got %s.", formatFloat(value)), nil)
	}
	if value, ok := integerField(fields, "top_k"); ok && value < -1 {
		return badRequest(fmt.Sprintf("top_k must be 0 (disable), or at least 1, got %d.", value), nil)
	}

	logprobsField := "top_logprobs"
	if path == "/v1/completions" {
		logprobsField = "logprobs"
	}
	if value, ok := integerField(fields, logprobsField); ok && value > 20 {
		return parameterError("logprobs", value,
			fmt.Sprintf("Requested sample logprobs of %d, which is greater than max allowed: 20", value))
	}

	if minTokens, ok := integerField(fields, "min_tokens"); ok {
		if minTokens < 0 {
			return badRequest(fmt.Sprintf("min_tokens must be greater than or equal to 0, got %d.", minTokens), nil)
		}
		if hasEffectiveMaxTokens && minTokens > effectiveMaxTokens {
			return badRequest(fmt.Sprintf("min_tokens must be less than or equal to max_tokens=%d, got %d.",
				effectiveMaxTokens, minTokens), nil)
		}
	}

	if raw, ok := fields["stop"]; ok && containsEmptyStop(raw) {
		return badRequest("stop cannot contain an empty string.", nil)
	}

	return nil
}

func badRequest(message string, param *string) *api.Error {
	err := api.NewError(message, fasthttp.StatusBadRequest, param)
	return &err
}

func parameterError(name string, value any, message string) *api.Error {
	param := name
	return badRequest(fmt.Sprintf("%s (parameter=%s, value=%v)", message, name, value), &param)
}

func numberField(fields map[string]json.RawMessage, name string) (float64, bool) {
	raw, ok := fields[name]
	if !ok || string(raw) == "null" {
		return 0, false
	}
	var number json.Number
	if err := json.Unmarshal(raw, &number); err != nil {
		return 0, false
	}
	value, err := number.Float64()
	return value, err == nil
}

func integerField(fields map[string]json.RawMessage, name string) (int64, bool) {
	raw, ok := fields[name]
	if !ok || string(raw) == "null" {
		return 0, false
	}
	var number json.Number
	if err := json.Unmarshal(raw, &number); err != nil {
		return 0, false
	}
	value, err := number.Int64()
	return value, err == nil
}

func containsEmptyStop(raw json.RawMessage) bool {
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		return single == ""
	}
	var multiple []string
	if err := json.Unmarshal(raw, &multiple); err != nil {
		return false
	}
	for _, stop := range multiple {
		if stop == "" {
			return true
		}
	}
	return false
}

func formatNumber(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}

func formatFloat(value float64) string {
	formatted := formatNumber(value)
	if !strings.ContainsAny(formatted, ".eE") {
		formatted += ".0"
	}
	return formatted
}
