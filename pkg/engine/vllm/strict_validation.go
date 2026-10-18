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

package vllm

import (
	"encoding/json"
	"fmt"
	"math/big"
	"sort"
	"strconv"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/valyala/fasthttp"
)

const (
	chatCompletionsPath = "/v1/chat/completions"
	completionsPath     = "/v1/completions"
)

// validateFields applies the bundled vLLM request-model and sampling rules.
func (v *strictRequestValidator) validateFields(fields map[string]json.RawMessage, path string) *api.Error {

	if path == completionsPath {
		if err := v.validatePromptCount(fields); err != nil {
			return err
		}
	}
	if err := validateRequestModel(fields, path); err != nil {
		return err
	}
	// Beam-search requests do not construct SamplingParams.
	if boolField(fields, "use_beam_search") {
		return nil
	}

	if n, ok := integerField(fields, "n"); ok && n.Cmp(big.NewInt(v.maxN)) > 0 {
		return badRequest(fmt.Sprintf("n must be at most %d, got %d. To increase this limit, set the VLLM_MAX_N_SEQUENCES environment variable.", v.maxN, n), nil)
	}
	var bias map[string]json.RawMessage
	if json.Unmarshal(fields["logit_bias"], &bias) == nil {
		keys := make([]string, 0, len(bias))
		for key := range bias {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if _, ok := new(big.Int).SetString(strings.TrimSpace(key), 10); !ok {
				return badRequest(fmt.Sprintf("invalid literal for int() with base 10: %q", key), nil)
			}
		}
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
	temperature, hasTemperature := numberField(fields, "temperature")
	if hasTemperature && temperature < 0 {
		formatted := formatFloat(temperature)
		return parameterError("temperature", formatted,
			fmt.Sprintf("temperature must be non-negative, got %s.", formatted))
	}
	if value, ok := numberField(fields, "top_p"); ok && (value <= 0 || value > 1) {
		formatted := formatFloat(value)
		return parameterError("top_p", formatted,
			fmt.Sprintf("top_p must be in (0, 1], got %s.", formatted))
	}
	if value, ok := integerField(fields, "top_k"); ok && value.Cmp(big.NewInt(-1)) < 0 {
		return badRequest(fmt.Sprintf("top_k must be 0 (disable), or at least 1, got %d.", value), nil)
	}
	if value, ok := numberField(fields, "min_p"); ok && (value < 0 || value > 1) {
		return badRequest(fmt.Sprintf("min_p must be in [0, 1], got %s.", formatFloat(value)), nil)
	}

	maxTokens, hasMaxTokens := integerField(fields, "max_tokens")
	maxCompletionTokens, hasMaxCompletionTokens := integerField(fields, "max_completion_tokens")
	effectiveMaxTokens, hasEffectiveMaxTokens := maxTokens, hasMaxTokens
	if path == chatCompletionsPath && hasMaxCompletionTokens {
		effectiveMaxTokens, hasEffectiveMaxTokens = maxCompletionTokens, true
	}

	if !hasEffectiveMaxTokens {
		if _, present := fields["max_tokens"]; !present {
			defaultMaximum, present := v.defaultMax[path]
			effectiveMaxTokens, hasEffectiveMaxTokens = big.NewInt(defaultMaximum), present
		}
	}
	if path == completionsPath && boolField(fields, "echo") && hasEffectiveMaxTokens && effectiveMaxTokens.Sign() == 0 {
		effectiveMaxTokens = big.NewInt(1)
	}

	if minTokens, ok := integerField(fields, "min_tokens"); ok {
		if minTokens.Sign() < 0 {
			return badRequest(fmt.Sprintf("min_tokens must be greater than or equal to 0, got %d.", minTokens), nil)
		}
		if hasEffectiveMaxTokens && effectiveMaxTokens.Sign() > 0 && minTokens.Cmp(effectiveMaxTokens) > 0 {
			return badRequest(fmt.Sprintf("min_tokens must be less than or equal to max_tokens=%d, got %d.",
				effectiveMaxTokens, minTokens), nil)
		}
	}

	if raw, ok := fields["stop"]; ok && nonNull(raw) && containsEmptyStop(raw) {
		return badRequest("stop cannot contain an empty string.", nil)
	}

	if n, ok := integerField(fields, "n"); ok && hasTemperature && temperature == 0 && n.Cmp(big.NewInt(1)) > 0 {
		return badRequest(fmt.Sprintf("n must be 1 when using greedy sampling, got %d.", n), nil)
	}

	if emptyArrayField(fields, "allowed_token_ids") {
		return parameterError("allowed_token_ids", "[]", "allowed_token_ids is not None and empty!")
	}

	return nil
}

// validatePromptCount applies the prompt count limit of the text completion request model.
func (v *strictRequestValidator) validatePromptCount(fields map[string]json.RawMessage) *api.Error {
	var prompts []json.RawMessage
	if json.Unmarshal(fields["prompt"], &prompts) == nil && len(prompts) > 0 &&
		!allIntegers(prompts) && int64(len(prompts)) > v.maxPrompts {
		return v.promptCountError("prompt", len(prompts))
	}
	var embeddings []json.RawMessage
	if json.Unmarshal(fields["prompt_embeds"], &embeddings) == nil && int64(len(embeddings)) > v.maxPrompts {
		return v.promptCountError("prompt_embeds", len(embeddings))
	}
	return nil
}

func (v *strictRequestValidator) promptCountError(parameter string, count int) *api.Error {
	message := fmt.Sprintf("%s list length %d exceeds the maximum allowed count of %d. "+
		"To increase this limit, set the VLLM_MAX_COMPLETION_PROMPTS environment variable.", parameter, count, v.maxPrompts)
	param := parameter
	return badRequest(message, &param)
}

// allIntegers reports whether every element is an integer, which vLLM reads as one
// tokenized prompt rather than as several prompts.
func allIntegers(values []json.RawMessage) bool {
	for _, value := range values {
		if _, ok := integerValue(value); !ok {
			return false
		}
	}
	return true
}

func badRequest(message string, param *string) *api.Error {
	err := api.NewError(message, fasthttp.StatusBadRequest, param)
	return &err
}

func parameterError(name string, value any, message string) *api.Error {
	param := name
	if value == nil {
		return badRequest(fmt.Sprintf("%s (parameter=%s)", message, name), &param)
	}
	return badRequest(fmt.Sprintf("%s (parameter=%s, value=%v)", message, name, value), &param)
}

func numberField(fields map[string]json.RawMessage, name string) (float64, bool) {
	raw := fields[name]
	if !nonNull(raw) {
		return 0, false
	}
	var number json.Number
	if err := json.Unmarshal(raw, &number); err != nil {
		return 0, false
	}
	value, err := number.Float64()
	return value, err == nil
}

func integerField(fields map[string]json.RawMessage, name string) (*big.Int, bool) {
	return integerValue(fields[name])
}

// integerValue parses a JSON Schema integer, which is unbounded and may use decimal or
// exponent notation.
func integerValue(raw json.RawMessage) (*big.Int, bool) {
	rational, ok := new(big.Rat).SetString(string(raw))
	if !ok || !rational.IsInt() {
		return new(big.Int), false
	}
	return rational.Num(), true
}

func emptyArrayField(fields map[string]json.RawMessage, name string) bool {
	raw := fields[name]
	if !nonNull(raw) {
		return false
	}
	var values []json.RawMessage
	return json.Unmarshal(raw, &values) == nil && values != nil && len(values) == 0
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

// validateRequestModel covers validators outside the vLLM request JSON schema.
func validateRequestModel(fields map[string]json.RawMessage, path string) *api.Error {
	if truthyField(fields, "stream_options") && !boolField(fields, "stream") {
		return parameterError("stream_options", nil, "Stream options can only be defined when `stream=True`.")
	}
	if value, ok := integerField(fields, "prompt_logprobs"); ok {
		if boolField(fields, "stream") && (value.Sign() > 0 || value.Cmp(big.NewInt(-1)) == 0) {
			return parameterError("prompt_logprobs", nil, "`prompt_logprobs` are not available when `stream=True`.")
		}
		if value.Cmp(big.NewInt(-1)) < 0 {
			return parameterError("prompt_logprobs", value, "`prompt_logprobs` must be a positive value or -1.")
		}
	}
	if path == chatCompletionsPath {
		if value, ok := integerField(fields, "top_logprobs"); ok {
			if value.Cmp(big.NewInt(-1)) < 0 {
				return parameterError("top_logprobs", value, "`top_logprobs` must be a positive value or -1.")
			}
			if (value.Sign() > 0 || value.Cmp(big.NewInt(-1)) == 0) && !boolField(fields, "logprobs") {
				return parameterError("top_logprobs", nil, "when using `top_logprobs`, `logprobs` must be set to true.")
			}
		}
		if err := validateTools(fields); err != nil {
			return err
		}
		if boolField(fields, "continue_final_message") && boolField(fields, "add_generation_prompt") {
			return badRequest("Cannot set both `continue_final_message` and `add_generation_prompt` to True.", nil)
		}
	} else {
		if value, ok := integerField(fields, "logprobs"); ok && value.Sign() < 0 {
			return parameterError("logprobs", value, "`logprobs` must be a positive value.")
		}
		prompt := fields["prompt"]
		if (!nonNull(prompt) || string(prompt) == `""`) && !truthyField(fields, "prompt_embeds") {
			return badRequest("Either prompt or prompt_embeds must be provided and non-empty.", nil)
		}
	}
	if raw := fields["cache_salt"]; nonNull(raw) && string(raw) == `""` {
		return badRequest("Parameter 'cache_salt' must be a non-empty string if provided.", nil)
	}
	var format map[string]json.RawMessage
	_ = json.Unmarshal(fields["response_format"], &format)
	if stringField(format, "type") == "json_schema" && !nonNull(format["json_schema"]) {
		return parameterError("response_format", nil, "When response_format type is 'json_schema', the 'json_schema' field must be provided.")
	}
	if err := validateStructuredOutputs(fields, format); err != nil {
		return err
	}
	var repetition map[string]json.RawMessage
	if json.Unmarshal(fields["repetition_detection"], &repetition) == nil && repetition != nil {
		maximum, _ := integerField(repetition, "max_pattern_size")
		minimum, _ := integerField(repetition, "min_pattern_size")
		count, _ := integerField(repetition, "min_count")
		if maximum.Sign() < 0 || minimum.Sign() < 0 || minimum.Cmp(maximum) > 0 {
			return badRequest("max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size. Set both to 0 to disable repetitive pattern detection.", nil)
		}
		if maximum.Sign() > 0 && count.Cmp(big.NewInt(2)) < 0 {
			return badRequest("min_count must be >= 2 to detect repetitive patterns in engine output. If you do not wish to detect repetitive patterns, set max_pattern_size to 0.", nil)
		}
	}
	return nil
}

func validateTools(fields map[string]json.RawMessage) *api.Error {
	if emptyArrayField(fields, "tools") {
		return badRequest("`tools` must not be an empty array. Either provide at least one tool or omit the field entirely.", nil)
	}
	choice := fields["tool_choice"]
	if !nonNull(choice) || string(choice) == `"none"` {
		return nil
	}
	if !nonNull(fields["tools"]) {
		return badRequest("When using `tool_choice`, `tools` must be set.", nil)
	}
	var named struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(choice, &named) != nil {
		return nil
	}
	if named.Function.Name == "" {
		return badRequest("Invalid `name` in `function`: `` in `tool_choice`! Correct usage: `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`", nil)
	}
	var tools []struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	_ = json.Unmarshal(fields["tools"], &tools)
	for _, tool := range tools {
		if tool.Function.Name == named.Function.Name {
			return nil
		}
	}
	return badRequest("The tool specified in `tool_choice` does not match any of the specified `tools`", nil)
}

func validateStructuredOutputs(fields, format map[string]json.RawMessage) *api.Error {
	var constraints map[string]json.RawMessage
	if json.Unmarshal(fields["structured_outputs"], &constraints) != nil || constraints == nil {
		return nil
	}
	keys := []string{"json", "regex", "choice", "grammar", "json_object", "structural_tag"}
	count := 0
	for _, key := range keys {
		if nonNull(constraints[key]) {
			count++
		}
	}
	if count != 1 {
		return badRequest("structured_outputs must specify exactly one of json, regex, choice, grammar, json_object or structural_tag", nil)
	}
	if boolField(fields, "use_beam_search") {
		return nil
	}
	// response_format is merged into the existing constraints by to_sampling_params.
	formatKey := map[string]string{"json_object": "json_object", "json_schema": "json", "structural_tag": "structural_tag"}[stringField(format, "type")]
	if formatKey != "" && !nonNull(constraints[formatKey]) {
		return badRequest("response_format and structured_outputs specify conflicting constraints", nil)
	}
	if emptyArrayField(constraints, "choice") {
		return badRequest("Choice '[]' cannot be an empty list", nil)
	}
	if raw := constraints["grammar"]; nonNull(raw) && strings.TrimSpace(stringField(constraints, "grammar")) == "" {
		return badRequest("structured_outputs.grammar cannot be an empty string", nil)
	}
	return nil
}

func nonNull(raw json.RawMessage) bool { return len(raw) != 0 && string(raw) != "null" }
func boolField(fields map[string]json.RawMessage, name string) bool {
	return string(fields[name]) == "true"
}
func stringField(fields map[string]json.RawMessage, name string) string {
	var value string
	_ = json.Unmarshal(fields[name], &value)
	return value
}
func truthyField(fields map[string]json.RawMessage, name string) bool {
	raw := fields[name]
	if !nonNull(raw) {
		return false
	}
	var value any
	_ = json.Unmarshal(raw, &value)
	switch v := value.(type) {
	case bool:
		return v
	case string:
		return v != ""
	case []any:
		return len(v) != 0
	case map[string]any:
		return len(v) != 0
	default:
		return true
	}
}
