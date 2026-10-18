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
	"bytes"
	"strings"
	"testing"
)

func TestBundledStrictSchema(t *testing.T) {
	v, err := compileStrictRequestValidator(strictOpenAPI, 16384, 1024)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, path, body, message string }{
		{"chat required messages", chatCompletionsPath, `{}`, "schema validation failed"},
		{"chat structure", chatCompletionsPath, `{"messages":"hi"}`, "schema validation failed"},
		{"nested field", chatCompletionsPath, `{"messages":[{"role":"user","content":12}]}`, "schema validation failed"},
		{"wrong stream type", chatCompletionsPath, `{"messages":[],"stream":"false"}`, "schema validation failed"},
		{"fractional n", chatCompletionsPath, `{"messages":[],"n":1.5}`, "schema validation failed"},
		{"seed boundary", chatCompletionsPath, `{"messages":[],"seed":9223372036854775807}`, ""},
		{"negative seed boundary", chatCompletionsPath, `{"messages":[],"seed":-9223372036854775808}`, ""},
		{"integer notation", chatCompletionsPath, `{"messages":[],"top_k":-2.0}`, "top_k must be 0 (disable), or at least 1, got -2."},
		{"int64 seed", chatCompletionsPath, `{"messages":[],"seed":9223372036854775808}`, "schema validation failed"},
		{"negative prompt token", completionsPath, `{"prompt":[-1]}`, "schema validation failed"},
		{"stop token type", completionsPath, `{"prompt":"hi","stop_token_ids":["1"]}`, "schema validation failed"},
		{"truncate lower bound", completionsPath, `{"prompt":"hi","truncate_prompt_tokens":-2}`, "schema validation failed"},
		{"null min tokens", chatCompletionsPath, `{"messages":[],"min_tokens":null}`, "schema validation failed"},
		{"large n", chatCompletionsPath, `{"messages":[],"n":9223372036854775808}`, "n must be at most 16384, got 9223372036854775808."},
		{"large min tokens", completionsPath, `{"prompt":"hi","min_tokens":9223372036854775808}`, "min_tokens must be less than or equal to max_tokens=16, got 9223372036854775808."},
		{"n limit", chatCompletionsPath, `{"messages":[],"n":16385}`, "n must be at most 16384, got 16385. To increase this limit, set the VLLM_MAX_N_SEQUENCES environment variable."},
		{"text omitted maximum", completionsPath, `{"prompt":"hi","min_tokens":17}`, "min_tokens must be less than or equal to max_tokens=16, got 17."},
		{"text ignores chat extension", completionsPath, `{"prompt":"hi","min_tokens":17,"max_completion_tokens":32}`, "min_tokens must be less than or equal to max_tokens=16, got 17."},
		{"chat explicit maximum", chatCompletionsPath, `{"messages":[],"max_tokens":32,"max_completion_tokens":4,"min_tokens":5}`, "min_tokens must be less than or equal to max_tokens=4, got 5."},
		{"chat null override", chatCompletionsPath, `{"messages":[],"max_tokens":4,"max_completion_tokens":null,"min_tokens":5}`, "min_tokens must be less than or equal to max_tokens=4, got 5."},
		{"valid chat", chatCompletionsPath, `{"messages":[{"role":"user","content":"hi"}]}`, ""},
		{"unknown extensions", chatCompletionsPath, `{"messages":[],"custom_extension":{"key":1}}`, ""},
		{"chat omitted maximum", chatCompletionsPath, `{"messages":[],"min_tokens":17}`, ""},
		{"text null maximum", completionsPath, `{"prompt":"hi","max_tokens":null,"min_tokens":17}`, ""},
		{"text explicit maximum", completionsPath, `{"prompt":"hi","max_tokens":32,"min_tokens":17}`, ""},
		{"chat override", chatCompletionsPath, `{"messages":[],"max_tokens":4,"max_completion_tokens":32,"min_tokens":17}`, ""},
		{"stop null", chatCompletionsPath, `{"messages":[],"stop":null}`, ""},
		{"trailing JSON", chatCompletionsPath, `{"messages":[]} {}`, "Request body must contain one JSON value"},
		{"invalid JSON", completionsPath, `{`, "Invalid JSON request body"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := v.Validate([]byte(tc.body), tc.path)
			if tc.message == "" {
				if got != nil {
					t.Fatal(got)
				}
				return
			}
			if got == nil || !strings.Contains(got.Message, tc.message) {
				t.Fatalf("want %q; got %v", tc.message, got)
			}
		})
	}
}

func TestStrictSchemaStartup(t *testing.T) {
	t.Setenv("VLLM_MAX_N_SEQUENCES", "4")
	v, err := loadStrictRequestValidator()
	if err != nil {
		t.Fatal(err)
	}
	if v.Validate([]byte(`{"messages":[],"n":5}`), chatCompletionsPath) == nil {
		t.Fatal("environment limit ignored")
	}
	for _, value := range []string{"0", "-1", "many", ""} {
		t.Setenv("VLLM_MAX_N_SEQUENCES", value)
		if _, err := loadStrictRequestValidator(); err == nil {
			t.Fatalf("accepted limit %q", value)
		}
	}
	document := bytes.ReplaceAll(strictOpenAPI, []byte("#/components/schemas/ChatCompletionRequest"), []byte("https://example.invalid/request.json"))
	if _, err := compileStrictRequestValidator(document, 16384, 1024); err == nil {
		t.Fatal("external reference accepted")
	}
}

func TestStrictPromptCountLimit(t *testing.T) {
	t.Setenv("VLLM_MAX_COMPLETION_PROMPTS", "2")
	v, err := loadStrictRequestValidator()
	if err != nil {
		t.Fatal(err)
	}
	if err := v.Validate([]byte(`{"prompt":["a","b","c"]}`), completionsPath); err == nil ||
		!strings.Contains(err.Message, "prompt list length 3 exceeds the maximum allowed count of 2. "+
			"To increase this limit, set the VLLM_MAX_COMPLETION_PROMPTS environment variable.") {
		t.Fatalf("prompt list limit ignored: %v", err)
	}
	// A list of token IDs is one tokenized prompt, not a list of prompts.
	if err := v.Validate([]byte(`{"prompt":[1,2,3]}`), completionsPath); err != nil {
		t.Fatalf("tokenized prompt rejected: %v", err)
	}
	if err := v.Validate([]byte(`{"prompt":"a","prompt_embeds":["e","e","e"]}`), completionsPath); err == nil ||
		!strings.Contains(err.Message, "prompt_embeds list length 3 exceeds the maximum allowed count of 2.") {
		t.Fatalf("prompt_embeds limit ignored: %v", err)
	}
	if err := v.Validate([]byte(`{"prompt":["a","b"]}`), completionsPath); err != nil {
		t.Fatalf("request within the limit rejected: %v", err)
	}
	chat := `{"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"},{"role":"user","content":"c"}]}`
	if err := v.Validate([]byte(chat), chatCompletionsPath); err != nil {
		t.Fatalf("chat request limited by the completion prompt count: %v", err)
	}
	for _, value := range []string{"0", "-1", "many", ""} {
		t.Setenv("VLLM_MAX_COMPLETION_PROMPTS", value)
		if _, err := loadStrictRequestValidator(); err == nil {
			t.Fatalf("accepted limit %q", value)
		}
	}
}

func TestStrictModelConstraints(t *testing.T) {
	v, err := loadStrictRequestValidator()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, fields, message string
		chatOnly, textOnly    bool
	}{
		{name: "stream options", fields: `"stream_options":{"include_usage":true}`, message: "Stream options can only be defined when `stream=True`. (parameter=stream_options)"},
		{name: "empty stream options", fields: `"stream_options":{}`},
		{name: "stream options null", fields: `"stream_options":null`},
		{name: "streaming options", fields: `"stream_options":{"include_usage":true},"stream":true`},
		{name: "stream prompt logprobs", fields: `"prompt_logprobs":1,"stream":true`, message: "`prompt_logprobs` are not available when `stream=True`. (parameter=prompt_logprobs)"},
		{name: "stream all prompt logprobs", fields: `"prompt_logprobs":-1,"stream":true`, message: "`prompt_logprobs` are not available when `stream=True`. (parameter=prompt_logprobs)"},
		{name: "stream zero prompt logprobs", fields: `"prompt_logprobs":0,"stream":true`},
		{name: "negative prompt logprobs", fields: `"prompt_logprobs":-2`, message: "`prompt_logprobs` must be a positive value or -1. (parameter=prompt_logprobs, value=-2)"},
		{name: "chat missing logprobs", fields: `"top_logprobs":2`, message: "when using `top_logprobs`, `logprobs` must be set to true. (parameter=top_logprobs)", chatOnly: true},
		{name: "chat false logprobs", fields: `"top_logprobs":-1,"logprobs":false`, message: "when using `top_logprobs`, `logprobs` must be set to true. (parameter=top_logprobs)", chatOnly: true},
		{name: "chat zero logprobs", fields: `"top_logprobs":0,"logprobs":false`, chatOnly: true},
		{name: "chat all logprobs", fields: `"top_logprobs":-1,"logprobs":true`, chatOnly: true},
		{name: "chat negative top logprobs", fields: `"top_logprobs":-2`, message: "`top_logprobs` must be a positive value or -1. (parameter=top_logprobs, value=-2)", chatOnly: true},
		{name: "text negative logprobs", fields: `"logprobs":-1`, message: "`logprobs` must be a positive value. (parameter=logprobs, value=-1)", textOnly: true},
		{name: "invalid logit bias key", fields: `"logit_bias":{"word":1}`, message: "invalid literal for int()"},
		{name: "clamped logit bias", fields: `"logit_bias":{"1":101}`},
		{name: "empty salt", fields: `"cache_salt":""`, message: "Parameter 'cache_salt' must be a non-empty string if provided."},
		{name: "null salt", fields: `"cache_salt":null`},
		{name: "generation prompt conflict", fields: `"continue_final_message":true,"add_generation_prompt":true`, message: "Cannot set both `continue_final_message` and `add_generation_prompt` to True.", chatOnly: true},
		{name: "generation prompt omitted", fields: `"continue_final_message":true`, chatOnly: true},
		{name: "response format missing schema", fields: `"response_format":{"type":"json_schema"}`, message: "When response_format type is 'json_schema', the 'json_schema' field must be provided. (parameter=response_format)"},
		{name: "response format schema", fields: `"response_format":{"type":"json_schema","json_schema":{"name":"out","schema":{"type":"object"}}}`},
		{name: "empty tools", fields: `"tools":[]`, message: "`tools` must not be an empty array. Either provide at least one tool or omit the field entirely.", chatOnly: true},
		{name: "missing tools", fields: `"tool_choice":"auto"`, message: "When using `tool_choice`, `tools` must be set.", chatOnly: true},
		{name: "tool mismatch", fields: `"tools":[{"function":{"name":"a"}}],"tool_choice":{"function":{"name":"b"}}`, message: "The tool specified in `tool_choice` does not match any of the specified `tools`", chatOnly: true},
		{name: "matching tool", fields: `"tools":[{"function":{"name":"a"}}],"tool_choice":{"function":{"name":"a"}}`, chatOnly: true},
		{name: "tool none", fields: `"tool_choice":"none"`, chatOnly: true},
		{name: "no structured constraint", fields: `"structured_outputs":{}`, message: "structured_outputs must specify exactly one"},
		{name: "multiple structured constraints", fields: `"structured_outputs":{"regex":"a","grammar":"b"}`, message: "structured_outputs must specify exactly one"},
		{name: "empty choices", fields: `"structured_outputs":{"choice":[]}`, message: "Choice '[]' cannot be an empty list"},
		{name: "blank grammar", fields: `"structured_outputs":{"grammar":"  "}`, message: "structured_outputs.grammar cannot be an empty string"},
		{name: "structured choice", fields: `"structured_outputs":{"choice":["a","b"]}`},
		{name: "conflicting response format", fields: `"structured_outputs":{"regex":"a"},"response_format":{"type":"json_object"}`, message: "response_format and structured_outputs specify conflicting constraints"},
		{name: "matching response format", fields: `"structured_outputs":{"json_object":true},"response_format":{"type":"json_object"}`},
		{name: "negative repetition size", fields: `"repetition_detection":{"max_pattern_size":-1}`, message: "max_pattern_size, min_pattern_size must be >=0"},
		{name: "repetition range", fields: `"repetition_detection":{"max_pattern_size":2,"min_pattern_size":3}`, message: "max_pattern_size, min_pattern_size must be >=0"},
		{name: "repetition count", fields: `"repetition_detection":{"max_pattern_size":2,"min_count":1}`, message: "min_count must be >= 2"},
		{name: "repetition boundary", fields: `"repetition_detection":{"max_pattern_size":2,"min_pattern_size":2,"min_count":2}`},
		{name: "repetition disabled", fields: `"repetition_detection":{}`},
		{name: "beam ignores sampling-only ranges", fields: `"use_beam_search":true,"top_p":-1,"min_tokens":-1,"temperature":0,"n":2`},
	} {
		for _, path := range []string{chatCompletionsPath, completionsPath} {
			if tc.chatOnly && path == completionsPath || tc.textOnly && path == chatCompletionsPath {
				continue
			}
			t.Run(tc.name+path, func(t *testing.T) {
				prefix := `{"model":"m","prompt":"hi",`
				if path == chatCompletionsPath {
					prefix = `{"model":"m","messages":[],`
				}
				got := v.Validate([]byte(prefix+tc.fields+"}"), path)
				if tc.message == "" {
					if got != nil {
						t.Fatal(got)
					}
					return
				}
				if got == nil || !strings.Contains(got.Message, tc.message) {
					t.Fatalf("want %q; got %v", tc.message, got)
				}
			})
		}
	}
}
