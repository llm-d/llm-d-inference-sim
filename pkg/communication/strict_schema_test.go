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
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func strictSchemaFixture(t *testing.T) []byte {
	t.Helper()
	paths := map[string]any{}
	for _, path := range []string{"/v1/chat/completions", "/v1/completions"} {
		paths[path] = map[string]any{"post": map[string]any{
			"requestBody": map[string]any{"content": map[string]any{
				"application/json": map[string]any{"schema": map[string]any{"$ref": "#/components/schemas/Request"}},
			}},
		}}
	}
	document := map[string]any{
		"openapi": "3.1.0", "paths": paths,
		"components": map[string]any{"schemas": map[string]any{
			"Request": map[string]any{
				"type": "object", "required": []string{"model", "messages"},
				"properties": map[string]any{
					"model": map[string]any{"type": "string"},
					"messages": map[string]any{"type": "array", "items": map[string]any{
						"type": "object", "required": []string{"role"},
						"properties": map[string]any{"role": map[string]any{"enum": []string{"user", "assistant"}}},
					}},
					"n":          map[string]any{"type": "integer"},
					"max_tokens": map[string]any{"type": []string{"integer", "null"}, "default": 16},
					"min_tokens": map[string]any{"type": "integer"},
					"stream":     map[string]any{"type": "boolean"},
				},
			},
		}},
	}
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestStrictSchemaValidation(t *testing.T) {
	v, err := compileStrictRequestValidator(strictSchemaFixture(t), 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, body string
		valid      bool
	}{
		{"valid", `{"model":"m","messages":[{"role":"user"}],"n":2}`, true},
		{"missing model", `{"messages":[]}`, false},
		{"missing messages", `{"model":"m"}`, false},
		{"invalid role", `{"model":"m","messages":[{"role":"invalid"}]}`, false},
		{"wrong type", `{"model":"m","messages":[],"stream":"false"}`, false},
		{"n runtime limit", `{"model":"m","messages":[],"n":3}`, false},
		{"omitted maximum", `{"model":"m","messages":[],"min_tokens":17}`, false},
		{"maximum boundary", `{"model":"m","messages":[],"min_tokens":16}`, true},
		{"explicit maximum", `{"model":"m","messages":[],"max_tokens":32,"min_tokens":17}`, true},
		{"null maximum", `{"model":"m","messages":[],"max_tokens":null,"min_tokens":17}`, true},
		{"trailing JSON", `{"model":"m","messages":[]}{}`, false},
		{"not object", `[]`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, path := range []string{"/v1/chat/completions", "/v1/completions"} {
				if got := v.validate([]byte(tc.body), path); (got == nil) != tc.valid {
					t.Fatalf("%s: valid=%v, error=%v", path, tc.valid, got)
				}
			}
		})
	}
}

func TestStrictSchemaStartup(t *testing.T) {
	if _, err := loadStrictRequestValidator(""); err == nil {
		t.Fatal("missing schema accepted")
	}
	file := filepath.Join(t.TempDir(), "openapi.json")
	if err := os.WriteFile(file, strictSchemaFixture(t), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_MAX_N_SEQUENCES", "4")
	v, err := loadStrictRequestValidator(file)
	if err != nil || v.maxN != 4 {
		t.Fatalf("validator=%v error=%v", v, err)
	}
	t.Setenv("VLLM_MAX_N_SEQUENCES", "0")
	if _, err := loadStrictRequestValidator(file); err == nil {
		t.Fatal("invalid limit accepted")
	}
}

func TestStrictSchemaRejectsExternalReferences(t *testing.T) {
	document := strings.ReplaceAll(string(strictSchemaFixture(t)), "#/components/schemas/Request", "https://example.invalid/request.json")
	if _, err := compileStrictRequestValidator([]byte(document), 2); err == nil {
		t.Fatal("external reference accepted")
	}
}

func TestStrictCompletionMaximumDoesNotUseChatExtension(t *testing.T) {
	v, err := compileStrictRequestValidator(strictSchemaFixture(t), 2)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"m","messages":[],"max_completion_tokens":32,"min_tokens":17}`)
	if err := v.validate(body, "/v1/completions"); err == nil {
		t.Fatal("chat-only limit bypassed completion default")
	}
	if err := v.validate(body, "/v1/chat/completions"); err != nil {
		t.Fatal(err)
	}
	body = []byte(`{"max_tokens":16,"max_completion_tokens":32,"min_tokens":17}`)
	if err := validateStrictCompletionBody(body, "/v1/completions"); err == nil {
		t.Fatal("chat-only limit bypassed explicit completion maximum")
	}
	if err := validateStrictCompletionBody(body, "/v1/chat/completions"); err != nil {
		t.Fatal(err)
	}
}
