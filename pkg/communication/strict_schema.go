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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/santhosh-tekuri/jsonschema/v5"
)

type strictRequestValidator struct {
	schemas    map[string]*jsonschema.Schema
	maxN       int64
	defaultMax map[string]int64
}

func loadStrictRequestValidator(filename string) (*strictRequestValidator, error) {
	if filename == "" {
		return nil, errors.New("--strict requires --strict-openapi with a target vLLM OpenAPI 3.1 document")
	}
	document, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	maxN := int64(16384)
	if value, exists := os.LookupEnv("VLLM_MAX_N_SEQUENCES"); exists {
		maxN, err = strconv.ParseInt(value, 10, 64)
		if err != nil || maxN < 1 {
			return nil, errors.New("VLLM_MAX_N_SEQUENCES must be a positive integer")
		}
	}
	return compileStrictRequestValidator(document, maxN)
}

func compileStrictRequestValidator(document []byte, maxN int64) (*strictRequestValidator, error) {
	var spec struct {
		OpenAPI string `json:"openapi"`
		Paths   map[string]struct {
			Post struct {
				RequestBody struct {
					Content map[string]struct {
						Schema json.RawMessage `json:"schema"`
					} `json:"content"`
				} `json:"requestBody"`
			} `json:"post"`
		} `json:"paths"`
	}
	if err := json.Unmarshal(document, &spec); err != nil {
		return nil, err
	}
	if !strings.HasPrefix(spec.OpenAPI, "3.1.") {
		return nil, errors.New("strict validation requires OpenAPI 3.1 (JSON Schema 2020-12)")
	}
	compiler := jsonschema.NewCompiler()
	compiler.Draft = jsonschema.Draft2020
	compiler.ExtractAnnotations = true
	// Resolve only the supplied snapshot, never remote references at runtime.
	compiler.LoadURL = func(url string) (io.ReadCloser, error) {
		return nil, fmt.Errorf("external schema reference is not allowed: %s", url)
	}
	const source = "https://strict.invalid/openapi.json"
	if err := compiler.AddResource(source, bytes.NewReader(document)); err != nil {
		return nil, err
	}
	validator := &strictRequestValidator{schemas: make(map[string]*jsonschema.Schema), maxN: maxN, defaultMax: make(map[string]int64)}
	for _, path := range []string{"/v1/chat/completions", "/v1/completions"} {
		if len(spec.Paths[path].Post.RequestBody.Content["application/json"].Schema) == 0 {
			return nil, fmt.Errorf("missing JSON request schema for %s", path)
		}
		pointer := strings.ReplaceAll(path, "/", "~1")
		schema, err := compiler.Compile(source + "#/paths/" + pointer + "/post/requestBody/content/application~1json/schema")
		if err != nil {
			return nil, err
		}
		validator.schemas[path] = schema
		for schema.Ref != nil {
			schema = schema.Ref
		}
		if property := schema.Properties["max_tokens"]; property != nil && property.Default != nil {
			value, err := strconv.ParseInt(fmt.Sprint(property.Default), 10, 64)
			if err != nil || value < 1 {
				return nil, fmt.Errorf("invalid max_tokens default for %s", path)
			}
			validator.defaultMax[path] = value
		}
	}
	return validator, nil
}

func (v *strictRequestValidator) validate(body []byte, path string) *api.Error {
	var value any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return badRequest("Invalid JSON request body", nil)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return badRequest("Request body must contain one JSON value", nil)
	}
	if err := v.schemas[path].Validate(value); err != nil {
		return badRequest(fmt.Sprintf("Request schema validation failed: %s", err), nil)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return badRequest("Expected a JSON object", nil)
	}
	if n, ok := integerField(fields, "n"); ok && n > v.maxN {
		return badRequest(fmt.Sprintf("n must be at most %d, got %d.", v.maxN, n), nil)
	}
	if maximum := v.defaultMax[path]; maximum > 0 {
		if _, present := fields["max_tokens"]; !present {
			if _, explicit := integerField(fields, "max_completion_tokens"); path != "/v1/chat/completions" || !explicit {
				if minTokens, ok := integerField(fields, "min_tokens"); ok && minTokens > maximum {
					return badRequest(fmt.Sprintf("min_tokens must be less than or equal to max_tokens=%d, got %d.", maximum, minTokens), nil)
				}
			}
		}
	}
	return nil
}
