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

package openaiserverapi

import "encoding/json"

// EmbeddingRequest is the request body for POST /v1/embeddings (OpenAI-compatible).
type EmbeddingRequest struct {
	// Model is the embedding model name.
	Model string `json:"model"`
	// Input is the text(s) to embed. Can be a single string or a slice of strings.
	Input EmbeddingInput `json:"input"`
	// EncodingFormat is optional: "float" (default) or "base64".
	EncodingFormat string `json:"encoding_format,omitempty"`
	// Dimensions is optional; if set, truncate or pad embedding to this size.
	Dimensions *int `json:"dimensions,omitempty"`
	// User is an optional end-user identifier for abuse monitoring.
	User string `json:"user,omitempty"`
}

// EmbeddingInput supports either a single string or multiple strings.
// UnmarshalJSON accepts: "a string" or ["s1", "s2"].
type EmbeddingInput []string

// UnmarshalJSON implements json.Unmarshaler so input can be string or []string.
func (e *EmbeddingInput) UnmarshalJSON(data []byte) error {
	if len(data) == 0 {
		*e = nil
		return nil
	}
	switch data[0] {
	case '"':
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		*e = []string{s}
		return nil
	case '[':
		var ss []string
		if err := json.Unmarshal(data, &ss); err != nil {
			return err
		}
		*e = ss
		return nil
	default:
		*e = nil
		return nil
	}
}

// MarshalJSON implements json.Marshaler: single element is serialized as string.
func (e EmbeddingInput) MarshalJSON() ([]byte, error) {
	if len(e) == 1 {
		return json.Marshal(e[0])
	}
	return json.Marshal([]string(e))
}

// EmbeddingResponse is the response for POST /v1/embeddings (OpenAI-compatible).
type EmbeddingResponse struct {
	Object string                  `json:"object"`
	Data   []EmbeddingDataItem     `json:"data"`
	Model  string                  `json:"model"`
	Usage  EmbeddingResponseUsage  `json:"usage"`
}

// EmbeddingDataItem is a single embedding in the response.
type EmbeddingDataItem struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// EmbeddingResponseUsage reports token usage.
type EmbeddingResponseUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
