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

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
)

// EmbeddingRequest is the request body for POST /v1/embeddings (OpenAI-compatible).
// See https://developers.openai.com/api/reference/resources/embeddings/methods/create
type EmbeddingRequest struct {
	// Input: string, array of string, array of number, or array of array of number.
	Input EmbeddingInput `json:"input"`
	// Model ID (e.g. text-embedding-3-small).
	Model string `json:"model"`
	// Dimensions: optional; only supported for text-embedding-3 and later. Minimum 1.
	Dimensions *int `json:"dimensions,omitempty"`
	// EncodingFormat: "float" (default) or "base64".
	EncodingFormat string `json:"encoding_format,omitempty"`
	// User: optional end-user identifier for abuse monitoring.
	User string `json:"user,omitempty"`
}

// EmbeddingInput represents input per OpenAI spec: string, []string, []number (token ids), or [][]number.
// Use TextInputs() for text to tokenize; use TokenInputs() when input was already token ids.
type EmbeddingInput struct {
	textInputs  []string
	tokenInputs [][]int64
}

// UnmarshalJSON implements json.Unmarshaler.
// Accepts: "string", ["s1","s2"], [1,2,3] (one token sequence), [[1,2],[3,4]] (multiple token sequences).
func (e *EmbeddingInput) UnmarshalJSON(data []byte) error {
	if len(data) == 0 {
		e.textInputs = nil
		e.tokenInputs = nil
		return nil
	}
	switch data[0] {
	case '"':
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		e.textInputs = []string{s}
		e.tokenInputs = nil
		return nil
	case '[':
		var raw []interface{}
		if err := json.Unmarshal(data, &raw); err != nil {
			return err
		}
		if len(raw) == 0 {
			e.textInputs = nil
			e.tokenInputs = nil
			return nil
		}
		switch raw[0].(type) {
		case string:
			e.textInputs = make([]string, len(raw))
			for i, v := range raw {
				if s, ok := v.(string); ok {
					e.textInputs[i] = s
				} else {
					e.textInputs = nil
					e.tokenInputs = nil
					return nil
				}
			}
			e.tokenInputs = nil
			return nil
		case float64:
			// Single token sequence: array of number
			tok := sliceToInt64(raw)
			if tok == nil {
				return nil
			}
			e.textInputs = nil
			e.tokenInputs = [][]int64{tok}
			return nil
		case []interface{}:
			// Multiple token sequences: array of array of number
			e.tokenInputs = make([][]int64, len(raw))
			for i, v := range raw {
				arr, ok := v.([]interface{})
				if !ok {
					e.textInputs = nil
					e.tokenInputs = nil
					return nil
				}
				tok := sliceToInt64(arr)
				if tok == nil {
					return nil
				}
				e.tokenInputs[i] = tok
			}
			e.textInputs = nil
			return nil
		default:
			e.textInputs = nil
			e.tokenInputs = nil
			return nil
		}
	default:
		e.textInputs = nil
		e.tokenInputs = nil
		return nil
	}
}

func sliceToInt64(arr []interface{}) []int64 {
	out := make([]int64, len(arr))
	for i, v := range arr {
		switch n := v.(type) {
		case float64:
			out[i] = int64(n)
		case int:
			out[i] = int64(n)
		case int64:
			out[i] = n
		default:
			return nil
		}
	}
	return out
}

// TextInputs returns text inputs when input was string or array of string; nil otherwise.
func (e *EmbeddingInput) TextInputs() []string { return e.textInputs }

// TokenInputs returns token id inputs when input was array of number or array of array of number; nil otherwise.
func (e *EmbeddingInput) TokenInputs() [][]int64 { return e.tokenInputs }

// IsTokenInput returns true when input was token ids (array of number or array of array of number).
func (e *EmbeddingInput) IsTokenInput() bool { return len(e.tokenInputs) > 0 }

// Len returns the number of inputs (text or token sequences).
func (e *EmbeddingInput) Len() int {
	if len(e.textInputs) > 0 {
		return len(e.textInputs)
	}
	return len(e.tokenInputs)
}

// EmbeddingResponse is the response for POST /v1/embeddings (OpenAI-compatible).
// object "list", data[], model, usage.
type EmbeddingResponse struct {
	Object string                 `json:"object"`
	Data   []EmbeddingDataItem    `json:"data"`
	Model  string                 `json:"model"`
	Usage  EmbeddingResponseUsage `json:"usage"`
}

// EmbeddingDataItem is a single embedding: object "embedding", index, embedding (float array or base64 string).
type EmbeddingDataItem struct {
	Object    string      `json:"object"`
	Index     int         `json:"index"`
	Embedding interface{} `json:"embedding"` // []float32 when encoding_format is "float", base64 string when "base64"
}

// EmbeddingResponseUsage reports token usage (prompt_tokens, total_tokens).
type EmbeddingResponseUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// EncodeEmbeddingBase64 returns the embedding vector as base64 (little-endian float32 bytes).
// Used when encoding_format is "base64" per OpenAI API.
func EncodeEmbeddingBase64(emb []float32) string {
	b := make([]byte, 4*len(emb))
	for i, v := range emb {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return base64.StdEncoding.EncodeToString(b)
}
