/*
Copyright 2025 The llm-d-inference-sim Authors.

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

package common

// LogprobsContent represents logprobs for a single token in chat completions
type LogprobsContent struct {
	// Token is the token string
	Token string `json:"token"`
	// Logprob is the log probability of the token
	Logprob float64 `json:"logprob"`
	// Bytes is the byte representation of the token
	Bytes []int `json:"bytes"`
	// TopLogprobs is the list of top alternative tokens along their log probabilities
	TopLogprobs []LogprobsContent `json:"top_logprobs,omitempty"`
}

// ChatLogprobs represents logprobs for chat completion responses
type ChatLogprobs struct {
	// Content is an array of logprobs for each token in the content
	Content []LogprobsContent `json:"content"`
}

// TextLogprobs represents logprobs for text completion responses
type TextLogprobs struct {
	// Tokens is an array of tokens
	Tokens []string `json:"tokens"`
	// TokenLogprobs is an array of log probabilities for each token
	TokenLogprobs []float64 `json:"token_logprobs"`
	// TopLogprobs is an array of objects containing the top alternative tokens
	TopLogprobs []map[string]float64 `json:"top_logprobs"`
	// TextOffset is an array of character offsets
	TextOffset []int `json:"text_offset"`
}
