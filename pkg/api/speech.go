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

package api

// CreateSpeechRequest is the request body for POST /v1/audio/speech.
// Fields not needed by the simulator are accepted and ignored by JSON decoding.
type CreateSpeechRequest struct {
	Input          *string  `json:"input"`
	Model          string   `json:"model,omitempty"`
	Instructions   string   `json:"instructions,omitempty"`
	ResponseFormat *string  `json:"response_format,omitempty"`
	Speed          *float64 `json:"speed,omitempty"`
	Stream         bool     `json:"stream,omitempty"`
	StreamFormat   *string  `json:"stream_format,omitempty"`
}

// SpeechInputTokenDetails reports simulated token usage by input modality.
type SpeechInputTokenDetails struct {
	TextTokens  int `json:"text_tokens"`
	AudioTokens int `json:"audio_tokens"`
}

// SpeechTokenUsage reports simulated token usage for speech generation.
type SpeechTokenUsage struct {
	InputTokens       int                     `json:"input_tokens"`
	OutputTokens      int                     `json:"output_tokens"`
	TotalTokens       int                     `json:"total_tokens"`
	InputTokenDetails SpeechInputTokenDetails `json:"input_token_details"`
}

// SpeechAudioDeltaEvent carries one base64-encoded audio chunk.
type SpeechAudioDeltaEvent struct {
	Type           string `json:"type"`
	Audio          string `json:"audio"`
	ResponseFormat string `json:"response_format"`
}

// SpeechAudioDoneEvent terminates an SSE speech stream.
type SpeechAudioDoneEvent struct {
	Type  string           `json:"type"`
	Usage SpeechTokenUsage `json:"usage"`
}
