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

package api

// SyntheticWAVData is a base64-encoded 44-byte silent WAV file (24 kHz, mono, 16-bit, zero frames)
// used as the synthetic audio payload for /v1/audio/speech responses.
const SyntheticWAVData = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQAAAAA="

// AudioSpeechRequest is the request body for POST /v1/audio/speech (OpenAI-compatible TTS).
type AudioSpeechRequest struct {
	// Input is the text to synthesize (required).
	Input string `json:"input"`
	// Model is the TTS model to use.
	Model string `json:"model,omitempty"`
	// Voice is the speaker voice (e.g. alloy, echo, fable, onyx, nova, shimmer for OpenAI;
	// vivian, ryan, aiden, etc. for Qwen3-TTS).
	Voice string `json:"voice,omitempty"`
	// ResponseFormat is the audio encoding: mp3 (default), opus, aac, flac, wav, pcm.
	ResponseFormat string `json:"response_format,omitempty"`
	// Speed controls playback speed in the range [0.25, 4.0]. Default is 1.0.
	Speed float64 `json:"speed,omitempty"`
	// Stream enables streaming. When true the response is a series of SSE events
	// (speech.audio.delta, speech.audio.done).
	Stream bool `json:"stream,omitempty"`
	// StreamFormat selects the streaming wire format: "sse" (default) or "audio" (raw bytes).
	StreamFormat string `json:"stream_format,omitempty"`

	// Qwen3-TTS / vllm-omni extended fields — accepted and ignored by the simulator.
	TaskType         string      `json:"task_type,omitempty"`
	Language         string      `json:"language,omitempty"`
	Instructions     string      `json:"instructions,omitempty"`
	SampleRate       *int        `json:"sample_rate,omitempty"`
	RefAudio         interface{} `json:"ref_audio,omitempty"`
	RefText          string      `json:"ref_text,omitempty"`
	XVectorOnly      *bool       `json:"x_vector_only_mode,omitempty"`
	SpeakerEmbedding interface{} `json:"speaker_embedding,omitempty"`
	MaxNewTokens     *int64      `json:"max_new_tokens,omitempty"`
	Seed             *int64      `json:"seed,omitempty"`
}

// AudioSpeechStreamDeltaEvent is the SSE payload for a speech.audio.delta event.
type AudioSpeechStreamDeltaEvent struct {
	Type  string `json:"type"`
	Audio string `json:"audio"` // base64-encoded audio chunk
}

// AudioSpeechStreamDoneEvent is the SSE payload for a speech.audio.done event.
type AudioSpeechStreamDoneEvent struct {
	Type  string            `json:"type"`
	Usage *AudioSpeechUsage `json:"usage,omitempty"`
}

// AudioSpeechUsage reports token usage for a TTS request.
type AudioSpeechUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
