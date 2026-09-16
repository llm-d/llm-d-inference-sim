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

// ImagesGenerationsRequest is the request body for POST /v1/images/generations
// (OpenAI DALL-E compatible, with vllm-omni diffusion extensions).
type ImagesGenerationsRequest struct {
	// Prompt is the text description of the desired image (required).
	Prompt string `json:"prompt"`
	// Model is the image generation model to use.
	Model string `json:"model,omitempty"`
	// N is the number of images to generate. Default is 1.
	N int `json:"n,omitempty"`
	// Size is the output dimensions in WIDTHxHEIGHT format (e.g. "1024x1024").
	Size string `json:"size,omitempty"`
	// ResponseFormat selects the image encoding: b64_json (default) or url.
	ResponseFormat string `json:"response_format,omitempty"`

	// vllm-omni / diffusion extensions — accepted and ignored by the simulator.
	NegativePrompt    string   `json:"negative_prompt,omitempty"`
	NumInferenceSteps *int     `json:"num_inference_steps,omitempty"`
	GuidanceScale     *float64 `json:"guidance_scale,omitempty"`
	Seed              *int64   `json:"seed,omitempty"`
	OutputFormat      string   `json:"output_format,omitempty"`
	Layers            *int     `json:"layers,omitempty"`
	FlowShift         *float64 `json:"flow_shift,omitempty"`
}

// ImageData is a single generated image in the response.
type ImageData struct {
	B64JSON string `json:"b64_json,omitempty"`
	URL     string `json:"url,omitempty"`
}

// ImagesGenerationsResponse is the response for POST /v1/images/generations.
type ImagesGenerationsResponse struct {
	Created      int64       `json:"created"`
	Data         []ImageData `json:"data"`
	OutputFormat string      `json:"output_format,omitempty"`
	Size         string      `json:"size,omitempty"`
}
