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

package vllmapi

const (
	ObjectModel = "model"
)

const (
	PromLabelWaitingLoraAdapters = "waiting_lora_adapters"
	PromLabelRunningLoraAdapters = "running_lora_adapters"
	PromLabelMaxLora             = "max_lora"
	PromLabelModelName           = "model_name"

	VllmLoraRequestInfo    = "vllm:lora_requests_info"
	VllmNumRequestsRunning = "vllm:num_requests_running"
)

// modelInfo defines data about model returned by /models API
type ModelsResponseModelInfo struct {
	// ID the ID of this model
	ID string `json:"id"`
	// Object is the Object type, "model"
	Object string `json:"object"`
	// Created is model creation type - in simulator contains "now" timestamp
	Created int64 `json:"created"`
	// OwnedBy is "vllm"
	OwnedBy string `json:"owned_by"`
	// Root is the model path
	Root string `json:"root"`
	// Parent is name of base model when the model is LoRA adapter, if the model is not a LoRA - null
	Parent *string `json:"parent"`
}

// modelsResponse is the response of /models API
type ModelsResponse struct {
	// Object is type of the data, "list"
	Object string `json:"object"`
	// Data contains list of model infos
	Data []ModelsResponseModelInfo `json:"data"`
}
