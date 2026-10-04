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

// Package sglang implements the engine.Engine interface for the SGLang
// backend. It covers the engine-neutral surface only: the OpenAI-compatible
// endpoints, the latency model, and dataset-backed response generation. SGLang's
// own HTTP routes, its Prometheus metrics, its LoRA adapters, and its KV-cache
// event format are not implemented, and a configuration asking for one of them
// is rejected rather than silently ignored.
package sglang

// Engine implements engine.Engine for the SGLang backend.
type Engine struct{}

// New returns the SGLang engine.
func New() Engine { return Engine{} }

// Name identifies this engine backend.
func (Engine) Name() string { return "sglang" }
