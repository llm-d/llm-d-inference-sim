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

package engine

import (
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
)

type vllmEngine struct{}

func (vllmEngine) Name() string {
	return VLLM
}

func (vllmEngine) NewConfiguration() *common.Configuration {
	cfg := common.NewConfiguration()
	cfg.Engine = VLLM
	return cfg
}

// Routes returns vLLM's HTTP-specific endpoints: the render debug endpoints, the
// Responses and Messages APIs, the vLLM-specific generate endpoint, LoRA adapter
// load/unload, the Mooncake bootstrap query, and the sleep/wake_up/is_sleeping
// dev-mode endpoints.
func (vllmEngine) Routes(c *communication.Communication) []communication.Route {
	return []communication.Route{
		{Method: fasthttp.MethodPost, Path: "/v1/chat/completions/render", Handler: c.HandleChatCompletionsRender},
		{Method: fasthttp.MethodPost, Path: "/v1/completions/render", Handler: c.HandleTextCompletionsRender},
		{Method: fasthttp.MethodPost, Path: "/v1/responses", Handler: c.HandleResponses},
		{Method: fasthttp.MethodPost, Path: "/v1/messages", Handler: c.HandleMessages},
		{Method: fasthttp.MethodPost, Path: "/inference/v1/generate", Handler: c.HandleGenerate},
		{Method: fasthttp.MethodPost, Path: "/v1/load_lora_adapter", Handler: c.HandleLoadLora},
		{Method: fasthttp.MethodPost, Path: "/v1/unload_lora_adapter", Handler: c.HandleUnloadLora},
		{Method: fasthttp.MethodGet, Path: "/query", Handler: c.HandleMooncakeQuery},
		{Method: fasthttp.MethodPost, Path: "/sleep", Handler: c.HandleSleep},
		{Method: fasthttp.MethodPost, Path: "/wake_up", Handler: c.HandleWakeUp},
		{Method: fasthttp.MethodGet, Path: "/is_sleeping", Handler: c.HandleIsSleeping},
	}
}

// GRPC registers vLLM's gRPC engine service (generate, embed, health check, abort,
// model/server info).
func (vllmEngine) GRPC() communication.GRPCRegistrar {
	return func(server *grpc.Server, c *communication.Communication) {
		pb.RegisterVllmEngineServer(server, c)
	}
}
