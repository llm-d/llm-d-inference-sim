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

package vllm

import (
	"github.com/buaazp/fasthttprouter"
	"google.golang.org/grpc"

	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
)

// BindHTTP registers the vLLM-specific HTTP routes on r, on top of the
// common routes comm's own HTTP server already registers.
func (Engine) BindHTTP(r *fasthttprouter.Router, comm *communication.Communication) {
	r.POST("/inference/v1/generate", comm.HandleGenerate)
	r.POST("/v1/load_lora_adapter", comm.HandleLoadLora)
	r.POST("/v1/unload_lora_adapter", comm.HandleUnloadLora)
	r.POST("/fake_metrics", comm.HandleFakeMetrics) //nolint:staticcheck // deprecated endpoint, still served until v0.12.0
	// emulates vLLM's Mooncake bootstrap endpoint on the prefill pod; the routing sidecar queries it to resolve remote engine ids
	r.GET("/query", comm.HandleMooncakeQuery)
	r.POST("/sleep", comm.HandleSleep)
	r.POST("/wake_up", comm.HandleWakeUp)
	r.GET("/is_sleeping", comm.HandleIsSleeping)
}

// BindGRPC registers comm as the vLLM gRPC engine service on server.
func (Engine) BindGRPC(server *grpc.Server, comm *communication.Communication) bool {
	pb.RegisterVllmEngineServer(server, comm)
	return true
}
