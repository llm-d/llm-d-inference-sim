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

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
)

// BindHTTP registers the vLLM-specific HTTP routes on r, on top of the
// common routes comm's own HTTP server already registers.
func (Engine) BindHTTP(r *fasthttprouter.Router, comm *communication.Communication) {
	r.POST("/v1/systemone", comm.HandleSystemOne)
	r.POST("/v1/decisions", comm.HandleDecisions)
	r.POST("/inference/v1/generate", comm.HandleGenerate)
	r.POST("/v1/chat/completions/render", comm.HandleChatCompletionsRender)
	r.POST("/v1/completions/render", comm.HandleTextCompletionsRender)
	// vLLM versions that predate the Responses and Messages render endpoints do not serve them
	if !comm.Config().EnableLegacyRender {
		r.POST("/v1/responses/render", comm.HandleResponsesRender)
		r.POST("/v1/messages/render", comm.HandleMessagesRender)
	}
	r.POST("/v1/chat/completions/derender", comm.HandleChatCompletionsDerender)
	r.POST("/v1/completions/derender", comm.HandleTextCompletionsDerender)
	r.POST("/v1/load_lora_adapter", comm.HandleLoadLora)
	r.POST("/v1/unload_lora_adapter", comm.HandleUnloadLora)
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

// ErrorBody frames err the way vLLM does: wrapped under an "error" key as the
// OpenAI API specifies, on every route but the Messages API's. There vLLM
// translates the same error object into the Anthropic envelope field by field,
// keeping its own error-type spelling rather than mapping it to Anthropic's
// vocabulary.
func (Engine) ErrorBody(err api.Error, route api.ErrorRoute) any {
	if route == api.ErrorRouteMessages {
		return api.NewMessagesErrorResponse(err.Type, err.Message)
	}
	return api.ErrorResponse{Error: err}
}

// StreamErrorBody frames a streaming error frame the same way ErrorBody does:
// vLLM sends one shape in both cases.
func (e Engine) StreamErrorBody(err api.Error, route api.ErrorRoute) any {
	return e.ErrorBody(err, route)
}

// DecorateChatChoice adds no fields: vLLM has no equivalent of sglang's
// matched_stop.
func (Engine) DecorateChatChoice(choice api.ChatRespChoice) api.ChatRespChoice { return choice }

// DecorateChatChunkChoice adds no fields: vLLM has no equivalent of sglang's
// matched_stop.
func (Engine) DecorateChatChunkChoice(choice api.ChatRespChunkChoice) api.ChatRespChunkChoice {
	return choice
}

// DecorateTextChoice adds no fields: vLLM has no equivalent of sglang's
// matched_stop.
func (Engine) DecorateTextChoice(choice api.TextRespChoice) api.TextRespChoice { return choice }

// DecorateMessagesCountTokens adds context_management, which vLLM reports with
// the same count as input_tokens.
func (Engine) DecorateMessagesCountTokens(resp api.MessagesCountTokensResponse) api.MessagesCountTokensResponse {
	resp.ContextManagement = &api.MessagesContextManagement{OriginalInputTokens: resp.InputTokens}
	return resp
}
