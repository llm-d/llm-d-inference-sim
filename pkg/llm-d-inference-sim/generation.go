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

package llmdinferencesim

import (
	"time"

	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

// Implementation of request for generation requests
type generationRequest struct {
	openaiserverapi.GenerationRequest
}

func (g *generationRequest) unmarshal(data []byte) error {
	return nil
}

func (g *generationRequest) validate(toolsValidator *toolsValidator) (string, int) {
	return validateRequest(g)
}

func (g *generationRequest) buildRequestContext(simCtx *simContext, channel chan *responseInfo) requestContext {
	reqCtx := &generationReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel),
		req:                g,
	}
	// wire generationReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (g *generationRequest) asString() string {
	return "generation request (req id " + g.RequestID + ")"
}

func (g *generationRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *openaiserverapi.Tokenized, finishReason *string, usageData *openaiserverapi.Usage,
	sendUsageData bool, logprobs *int, toolCalls []openaiserverapi.ToolCall) responseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, g.GetRequestID(), g.IsDoRemotePrefill(), g.IsDoRemoteDecode(), g.GetNumberOfCachedPromptTokens())
	return &generationResponseCtx{
		baseResponseContext: base,
	}
}

var _ request = (*generationRequest)(nil)

// Implementation of requestContext for generation requests
type generationReqCtx struct {
	baseRequestContext
	req *generationRequest
}

func (g *generationReqCtx) request() request {
	return g.req
}

func (g *generationReqCtx) kvCacheOnRequestStart() (hitRate float64, oaiServerError *openaiserverapi.Error) {
	if g.sim.config.EnableKVCache {
		var err error
		hitRate, err = g.sim.kvcacheHelper.OnRequestStart(g.request())
		if err != nil {
			serverError := openaiserverapi.NewError(err.Error(), fasthttp.StatusInternalServerError, nil)
			return 0, &serverError
		}
		return hitRate, nil
	}
	return 0, nil
}

func (g *generationReqCtx) kvCacheOnRequestEnd() {
	if g.sim.config.EnableKVCache {
		if err := g.sim.kvcacheHelper.OnRequestEnd(g.request().GetRequestID()); err != nil {
			g.sim.logger.Error(err, "kv cache failed to process request end")
		}
	}
}

func (g *generationReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

var _ requestContext = (*generationReqCtx)(nil)

// Implementation of responseContext for generation requests
type generationResponseCtx struct {
	baseResponseContext
}

// createResponse creates the response for chat completion requests
func (respCtx *generationResponseCtx) createResponse(tokens *openaiserverapi.Tokenized) openaiserverapi.CompletionResponse {
	baseResp := openaiserverapi.CreateBaseCompletionResponse(
		time.Now().Unix(), respCtx.displayModelName, respCtx.usage, respCtx.id, respCtx.remoteDecode)
	return openaiserverapi.CreateGenerationResponse(baseResp, tokens)
}

func (respCtx *generationResponseCtx) createUsageChunk() openaiserverapi.CompletionRespChunk {
	return nil
}

// createChatCompletionChunk creates and returns a CompletionRespChunk, a single chunk of streamed completion
// API response, for chat completion. It sets either role, or token, or tool call info in the message.
func (respCtx *generationResponseCtx) createCompletionChunk(token string, tool *openaiserverapi.ToolCall,
	role string, finishReason *string) openaiserverapi.CompletionRespChunk {
	return nil
}

// in chat completion first chunk contains the role
func (respCtx *generationResponseCtx) createFirstCompletionChunk() openaiserverapi.CompletionRespChunk {
	return nil
}

func (respCtx *generationResponseCtx) toolCalls() []openaiserverapi.ToolCall {
	return nil
}

var _ responseContext = (*generationResponseCtx)(nil)
