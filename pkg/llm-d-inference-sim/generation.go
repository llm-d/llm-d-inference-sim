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
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

// Implementation of request for generation requests
type GenerationRequest struct {
	openaiserverapi.GenerationRequest
}

func (g *GenerationRequest) Unmarshal(data []byte) error {
	return nil
}

func (g *GenerationRequest) validate(toolsValidator *toolsValidator) (string, int) {
	return validateRequest(g)
}

func (g *GenerationRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo]) requestContext {
	reqCtx := &generationReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel),
		req:                g,
	}
	// wire generationReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx
	return reqCtx
}

func (g *GenerationRequest) AsString() string {
	return "generation request (req id " + g.RequestID + ")"
}

func (g *GenerationRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *openaiserverapi.Tokenized, finishReason *string, usageData *openaiserverapi.Usage,
	sendUsageData bool, logprobs *int, toolCalls []openaiserverapi.ToolCall) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, g.GetRequestID(), g.IsDoRemotePrefill(), g.IsDoRemoteDecode(), g.GetNumberOfCachedPromptTokens())
	return &generationResponseCtx{
		baseResponseContext: base,
	}
}

var _ Request = (*GenerationRequest)(nil)

// Implementation of requestContext for generation requests
type generationReqCtx struct {
	baseRequestContext
	req *GenerationRequest
}

func (g *generationReqCtx) request() Request {
	return g.req
}

func (g *generationReqCtx) tokenizedPromptForEcho() (*openaiserverapi.Tokenized, error) {
	return g.req.TokenizedPrompt(), nil
}

func (g *generationReqCtx) encode() ([]uint32, []string, *openaiserverapi.RenderMMFeatures, error) {
	tokenizedPrompt := g.req.TokenizedPrompt()
	if tokenizedPrompt != nil {
		return tokenizedPrompt.Tokens, tokenizedPrompt.Strings, nil, nil
	}
	tokens, strTokens, err := g.sim.Tokenizer.RenderText(g.req.Prompt)
	return tokens, strTokens, nil, err
}

func (g *generationReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

var _ requestContext = (*generationReqCtx)(nil)

// Implementation of responseContext for generation requests
type generationResponseCtx struct {
	baseResponseContext
}

func (respCtx *generationResponseCtx) ToolCalls() []openaiserverapi.ToolCall {
	return nil
}

var _ ResponseContext = (*generationResponseCtx)(nil)
