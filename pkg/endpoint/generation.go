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

package endpoint

import (
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// Implementation of request for generation requests
type GenerationRequest struct {
	api.GenerationRequest
}

func (g *GenerationRequest) Unmarshal(data []byte) error {
	return nil
}

func (g *GenerationRequest) Validate(toolsValidator *ToolsValidator) *api.Error {
	return validateRequest(g)
}

func (g *GenerationRequest) BuildRequestContext(runtime Runtime, channel common.Channel[*ResponseInfo],
	choiceIdx int, doneFn func()) RequestContext {
	reqCtx := &generationReqCtx{
		baseRequestContext: newBaseRequestContext(runtime, channel, choiceIdx, doneFn),
		req:                g,
	}
	// wire generationReqCtx into embedded RequestContext interface
	reqCtx.RequestContext = reqCtx
	return reqCtx
}

func (g *GenerationRequest) AsString() string {
	return "generation request (req id " + g.RequestID + ")"
}

func (g *GenerationRequest) createResponseContext(reqCtx RequestContext, displayModel string,
	responseTokens *api.Tokenized, finishReason *string, usageData *api.Usage,
	sendUsageData bool, logprobs *int, toolCalls []api.ToolCall, _ bool) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, g.GetRequestID(), g.IsDoRemotePrefill(), g.IsDoRemoteDecode(), g.GetNumberOfCachedPromptTokens())
	return &generationResponseCtx{
		baseResponseContext: base,
	}
}

// Split is a no-op: generation requests always carry a single prompt.
func (g *GenerationRequest) Split() []Request {
	return []Request{g}
}

var _ Request = (*GenerationRequest)(nil)

// Implementation of RequestContext for generation requests
type generationReqCtx struct {
	baseRequestContext
	req *GenerationRequest
}

func (g *generationReqCtx) Request() Request {
	return g.req
}

func (g *generationReqCtx) tokenizedPromptForEcho() (*api.Tokenized, error) {
	return g.req.TokenizedPrompt(), nil
}

func (g *generationReqCtx) encode() ([]uint32, []string, *api.RenderMMFeatures, error) {
	tokenizedPrompt := g.req.TokenizedPrompt()
	if tokenizedPrompt != nil {
		return tokenizedPrompt.Tokens, tokenizedPrompt.Strings, nil, nil
	}
	tokens, strTokens, err := g.runtime.GetTokenizer().RenderText(g.req.Prompt)
	return tokens, strTokens, nil, err
}

func (g *generationReqCtx) createToolCalls() ([]api.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

var _ RequestContext = (*generationReqCtx)(nil)

// Implementation of ResponseContext for generation requests
type generationResponseCtx struct {
	baseResponseContext
}

func (respCtx *generationResponseCtx) ToolCalls() []api.ToolCall {
	return nil
}

var _ ResponseContext = (*generationResponseCtx)(nil)
