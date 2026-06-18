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
	"encoding/json"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/valyala/fasthttp"
)

// Implementation of request for generate requests
type GenerateRequest struct {
	api.GenerateRequest
}

func (g *GenerateRequest) Unmarshal(data []byte) error {
	if err := json.Unmarshal(data, g); err != nil {
		return err
	}
	// Fall back to sampling_params.extra_args.kv_transfer_params when not provided at the root.
	// Real vLLM has a bug where the parameters land in extra_args; accept both locations.
	if g.KVParams == nil && g.SamplingParams != nil && g.SamplingParams.ExtraArgs != nil {
		g.KVParams = g.SamplingParams.ExtraArgs.KVTransferParams
	}
	g.SetTokenizedPrompt(&api.Tokenized{Tokens: g.TokenIDs, Strings: []string{}})
	return nil
}

func (g *GenerateRequest) validate(toolsValidator *toolsValidator) *api.Error {
	if g.TokenIDs == nil {
		err := api.NewError("Missing input token_ids", fasthttp.StatusBadRequest, nil)
		return &err
	}

	if g.SamplingParams == nil {
		err := api.NewError("Missing sampling_params field", fasthttp.StatusBadRequest, nil)
		return &err
	}

	return validateRequest(g)
}

func (g *GenerateRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo],
	choiceIdx int, doneFn func()) requestContext {
	reqCtx := &generateReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel, choiceIdx, doneFn),
		req:                g,
	}
	// wire generateReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx

	return reqCtx
}

// split is a no-op: generate requests always carry a single prompt.
func (g *GenerateRequest) split() []Request {
	return []Request{g}
}

func (g *GenerateRequest) AsString() string {
	return "generate request (req id " + g.RequestID + ")"
}

func (g *GenerateRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *api.Tokenized, finishReason *string, usageData *api.Usage,
	sendUsageData bool, logprobs *int, toolCalls []api.ToolCall, mmEncoderOnlyMode bool) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, g.GetRequestID(), g.IsDoRemotePrefill(), g.IsDoRemoteDecode(), g.GetNumberOfCachedPromptTokens())

	var ecParams map[string]api.ECTransferParams
	if mmEncoderOnlyMode && g.Features != nil {
		ecParams = buildECTransferParams(g.Features.MMHashes)
	}

	return &generateResponseCtx{
		baseResponseContext: base,
		ecTransferParams:    ecParams,
	}
}

var _ Request = (*GenerateRequest)(nil)

// Implementation of requestContext for generation requests
type generateReqCtx struct {
	baseRequestContext
	req *GenerateRequest
}

func (g *generateReqCtx) request() Request {
	return g.req
}

func (g *generateReqCtx) tokenizedPromptForEcho() (*api.Tokenized, error) {
	return g.req.TokenizedPrompt(), nil
}

func (g *generateReqCtx) encode() ([]uint32, []string, *api.RenderMMFeatures, error) {
	return g.req.TokenizedPrompt().Tokens, g.req.TokenizedPrompt().Strings, nil, nil
}

func (g *generateReqCtx) createToolCalls() ([]api.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

var _ requestContext = (*generateReqCtx)(nil)

// Implementation of responseContext for generation requests
type generateResponseCtx struct {
	baseResponseContext
	ecTransferParams map[string]api.ECTransferParams
}

func (respCtx *generateResponseCtx) ToolCalls() []api.ToolCall {
	return nil
}

func (respCtx *generateResponseCtx) ECTransferParams() map[string]api.ECTransferParams {
	return respCtx.ecTransferParams
}

var _ ResponseContext = (*generateResponseCtx)(nil)
