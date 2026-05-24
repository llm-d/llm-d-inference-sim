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

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

// Implementation of request for generate requests
type GenerateRequest struct {
	openaiserverapi.GenerateRequest
}

func (g *GenerateRequest) Unmarshal(data []byte) error {
	if err := json.Unmarshal(data, g); err != nil {
		return err
	}
	g.SetTokenizedPrompt(&openaiserverapi.Tokenized{Tokens: g.TokenIDs, Strings: []string{}})
	return nil
}

func (g *GenerateRequest) validate(toolsValidator *toolsValidator) (string, int) {
	if g.TokenIDs == nil {
		return "Missing input token_ids", fasthttp.StatusBadRequest
	}

	if g.SamplingParams == nil {
		return "Missing sampling_params field", fasthttp.StatusBadRequest
	}

	return validateRequest(g)
}

func (g *GenerateRequest) buildRequestContext(simCtx *SimContext, channel common.Channel[*ResponseInfo]) requestContext {
	reqCtx := &generateReqCtx{
		baseRequestContext: newBaseRequestContext(simCtx, channel),
		req:                g,
	}
	// wire generateReqCtx into embedded requestContext interface
	reqCtx.requestContext = reqCtx

	return reqCtx
}

func (g *GenerateRequest) AsString() string {
	return "generate request (req id " + g.RequestID + ")"
}

func (g *GenerateRequest) createResponseContext(reqCtx requestContext, displayModel string,
	responseTokens *openaiserverapi.Tokenized, finishReason *string, usageData *openaiserverapi.Usage,
	sendUsageData bool, logprobs *int, toolCalls []openaiserverapi.ToolCall, mmEncoderOnlyMode bool) ResponseContext {
	base := newBaseResponseContext(reqCtx, displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, g.GetRequestID(), g.IsDoRemotePrefill(), g.IsDoRemoteDecode(), g.GetNumberOfCachedPromptTokens())

	var ecParams map[string]openaiserverapi.ECTransferParams
	if mmEncoderOnlyMode && g.Features != nil {
		ecParams = buildECTransferParams(g.Features)
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

func (g *generateReqCtx) tokenizedPromptForEcho() (*openaiserverapi.Tokenized, error) {
	return g.req.TokenizedPrompt(), nil
}

func (g *generateReqCtx) encode() ([]uint32, []string, *openaiserverapi.RenderMMFeatures, error) {
	return g.req.TokenizedPrompt().Tokens, g.req.TokenizedPrompt().Strings, nil, nil
}

func (g *generateReqCtx) createToolCalls() ([]openaiserverapi.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

var _ requestContext = (*generateReqCtx)(nil)

// Implementation of responseContext for generation requests
type generateResponseCtx struct {
	baseResponseContext
	ecTransferParams map[string]openaiserverapi.ECTransferParams
}

func (respCtx *generateResponseCtx) ToolCalls() []openaiserverapi.ToolCall {
	return nil
}

func (respCtx *generateResponseCtx) ECTransferParams() map[string]openaiserverapi.ECTransferParams {
	return respCtx.ecTransferParams
}

var _ ResponseContext = (*generateResponseCtx)(nil)

// buildECTransferParams creates simulated ECTransferParams for each MM hash in the request features.
func buildECTransferParams(features *openaiserverapi.EncodeMMFeatures) map[string]openaiserverapi.ECTransferParams {
	params := make(map[string]openaiserverapi.ECTransferParams)

	for _, hashes := range features.MMHashes {
		for _, hash := range hashes {
			params[hash] = openaiserverapi.ECTransferParams{
				PeerHost:      "DUMMY",
				PeerPort:      1234,
				SizeBytes:     2359296,
				NixlAgentData: []byte("NIXL_METADATA_PLACEHOLDER"),
			}
		}
	}
	return params
}
