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

package llmdinferencesim

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

type textCompletionReqCtx struct {
	baseRequestContext
	req *textCompletionRequest
}

func (t *textCompletionReqCtx) request() request {
	return t.req
}

type textCompletionRequest struct {
	openaiserverapi.TextCompletionRequest
}

// reads and parses data from the body of the given request
func (t *textCompletionRequest) unmarshal(data []byte) error {
	return json.Unmarshal(data, t)
}

func (t *textCompletionRequest) validate(config *common.Configuration, toolsValidator *common.ToolsValidator) (string, int) {
	return validateRequest(t, config)
}

func (t *textCompletionRequest) buildRequestContext(simCtx *simContext, ctx *fasthttp.RequestCtx, wg *sync.WaitGroup) requestContext {
	reqCtx := &textCompletionReqCtx{
		baseRequestContext: baseRequestContext{
			sim:             simCtx,
			startProcessing: time.Now(),
			wg:              wg,
			httpReqCtx:      ctx,
		},
		req: t,
	}
	// wire textCompletionReqCtx into embedded requestContext interface
	reqCtx.baseRequestContext.requestContext = reqCtx
	return reqCtx
}

func (t *textCompletionRequest) setID(id string) {
	t.RequestID = id
}

func (t *textCompletionRequest) asString() string {
	return "text completion request (req id " + t.RequestID + ")"
}

func (t *textCompletionRequest) createResponseContext(displayModel string, responseTokens []string, finishReason *string,
	usageData *openaiserverapi.Usage, sendUsageData bool, logprobs *int, toolCalls []openaiserverapi.ToolCall) responseContext {
	base := newBaseResponseContext(displayModel, responseTokens, finishReason, usageData, sendUsageData,
		logprobs, t.GetRequestID(), t.IsDoRemotePrefill(), t.IsDoRemoteDecode(), t.GetNumberOfCachedPromptTokens())
	return &textCompletionResponseCtx{
		baseResponseContext: base,
	}
}

var _ request = (*textCompletionRequest)(nil)
