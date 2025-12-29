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
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

type requestBuilder interface {
	unmarshal(data []byte) error
	validate(config *common.Configuration, toolsValidator *common.ToolsValidator) (string, int)
	buildRequestContext(simCtx *simContext, ctx *fasthttp.RequestCtx, wg *sync.WaitGroup) requestContext
	setID(string)
	asString() string
	createResponseContext(displayModel string, responseTokens []string, finishReason *string,
		usageData *openaiserverapi.Usage, sendUsageData bool, logprobs *int, toolCalls []openaiserverapi.ToolCall) responseContext
}

type request interface {
	requestBuilder
	openaiserverapi.Request
}

type requestProcessor interface {
	kvCacheOnRequestStart(reqCtx requestContext) *openaiserverapi.Error
	kvCacheOnRequestEnd(reqCtx requestContext)
	createToolCalls(reqCtx requestContext) ([]openaiserverapi.ToolCall, int, string, error)
}

type requestContext interface {
	processor() requestProcessor
	request() request
	httpRequestCtx() *fasthttp.RequestCtx
	done()
	startProcessingTime() time.Time
}

type baseRequestContext struct {
	httpReqCtx      *fasthttp.RequestCtx
	wg              *sync.WaitGroup
	startProcessing time.Time
}

func (b *baseRequestContext) httpRequestCtx() *fasthttp.RequestCtx {
	return b.httpReqCtx
}

func (b *baseRequestContext) startProcessingTime() time.Time {
	return b.startProcessing
}

func (b *baseRequestContext) done() {
	b.wg.Done()
}
