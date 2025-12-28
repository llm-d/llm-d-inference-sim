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

	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

type chatCompletionRequest struct {
	openaiserverapi.ChatCompletionRequest
}

type chatCompletionReqCtx struct {
	baseRequestContext
	req          *chatCompletionRequest
	reqProcessor *chatRequestProcessor
}

func (c *chatCompletionReqCtx) request() request {
	return c.req
}
func (c *chatCompletionReqCtx) processor() requestProcessor {
	return c.reqProcessor
}

// reads and parses data from the body of the given request
func (c *chatCompletionRequest) unmarshal(data []byte) error {
	return json.Unmarshal(data, c)
}

func (c *chatCompletionRequest) validate(config *common.Configuration, toolsValidator *common.ToolsValidator) (string, int) {
	for _, tool := range c.Tools {
		toolJson, err := json.Marshal(tool.Function)
		if err != nil {
			return "Failed to marshal request tools: " + err.Error(), fasthttp.StatusBadRequest
		}
		err = toolsValidator.ValidateTool(toolJson)
		if err != nil {
			return "Tool validation failed: " + err.Error(), fasthttp.StatusBadRequest
		}
	}

	return validateRequest(c, config)
}

func (c *chatCompletionRequest) buildRequestContext(simCtx *simContext, ctx *fasthttp.RequestCtx, wg *sync.WaitGroup) requestContext {
	reqCtx := &chatCompletionReqCtx{
		baseRequestContext: baseRequestContext{
			startProcessing: time.Now(),
			wg:              wg,
			httpReqCtx:      ctx,
		},
		req: c,
		reqProcessor: &chatRequestProcessor{
			baseRequestProcessor{simCtx},
		},
	}
	return reqCtx
}

func (c *chatCompletionRequest) setID(id string) {
	c.RequestID = id
}

func (c *chatCompletionRequest) asString() string {
	return "chat completion request (req id " + c.RequestID + ")"
}

var _ request = (*chatCompletionRequest)(nil)
