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
	"errors"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

type baseRequestProcessor struct {
	sim *simContext
}

type chatRequestProcessor struct {
	baseRequestProcessor
}

type textRequestProcessor struct {
	baseRequestProcessor
}

func (c *chatRequestProcessor) kvCacheOnRequestStart(reqCtx requestContext) *openaiserverapi.Error {
	// kv cache is currently supported for /completion API only
	return nil
}

func (c *chatRequestProcessor) createToolCalls(reqCtx requestContext) ([]openaiserverapi.ToolCall, int, string, error) {
	req, ok := reqCtx.request().(*chatCompletionRequest)
	if !ok {
		return nil, 0, "", errors.New("invalid type of request")
	}
	if !common.IsToolChoiceNone(req.GetToolChoice()) &&
		req.GetTools() != nil {
		toolCalls, completionTokens, err :=
			common.CreateToolCalls(req.GetTools(), req.GetToolChoice(), c.sim.config, c.sim.random)
		finishReason := common.ToolsFinishReason
		return toolCalls, completionTokens, finishReason, err
	}
	return nil, 0, "", nil

}

func (t *textRequestProcessor) kvCacheOnRequestStart(reqCtx requestContext) *openaiserverapi.Error {
	if t.sim.config.EnableKVCache {
		if err := t.sim.kvcacheHelper.OnRequestStart(reqCtx.request()); err != nil {
			serverError := openaiserverapi.NewError(err.Error(), fasthttp.StatusInternalServerError, nil)
			return &serverError
		}
	}
	return nil

}

func (t *textRequestProcessor) createToolCalls(reqCtx requestContext) ([]openaiserverapi.ToolCall, int, string, error) {
	return nil, 0, "", nil
}
