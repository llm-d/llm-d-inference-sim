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
	"fmt"
	"time"

	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

type chatCompletionRequest struct {
	req openaiserverapi.ChatCompletionRequest
}

// reads and parses data from the body of the given request
func (c *chatCompletionRequest) Unmarshal(data []byte) error {
	var req openaiserverapi.ChatCompletionRequest

	err := json.Unmarshal(data, &req)
	if err != nil {
		return err
	}

	c.req = req
	return nil
}

func (c *chatCompletionRequest) Validate(config *common.Configuration, toolsValidator *common.ToolsValidator) (string, int) {
	for _, tool := range c.req.Tools {
		toolJson, err := json.Marshal(tool.Function)
		if err != nil {
			return fmt.Sprintf("Failed to marshal request tools: %s", err.Error()), fasthttp.StatusBadRequest
		}
		err = toolsValidator.ValidateTool(toolJson)
		if err != nil {
			return fmt.Sprintf("Tool validation failed: %s", err.Error()), fasthttp.StatusBadRequest
		}
	}

	return validateRequest(&c.req, config)
}

func (c *chatCompletionRequest) BuildRequestContext() *openaiserverapi.CompletionReqCtx {
	reqCtx := &openaiserverapi.CompletionReqCtx{
		CompletionReq:    &c.req,
		IsChatCompletion: true,
		StartProcessing:  time.Now(),
	}
	return reqCtx
}

func (c *chatCompletionRequest) SetID(id string) {
	c.req.RequestID = id
}

func (c *chatCompletionRequest) ID() string {
	return c.req.RequestID
}

func (c *chatCompletionRequest) Model() string {
	return c.req.Model
}

func (c *chatCompletionRequest) String() string {
	return "chat completion request (req id " + c.ID() + ")"
}
