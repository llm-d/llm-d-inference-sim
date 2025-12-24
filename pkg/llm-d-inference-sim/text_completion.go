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
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

type textCompletionRequest struct {
	req openaiserverapi.TextCompletionRequest
}

// reads and parses data from the body of the given request
func (t *textCompletionRequest) Unmarshal(data []byte) error {
	var req openaiserverapi.TextCompletionRequest
	err := json.Unmarshal(data, &req)

	t.req = req
	return err
}

func (t *textCompletionRequest) Validate(config *common.Configuration, toolsValidator *common.ToolsValidator) (string, int) {
	return validateRequest(&t.req, config)
}

func (t *textCompletionRequest) BuildRequestContext() *openaiserverapi.CompletionReqCtx {
	reqCtx := &openaiserverapi.CompletionReqCtx{
		CompletionReq:    &t.req,
		IsChatCompletion: false,
		StartProcessing:  time.Now(),
	}
	return reqCtx
}

func (t *textCompletionRequest) SetID(id string) {
	t.req.RequestID = id
}

func (t *textCompletionRequest) ID() string {
	return t.req.RequestID
}

func (t *textCompletionRequest) Model() string {
	return t.req.Model
}

func (c *textCompletionRequest) String() string {
	return "text completion request (req id " + c.ID() + ")"
}
