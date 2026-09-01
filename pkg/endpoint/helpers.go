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

package endpoint

import (
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/valyala/fasthttp"
)

func getNumberOfPromptTokens(req api.Request) int {
	return req.TokenizedPrompt().Length()
}

func validateRequest(req api.Request) *api.Error {
	if n := req.GetRawN(); n != nil && *n <= 0 {
		err := api.NewError("n must be at least 1", fasthttp.StatusBadRequest, nil)
		return &err
	}

	if req.GetMaxCompletionTokens() != nil && *req.GetMaxCompletionTokens() <= 0 {
		err := api.NewError(common.InvalidMaxTokensErrMsg, fasthttp.StatusBadRequest, nil)
		return &err
	}

	if req.IsDoRemoteDecode() && req.IsStream() {
		err := api.NewError("Prefill does not support streaming", fasthttp.StatusBadRequest, nil)
		return &err
	}

	if req.GetIgnoreEOS() && req.GetMaxCompletionTokens() == nil {
		err := api.NewError("Ignore_eos is true but max_completion_tokens (or max_tokens) is not set",
			fasthttp.StatusBadRequest, nil)
		return &err
	}

	return nil
}

// buildECTransferParams creates simulated ECTransferParams for each MM hash.
func buildECTransferParams(mmHashes map[string][]string) map[string]api.ECTransferParams {
	params := make(map[string]api.ECTransferParams)

	for _, hashes := range mmHashes {
		for _, hash := range hashes {
			params[hash] = api.ECTransferParams{
				PeerHost:      "DUMMY",
				PeerPort:      1234,
				SizeBytes:     2359296,
				NixlAgentData: []byte("NIXL_METADATA_PLACEHOLDER"),
			}
		}
	}
	return params
}
