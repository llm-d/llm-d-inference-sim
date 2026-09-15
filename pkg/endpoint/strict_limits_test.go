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
	"strings"
	"testing"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/valyala/fasthttp"
)

func TestStrictEffectiveTokenLimit(t *testing.T) {
	for _, tc := range []struct {
		name                string
		strict              bool
		minimum, defaultMax int64
		explicit            *int64
		prompt              int
		want                string
	}{
		{"schema default", true, 17, 16, nil, 4, "max_tokens=16"},
		{"remaining context", true, 29, 0, nil, 4, "max_tokens=28"},
		{"context caps explicit", true, 29, 0, ptrInt64(40), 4, "max_tokens=28"},
		{"explicit replaces default", true, 20, 16, ptrInt64(24), 4, ""},
		{"boundary", true, 28, 0, nil, 4, ""},
		{"lenient", false, 29, 16, nil, 4, ""},
		{"invalid context", true, 0, 0, nil, 33, "maximum context length"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := &TextCompletionsRequest{}
			req.MaxTokens = tc.explicit
			req.SetStrictTokenLimits(tc.minimum, tc.defaultMax)
			req.SetTokenizedPrompt(&api.Tokenized{Tokens: make([]uint32, tc.prompt)})
			runtime := &fakeRuntime{config: &common.Configuration{StrictRequestValidation: tc.strict, MaxModelLen: 32}}
			ctx := req.BuildRequestContext(runtime, common.Channel[*ResponseInfo]{}, 0, nil).(*textCompletionReqCtx)
			message, code := ctx.validateTokenizedRequest()
			if tc.want == "" {
				if message != "" || code != fasthttp.StatusOK {
					t.Fatalf("message=%q code=%d", message, code)
				}
			} else if !strings.Contains(message, tc.want) || code != fasthttp.StatusBadRequest {
				t.Fatalf("message=%q code=%d", message, code)
			}
		})
	}
}
