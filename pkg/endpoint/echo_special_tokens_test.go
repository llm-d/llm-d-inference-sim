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
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// bosTokenizer renders like a model with a BOS token: RenderText starts with
// BOS, RenderOutputText does not.
type bosTokenizer struct{ recordingTokenizer }

func (bosTokenizer) RenderText(string) ([]uint32, []string, error) {
	return []uint32{1, 7}, []string{"<s>", "hi"}, nil
}
func (bosTokenizer) RenderOutputText(string) ([]uint32, []string, error) {
	return []uint32{7}, []string{"hi"}, nil
}

var _ = Describe("Echo mode with a BOS model", func() {
	It("echoes a chat message without BOS", func() {
		var req ChatCompletionsRequest
		Expect(req.Unmarshal([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))).To(Succeed())
		reqCtx := req.BuildRequestContext(&fakeRuntime{tokenizer: &bosTokenizer{}}, common.Channel[*ResponseInfo]{}, 0, func() {})
		echo, err := reqCtx.(*chatCompletionReqCtx).tokenizedPromptForEcho()
		Expect(err).NotTo(HaveOccurred())
		Expect(echo).To(Equal(&api.Tokenized{Tokens: []uint32{7}, Strings: []string{"hi"}}))
	})

	It("echoes a responses input without BOS", func() {
		var req ResponsesRequest
		Expect(req.Unmarshal([]byte(`{"model":"m","input":"hi"}`))).To(Succeed())
		reqCtx := req.BuildRequestContext(&fakeRuntime{tokenizer: &bosTokenizer{}}, common.Channel[*ResponseInfo]{}, 0, func() {})
		echo, err := reqCtx.(*responsesReqCtx).tokenizedPromptForEcho()
		Expect(err).NotTo(HaveOccurred())
		Expect(echo).To(Equal(&api.Tokenized{Tokens: []uint32{7}, Strings: []string{"hi"}}))
	})
})
