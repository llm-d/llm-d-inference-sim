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

package tokenizer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

const testBOS = 1

// newBOSRenderServer serves /v1/completions/render like vLLM for a model with a
// BOS token: the token ids start with BOS unless add_special_tokens is false.
func newBOSRenderServer(sent *map[string]json.RawMessage) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer GinkgoRecover()
		Expect(r.URL.Path).To(Equal("/v1/completions/render"))
		*sent = nil
		Expect(json.NewDecoder(r.Body).Decode(sent)).To(Succeed())
		ids := []uint32{7, 8}
		if string((*sent)["add_special_tokens"]) != "false" {
			ids = append([]uint32{testBOS}, ids...)
		}
		Expect(json.NewEncoder(w).Encode([]map[string][]uint32{{"token_ids": ids}})).To(Succeed())
	}))
}

var _ = Describe("HF tokenizer output text", func() {
	var (
		srv  *httptest.Server
		sent map[string]json.RawMessage
		tk   *HFTokenizer
	)

	BeforeEach(func() {
		srv = newBOSRenderServer(&sent)
		var err error
		tk, err = NewHFTokenizer(context.Background(), klog.Background(), srv.URL, "m", time.Second, time.Second)
		Expect(err).NotTo(HaveOccurred())
	})
	AfterEach(func() { srv.Close() })

	It("renders a prompt with the render service's special tokens", func() {
		tokens, _, err := tk.RenderText("hello world")
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).To(Equal([]uint32{testBOS, 7, 8}))
		Expect(sent).NotTo(HaveKey("add_special_tokens"))
	})

	It("renders generated text without special tokens", func() {
		tokens, strs, err := tk.RenderOutputText("hello world")
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).To(Equal([]uint32{7, 8}))
		Expect(strs).To(Equal([]string{"hello ", "world"}))
		Expect(string(sent["add_special_tokens"])).To(Equal("false"))
	})
})
