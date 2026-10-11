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
	"io"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

var _ = Describe("HF tokenizer chat render tools", func() {
	var (
		srv  *httptest.Server
		body []byte
		tk   *HFTokenizer
	)
	messages := []api.Message{{Role: api.RoleUser, Content: api.ChatComplContent{Raw: "hi"}}}

	BeforeEach(func() {
		body = nil
		srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			Expect(r.URL.Path).To(Equal("/v1/chat/completions/render"))
			body, _ = io.ReadAll(r.Body)
			_, _ = w.Write([]byte(`{"token_ids":[1,2,3]}`))
		}))
		var err error
		tk, err = NewHFTokenizer(context.Background(), klog.Background(), srv.URL, "m", time.Second, time.Second)
		Expect(err).NotTo(HaveOccurred())
	})
	AfterEach(func() { srv.Close() })

	It("forwards tools and tool_choice, keeping their fields and key order", func() {
		// Keys deliberately out of alphabetical order, plus a field the
		// simulator's Tool type does not model: re-encoding would change both.
		tools := api.RenderTools{
			Tools: json.RawMessage(`[{"type":"function","function":{"name":"get_weather","strict":true,` +
				`"parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}}]`),
			ToolChoice: json.RawMessage(`{"type":"function","function":{"name":"get_weather"}}`),
		}

		tokens, _, _, err := tk.RenderMessages(messages, tools)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).To(Equal([]uint32{1, 2, 3}))

		var sent struct {
			Tools      json.RawMessage `json:"tools"`
			ToolChoice json.RawMessage `json:"tool_choice"`
		}
		Expect(json.Unmarshal(body, &sent)).To(Succeed())
		Expect(string(sent.Tools)).To(Equal(string(tools.Tools)))
		Expect(string(sent.ToolChoice)).To(Equal(string(tools.ToolChoice)))
	})

	It("omits both when the request has neither", func() {
		_, _, _, err := tk.RenderMessages(messages, api.RenderTools{})
		Expect(err).NotTo(HaveOccurred())
		var sent map[string]json.RawMessage
		Expect(json.Unmarshal(body, &sent)).To(Succeed())
		Expect(sent).NotTo(HaveKey("tools"))
		Expect(sent).NotTo(HaveKey("tool_choice"))
	})
})
