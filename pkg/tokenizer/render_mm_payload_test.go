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

var _ = Describe("HF tokenizer chat render mm_payload", func() {
	var (
		srv  *httptest.Server
		body []byte
		tk   *HFTokenizer
	)
	messages := []api.Message{{Role: api.RoleUser, Content: api.ChatComplContent{Raw: "hi"}}}

	BeforeEach(func() {
		body = nil
		srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ = io.ReadAll(r.Body)
			_, _ = w.Write([]byte(`{"token_ids":[1,2,3],"features":{"mm_hashes":{"image":["h"]},` +
				`"mm_placeholders":{"image":[{"offset":0,"length":3}]},` +
				`"kwargs_data":{"image":["k"]},"mm_metadata":{"image":["m"]}}}`))
		}))
		var err error
		tk, err = NewHFTokenizer(context.Background(), klog.Background(), srv.URL, "m", time.Second, time.Second)
		Expect(err).NotTo(HaveOccurred())
	})
	AfterEach(func() { srv.Close() })

	DescribeTable("forwards the client's mm_payload and passes the renderer's features back",
		func(mmPayload string) {
			_, _, features, err := tk.RenderMessages(messages, api.RenderTools{}, mmPayload)
			Expect(err).NotTo(HaveOccurred())

			var sent map[string]json.RawMessage
			Expect(json.Unmarshal(body, &sent)).To(Succeed())
			Expect(string(sent["mm_payload"])).To(Equal(`"` + mmPayload + `"`))

			Expect(features).NotTo(BeNil())
			Expect(features.KwargsData).To(HaveKeyWithValue("image", []string{"k"}))
			Expect(features.MMMetadata).To(HaveKeyWithValue("image", []string{"m"}))
		},
		Entry("none", api.MMPayloadNone),
		Entry("tensors", api.MMPayloadTensors),
	)

	It("omits mm_payload when the client sent none", func() {
		_, _, _, err := tk.RenderMessages(messages, api.RenderTools{}, "")
		Expect(err).NotTo(HaveOccurred())
		var sent map[string]json.RawMessage
		Expect(json.Unmarshal(body, &sent)).To(Succeed())
		Expect(sent).NotTo(HaveKey("mm_payload"))
	})
})
