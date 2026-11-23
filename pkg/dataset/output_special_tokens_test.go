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

package dataset

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

var _ = Describe("Dataset responses with a BOS model", func() {
	const bos = 1
	var tk tokenizer.Tokenizer

	BeforeEach(func() {
		// Renders like vLLM for a model with a BOS token: one id per prompt,
		// preceded by BOS unless add_special_tokens is false.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer GinkgoRecover()
			var req map[string]json.RawMessage
			Expect(json.NewDecoder(r.Body).Decode(&req)).To(Succeed())
			ids := []uint32{100 + uint32(len(req["prompt"]))}
			if string(req["add_special_tokens"]) != "false" {
				ids = append([]uint32{bos}, ids...)
			}
			Expect(json.NewEncoder(w).Encode([]map[string][]uint32{{"token_ids": ids}})).To(Succeed())
		}))
		DeferCleanup(srv.Close)
		var err error
		tk, err = tokenizer.NewHFTokenizer(context.Background(), log.FromContext(context.Background()),
			srv.URL, "m", time.Second, time.Second)
		Expect(err).NotTo(HaveOccurred())
	})

	It("does not start the default responses with BOS", func() {
		ds := &DefaultDataset{}
		Expect(ds.Init(context.Background(), log.FromContext(context.Background()),
			common.NewRandom(1, 8080), 1024, tk)).To(Succeed())
		for _, resp := range ds.tokenizedResponses {
			Expect(resp.Tokens).NotTo(ContainElement(uint32(bos)))
		}
	})

	It("returns the token of '!' in mm-encoder-only mode", func() {
		ds, err := NewMMEncoderOnlyDataset(log.FromContext(context.Background()), tk)
		Expect(err).NotTo(HaveOccurred())
		Expect(ds.tokens.Tokens).To(Equal([]uint32{100 + uint32(len(`"!"`))}))
	})
})
