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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

var _ = Describe("HF tokenizer Tokenize", func() {
	var (
		srv           *httptest.Server
		paths         []string
		body          []byte
		status        int
		version       string
		versionStatus int
		versionCalls  int
		hft           *HFTokenizer
	)

	received := func() map[string]any {
		var m map[string]any
		Expect(json.Unmarshal(body, &m)).To(Succeed())
		return m
	}

	BeforeEach(func() {
		paths = nil
		status = http.StatusOK
		version = "0.34.0"
		versionStatus = http.StatusOK
		versionCalls = 0
		srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/version" {
				versionCalls++
				w.WriteHeader(versionStatus)
				_, _ = w.Write([]byte(`{"version":"` + version + `"}`))
				return
			}
			paths = append(paths, r.URL.Path)
			body, _ = io.ReadAll(r.Body)
			w.WriteHeader(status)
			switch r.URL.Path {
			case "/v1/completions/render":
				_, _ = w.Write([]byte(`[{"token_ids":[5,6]}]`))
			case "/v1/chat/completions/render":
				_, _ = w.Write([]byte(`{"token_ids":[5,6,7]}`))
			default:
				_, _ = w.Write([]byte(`{"count":2,"max_model_len":32,"tokens":[5,6],"token_strs":["Hi","!"]}`))
			}
		}))
		DeferCleanup(srv.Close)
		var err error
		hft, err = NewHFTokenizer(context.Background(), klog.Background(), srv.URL, "base-model",
			5*time.Second, 5*time.Second)
		Expect(err).NotTo(HaveOccurred())
	})

	It("forwards a prompt to the render service's /tokenize", func() {
		prompt := "Hi!"
		tokens, strs, err := hft.Tokenize(api.TokenizeRequest{Model: "lora-1", Prompt: &prompt, ReturnTokenStrs: true})
		Expect(err).NotTo(HaveOccurred())
		Expect(paths).To(Equal([]string{"/tokenize"}))
		Expect(received()).To(Equal(map[string]any{
			"model": "base-model", "prompt": "Hi!", "return_token_strs": true,
		}))
		Expect(tokens).To(Equal([]uint32{5, 6}))
		Expect(strs).To(Equal([]string{"Hi", "!"}))
	})

	It("forwards messages without a prompt field", func() {
		_, _, err := hft.Tokenize(api.TokenizeRequest{
			Messages: []api.Message{{Role: "user", Content: api.ChatComplContent{Raw: "Hi!"}}},
		})
		Expect(err).NotTo(HaveOccurred())
		got := received()
		Expect(got).To(HaveKey("messages"))
		Expect(got).NotTo(HaveKey("prompt"))
		Expect(got).NotTo(HaveKey("return_token_strs"))
		Expect(got["model"]).To(Equal("base-model"))
	})

	It("forwards an empty messages list as an empty list", func() {
		_, _, err := hft.Tokenize(api.TokenizeRequest{Messages: []api.Message{}})
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal(`{"model":"base-model","messages":[]}`))
	})

	It("returns an error when the render service rejects the request", func() {
		status = http.StatusBadRequest
		prompt := "Hi!"
		_, _, err := hft.Tokenize(api.TokenizeRequest{Prompt: &prompt})
		Expect(err).To(HaveOccurred())
	})

	It("reads the render service version once", func() {
		prompt := "Hi!"
		for range 3 {
			_, _, err := hft.Tokenize(api.TokenizeRequest{Prompt: &prompt})
			Expect(err).NotTo(HaveOccurred())
		}
		Expect(versionCalls).To(Equal(1))
		Expect(paths).To(Equal([]string{"/tokenize", "/tokenize", "/tokenize"}))
	})

	Context("with a render service older than v0.30.0", func() {
		BeforeEach(func() {
			version = "0.29.0"
		})

		It("renders a prompt and splits its text", func() {
			prompt := "Hi there"
			tokens, strs, err := hft.Tokenize(api.TokenizeRequest{Prompt: &prompt, ReturnTokenStrs: true})
			Expect(err).NotTo(HaveOccurred())
			Expect(paths).To(Equal([]string{"/v1/completions/render"}))
			Expect(tokens).To(Equal([]uint32{5, 6}))
			Expect(strs).To(Equal([]string{"Hi ", "there"}))
		})

		It("renders messages", func() {
			tokens, _, err := hft.Tokenize(api.TokenizeRequest{
				Messages: []api.Message{{Role: "user", Content: api.ChatComplContent{Raw: "Hi!"}}},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(paths).To(Equal([]string{"/v1/chat/completions/render"}))
			Expect(tokens).To(Equal([]uint32{5, 6, 7}))
		})
	})

	It("treats a render service whose /version answers 404 as older and does not read it again", func() {
		versionStatus = http.StatusNotFound
		prompt := "Hi!"
		for range 2 {
			_, _, err := hft.Tokenize(api.TokenizeRequest{Prompt: &prompt})
			Expect(err).NotTo(HaveOccurred())
		}
		Expect(versionCalls).To(Equal(1))
		Expect(paths).To(Equal([]string{"/v1/completions/render", "/v1/completions/render"}))
	})

	It("renders the request and reads the version again when /version fails", func() {
		versionStatus = http.StatusServiceUnavailable
		prompt := "Hi!"
		_, _, err := hft.Tokenize(api.TokenizeRequest{Prompt: &prompt})
		Expect(err).NotTo(HaveOccurred())
		Expect(paths).To(Equal([]string{"/v1/completions/render"}))

		versionStatus = http.StatusOK
		_, _, err = hft.Tokenize(api.TokenizeRequest{Prompt: &prompt})
		Expect(err).NotTo(HaveOccurred())
		Expect(versionCalls).To(Equal(2))
		Expect(paths).To(Equal([]string{"/v1/completions/render", "/tokenize"}))
	})
})

var _ = DescribeTable("versionAtLeast v0.30.0",
	func(version string, want bool) {
		Expect(versionAtLeast(version, minTokenizeVersion)).To(Equal(want))
	},
	Entry("same version", "0.30.0", true),
	Entry("newer minor", "0.34.0", true),
	Entry("v prefix", "v0.34.0", true),
	Entry("local build suffix", "0.34.0+cpu", true),
	Entry("newer major", "1.0.0", true),
	Entry("older patch line", "0.29.9", false),
	Entry("dev build without a release tag", "0.1.dev1+gabc", false),
	Entry("not a version", "unknown", false),
)

var _ = Describe("Simple tokenizer Tokenize", func() {
	It("renders the prompt like RenderText", func() {
		st := NewSimpleTokenizer()
		prompt := "This is a test"
		tokens, strs, err := st.Tokenize(api.TokenizeRequest{Prompt: &prompt})
		Expect(err).NotTo(HaveOccurred())
		wantTokens, wantStrs, err := st.RenderText(prompt)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).To(Equal(wantTokens))
		Expect(strs).To(Equal(wantStrs))
	})

	It("renders the messages like RenderMessages when there is no prompt", func() {
		st := NewSimpleTokenizer()
		messages := []api.Message{{Role: "user", Content: api.ChatComplContent{Raw: "This is a test"}}}
		tokens, strs, err := st.Tokenize(api.TokenizeRequest{Messages: messages})
		Expect(err).NotTo(HaveOccurred())
		wantTokens, wantStrs, _, err := st.RenderMessages(messages, api.RenderTools{}, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).To(Equal(wantTokens))
		Expect(strs).To(Equal(wantStrs))
	})
})
