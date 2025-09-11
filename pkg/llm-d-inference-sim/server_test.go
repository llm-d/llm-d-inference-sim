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
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Server", func() {
	It("Should respond to /health", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Get("http://localhost/health")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
	})

	It("Should respond to /ready", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Get("http://localhost/ready")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
	})

	Context("tokenize", Ordered, func() {
		tmpDir := "./tests-tmp/"
		AfterAll(func() {
			err := os.RemoveAll(tmpDir)
			Expect(err).NotTo(HaveOccurred())
		})

		It("Should return correct response to /tokenize chat", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", qwenModelName, "--mode", common.ModeRandom,
				"--tokenizers-cache-dir", tmpDir, "--max-model-len", "2048"}
			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
			Expect(err).NotTo(HaveOccurred())

			reqBody := `{
				"messages": [{"role": "user", "content": "This is a test"}],
				"model": "Qwen/Qwen2-0.5B"
			}`
			resp, err := client.Post("http://localhost/tokenize", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var tokenizeResp vllmapi.TokenizeResponse
			err = json.Unmarshal(body, &tokenizeResp)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokenizeResp.Count).To(Equal(4))
			Expect(tokenizeResp.Tokens).To(HaveLen(4))
			Expect(tokenizeResp.MaxModelLen).To(Equal(2048))
		})

		It("Should return correct response to /tokenize text", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", qwenModelName, "--mode", common.ModeRandom,
				"--tokenizers-cache-dir", tmpDir, "--max-model-len", "2048"}
			client, err := startServerWithArgs(ctx, common.ModeRandom, args, nil)
			Expect(err).NotTo(HaveOccurred())

			reqBody := `{
				"prompt": "This is a test",
				"model": "Qwen/Qwen2-0.5B"
			}`
			resp, err := client.Post("http://localhost/tokenize", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var tokenizeResp vllmapi.TokenizeResponse
			err = json.Unmarshal(body, &tokenizeResp)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokenizeResp.Count).To(Equal(4))
			Expect(tokenizeResp.Tokens).To(HaveLen(4))
			Expect(tokenizeResp.MaxModelLen).To(Equal(2048))
		})
	})
})
