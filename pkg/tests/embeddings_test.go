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

package tests

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// postEmbeddings sends a raw /v1/embeddings request, for the fields the OpenAI
// client's own types cannot carry.
func postEmbeddings(client *http.Client, body string) *api.EmbeddingResponse {
	resp, err := client.Post("http://localhost/v1/embeddings", "application/json", strings.NewReader(body))
	Expect(err).NotTo(HaveOccurred())
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	Expect(resp.StatusCode).To(Equal(http.StatusOK), "response body: %s", string(data))

	var embeddings api.EmbeddingResponse
	Expect(json.Unmarshal(data, &embeddings)).To(Succeed())
	return &embeddings
}

var _ = Describe("Simulator for /v1/embeddings", forEachEngine(func() {
	var ctx context.Context

	BeforeEach(func() {
		ctx = context.TODO()
	})

	It("returns one vector of the default width for a single input", func() {
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client),
			option.WithMaxRetries(0))

		resp, err := openaiclient.Embeddings.New(ctx, openai.EmbeddingNewParams{
			Input: openai.EmbeddingNewParamsInputUnion{OfString: param.NewOpt(testUserMessage)},
			Model: common.TestModelName,
		})
		Expect(err).NotTo(HaveOccurred())

		// The client types both fields as constants rather than strings.
		Expect(resp.Object).To(BeEquivalentTo("list"))
		Expect(resp.Model).To(Equal(common.TestModelName))
		Expect(resp.Data).To(HaveLen(1))
		Expect(resp.Data[0].Object).To(BeEquivalentTo("embedding"))
		Expect(resp.Data[0].Index).To(BeZero())
		Expect(resp.Data[0].Embedding).To(HaveLen(384))
		Expect(resp.Usage.PromptTokens).To(BeNumerically(">", 0))
		Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.PromptTokens))
	})

	It("returns one vector per input, indexed in request order", func() {
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		openaiclient := openai.NewClient(
			option.WithBaseURL(baseURL),
			option.WithHTTPClient(client),
			option.WithMaxRetries(0))

		resp, err := openaiclient.Embeddings.New(ctx, openai.EmbeddingNewParams{
			Input: openai.EmbeddingNewParamsInputUnion{
				OfArrayOfStrings: []string{"first input", "second input", "third input"},
			},
			Model: common.TestModelName,
		})
		Expect(err).NotTo(HaveOccurred())

		Expect(resp.Data).To(HaveLen(3))
		for i, item := range resp.Data {
			Expect(item.Index).To(BeNumerically("==", i))
			Expect(item.Embedding).To(HaveLen(384))
		}
	})

	It("honors the requested number of dimensions", func() {
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		embeddings := postEmbeddings(client, `{"model":"`+common.TestModelName+
			`","input":"`+testUserMessage+`","dimensions":8}`)

		Expect(embeddings.Data).To(HaveLen(1))
		vector, ok := embeddings.Data[0].Embedding.([]any)
		Expect(ok).To(BeTrue(), "embedding: %v", embeddings.Data[0].Embedding)
		Expect(vector).To(HaveLen(8))
	})

	It("returns a base64 vector when that encoding format is requested", func() {
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		embeddings := postEmbeddings(client, `{"model":"`+common.TestModelName+
			`","input":"`+testUserMessage+`","dimensions":8,"encoding_format":"base64"}`)

		Expect(embeddings.Data).To(HaveLen(1))
		encoded, ok := embeddings.Data[0].Embedding.(string)
		Expect(ok).To(BeTrue(), "embedding: %v", embeddings.Data[0].Embedding)
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		Expect(err).NotTo(HaveOccurred())
		// Little-endian float32s, so four bytes per dimension.
		Expect(decoded).To(HaveLen(8 * 4))
	})

	It("accepts token ids as input", func() {
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		embeddings := postEmbeddings(client, `{"model":"`+common.TestModelName+
			`","input":[1037, 2024, 3899],"dimensions":8}`)

		Expect(embeddings.Data).To(HaveLen(1))
		Expect(embeddings.Usage.PromptTokens).To(BeNumerically(">", 0))
	})
}))
