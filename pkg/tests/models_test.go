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
	"encoding/json"
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// getModels reads /v1/models and returns the parsed response.
func getModels(client *http.Client) *api.ModelsResponse {
	resp, err := client.Get("http://localhost/v1/models")
	Expect(err).NotTo(HaveOccurred())
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	Expect(resp.StatusCode).To(Equal(http.StatusOK), "response body: %s", string(data))

	var models api.ModelsResponse
	Expect(json.Unmarshal(data, &models)).To(Succeed())
	return &models
}

// The entries this route serves for LoRA adapters are vLLM's, since the engine
// owns the adapters themselves; they are covered in the LoRA suite.
var _ = Describe("Simulator for /v1/models", forEachEngine(func() {
	var ctx context.Context

	BeforeEach(func() {
		ctx = context.TODO()
	})

	It("advertises the base model as owned by the running engine", func() {
		client, err := startServer(ctx, common.ModeEcho)
		Expect(err).NotTo(HaveOccurred())

		models := getModels(client)

		Expect(models.Object).To(Equal("list"))
		Expect(models.Data).To(HaveLen(1))

		model := models.Data[0]
		Expect(model.ID).To(Equal(common.TestModelName))
		Expect(model.Object).To(Equal(api.ObjectModel))
		// The engine's own name, the one field of this route an engine decides.
		Expect(model.OwnedBy).To(Equal(currentEngine))
		Expect(model.Root).To(Equal(common.TestModelName))
		// Null for a base model, and set only for a LoRA adapter.
		Expect(model.Parent).To(BeNil())
		Expect(model.Created).To(BeNumerically(">", 0))
		Expect(model.MaxModelLen).To(BeNumerically(">", 0))
	})

	It("advertises every served model name, each rooted at the real model", func() {
		client, err := startServerWithArgs(ctx, []string{"cmd",
			"--model", common.TestModelName, "--mode", common.ModeEcho,
			"--served-model-name", "alias-one", "alias-two"})
		Expect(err).NotTo(HaveOccurred())

		models := getModels(client)

		Expect(models.Data).To(HaveLen(2))
		ids := []string{models.Data[0].ID, models.Data[1].ID}
		Expect(ids).To(ConsistOf("alias-one", "alias-two"))
		for _, model := range models.Data {
			Expect(model.Root).To(Equal(common.TestModelName))
			Expect(model.OwnedBy).To(Equal(currentEngine))
			Expect(model.Parent).To(BeNil())
		}
	})

	It("reports the configured context window", func() {
		client, err := startServerWithArgs(ctx, []string{"cmd",
			"--model", common.TestModelName, "--mode", common.ModeEcho,
			engineFlag(common.FieldContextWindow), "2048"})
		Expect(err).NotTo(HaveOccurred())

		models := getModels(client)

		Expect(models.Data).To(HaveLen(1))
		Expect(models.Data[0].MaxModelLen).To(Equal(2048))
	})
}))
