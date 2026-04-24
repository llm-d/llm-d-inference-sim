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

package tests

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

var _ = Describe("LoRAs", func() {
	Context("LoRAs config and load", func() {
		It("Should config, load and unload LoRAs correctly", func() {
			ctx := context.TODO()
			client, err := startServerWithArgs(ctx,
				[]string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
					"--lora-modules", "{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
					"{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}"})
			Expect(err).NotTo(HaveOccurred())

			// Request to lora3
			openaiclient, params := getOpenAIClientAndChatParams(client, "lora3", testUserMessage, false)
			resp, err := openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).ToNot(HaveOccurred())

			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(openaiserverapi.ChatCompletionObject))

			msg := resp.Choices[0].Message.Content
			Expect(msg).Should(Equal(testUserMessage))

			// Unknown model, should return 404
			params.Model = "lora1"
			_, err = openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).To(HaveOccurred())
			var openaiError *openai.Error
			ok := errors.As(err, &openaiError)
			Expect(ok).To(BeTrue())
			Expect(openaiError.StatusCode).To(Equal(404))

			// Add lora1
			payload := map[string]string{
				"lora_name": "lora1",          // Name to register the adapter as
				"lora_path": "/path/to/lora1", // Local or remote path
			}

			loraParams, err := json.Marshal(payload)
			Expect(err).ToNot(HaveOccurred())

			options := option.WithHeader("Content-Type", "application/json")
			err = openaiclient.Post(ctx, "/load_lora_adapter", loraParams, nil, options)
			Expect(err).ToNot(HaveOccurred())

			// Should be four models: base model and three LoRAs
			var modelsResp vllmapi.ModelsResponse
			err = openaiclient.Get(ctx, "/models", nil, &modelsResp)
			Expect(err).ToNot(HaveOccurred())
			Expect(modelsResp).NotTo(BeNil())
			Expect(modelsResp.Data).To(HaveLen(4))

			// Root must reflect the LoRA's filesystem path, not its name.
			// lora3/lora4 come from --lora-modules; lora1 from /load_lora_adapter.
			expectedRoots := map[string]string{
				"lora1": "/path/to/lora1",
				"lora3": "/path/to/lora3",
				"lora4": "/path/to/lora4",
			}
			for _, model := range modelsResp.Data {
				if strings.HasPrefix(model.ID, "lora") {
					Expect(model.Parent).ToNot(BeNil())
					Expect(*model.Parent).To(Equal(common.TestModelName))
					Expect(model.Root).To(Equal(expectedRoots[model.ID]))
				} else {
					Expect(model.Parent).To(BeNil())
				}
			}

			// Request to lora1, should work now
			resp, err = openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).ToNot(HaveOccurred())

			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(openaiserverapi.ChatCompletionObject))

			msg = resp.Choices[0].Message.Content
			Expect(msg).Should(Equal(testUserMessage))

			// Unload lora3
			payload = map[string]string{
				"lora_name": "lora3",          // Name to register the adapter as
				"lora_path": "/path/to/lora3", // Local or remote path
			}

			loraParams, err = json.Marshal(payload)
			Expect(err).ToNot(HaveOccurred())
			options = option.WithHeader("Content-Type", "application/json")
			err = openaiclient.Post(ctx, "/unload_lora_adapter", loraParams, nil, options)
			Expect(err).ToNot(HaveOccurred())

			// We should now get an error now
			params.Model = "lora3"
			_, err = openaiclient.Chat.Completions.New(ctx, params)
			Expect(err).To(HaveOccurred())
			ok = errors.As(err, &openaiError)
			Expect(ok).To(BeTrue())
			Expect(openaiError.StatusCode).To(Equal(404))

			// Should be three models: base model and two LoRAs
			err = openaiclient.Get(ctx, "/models", nil, &modelsResp)
			Expect(err).ToNot(HaveOccurred())
			Expect(modelsResp).NotTo(BeNil())
			Expect(modelsResp.Data).To(HaveLen(3))
		})
	})
})
