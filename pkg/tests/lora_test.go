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
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
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
			Expect(string(resp.Object)).To(Equal(api.ChatCompletionObject))

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
			var modelsResp api.ModelsResponse
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
			Expect(string(resp.Object)).To(Equal(api.ChatCompletionObject))

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

	Context("LoRA unloaded while in use", func() {
		var openaiclient openai.Client

		BeforeEach(func() {
			client, err := startServerWithArgs(context.TODO(),
				[]string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
					"--time-to-first-token", "500ms", "--max-num-seqs", "2", "--max-loras", "1",
					"--lora-modules", "{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
					"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}",
					"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}"})
			Expect(err).NotTo(HaveOccurred())
			openaiclient = openai.NewClient(option.WithBaseURL(baseURL), option.WithHTTPClient(client))
		})

		It("Should free the slot of a LoRA unloaded while its request runs", func() {
			ctx := context.TODO()
			done := make(chan error, 1)
			go func() {
				_, err := openaiclient.Chat.Completions.New(ctx, paramsLora1)
				done <- err
			}()
			time.Sleep(100 * time.Millisecond)
			unloadLora(ctx, openaiclient, "lora1")
			Eventually(done, "2s").Should(Receive(BeNil()))

			reqCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()
			_, err := openaiclient.Chat.Completions.New(reqCtx, paramsLora2)
			Expect(err).NotTo(HaveOccurred())
		})

		It("Should keep max-loras for a queued request whose LoRA was unloaded", func() {
			ctx := context.TODO()
			var wg sync.WaitGroup
			var finishedLora1, finishedLora2, finishedLora3 time.Time
			send := func(params openai.ChatCompletionNewParams, finished *time.Time) {
				defer GinkgoRecover()
				defer wg.Done()
				_, err := openaiclient.Chat.Completions.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				*finished = time.Now()
			}

			// the lora1 request holds the only slot, so the lora2 and lora3 requests wait
			wg.Add(3)
			go send(paramsLora1, &finishedLora1)
			time.Sleep(100 * time.Millisecond)
			go send(paramsLora2, &finishedLora2)
			time.Sleep(100 * time.Millisecond)
			unloadLora(ctx, openaiclient, "lora2")
			go send(paramsLora3, &finishedLora3)
			wg.Wait()

			// the lora3 request can start only after the lora2 request releases the slot
			Expect(finishedLora3.Sub(finishedLora2)).To(BeNumerically(">=", 400*time.Millisecond))
		})
	})
})

func unloadLora(ctx context.Context, openaiclient openai.Client, name string) {
	body, err := json.Marshal(map[string]string{"lora_name": name})
	Expect(err).ToNot(HaveOccurred())
	err = openaiclient.Post(ctx, "/unload_lora_adapter", body, nil,
		option.WithHeader("Content-Type", "application/json"))
	Expect(err).ToNot(HaveOccurred())
}
