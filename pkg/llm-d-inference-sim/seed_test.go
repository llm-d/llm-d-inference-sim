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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

var _ = Describe("Simulator with seed", func() {
	firstText := ""
	DescribeTable("text completions",
		// use a function so that httpClient is captured when running
		func() {
			ctx := context.TODO()
			client, err := startServerWithArgs(ctx, modeRandom,
				[]string{"cmd", "--model", model, "--mode", modeRandom, "--seed", "100"})
			Expect(err).NotTo(HaveOccurred())

			openaiclient := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(client))
			params := openai.CompletionNewParams{
				Prompt: openai.CompletionNewParamsPromptUnion{
					OfString: openai.String(userMessage),
				},
				Model: openai.CompletionNewParamsModel(model),
			}

			resp, err := openaiclient.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(textCompletionObject))

			text := resp.Choices[0].Text
			Expect(text).ShouldNot(BeEmpty())
			if firstText == "" {
				firstText = text
			} else {
				Expect(text).Should(Equal(firstText))
			}
		},
		Entry("first time text completion with seed"),
		Entry("second time text completion with seed"),
		Entry("third time text completion with seed"),
		Entry("fourth time text completion with seed"),
		Entry("fifth time text completion with seed"),
		Entry("sixth time text completion with seed"),
		Entry("seventh time text completion with seed"),
		Entry("eighth time text completion with seed"),
	)
})
