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

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"
)

var _ = Describe("Simulator with seed", func() {
	firstText := ""
	DescribeTable("text completions with the same seed",
		// use a function so that httpClient is captured when running
		func() {
			ctx := context.TODO()
			client, err := startServerWithArgs(ctx, common.ModeRandom,
				[]string{"cmd", "--model", model, "--mode", common.ModeRandom, "--seed", "100"}, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, model, userMessage, false)
			params.MaxTokens = openai.Int(10)
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

	texts := make([]string, 0)
	DescribeTable("text completions with different seeds",
		func(lastTest bool) {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClentAndCompletionParams(client, model, userMessage, false)
			resp, err := openaiclient.Completions.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Choices).ShouldNot(BeEmpty())
			Expect(string(resp.Object)).To(Equal(textCompletionObject))

			text := resp.Choices[0].Text
			Expect(text).ShouldNot(BeEmpty())
			texts = append(texts, text)
			if lastTest {
				Expect(hasAtLeastTwoDifferentTexts(texts)).To(BeTrue())
			}
		},
		Entry("first time text completion without seed", false),
		Entry("second time text completion without seed", false),
		Entry("third time text completion without seed", false),
		Entry("fourth time text completion without seed", false),
		Entry("fifth time text completion without seed", false),
		Entry("sixth time text completion without seed", false),
		Entry("seventh time text completion without seed", false),
		Entry("eighth time text completion without seed", true),
	)
})

func hasAtLeastTwoDifferentTexts(texts []string) bool {
	unique := make(map[string]struct{})
	for _, s := range texts {
		unique[s] = struct{}{}
		if len(unique) > 1 {
			return true
		}
	}
	return false
}
