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

package tokenizer

import (
	"strings"

	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const (
	input = "The purple giraffe sang opera while riding a bicycle through the crowded market."
)

var _ = Describe("tokenizer", func() {
	messages := []openaiserverapi.Message{
		{Role: openaiserverapi.RoleUser, Content: openaiserverapi.ChatComplContent{Raw: "q1"}},
		{Role: openaiserverapi.RoleAssistant, Content: openaiserverapi.ChatComplContent{Raw: "a1"}},
		{Role: openaiserverapi.RoleUser, Content: openaiserverapi.ChatComplContent{Raw: "q2"}},
	}

	It("should tokenize with simple tokenizer", func() {
		tokens, strTokens, err := tokenizerMngr.TestTokenizer().RenderText(input)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))

		output := strings.Join(strTokens, "")
		Expect(output).To(Equal(input))
	})

	It("should tokenize chat with simple tokenizer", func() {
		tokens, strTokens, _, err := tokenizerMngr.TestTokenizer().RenderMessages(messages)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))
	})

	It("should tokenize with real tokenizer", func() {
		tokens, strTokens, err := tokenizerMngr.RealTokenizer().RenderText(input)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))

		output := strings.Join(strTokens, "")
		Expect(output).To(Equal(input))
	})

	It("should tokenize chat with real tokenizer", func() {
		// in /chat/completions case the string tokens are not returned
		tokens, _, _, err := tokenizerMngr.RealTokenizer().RenderMessages(messages)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
	})
})
