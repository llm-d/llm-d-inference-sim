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
	"context"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

const (
	input         = "The purple giraffe sang opera while riding a bicycle through the crowded market."
	testModel     = "testmodel"
	qwenModelName = "Qwen/Qwen2-0.5B"
)

var _ = Describe("tokenizer", func() {
	messages := []openaiserverapi.Message{
		{Role: openaiserverapi.RoleUser, Content: openaiserverapi.Content{Raw: "q1"}},
		{Role: openaiserverapi.RoleAssistant, Content: openaiserverapi.Content{Raw: "a1"}},
		{Role: openaiserverapi.RoleUser, Content: openaiserverapi.Content{Raw: "q2"}},
	}

	It("should tokenize with simple tokenizer", func() {
		tokenizer, err := New(context.Background(), &common.Configuration{Model: testModel}, klog.Background())
		Expect(err).NotTo(HaveOccurred())
		tokens, strTokens, err := tokenizer.RenderText(input)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))

		output := strings.Join(strTokens, "")
		Expect(output).To(Equal(input))
	})

	It("should tokenize chat with simple tokenizer", func() {
		tokenizer, err := New(context.Background(), &common.Configuration{Model: testModel}, klog.Background())
		Expect(err).NotTo(HaveOccurred())

		tokens, strTokens, err := tokenizer.RenderChatCompletion(messages)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))
	})

	It("should tokenize with real tokenizer", func() {
		tokenizer, err := New(context.Background(), &common.Configuration{Model: qwenModelName}, klog.Background())
		Expect(err).NotTo(HaveOccurred())
		tokens, strTokens, err := tokenizer.RenderText(input)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))

		output := strings.Join(strTokens, "")
		Expect(output).To(Equal(input))
	})

	It("should tokenize chat with real tokenizer", func() {
		tokenizer, err := New(context.Background(), &common.Configuration{Model: qwenModelName}, klog.Background())
		Expect(err).NotTo(HaveOccurred())
		// in /chat/completions case the string tokens are not returned
		tokens, _, err := tokenizer.RenderChatCompletion(messages)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
	})
})
