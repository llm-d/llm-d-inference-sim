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
	"os"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

const (
	input           = "The purple giraffe sang opera while riding a bicycle through the crowded market."
	qwenModelName   = "Qwen/Qwen2-0.5B"
	tokenizerTmpDir = "./test_tokenizers"
)

var _ = Describe("tokenizer", func() {

	It("should tokenize with simple tokenizer", func() {
		tokenizer, err := New(&common.Configuration{Model: "test"}, klog.Background())
		Expect(err).NotTo(HaveOccurred())
		tokens, strTokens, err := tokenizer.Encode(input, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))

		output := strings.Join(strTokens, "")
		Expect(output).To(Equal(input))
	})

	It("should tokenize with real tokenizer", func() {
		config := &common.Configuration{Model: qwenModelName, TokenizersCacheDir: tokenizerTmpDir}
		tokenizer, err := New(config, klog.Background())
		Expect(err).NotTo(HaveOccurred())
		tokens, strTokens, err := tokenizer.Encode(input, qwenModelName)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))

		output := strings.Join(strTokens, "")
		Expect(output).To(Equal(input))

		err = os.RemoveAll(tokenizerTmpDir)
		Expect(err).NotTo(HaveOccurred())
	})
})
