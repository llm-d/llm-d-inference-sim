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

package common

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Utils", Ordered, func() {
	Context("validateContextWindow", func() {
		Context("random mode", func() {
			It("should pass when prompt leaves room for at least one response token", func() {
				Expect(ValidateContextWindow(198, 200, ModeRandom)).Should(BeTrue())
			})

			It("should pass at the boundary, when the prompt plus one response token exactly fills the window", func() {
				Expect(ValidateContextWindow(199, 200, ModeRandom)).Should(BeTrue())
			})

			It("should fail when prompt alone fills or exceeds the context window", func() {
				Expect(ValidateContextWindow(200, 200, ModeRandom)).Should(BeFalse())
				Expect(ValidateContextWindow(201, 200, ModeRandom)).Should(BeFalse())
			})

			It("should ignore max-tokens entirely", func() {
				// even though this would previously fail on prompt+maxTokens > maxModelLen,
				// random mode no longer considers max-tokens at all
				Expect(ValidateContextWindow(100, 200, ModeRandom)).Should(BeTrue())
			})
		})

		Context("echo mode", func() {
			It("should pass when the prompt echoed back still fits within the window", func() {
				Expect(ValidateContextWindow(99, 200, ModeEcho)).Should(BeTrue())
			})

			It("should pass at the boundary, when the prompt echoed back exactly fills the window", func() {
				Expect(ValidateContextWindow(100, 200, ModeEcho)).Should(BeTrue())
			})

			It("should fail when the prompt echoed back would not fit within the window", func() {
				Expect(ValidateContextWindow(101, 200, ModeEcho)).Should(BeFalse())
				Expect(ValidateContextWindow(150, 200, ModeEcho)).Should(BeFalse())
			})
		})
	})

})
