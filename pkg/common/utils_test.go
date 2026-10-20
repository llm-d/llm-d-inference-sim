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
		It("should pass when prompt leaves room for at least one response token", func() {
			Expect(ValidateContextWindow(198, 200)).Should(BeTrue())
		})

		It("should pass at the boundary, when the prompt plus one response token exactly fills the window", func() {
			Expect(ValidateContextWindow(199, 200)).Should(BeTrue())
		})

		It("should fail when prompt alone fills or exceeds the context window", func() {
			Expect(ValidateContextWindow(200, 200)).Should(BeFalse())
			Expect(ValidateContextWindow(201, 200)).Should(BeFalse())
		})
	})

	DescribeTable("EchoResponseLen",
		func(echoLen int, minTokens *int64, maxLen int, padToMax bool, expected int) {
			Expect(EchoResponseLen(echoLen, minTokens, maxLen, padToMax)).To(Equal(expected))
		},
		Entry("max len above the echo length", 10, nil, 20, false, 10),
		Entry("max len below the echo length", 10, nil, 5, false, 5),
		Entry("max len below the echo length with pad to max", 10, nil, 5, true, 5),
		Entry("max len above the echo length with pad to max", 10, nil, 20, true, 20),
		Entry("max len equal to the echo length with pad to max", 10, nil, 10, true, 10),
		Entry("min tokens below the echo length", 10, new(int64(5)), 20, false, 10),
		Entry("min tokens above the echo length and below max len", 4, new(int64(20)), 30, false, 20),
		Entry("min tokens above the echo length and below max len with pad to max", 4, new(int64(20)), 30, true, 30),
		Entry("min tokens below the echo truncated to max len", 10, new(int64(3)), 5, false, 5),
	)

})
