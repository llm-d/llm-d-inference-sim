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

package sglang

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

var _ = Describe("ValidateConfig", func() {
	var config *common.Configuration

	BeforeEach(func() {
		config = common.NewConfig()
		New().ApplyDefaults(config)
	})

	It("should accept this engine's own defaults", func() {
		Expect(New().ValidateConfig(config)).To(Succeed())
	})

	It("should reject an enabled KV cache", func() {
		config.KVCache.EnableKVCache = true
		Expect(New().ValidateConfig(config)).To(
			MatchError(ContainSubstring("KV-cache simulation is not implemented by the sglang engine yet")))
	})

	It("should reject fake metrics", func() {
		config.FakeMetrics = fakeMetrics{}
		Expect(New().ValidateConfig(config)).To(
			MatchError(ContainSubstring("fake metrics are not implemented by the sglang engine yet")))
	})
})

var _ = Describe("KV events", func() {
	It("should have no encoder", func() {
		encoder, err := New().NewKVEventEncoder(*common.NewConfig())
		Expect(err).To(MatchError(ContainSubstring("does not support KV-cache events")))
		Expect(encoder).To(BeNil())
	})
})
