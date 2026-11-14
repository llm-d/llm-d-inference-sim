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

	It("should accept an enabled KV cache with a page size of 1", func() {
		config.KVCache.EnableKVCache = true
		config.KVCache.TokenBlockSize = 1
		Expect(New().ValidateConfig(config)).To(Succeed())
		Expect(config.KVCache.TokenBlockSize).To(Equal(1))
	})

	It("should reject a page size below 1 with the KV cache enabled", func() {
		config.KVCache.EnableKVCache = true
		config.KVCache.TokenBlockSize = 0
		Expect(New().ValidateConfig(config)).To(MatchError(ContainSubstring("page size must be at least 1")))
	})

	It("should clear the KV-cache settings while the cache is disabled", func() {
		Expect(config.KVCache.TokenBlockSize).To(Equal(16))
		Expect(New().ValidateConfig(config)).To(Succeed())
		Expect(config.KVCache).To(Equal(common.KVCacheConfig{}))
	})

	It("should reject fake metrics", func() {
		config.FakeMetrics = fakeMetrics{}
		Expect(New().ValidateConfig(config)).To(
			MatchError(ContainSubstring("fake metrics are not implemented by the sglang engine yet")))
	})
})

var _ = Describe("strict request validation", func() {
	It("should leave strict validation disabled for the experimental engine", func() {
		validator, err := New().NewRequestValidator()
		Expect(err).NotTo(HaveOccurred())
		Expect(validator).To(BeNil())
	})
})
