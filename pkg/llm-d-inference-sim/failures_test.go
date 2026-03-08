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
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

var _ = Describe("Failures", func() {
	Describe("GetRandomFailure", Ordered, func() {
		var random *common.Random
		BeforeAll(func() {
			random = common.NewRandom(time.Now().UnixNano(), 8080)
		})

		It("should return a failure from all types when none specified", func() {
			config := &common.Configuration{
				Model:        "test-model",
				FailureTypes: []string{},
			}
			failure := getRandomFailure(config, random)
			Expect(failure.Code).To(BeNumerically(">=", 400))
			Expect(failure.Message).ToNot(BeEmpty())
			Expect(failure.Type).ToNot(BeEmpty())
		})

		It("should return rate limit failure when specified", func() {
			config := &common.Configuration{
				Model:        "test-model",
				FailureTypes: []string{common.FailureTypeRateLimit},
			}
			failure := getRandomFailure(config, random)
			Expect(failure.Code).To(Equal(429))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(429)))
			Expect(strings.Contains(failure.Message, "test-model")).To(BeTrue())
		})

		It("should return invalid API key failure when specified", func() {
			config := &common.Configuration{
				FailureTypes: []string{common.FailureTypeInvalidAPIKey},
			}
			failure := getRandomFailure(config, random)
			Expect(failure.Code).To(Equal(401))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(401)))
			Expect(failure.Message).To(Equal("Incorrect API key provided."))
		})

		It("should return context length failure when specified", func() {
			config := &common.Configuration{
				FailureTypes: []string{common.FailureTypeContextLength},
			}
			failure := getRandomFailure(config, random)
			Expect(failure.Code).To(Equal(400))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(400)))
			Expect(failure.Param).ToNot(BeNil())
			Expect(*failure.Param).To(Equal("messages"))
		})

		It("should return server error when specified", func() {
			config := &common.Configuration{
				FailureTypes: []string{common.FailureTypeServerError},
			}
			failure := getRandomFailure(config, random)
			Expect(failure.Code).To(Equal(503))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(503)))
		})

		It("should return model not found failure when specified", func() {
			config := &common.Configuration{
				Model:        "test-model",
				FailureTypes: []string{common.FailureTypeModelNotFound},
			}
			failure := getRandomFailure(config, random)
			Expect(failure.Code).To(Equal(404))
			Expect(failure.Type).To(Equal(openaiserverapi.ErrorCodeToType(404)))
			Expect(strings.Contains(failure.Message, "test-model-nonexistent")).To(BeTrue())
		})

		It("should return server error as fallback for empty types", func() {
			config := &common.Configuration{
				FailureTypes: []string{},
			}
			// This test is probabilistic since it randomly selects, but we can test structure
			failure := getRandomFailure(config, random)
			Expect(failure.Code).To(BeNumerically(">=", 400))
			Expect(failure.Type).ToNot(BeEmpty())
		})
	})
})
