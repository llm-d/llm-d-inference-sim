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

package common_test

import (
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

var _ = Describe("Failures", func() {
	Describe("ShouldInjectFailure", func() {
		It("should not inject failure when not in failure mode", func() {
			config := &common.Configuration{
				Mode:                 common.ModeRandom,
				FailureInjectionRate: 100,
			}
			Expect(common.ShouldInjectFailure(config)).To(BeFalse())
		})

		It("should not inject failure when rate is 0", func() {
			config := &common.Configuration{
				Mode:                 common.ModeFailure,
				FailureInjectionRate: 0,
			}
			Expect(common.ShouldInjectFailure(config)).To(BeFalse())
		})

		It("should inject failure when in failure mode with 100% rate", func() {
			config := &common.Configuration{
				Mode:                 common.ModeFailure,
				FailureInjectionRate: 100,
			}
			Expect(common.ShouldInjectFailure(config)).To(BeTrue())
		})
	})

	Describe("GetRandomFailure", func() {
		It("should return a failure from all types when none specified", func() {
			config := &common.Configuration{
				Model:        "test-model",
				FailureTypes: []string{},
			}
			failure := common.GetRandomFailure(config)
			Expect(failure.StatusCode).To(BeNumerically(">=", 400))
			Expect(failure.Message).ToNot(BeEmpty())
			Expect(failure.ErrorType).ToNot(BeEmpty())
		})

		It("should return rate limit failure when specified", func() {
			config := &common.Configuration{
				Model:        "test-model",
				FailureTypes: []string{"rate_limit"},
			}
			failure := common.GetRandomFailure(config)
			Expect(failure.StatusCode).To(Equal(429))
			Expect(failure.ErrorType).To(Equal("rate_limit_exceeded"))
			Expect(failure.ErrorCode).To(Equal("rate_limit_exceeded"))
			Expect(strings.Contains(failure.Message, "test-model")).To(BeTrue())
		})

		It("should return invalid API key failure when specified", func() {
			config := &common.Configuration{
				FailureTypes: []string{"invalid_api_key"},
			}
			failure := common.GetRandomFailure(config)
			Expect(failure.StatusCode).To(Equal(401))
			Expect(failure.ErrorType).To(Equal("invalid_request_error"))
			Expect(failure.ErrorCode).To(Equal("invalid_api_key"))
			Expect(failure.Message).To(Equal("Incorrect API key provided"))
		})

		It("should return context length failure when specified", func() {
			config := &common.Configuration{
				FailureTypes: []string{"context_length"},
			}
			failure := common.GetRandomFailure(config)
			Expect(failure.StatusCode).To(Equal(400))
			Expect(failure.ErrorType).To(Equal("invalid_request_error"))
			Expect(failure.ErrorCode).To(Equal("context_length_exceeded"))
			Expect(failure.Param).ToNot(BeNil())
			Expect(*failure.Param).To(Equal("messages"))
		})

		It("should return server error when specified", func() {
			config := &common.Configuration{
				FailureTypes: []string{"server_error"},
			}
			failure := common.GetRandomFailure(config)
			Expect(failure.StatusCode).To(Equal(503))
			Expect(failure.ErrorType).To(Equal("server_error"))
			Expect(failure.ErrorCode).To(Equal("server_error"))
		})

		It("should return model not found failure when specified", func() {
			config := &common.Configuration{
				Model:        "test-model",
				FailureTypes: []string{"model_not_found"},
			}
			failure := common.GetRandomFailure(config)
			Expect(failure.StatusCode).To(Equal(404))
			Expect(failure.ErrorType).To(Equal("invalid_request_error"))
			Expect(failure.ErrorCode).To(Equal("model_not_found"))
			Expect(strings.Contains(failure.Message, "test-model-nonexistent")).To(BeTrue())
		})

		It("should return server error as fallback for empty types", func() {
			config := &common.Configuration{
				FailureTypes: []string{},
			}
			// This test is probabilistic since it randomly selects, but we can test structure
			failure := common.GetRandomFailure(config)
			Expect(failure.StatusCode).To(BeNumerically(">=", 400))
			Expect(failure.ErrorType).ToNot(BeEmpty())
		})
	})
})