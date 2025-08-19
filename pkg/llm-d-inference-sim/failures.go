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
	"fmt"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const (
	// Error message templates
	RateLimitMessageTemplate    = "Rate limit reached for %s in organization org-xxx on requests per min (RPM): Limit 3, Used 3, Requested 1."
	ModelNotFoundMessageTemplate = "The model '%s-nonexistent' does not exist"
)

type FailureSpec struct {
	StatusCode int
	ErrorType  string
	ErrorCode  string
	Message    string
	Param      *string
}

var predefinedFailures = map[string]FailureSpec{
	common.FailureTypeRateLimit: {
		StatusCode: 429,
		ErrorType:  "rate_limit_exceeded",
		ErrorCode:  "rate_limit_exceeded",
		Message:    "Rate limit reached for model in organization org-xxx on requests per min (RPM): Limit 3, Used 3, Requested 1.",
		Param:      nil,
	},
	common.FailureTypeInvalidAPIKey: {
		StatusCode: 401,
		ErrorType:  "invalid_request_error",
		ErrorCode:  "invalid_api_key",
		Message:    "Incorrect API key provided",
		Param:      nil,
	},
	common.FailureTypeContextLength: {
		StatusCode: 400,
		ErrorType:  "invalid_request_error",
		ErrorCode:  "context_length_exceeded",
		Message:    "This model's maximum context length is 4096 tokens. However, your messages resulted in 4500 tokens.",
		Param:      stringPtr("messages"),
	},
	common.FailureTypeServerError: {
		StatusCode: 503,
		ErrorType:  "server_error",
		ErrorCode:  "server_error",
		Message:    "The server is overloaded or not ready yet.",
		Param:      nil,
	},
	common.FailureTypeInvalidRequest: {
		StatusCode: 400,
		ErrorType:  "invalid_request_error",
		ErrorCode:  "invalid_request_error",
		Message:    "Invalid request: missing required parameter 'model'.",
		Param:      stringPtr("model"),
	},
	common.FailureTypeModelNotFound: {
		StatusCode: 404,
		ErrorType:  "invalid_request_error",
		ErrorCode:  "model_not_found",
		Message:    "The model 'gpt-nonexistent' does not exist",
		Param:      stringPtr("model"),
	},
}

// ShouldInjectFailure determines whether to inject a failure based on configuration
func ShouldInjectFailure(config *common.Configuration) bool {
	if config.FailureInjectionRate == 0 {
		return false
	}
	
	return common.RandomInt(1, 100) <= config.FailureInjectionRate
}

// GetRandomFailure returns a random failure from configured types or all types if none specified
func GetRandomFailure(config *common.Configuration) FailureSpec {
	var availableFailures []string
	if len(config.FailureTypes) == 0 {
		// Use all failure types if none specified
		for failureType := range predefinedFailures {
			availableFailures = append(availableFailures, failureType)
		}
	} else {
		availableFailures = config.FailureTypes
	}
	
	if len(availableFailures) == 0 {
		// Fallback to server_error if no valid types
		return predefinedFailures[common.FailureTypeServerError]
	}
	
	randomIndex := common.RandomInt(0, len(availableFailures)-1)
	randomType := availableFailures[randomIndex]
	
	// Customize message with current model name
	failure := predefinedFailures[randomType]
	if randomType == common.FailureTypeRateLimit && config.Model != "" {
		failure.Message = fmt.Sprintf(RateLimitMessageTemplate, config.Model)
	} else if randomType == common.FailureTypeModelNotFound && config.Model != "" {
		failure.Message = fmt.Sprintf(ModelNotFoundMessageTemplate, config.Model)
	}
	
	return failure
}

func stringPtr(s string) *string {
	return &s
}