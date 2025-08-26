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
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

const (
	// Error message templates
	rateLimitMessageTemplate     = "Rate limit reached for %s in organization org-xxx on requests per min (RPM): Limit 3, Used 3, Requested 1."
	modelNotFoundMessageTemplate = "The model '%s-nonexistent' does not exist"
)

type FailureInfo struct {
	StatusCode int
	Error      openaiserverapi.CompletionError
}

var predefinedFailures = map[string]FailureInfo{
	common.FailureTypeRateLimit: {
		StatusCode: 429,
		Error: openaiserverapi.CompletionError{
			Message: rateLimitMessageTemplate,
			Type:    "rate_limit_exceeded",
			Code:    429,
			Param:   nil,
		},
	},
	common.FailureTypeInvalidAPIKey: {
		StatusCode: 401,
		Error: openaiserverapi.CompletionError{
			Message: "Incorrect API key provided",
			Type:    "invalid_request_error",
			Code:    401,
			Param:   nil,
		},
	},
	common.FailureTypeContextLength: {
		StatusCode: 400,
		Error: openaiserverapi.CompletionError{
			Message: "This model's maximum context length is 4096 tokens. However, your messages resulted in 4500 tokens.",
			Type:    "invalid_request_error",
			Code:    400,
			Param:   stringPtr("messages"),
		},
	},
	common.FailureTypeServerError: {
		StatusCode: 503,
		Error: openaiserverapi.CompletionError{
			Message: "The server is overloaded or not ready yet.",
			Type:    "server_error",
			Code:    503,
			Param:   nil,
		},
	},
	common.FailureTypeInvalidRequest: {
		StatusCode: 400,
		Error: openaiserverapi.CompletionError{
			Message: "Invalid request: missing required parameter 'model'.",
			Type:    "invalid_request_error",
			Code:    400,
			Param:   stringPtr("model"),
		},
	},
	common.FailureTypeModelNotFound: {
		StatusCode: 404,
		Error: openaiserverapi.CompletionError{
			Message: modelNotFoundMessageTemplate,
			Type:    "invalid_request_error",
			Code:    404,
			Param:   stringPtr("model"),
		},
	},
}

// shouldInjectFailure determines whether to inject a failure based on configuration
func shouldInjectFailure(config *common.Configuration) bool {
	if config.FailureInjectionRate == 0 {
		return false
	}

	return common.RandomInt(1, 100) <= config.FailureInjectionRate
}

// GetRandomFailure returns a random failure from configured types or all types if none specified
func GetRandomFailure(config *common.Configuration) FailureInfo {
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
		failure.Error.Message = fmt.Sprintf(rateLimitMessageTemplate, config.Model)
	} else if randomType == common.FailureTypeModelNotFound && config.Model != "" {
		failure.Error.Message = fmt.Sprintf(modelNotFoundMessageTemplate, config.Model)
	}

	return failure
}

func stringPtr(s string) *string {
	return &s
}
