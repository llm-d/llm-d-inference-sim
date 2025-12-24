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

var predefinedFailures = map[string]openaiserverapi.Error{
	common.FailureTypeRateLimit:     openaiserverapi.NewError(rateLimitMessageTemplate, 429, nil),
	common.FailureTypeInvalidAPIKey: openaiserverapi.NewError("Incorrect API key provided.", 401, nil),
	common.FailureTypeContextLength: openaiserverapi.NewError(
		"This model's maximum context length is 4096 tokens. However, your messages resulted in 4500 tokens.",
		400, stringPtr("messages")),
	common.FailureTypeServerError: openaiserverapi.NewError(
		"The server is overloaded or not ready yet.", 503, nil),
	common.FailureTypeInvalidRequest: openaiserverapi.NewError(
		"Invalid request: missing required parameter 'model'.", 400, stringPtr("model")),
	common.FailureTypeModelNotFound: openaiserverapi.NewError(modelNotFoundMessageTemplate,
		404, stringPtr("model")),
}

// shouldInjectFailure determines whether to inject a failure based on configuration
func shouldInjectFailure(config *common.Configuration, random *common.Random) bool {
	if config.FailureInjectionRate == 0 {
		return false
	}

	return random.RandomInt(1, 100) <= config.FailureInjectionRate
}

// getRandomFailure returns a random failure from configured types or all types if none specified
func getRandomFailure(config *common.Configuration, random *common.Random) openaiserverapi.Error {
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

	randomIndex := random.RandomInt(0, len(availableFailures)-1)
	randomType := availableFailures[randomIndex]

	// Customize message with current model name
	failure := predefinedFailures[randomType]
	if randomType == common.FailureTypeRateLimit && config.Model != "" {
		failure.Message = fmt.Sprintf(rateLimitMessageTemplate, config.Model)
	} else if randomType == common.FailureTypeModelNotFound && config.Model != "" {
		failure.Message = fmt.Sprintf(modelNotFoundMessageTemplate, config.Model)
	}

	return failure
}

func stringPtr(s string) *string {
	return &s
}
