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

package openaiserverapi

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

// ToolChoice is a wrapper around ChatCompletionToolChoiceOptionUnionParam that
// provides custom JSON unmarshalling logic to correctly handle
// the union type.
type ToolChoice struct {
	openai.ChatCompletionToolChoiceOptionUnionParam
}

// MarshalJSON forwards the marshalling process to the embedded
// ChatCompletionToolChoiceOptionUnionParam's MarshalJSON method,
// which is known to work correctly.
func (t ToolChoice) MarshalJSON() ([]byte, error) {
	return t.ChatCompletionToolChoiceOptionUnionParam.MarshalJSON()
}

// UnmarshalJSON provides custom logic to correctly deserialize the JSON data
// into the appropriate field of the embedded union type. It inspects the JSON
// structure to determine if it's a simple string or a complex object with a
// 'type' discriminator field.
func (t *ToolChoice) UnmarshalJSON(data []byte) error {
	// If the input is a simple string (e.g., "auto", "none", "required"),
	// unmarshal it into the OfAuto field.
	if data[0] == '"' {
		var strValue string
		if err := json.Unmarshal(data, &strValue); err != nil {
			return err
		}
		t.OfAuto = param.NewOpt(strValue)
		return nil
	}

	// If the input is a JSON object, we need to determine its type.
	// We use a temporary struct to detect the 'type' field.
	var typeDetector struct {
		Type string `json:"type"`
	}

	// We only care about the type field, ignore other fields
	if err := json.Unmarshal(data, &typeDetector); err != nil {
		return fmt.Errorf("failed to detect type for ToolChoice: %w", err)
	}

	// Based on the detected type, unmarshal the data into the correct struct.
	switch typeDetector.Type {
	case "function":
		var functionChoice openai.ChatCompletionNamedToolChoiceParam
		if err := functionChoice.UnmarshalJSON(data); err != nil {
			return err
		}
		t.OfFunctionToolChoice = &functionChoice
	case "custom":
		var customChoice openai.ChatCompletionNamedToolChoiceCustomParam
		if err := customChoice.UnmarshalJSON(data); err != nil {
			return err
		}
		t.OfCustomToolChoice = &customChoice
	case "allowed_tools":
		var allowedToolsChoice openai.ChatCompletionAllowedToolChoiceParam
		if err := allowedToolsChoice.UnmarshalJSON(data); err != nil {
			return err
		}
		t.OfAllowedTools = &allowedToolsChoice
	default:
		return fmt.Errorf("unknown ToolChoice type: %s", typeDetector.Type)
	}

	return nil
}
