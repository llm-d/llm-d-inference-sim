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
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/santhosh-tekuri/jsonschema/v5"
)

func countTokensForToolCalls(toolCalls []toolCall) int {
	numberOfTokens := 0
	for _, tc := range toolCalls {
		numberOfTokens += len(strings.Fields(tc.ID))
		numberOfTokens += len(strings.Fields(tc.Type))
		numberOfTokens += len(strings.Fields(*tc.Function.Name))
		numberOfTokens += len(strings.Fields(tc.Function.Arguments))
	}

	return numberOfTokens
}

var fakeStringArguments = []string{
	`testing`,
	`hello`,
	`Boston`,
	`sunny`,
	`temperature`,
	`cloudy`,
	`question`,
	`Yorick`,
	`silence`,
	`lifetime`,
}

func getStringArgument() string {
	index := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(fakeStringArguments))
	return fakeStringArguments[index]
}

func generateToolArguments(tool tool) (map[string]any, error) {
	arguments := make(map[string]any)
	properties, _ := tool.Function.Parameters["properties"].(map[string]any)

	requiredParams, ok := tool.Function.Parameters["required"]
	if ok {
		requiredArray, _ := requiredParams.([]any)
		for _, requiredParam := range requiredArray {
			param, _ := requiredParam.(string)
			property, _ := properties[param].(map[string]any)
			arg, err := createArgument(property)
			if err != nil {
				return nil, err
			}
			arguments[param] = arg
		}
	} else {
		// No required parameters
		numberOfParameters := randomInt(len(properties), false)
		if numberOfParameters > 0 {
			for param, property := range properties {
				arg, err := createArgument(property)
				if err != nil {
					return nil, err
				}
				arguments[param] = arg
				if len(arguments) == numberOfParameters {
					break
				}
			}
		}
	}
	return arguments, nil
}

func createArgument(property any) (any, error) {
	propertyMap, _ := property.(map[string]any)
	paramType := propertyMap["type"]

	// If there is an enum, choose from it
	enum, ok := propertyMap["enum"]
	if ok {
		enumArray, ok := enum.([]any)
		if !ok {
			// Support strings for tests
			enumString, ok := enum.(string)
			if ok {
				if err := json.Unmarshal([]byte(enumString), &enumArray); err != nil {
					return nil, err
				}
			}
		}
		if len(enumArray) > 0 {
			index := randomInt(len(enumArray)-1, false)
			return enumArray[index], nil
		}
	}

	switch paramType {
	case "string":
		return getStringArgument(), nil
	case "number":
		return randomInt(100, false), nil
	case "boolean":
		return randomInt(1, false) != 0, nil
	default:
		return nil, fmt.Errorf("tool parameters of type %s are currently not supported", paramType)
	}
}

func validateTool(tool []byte) error {
	sch, err := jsonschema.CompileString("schema.json", schema)
	if err != nil {
		return err
	}

	var v interface{}
	if err := json.Unmarshal(tool, &v); err != nil {
		return err
	}

	return sch.Validate(v)
}

const schema = `{
    "$schema": "https://json-schema.org/draft/2020-12/schema",
	"name": "function-metaschema",
	"schema": {
	  "type": "object",
	  "properties": {
		"name": {
		  "type": "string",
		  "description": "The name of the function"
		},
		"description": {
		  "type": "string",
		  "description": "A description of what the function does"
		},
		"parameters": {
		  "$ref": "#/$defs/schema_definition",
		  "description": "A JSON schema that defines the function's parameters"
		}
	  },
	  "required": [
		"name",
		"description",
		"parameters"
	  ],
	  "additionalProperties": false,
	  "$defs": {
		"schema_definition": {
		  "type": "object",
		  "properties": {
			"type": {
			  "type": "string",
			  "enum": [
				"object",
				"array",
				"string",
				"number",
				"boolean",
				"null"
			  ]
			},
			"properties": {
			  "type": "object",
			  "additionalProperties": {
				"$ref": "#/$defs/schema_definition"
			  }
			},
			"items": {
			  "anyOf": [
				{
				  "$ref": "#/$defs/schema_definition"
				},
				{
				  "type": "array",
				  "items": {
					"$ref": "#/$defs/schema_definition"
				  }
				}
			  ]
			},
			"required": {
			  "type": "array",
			  "items": {
				"type": "string"
			  }
			},
			"additionalProperties": {
			  "type": "boolean"
			}
		  },
		  "required": [
			"type"
		  ],
		  "additionalProperties": false,
		  "if": {
			"properties": {
			  "type": {
				"const": "object"
			  }
			}
		  },
		  "then": {
			"required": [
			  "properties"
			]
		  }
		}
	  }
	}
  }`
