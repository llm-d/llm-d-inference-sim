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
	"time"

	"github.com/santhosh-tekuri/jsonschema/v5"
)

func countTokensForToolCalls(toolCalls []toolCall) int {
	numberOfTokens := 0
	for _, tc := range toolCalls {
		// 3 - name, id, and type
		numberOfTokens += 3 + len(tc.Function.tokenizedArguments)
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

// createToolCalls creates and returns response payload based on this request
// (tool calls or nothing in case we randomly choose not to generate calls),
// and the number of generated completion token sand the finish reason
func createToolCalls(tools []tool, toolChoice string) ([]toolCall, string, int, error) {
	// This function is called if tool choice is either 'required' or 'auto'.
	// In case of 'required' at least one tool call has to be created, and we randomly choose
	// the number of calls starting from one. Otherwise, we start from 0, and in case we randomly
	// choose the number of calls to be 0, response text will be generated instead of a tool call.
	numberOfCalls := randomInt(len(tools), toolChoice == toolChoiceRequired)
	if numberOfCalls == 0 {
		return nil, "", 0, nil
	}

	calls := make([]toolCall, 0)
	for i := range numberOfCalls {
		// Randomly choose which tools to call. We may call the same tool more than once.
		index := randomInt(len(tools)-1, false)
		args, err := generateToolArguments(tools[index])
		if err != nil {
			return nil, "", 0, err
		}
		argsJson, err := json.Marshal(args)
		if err != nil {
			return nil, "", 0, err
		}

		call := toolCall{
			Function: functionCall{
				Arguments:          string(argsJson),
				tokenizedArguments: tokenize(string(argsJson)),
				Name:               &tools[index].Function.Name,
			},
			ID:    "chatcmpl-tool-" + randomNumericString(10),
			Type:  "function",
			Index: i,
		}
		calls = append(calls, call)
	}

	return calls, toolsFinishReason, countTokensForToolCalls(calls), nil
}

func getStringArgument() string {
	index := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(fakeStringArguments))
	return fakeStringArguments[index]
}

func generateToolArguments(tool tool) (map[string]any, error) {
	arguments := make(map[string]any)
	properties, _ := tool.Function.Parameters["properties"].(map[string]any)

	required := make(map[string]struct{})
	requiredParams, ok := tool.Function.Parameters["required"]
	if ok {
		requiredArray, _ := requiredParams.([]any)
		for _, requiredParam := range requiredArray {
			param, _ := requiredParam.(string)
			required[param] = struct{}{}
		}
	}

	for param, property := range properties {
		_, paramIsRequired := required[param]
		if !paramIsRequired && !flipCoin() {
			continue
		}
		arg, err := createArgument(property)
		if err != nil {
			return nil, err
		}
		arguments[param] = arg
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
		if ok && len(enumArray) > 0 {
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
		return flipCoin(), nil
	default:
		return nil, fmt.Errorf("tool parameters of type %s are currently not supported", paramType)
	}
}

type validator struct {
	schema *jsonschema.Schema
}

func createValidator() (*validator, error) {
	sch, err := jsonschema.CompileString("schema.json", schema)
	if err != nil {
		return nil, err
	}
	return &validator{schema: sch}, nil
}

func (v *validator) validateTool(tool []byte) error {
	var value interface{}
	if err := json.Unmarshal(tool, &value); err != nil {
		return err
	}

	return v.schema.Validate(value)
}

const schema = `{
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
      "$ref": "#/$defs/param_definition",
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
    "param_definition": {
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
            "$ref": "#/$defs/property_definition"
          }
        },
        "items": {
          "anyOf": [
            {
              "$ref": "#/$defs/property_definition"
            },
            {
              "type": "array",
              "items": {
                "$ref": "#/$defs/property_definition"
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
    },
    "property_definition": {
      "type": "object",
      "properties": {
        "type": {
          "type": "string",
          "enum": [
            "string",
            "number",
            "boolean",
            "null"
          ]
        },
        "description": {
          "type": "string"
        },
        "enum": {
          "type": "array",
          "items": {
            "type": [
              "string",
              "number",
              "boolean",
              "null"
            ]
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
      "allOf": [
        {
          "if": {
            "properties": {
              "type": {
                "const": "string"
              }
            }
          },
          "then": {
            "properties": {
              "enum": {
                "type": "array",
                "items": {
                  "type": "string"
                }
              }
            }
          }
        },
        {
          "if": {
            "properties": {
              "type": {
                "const": "number"
              }
            }
          },
          "then": {
            "properties": {
              "enum": {
                "type": "array",
                "items": {
                  "type": "number"
                }
              }
            }
          }
        },
        {
          "if": {
            "properties": {
              "type": {
                "const": "boolean"
              }
            }
          },
          "then": {
            "properties": {
              "enum": {
                "type": "array",
                "items": {
                  "type": "boolean"
                }
              }
            }
          }
        },
        {
          "if": {
            "properties": {
              "type": {
                "const": "null"
              }
            }
          },
          "then": {
            "not": {
              "required": [
                "enum"
              ]
            }
          }
        }
      ]
    }
  }
}
}`
