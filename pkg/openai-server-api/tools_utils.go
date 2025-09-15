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

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/santhosh-tekuri/jsonschema/v5"
)

const (
	ToolChoiceNone     = "none"
	ToolChoiceAuto     = "auto"
	ToolChoiceRequired = "required"
)

func CountTokensForToolCalls(toolCalls []ToolCall) int {
	numberOfTokens := 0
	for _, tc := range toolCalls {
		// 3 - name, id, and type
		numberOfTokens += 3 + len(tc.Function.TokenizedArguments)
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

// CreateToolCalls creates and returns response payload based on this request
// (tool calls or nothing in case we randomly choose not to generate calls),
// and the number of generated completion token sand the finish reason
func CreateToolCalls(tools []Tool, toolChoice string, config *common.Configuration) ([]ToolCall, int, error) {
	// This function is called if tool choice is either 'required' or 'auto'.
	// In case of 'required' at least one tool call has to be created, and we randomly choose
	// the number of calls starting from one. Otherwise, we start from 0, and in case we randomly
	// choose the number of calls to be 0, response text will be generated instead of a tool call.
	min := 0
	if toolChoice == ToolChoiceRequired {
		min = 1
	}
	numberOfCalls := common.RandomInt(min, len(tools))
	if numberOfCalls == 0 {
		return nil, 0, nil
	}

	calls := make([]ToolCall, 0)
	for i := range numberOfCalls {
		// Randomly choose which tools to call. We may call the same tool more than once.
		index := common.RandomInt(0, len(tools)-1)
		args, err := GenerateToolArguments(tools[index], config)
		if err != nil {
			return nil, 0, err
		}
		argsJson, err := json.Marshal(args)
		if err != nil {
			return nil, 0, err
		}

		call := ToolCall{
			Function: FunctionCall{
				Arguments:          string(argsJson),
				TokenizedArguments: common.Tokenize(string(argsJson)),
				Name:               &tools[index].Function.Name,
			},
			ID:    "chatcmpl-tool-" + common.RandomNumericString(10),
			Type:  "function",
			Index: i,
		}
		calls = append(calls, call)
	}

	return calls, CountTokensForToolCalls(calls), nil
}

func GetRequiredAsMap(property map[string]any) map[string]struct{} {
	required := make(map[string]struct{})
	requiredParams, ok := property["required"]
	if ok {
		requiredArray, _ := requiredParams.([]any)
		for _, requiredParam := range requiredArray {
			param, _ := requiredParam.(string)
			required[param] = struct{}{}
		}
	}
	return required
}

func GenerateToolArguments(tool Tool, config *common.Configuration) (map[string]any, error) {
	arguments := make(map[string]any)
	properties, _ := tool.Function.Parameters["properties"].(map[string]any)

	required := GetRequiredAsMap(tool.Function.Parameters)

	for param, property := range properties {
		_, paramIsRequired := required[param]
		if !paramIsRequired && !common.RandomBool(config.ToolCallNotRequiredParamProbability) {
			continue
		}
		arg, err := CreateArgument(property, config)
		if err != nil {
			return nil, err
		}
		arguments[param] = arg
	}

	return arguments, nil
}

func CreateArgument(property any, config *common.Configuration) (any, error) {
	propertyMap, _ := property.(map[string]any)
	paramType := propertyMap["type"]

	// If there is an enum, choose from it
	enum, ok := propertyMap["enum"]
	if ok {
		enumArray, ok := enum.([]any)
		if ok && len(enumArray) > 0 {
			index := common.RandomInt(0, len(enumArray)-1)
			return enumArray[index], nil
		}
	}

	switch paramType {
	case "string":
		return GetStringArgument(), nil
	case "integer":
		return common.RandomInt(config.MinToolCallIntegerParam, config.MaxToolCallIntegerParam), nil
	case "number":
		return common.RandomFloat(config.MinToolCallNumberParam, config.MaxToolCallNumberParam), nil
	case "boolean":
		return common.FlipCoin(), nil
	case "array":
		items := propertyMap["items"]
		itemsMap := items.(map[string]any)
		minItems := config.MinToolCallArrayParamLength
		maxItems := config.MaxToolCallArrayParamLength
		if value, ok := propertyMap["minItems"]; ok {
			minItems = int(value.(float64))
		}
		if value, ok := propertyMap["maxItems"]; ok {
			maxItems = int(value.(float64))
		}
		if minItems > maxItems {
			return nil, fmt.Errorf("minItems (%d) is greater than maxItems(%d)", minItems, maxItems)
		}
		numberOfElements := common.RandomInt(minItems, maxItems)
		array := make([]any, numberOfElements)
		for i := range numberOfElements {
			elem, err := CreateArgument(itemsMap, config)
			if err != nil {
				return nil, err
			}
			array[i] = elem
		}
		return array, nil
	case "object":
		required := GetRequiredAsMap(propertyMap)
		objectProperties := propertyMap["properties"].(map[string]any)
		object := make(map[string]interface{})
		for fieldName, fieldProperties := range objectProperties {
			_, fieldIsRequired := required[fieldName]
			if !fieldIsRequired && !common.RandomBool(config.ObjectToolCallNotRequiredParamProbability) {
				continue
			}
			fieldValue, err := CreateArgument(fieldProperties, config)
			if err != nil {
				return nil, err
			}
			object[fieldName] = fieldValue
		}
		return object, nil
	default:
		return nil, fmt.Errorf("tool parameters of type %s are not supported", paramType)
	}
}

func GetStringArgument() string {
	index := common.RandomInt(0, len(fakeStringArguments)-1)
	return fakeStringArguments[index]
}

type Validator struct {
	schema *jsonschema.Schema
}

func CreateValidator() (*Validator, error) {
	sch, err := jsonschema.CompileString("schema.json", schema)
	if err != nil {
		return nil, err
	}
	return &Validator{schema: sch}, nil
}

func (v *Validator) ValidateTool(tool []byte) error {
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
            "integer",
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
              "integer",
              "boolean"
            ]
          }
        },
        "properties": {
          "type": "object",
          "additionalProperties": {
            "$ref": "#/$defs/param_definition"
          }
        },
        "items": {
          "anyOf": [
            {
              "$ref": "#/$defs/param_definition"
            },
            {
              "type": "array",
              "items": {
                "$ref": "#/$defs/param_definition"
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
        },
        "minItems": {
          "type": "integer",
          "minimum": 0
        },
        "maxItems": {
          "type": "integer",
          "minimum": 0
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
                "const": "integer"
              }
            }
          },
          "then": {
            "properties": {
              "enum": {
                "type": "array",
                "items": {
                  "type": "integer"
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
            "anyOf": [
              {
                "properties": {
                  "type": {
                    "const": "null"
                  }
                }
              },
              {
                "properties": {
                  "type": {
                    "const": "object"
                  }
                }
              },
              {
                "properties": {
                  "type": {
                    "const": "array"
                  }
                }
              }
            ]
          },
          "then": {
            "not": {
              "required": [
                "enum"
              ]
            }
          }
        },
        {
          "if": {
            "properties": {
              "type": {
                "const": "array"
              }
            }
          },
          "then": {
            "required": [
              "items"
            ]
          }
        },
        {
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
      ]
    }
  }
}`
