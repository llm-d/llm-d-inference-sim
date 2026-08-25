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

package simulator

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/santhosh-tekuri/jsonschema/v5"
)

const (
	toolChoiceNone     = "none"
	toolChoiceAuto     = "auto"
	toolChoiceRequired = "required"
)

// paramTypeNull is the JSON Schema type keyword for a property whose only
// permitted value is null.
const paramTypeNull = "null"

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

func countTokensForToolCalls(toolCalls []api.ToolCall) int {
	numberOfTokens := 0
	for _, tc := range toolCalls {
		// 3 - name, id, and type
		numberOfTokens += 3 + tc.Function.TokenizedArguments().Length()
	}
	return numberOfTokens
}

// isToolChoiceNone checks if the tool_choice is set to "none".
func isToolChoiceNone(toolChoice api.ToolChoice) bool {
	if !param.IsOmitted(toolChoice.OfAuto) {
		val := toolChoice.OfAuto.Or("")
		return val == toolChoiceNone
	}
	return false
}

type toolsValidator struct {
	schema *jsonschema.Schema
	// skip reports whether tool schema validation is currently disabled. It is read on
	// every call so that a configuration swapped in at runtime takes effect immediately.
	// A nil skip means validation is always performed.
	skip func() bool
}

func createToolsValidator() (*toolsValidator, error) {
	sch, err := jsonschema.CompileString("schema.json", schema)
	if err != nil {
		return nil, err
	}
	return &toolsValidator{schema: sch}, nil
}

func (v *toolsValidator) validateTool(tool []byte) error {
	if v.skip != nil && v.skip() {
		return nil
	}

	var value interface{}
	if err := json.Unmarshal(tool, &value); err != nil {
		return err
	}

	return v.schema.Validate(value)
}

// createToolCalls creates and returns tool calls based on the request's tool
// definitions and the tool_choice parameter.
//
// The [tool_choice](https://platform.openai.com/docs/guides/function-calling#tool-choice)
// parameter controls how the model responds to function calls.
//
// This function handles the following cases for tool_choice:
//   - "none": The model will not call any tools. In this scenario, this function
//     should ideally be bypassed, as no tool calls will be generated.
//   - "auto": This is the default behavior where the model autonomously decides
//     whether to generate a message or call one or more tools from the provided list.
//   - "required": The model is constrained to call one or more of the available tools.
//   - Forced Function: A specific tool can be forced by providing an object with the
//     structure `{"type": "function", "function": {"name": "my_function"}}`.
//     The model will be restricted to calling only that specified tool.
//
// This function currently does not handle the following `tool_choice` scenarios:
//   - Forced Custom Tool: If `tool_choice` is set to `{"type": "custom", "name": "my_custom"}`,
//     this function will not be able to enforce the calling of a custom tool, as custom
//     tool types are not yet supported.
//   - Allowed Tools Subset: The functionality to restrict the model's tool-calling
//     capabilities to a specific subset of the available tools has not been implemented.
//
// This function returns the generated tool calls, the number of completion
// tokens used, and an error if one occurs (e.g., if a specified tool is not found).
func createToolCalls(
	tools []api.Tool,
	toolChoice api.ToolChoice,
	config *common.Configuration,
	random *common.Random,
	tokenizer tokenizer.Tokenizer,
	idPrefix string,
) ([]api.ToolCall, int, error) {
	generateCalls := func(availableTools []api.Tool, minCalls int) ([]api.ToolCall, int, error) {
		if len(availableTools) == 0 {
			// If no tools are available to choose from, no calls can be made.
			return nil, 0, errors.New("no tools available to create tool calls")
		}

		numberOfCalls := minCalls
		maxCalls := len(availableTools)
		// Truncated geometric: start at minCalls and add one extra call with
		// probability ToolCallExtraCallProbability, repeating until a roll
		// fails or we reach maxCalls. The minimum is always possible and the
		// maximum is always reachable, but higher counts are increasingly rare.
		for numberOfCalls < maxCalls && random.RandomBool(config.ToolCallExtraCallProbability) {
			numberOfCalls++
		}

		if numberOfCalls == 0 {
			return nil, 0, nil
		}

		calls := make([]api.ToolCall, 0, numberOfCalls)
		for i := range numberOfCalls {
			// Randomly choose which tool to call. We may call the same tool more than once.
			index := 0
			if len(availableTools) > 1 {
				index = random.RandomInt(0, len(availableTools)-1)
			}
			chosenTool := availableTools[index]

			args, err := generateToolArguments(chosenTool, config, random)
			if err != nil {
				return nil, 0, err
			}
			argsJson, err := json.Marshal(args)
			if err != nil {
				return nil, 0, err
			}

			tokens, strs, err := tokenizer.RenderText(string(argsJson))
			if err != nil {
				return nil, 0, err
			}
			tokenizedArgs := &api.Tokenized{Tokens: tokens, Strings: strs}

			call := api.ToolCall{
				Function: api.FunctionCall{
					Arguments: string(argsJson),
					Name:      &chosenTool.Function.Name,
				},
				ID:    idPrefix + random.RandomNumericString(10),
				Type:  "function",
				Index: i,
			}
			call.Function.SetTokenizedArguments(tokenizedArgs)
			calls = append(calls, call)
		}
		return calls, countTokensForToolCalls(calls), nil
	}

	// A specific function is forced.
	if functionChoice := toolChoice.GetFunction(); functionChoice != nil {
		requiredFuncName := functionChoice.Name
		var targetTool *api.Tool

		// Find the specified tool in the list of available tools.
		for i, tool := range tools {
			if tool.Function.Name == requiredFuncName {
				targetTool = &tools[i]
				break
			}
		}

		if targetTool == nil {
			return nil, 0, fmt.Errorf("tool with name '%s' requested in tool_choice but not found in the tools list", requiredFuncName)
		}

		specificTools := []api.Tool{*targetTool}

		// Generate arguments for the specific tool.
		return generateCalls(specificTools, len(specificTools))
	}

	// Default behavior for "auto" or "required".
	// The model can choose from any of the provided tools.
	min := 0
	if !param.IsOmitted(toolChoice.OfAuto) && toolChoice.OfAuto.Or("") == toolChoiceRequired {
		min = 1
	}

	return generateCalls(tools, min)
}

// createSingleToolCall generates at most one tool call. Used by the Responses API
// path where Praxis requires exactly one function_call per round.
func createSingleToolCall(
	tools []api.Tool,
	toolChoice api.ToolChoice,
	config *common.Configuration,
	random *common.Random,
	tokenizer tokenizer.Tokenizer,
	idPrefix string,
) ([]api.ToolCall, int, error) {
	calls, tokens, err := createToolCalls(tools, toolChoice, config, random, tokenizer, idPrefix)
	if err != nil || len(calls) <= 1 {
		return calls, tokens, err
	}
	calls = calls[:1]
	return calls, countTokensForToolCalls(calls), nil
}

func generateToolArguments(tool api.Tool, config *common.Configuration, random *common.Random) (map[string]any, error) {
	arguments := make(map[string]any)
	properties, _ := tool.Function.Parameters["properties"].(map[string]any)

	required := getRequiredAsMap(tool.Function.Parameters)

	for param, property := range properties {
		_, paramIsRequired := required[param]
		if !paramIsRequired && !random.RandomBool(config.ToolCallNotRequiredParamProbability) {
			continue
		}
		arg, err := createArgument(property, config, random)
		if err != nil {
			return nil, err
		}
		arguments[param] = arg
	}

	return arguments, nil
}

func getRequiredAsMap(property map[string]any) map[string]struct{} {
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

// resolveParamType normalizes the JSON Schema type keyword, which may be a single
// string, a union of strings, or absent, into the one type the generator handles.
// A union resolves to its first non-null member, matching the value a model would
// most likely produce for an optional field.
func resolveParamType(paramType any) any {
	switch typeValue := paramType.(type) {
	case nil:
		return "string"
	case []any:
		for _, member := range typeValue {
			if name, ok := member.(string); ok && name != paramTypeNull {
				return name
			}
		}
		return paramTypeNull
	default:
		return paramType
	}
}

func createArgument(property any, config *common.Configuration, random *common.Random) (any, error) {
	propertyMap, _ := property.(map[string]any)
	paramType := resolveParamType(propertyMap["type"])

	// If there is an enum, choose from it
	enum, ok := propertyMap["enum"]
	if ok {
		enumArray, ok := enum.([]any)
		if ok && len(enumArray) > 0 {
			index := random.RandomInt(0, len(enumArray)-1)
			return enumArray[index], nil
		}
	}

	switch paramType {
	case "string":
		return getStringArgument(random), nil
	case "integer":
		return random.RandomInt(config.MinToolCallIntegerParam, config.MaxToolCallIntegerParam), nil
	case "number":
		return random.RandomFloat(config.MinToolCallNumberParam, config.MaxToolCallNumberParam), nil
	case "boolean":
		return random.FlipCoin(), nil
	case paramTypeNull:
		return nil, nil
	case "array":
		// A schema may omit items; there is no element shape to generate from.
		itemsMap, ok := propertyMap["items"].(map[string]any)
		if !ok {
			return []any{}, nil
		}
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
		numberOfElements := random.RandomInt(minItems, maxItems)
		array := make([]any, numberOfElements)
		for i := range numberOfElements {
			elem, err := createArgument(itemsMap, config, random)
			if err != nil {
				return nil, err
			}
			array[i] = elem
		}
		return array, nil
	case "object":
		required := getRequiredAsMap(propertyMap)
		// A schema may omit properties; there are no fields to generate.
		objectProperties, ok := propertyMap["properties"].(map[string]any)
		if !ok {
			return map[string]any{}, nil
		}
		object := make(map[string]interface{})
		for fieldName, fieldProperties := range objectProperties {
			_, fieldIsRequired := required[fieldName]
			if !fieldIsRequired && !random.RandomBool(config.ObjectToolCallNotRequiredParamProbability) {
				continue
			}
			fieldValue, err := createArgument(fieldProperties, config, random)
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

func getStringArgument(random *common.Random) string {
	index := random.RandomInt(0, len(fakeStringArguments)-1)
	return fakeStringArguments[index]
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
        },
        "default": {
          "description": "A default value for the parameter"
        },
        "minimum": {
          "type": "number"
        },
        "maximum": {
          "type": "number"
        },
        "exclusiveMinimum": {
          "type": [
            "number",
            "boolean"
          ]
        },
        "exclusiveMaximum": {
          "type": [
            "number",
            "boolean"
          ]
        },
        "minLength": {
          "type": "integer",
          "minimum": 0
        },
        "maxLength": {
          "type": "integer",
          "minimum": 0
        },
        "pattern": {
          "type": "string"
        },
        "format": {
          "type": "string"
        },
        "title": {
          "type": "string"
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
