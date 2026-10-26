/*
Copyright 2026 The llm-d-inference-sim Authors.

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

package api

import (
	"bytes"
	"encoding/json"
)

type SystemOneRequest struct {
	Model              *string                    `json:"model"`
	State              json.RawMessage            `json:"state"`
	Questions          json.RawMessage            `json:"questions"`
	Instructions       *string                    `json:"instructions"`
	CacheSalt          *string                    `json:"cache_salt"`
	ChatTemplateKwargs map[string]json.RawMessage `json:"chat_template_kwargs"`
	Priority           json.RawMessage            `json:"priority"`
}

type SystemOneQuestion struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria"`
}

type DecisionRequest struct {
	Model            *string           `json:"model"`
	Input            json.RawMessage   `json:"input"`
	Questions        []json.RawMessage `json:"questions"`
	SafetyIdentifier *string           `json:"safety_identifier"`
	CacheSalt        *string           `json:"cache_salt"`
}

type DecisionQuestion struct {
	Type         string          `json:"type"`
	Name         json.RawMessage `json:"name"`
	Instructions *string         `json:"instructions"`
	Choices      json.RawMessage `json:"choices"`
	Levels       json.RawMessage `json:"levels"`
}

type DecisionChoiceOption struct {
	Value       json.RawMessage `json:"value"`
	Description json.RawMessage `json:"description"`
}

type DecisionScoreLevel struct {
	Label       *string         `json:"label"`
	Description json.RawMessage `json:"description"`
}

type DecisionUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type SystemOneAnswer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        *string            `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
}

type DecisionDiagnostics struct {
	LabelMass     float64 `json:"label_mass"`
	ArgmaxIsLabel bool    `json:"argmax_is_label"`
}

type SystemOneAnswerEntry struct {
	ID string `json:"-"`
	SystemOneAnswer
}

// SystemOneAnswers encodes an object in request order.
type SystemOneAnswers []SystemOneAnswerEntry

func (a SystemOneAnswers) MarshalJSON() ([]byte, error) {
	var result bytes.Buffer
	result.WriteByte('{')
	for i, entry := range a {
		if i > 0 {
			result.WriteByte(',')
		}
		key, err := json.Marshal(entry.ID)
		if err != nil {
			return nil, err
		}
		value, err := json.Marshal(entry.SystemOneAnswer)
		if err != nil {
			return nil, err
		}
		result.Write(key)
		result.WriteByte(':')
		result.Write(value)
	}
	result.WriteByte('}')
	return result.Bytes(), nil
}

type SystemOneResponse struct {
	ID          string                         `json:"id"`
	Object      string                         `json:"object"`
	Created     int64                          `json:"created"`
	Model       string                         `json:"model"`
	Answers     SystemOneAnswers               `json:"answers"`
	Usage       DecisionUsage                  `json:"usage"`
	Diagnostics map[string]DecisionDiagnostics `json:"diagnostics"`
}

type DecisionProbability struct {
	Value       any     `json:"value"`
	Label       *string `json:"label,omitempty"`
	Probability float64 `json:"probability"`
}

type DecisionAnswer struct {
	Type          string                `json:"type"`
	Name          *string               `json:"name"`
	Probability   *float64              `json:"probability,omitempty"`
	Choice        any                   `json:"choice,omitempty"`
	Score         *float64              `json:"score,omitempty"`
	Probabilities []DecisionProbability `json:"probabilities,omitempty"`
	Confidence    *float64              `json:"confidence,omitempty"`
}

type DecisionInputTokensDetails struct {
	CachedTokens     int `json:"cached_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
}

type DecisionOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type OpenAIDecisionUsage struct {
	DecisionUsage
	InputTokensDetails  DecisionInputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails DecisionOutputTokensDetails `json:"output_tokens_details"`
	TotalTokens         int                         `json:"total_tokens"`
}

type DecisionsResponse struct {
	Model   string              `json:"model"`
	Answers []DecisionAnswer    `json:"answers"`
	Usage   OpenAIDecisionUsage `json:"usage"`
}
