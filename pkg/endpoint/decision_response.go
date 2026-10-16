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

package endpoint

import (
	"slices"
	"strconv"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

func (d *DecisionRequest) SystemOneAnswers() (api.SystemOneAnswers, map[string]api.DecisionDiagnostics) {
	answers := make(api.SystemOneAnswers, 0, len(d.questions))
	diagnostics := make(map[string]api.DecisionDiagnostics, len(d.questions))
	for _, q := range d.questions {
		probabilities := decisionProbabilities(len(q.values))
		answers = append(answers, api.SystemOneAnswerEntry{ID: q.id, SystemOneAnswer: systemOneAnswer(q, probabilities, 1)})
		diagnostics[q.id] = api.DecisionDiagnostics{LabelMass: 1, ArgmaxIsLabel: true}
	}
	return answers, diagnostics
}

func (d *DecisionRequest) DecisionAnswers() []api.DecisionAnswer {
	answers := make([]api.DecisionAnswer, 0, len(d.questions))
	for _, q := range d.questions {
		probabilities := decisionProbabilities(len(q.values))
		answers = append(answers, decisionAnswer(q, probabilities, 1))
	}
	return answers
}

func systemOneAnswer(q decisionQuestion, probabilities []float64, labelMass float64) api.SystemOneAnswer {
	top := slices.Index(probabilities, slices.Max(probabilities))
	confidence := probabilities[top] * labelMass
	answer := api.SystemOneAnswer{
		Type: q.kind, Confidence: &confidence,
		Probabilities: make(map[string]float64, len(q.values)),
	}
	score := 0.0
	for i, value := range q.values {
		answer.Probabilities[value.(string)] = probabilities[i]
		score += float64(i) * probabilities[i]
	}
	switch q.kind {
	case "choice":
		value := q.values[top].(string)
		answer.Choice = &value
	case "score":
		answer.Score = &score
		if len(q.legend) > 0 {
			answer.Legend = make(map[string]string, len(q.legend))
			for i, level := range q.legend {
				answer.Legend[strconv.Itoa(i)] = decisionText(level)
			}
		}
	case "noul":
		answer.Noul = &probabilities[0]
	}
	return answer
}

func decisionAnswer(q decisionQuestion, probabilities []float64, labelMass float64) api.DecisionAnswer {
	answer := api.DecisionAnswer{Type: q.kind, Name: q.name}
	if q.kind == "predicate" {
		answer.Probability = &probabilities[1]
	} else {
		top := slices.Index(probabilities, slices.Max(probabilities))
		confidence := probabilities[top] * labelMass
		answer.Confidence = &confidence
		answer.Probabilities = make([]api.DecisionProbability, len(q.values))
		score := 0.0
		for i, value := range q.values {
			p := api.DecisionProbability{Value: value, Probability: probabilities[i]}
			if q.kind == "score" {
				label := decisionText(q.legend[i])
				p.Label = &label
				score += float64(i) * probabilities[i]
			}
			answer.Probabilities[i] = p
		}
		if q.kind == "score" {
			answer.Score = &score
		} else {
			answer.Choice = q.values[top]
		}
	}
	return answer
}

// Synthetic descending weights preserve caller-supplied option order; they are
// not model probabilities.
func decisionProbabilities(n int) []float64 {
	probabilities := make([]float64, n)
	total := 0.0
	for i := range probabilities {
		probabilities[i] = float64(n - i)
		total += probabilities[i]
	}
	for i := range probabilities {
		probabilities[i] /= total
	}
	return probabilities
}
