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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/google/uuid"
	"github.com/valyala/fasthttp"
	kjson "sigs.k8s.io/json"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

type DecisionProtocol int

const (
	SystemOneProtocol DecisionProtocol = iota
	DecisionsProtocol
)

type decisionQuestion struct {
	id           string
	name         *string
	kind         string
	values       []any
	legend       []json.RawMessage
	prompt       string
	instructions *string
	descriptions []string
}

// DecisionRequest splits questions into independent worker requests, as batched
// text completions split prompts. Scores do not depend on the response dataset.
type DecisionRequest struct {
	TextCompletionsRequest
	Protocol        DecisionProtocol
	DefaultModel    string
	ClientRequestID *string
	questions       []decisionQuestion
	input           string
	// Retain wire fields whose presence or value matters after normalization.
	systemOne           api.SystemOneRequest
	decisions           api.DecisionsRequest
	unsupportedControls bool
}

func (d *DecisionRequest) Unmarshal(data []byte) error {
	var controls struct {
		Stream               bool              `json:"stream"`
		Images               []json.RawMessage `json:"images"`
		Audio                []json.RawMessage `json:"audio"`
		Videos               []json.RawMessage `json:"videos"`
		ReturnPromptTokenIDs bool              `json:"return_prompt_token_ids"`
	}
	if err := kjson.UnmarshalCaseSensitivePreserveInts(data, &controls); err != nil {
		return err
	}
	d.unsupportedControls = controls.Stream || len(controls.Images)+len(controls.Audio)+len(controls.Videos) != 0 || controls.ReturnPromptTokenIDs
	d.questions = nil
	d.ClientRequestID = nil
	switch d.Protocol {
	case SystemOneProtocol:
		return d.parseSystemOne(data)
	case DecisionsProtocol:
		return d.parseDecisions(data)
	default:
		return errors.New("unknown decision protocol")
	}
}

func (d *DecisionRequest) Validate(_ *ToolsValidator) *api.Error {
	var err error
	if d.unsupportedControls {
		err = errors.New("decision simulation does not support streaming, media inputs, or return_prompt_token_ids")
	} else {
		switch d.Protocol {
		case SystemOneProtocol:
			err = d.validateSystemOne()
		case DecisionsProtocol:
			err = d.validateDecisions()
		default:
			err = errors.New("unknown decision protocol")
		}
	}
	if err != nil {
		apiErr := api.NewError(err.Error(), fasthttp.StatusBadRequest, nil)
		return &apiErr
	}
	return nil
}

func (d *DecisionRequest) Split() []Request {
	out := make([]Request, len(d.questions))
	requestID := uuid.NewString()
	for i := range d.questions {
		cp := *d
		cp.SetRequestID(fmt.Sprintf("%s-%d", requestID, i))
		cp.questions = d.questions[i : i+1]
		out[i] = &cp
	}
	return out
}

func (d *DecisionRequest) AsString() string {
	return "decision request (req id " + d.GetRequestID() + ")"
}

func (d *DecisionRequest) BuildRequestContext(runtime Runtime, channel common.Channel[*ResponseInfo],
	choiceIdx int, doneFn func()) RequestContext {
	rc := &decisionReqCtx{
		baseRequestContext: newBaseRequestContext(runtime, channel, choiceIdx, doneFn),
		req:                d,
	}
	rc.RequestContext = rc
	return rc
}

var _ Request = (*DecisionRequest)(nil)

type decisionReqCtx struct {
	baseRequestContext
	req *DecisionRequest
}

func (r *decisionReqCtx) Request() Request { return r.req }

func (r *decisionReqCtx) encode() ([]uint32, []string, *api.RenderMMFeatures, error) {
	tokens, _, err := r.runtime.GetTokenizer().RenderText(r.req.input + "\n" + r.req.questions[0].prompt)
	// Decision responses need token IDs for usage and cache lookups, not prompt text.
	return tokens, nil, nil, err
}

func (r *decisionReqCtx) tokenizedPromptForEcho() (*api.Tokenized, error) {
	return r.req.TokenizedPrompt(), nil
}

func (*decisionReqCtx) createToolCalls() ([]api.ToolCall, int, string, error) {
	return nil, 0, "", nil
}

func (r *decisionReqCtx) HandleRequest() (ResponseContext, *api.Error) {
	r.runtime.RequestStarted(r.req)
	if err := r.tokenize(); err != nil {
		return nil, err
	}
	inputTokens := r.req.TokenizedPrompt().Length()
	output := &api.Tokenized{Tokens: []uint32{0}}
	if inputTokens+output.Length() > r.runtime.Config().MaxModelLen {
		apiErr := api.NewError("decision prompt exceeds the model context length", fasthttp.StatusBadRequest, nil)
		return nil, &apiErr
	}
	stats, apiErr := r.kvCacheOnRequestStart()
	if apiErr != nil {
		return nil, apiErr
	}
	usage := &api.Usage{
		PromptTokens: inputTokens, CompletionTokens: output.Length(), TotalTokens: inputTokens + output.Length(),
		PromptTokensDetails: &api.PromptTokensDetails{CachedTokens: stats.CachedPromptTokens},
	}
	finish := common.StopFinishReason
	return r.req.createResponseContext(r, r.req.GetDisplayedModel(), output, &finish, usage, true, nil, nil, false), nil
}

func (d *DecisionRequest) setModel(model *string) {
	if model == nil || *model == "" {
		d.Model = d.DefaultModel
	} else {
		d.Model = *model
	}
}

func (d *DecisionRequest) parseSystemOne(data []byte) error {
	var wire api.SystemOneRequest
	if err := kjson.UnmarshalCaseSensitivePreserveInts(data, &wire); err != nil {
		return err
	}
	if wire.RequestID != nil {
		if err := json.Unmarshal(wire.RequestID, &d.ClientRequestID); err != nil || d.ClientRequestID == nil {
			return errors.New("request_id must be a string")
		}
	}
	d.systemOne = wire
	d.setModel(wire.Model)
	d.CacheSalt = wire.CacheSalt
	d.input = decisionText(wire.State)
	if wire.Instructions != nil {
		d.input = *wire.Instructions + "\n" + d.input
	}
	if present(wire.Questions) == nil {
		return nil
	}
	fields, err := parseDecisionObject(wire.Questions)
	if err != nil {
		return fmt.Errorf("questions: %w", err)
	}
	for _, field := range fields {
		var spec api.SystemOneQuestion
		if err := decodeDecisionStrict(field.value, &spec); err != nil {
			return fmt.Errorf("question %q: %w", field.key, err)
		}
		q := decisionQuestion{id: field.key, kind: spec.Type, prompt: string(field.value)}
		switch spec.Type {
		case "choice":
			var options []decisionField
			if present(spec.Criteria) != nil {
				options, err = parseDecisionObject(spec.Criteria)
				if err != nil {
					return fmt.Errorf("choice criteria: %w", err)
				}
			}
			for _, option := range options {
				q.values = append(q.values, option.key)
			}
		case "score":
			if spec.Criteria != nil {
				if err := json.Unmarshal(spec.Criteria, &q.legend); err != nil {
					return errors.New("score criteria must be an ordered list of levels")
				}
			}
			for i, level := range q.legend {
				label, err := decisionPythonString(level)
				if err != nil {
					return err
				}
				q.legend[i], err = json.Marshal(label)
				if err != nil {
					return err
				}
				q.values = append(q.values, strconv.Itoa(i))
			}
		case "noul":
			criteria := []decisionField{}
			if len(spec.Criteria) > 0 && string(spec.Criteria) != "null" {
				var err error
				criteria, err = parseDecisionObject(spec.Criteria)
				if err != nil {
					return errors.New("noul criteria must be an object with true and false")
				}
			}
			for _, field := range criteria {
				if field.key != "true" && field.key != "false" {
					return errors.New("noul criteria must be an object with true and false")
				}
			}
			q.values = []any{"yes", "no"}
		default:
			return fmt.Errorf("unknown question type %q", spec.Type)
		}
		d.questions = append(d.questions, q)
	}
	return nil
}

func (d *DecisionRequest) parseDecisions(data []byte) error {
	var wire api.DecisionsRequest
	if err := decodeDecisionStrict(data, &wire); err != nil {
		return err
	}
	d.decisions = wire
	d.setModel(wire.Model)
	d.CacheSalt = wire.CacheSalt
	var err error
	d.input, err = parseDecisionInput(wire.Input)
	if err != nil {
		return err
	}
	for _, raw := range wire.Questions {
		q, err := parseDecisionQuestion(raw)
		if err != nil {
			return err
		}
		d.questions = append(d.questions, q)
	}
	return nil
}

func parseDecisionQuestion(raw json.RawMessage) (decisionQuestion, error) {
	var spec api.DecisionQuestion
	if err := decodeDecisionStrict(raw, &spec); err != nil {
		return decisionQuestion{}, err
	}
	q := decisionQuestion{kind: spec.Type, prompt: string(raw), instructions: spec.Instructions}
	if len(spec.Name) != 0 {
		if string(spec.Name) == "null" || json.Unmarshal(spec.Name, &q.name) != nil {
			return q, errors.New("name must be a string when provided")
		}
	}
	switch spec.Type {
	case "predicate":
		if spec.Choices != nil || spec.Levels != nil {
			return q, errors.New("predicate questions do not accept choices or levels")
		}
		q.values = []any{false, true}
	case "choice":
		var choices []api.DecisionChoiceOption
		if spec.Choices != nil {
			if err := decodeDecisionStrict(spec.Choices, &choices); err != nil {
				return q, err
			}
		}
		if spec.Levels != nil {
			return q, errors.New("choice questions do not accept levels")
		}
		for _, option := range choices {
			description, err := parseDecisionDescription(option.Description)
			if err != nil {
				return q, err
			}
			q.descriptions = append(q.descriptions, description)
			var value any
			if err := json.Unmarshal(option.Value, &value); err != nil {
				return q, err
			}
			switch value.(type) {
			case string, bool:
			default:
				return q, errors.New("choice values must be strings or booleans")
			}
			q.values = append(q.values, value)
		}
	case "score":
		var levels []api.DecisionScoreLevel
		if spec.Levels != nil {
			if err := decodeDecisionStrict(spec.Levels, &levels); err != nil {
				return q, err
			}
		}
		if spec.Choices != nil {
			return q, errors.New("score questions do not accept choices")
		}
		for i, level := range levels {
			description, err := parseDecisionDescription(level.Description)
			if err != nil {
				return q, err
			}
			q.descriptions = append(q.descriptions, description)
			label, err := json.Marshal(level.Label)
			if err != nil {
				return q, err
			}
			q.legend = append(q.legend, label)
			q.values = append(q.values, i)
		}
	default:
		return q, fmt.Errorf("unknown question type %q", spec.Type)
	}
	return q, nil
}

func parseDecisionDescription(raw json.RawMessage) (string, error) {
	if raw == nil {
		return "", nil
	}
	var text *string
	if json.Unmarshal(raw, &text) != nil || text == nil {
		return "", errors.New("description must be a string")
	}
	return *text, nil
}

func parseDecisionInput(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		if utf8.RuneCountInString(text) > 10485760 {
			return "", errors.New("input text is too long")
		}
		return text, nil
	}
	var messages []struct {
		Type    json.RawMessage `json:"type"`
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := decodeDecisionStrict(raw, &messages); err != nil {
		return "", errors.New("input must be text or user messages")
	}
	if len(messages) > 131072 {
		return "", errors.New("too many input messages")
	}
	var parts []string
	for _, message := range messages {
		messageType := "message"
		if message.Type != nil && (present(message.Type) == nil || json.Unmarshal(message.Type, &messageType) != nil) {
			return "", errors.New("message type must be message")
		}
		if message.Role != api.RoleUser || messageType != "message" {
			return "", errors.New("input messages must have role user and type message")
		}
		if len(message.Content) == 0 || string(message.Content) == "null" {
			return "", errors.New("message content is required")
		}
		if json.Unmarshal(message.Content, &text) == nil {
			parts = append(parts, text)
			continue
		}
		var content []struct {
			Type string  `json:"type"`
			Text *string `json:"text"`
		}
		if err := decodeDecisionStrict(message.Content, &content); err != nil || len(content) > 16384 {
			return "", errors.New("content must be text or input_text parts")
		}
		for _, part := range content {
			if part.Type != api.ResponsesInputText || part.Text == nil {
				return "", errors.New("only input_text content is supported")
			}
			parts = append(parts, *part.Text)
		}
	}
	for _, part := range parts {
		if utf8.RuneCountInString(part) > 10485760 {
			return "", errors.New("input text is too long")
		}
	}
	return strings.Join(parts, "\n"), nil
}

func (d *DecisionRequest) validateSystemOne() error {
	if len(d.systemOne.State) == 0 {
		return errors.New("state is required")
	}
	if err := validateDecisionPriority(d.systemOne.Priority); err != nil {
		return err
	}
	if err := validateDecisionChatOptions(d.systemOne.ChatTemplateKwargs); err != nil {
		return err
	}
	if err := validateDecisionCacheSalt(d.CacheSalt); err != nil {
		return err
	}
	if len(d.questions) == 0 || len(d.questions) > 64 {
		return errors.New("questions must contain 1 to 64 entries")
	}
	for _, q := range d.questions {
		if q.id == "" {
			return errors.New("question ids must not be empty")
		}
		switch q.kind {
		case "choice":
			if len(q.values) < 1 || len(q.values) > 26 {
				return errors.New("choice criteria must have 1 to 26 options")
			}
		case "score":
			if len(q.legend) < 2 || len(q.legend) > 10 {
				return errors.New("score criteria must contain 2 to 10 levels")
			}
			seen := map[string]bool{}
			for _, level := range q.legend {
				label := string(level)
				if seen[label] {
					return errors.New("score level names must be distinct")
				}
				seen[label] = true
			}
		}
	}
	return nil
}

func (d *DecisionRequest) validateDecisions() error {
	if d.decisions.Model == nil {
		return errors.New("model is required")
	}
	if utf8.RuneCountInString(d.Model) > 1048576 {
		return errors.New("model is too long")
	}
	if id := d.decisions.SafetyIdentifier; id != nil && utf8.RuneCountInString(*id) > 128 {
		return errors.New("safety_identifier must not exceed 128 characters")
	}
	if err := validateDecisionCacheSalt(d.CacheSalt); err != nil {
		return err
	}
	if present(d.decisions.Input) == nil {
		return errors.New("input is required")
	}
	if len(d.questions) < 1 || len(d.questions) > 200 {
		return errors.New("questions must contain 1 to 200 entries")
	}
	for _, q := range d.questions {
		if err := validateDecisionQuestion(q); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionQuestion(q decisionQuestion) error {
	if q.instructions == nil || utf8.RuneCountInString(*q.instructions) > 1048576 {
		return errors.New("instructions is required and must not exceed 1048576 characters")
	}
	if q.name != nil && utf8.RuneCountInString(*q.name) > 1048576 {
		return errors.New("name is too long")
	}
	switch q.kind {
	case "choice":
		if len(q.values) < 2 || len(q.values) > 255 {
			return errors.New("choice questions require 2 to 255 choices")
		}
		seen := map[any]bool{}
		for _, value := range q.values {
			if text, ok := value.(string); ok && utf8.RuneCountInString(text) > 1048576 {
				return errors.New("choice value is too long")
			}
			if seen[value] {
				return errors.New("choice values must be distinct")
			}
			seen[value] = true
		}
	case "score":
		if len(q.legend) < 2 || len(q.legend) > 10 {
			return errors.New("score questions require 2 to 10 levels")
		}
		for _, level := range q.legend {
			if string(level) == "null" || utf8.RuneCountInString(decisionText(level)) > 1048576 {
				return errors.New("score level label is required and must not exceed 1048576 characters")
			}
		}
	}
	for _, description := range q.descriptions {
		if utf8.RuneCountInString(description) > 1048576 {
			return errors.New("description must be a string of at most 1048576 characters")
		}
	}
	return nil
}

func validateDecisionCacheSalt(salt *string) error {
	if salt != nil && (*salt == "" || utf8.RuneCountInString(*salt) > 128 || strings.ContainsAny(*salt, "@/\\\x00")) {
		return errors.New("cache_salt must contain 1 to 128 characters and must not contain '@', '/', '\\', or NUL")
	}
	return nil
}

func validateDecisionChatOptions(options map[string]json.RawMessage) error {
	if value := options["enable_thinking"]; value != nil && string(value) != "null" && string(value) != "false" {
		return errors.New("thinking must be off for decisions")
	}
	return nil
}

var decisionIntegerString = regexp.MustCompile(`^[+-]?[0-9]+(?:_[0-9]+)*(?:\.0+)?$`)

// Match vLLM's Pydantic integer coercion, including booleans and numeric strings.
func validateDecisionPriority(raw json.RawMessage) error {
	if raw == nil || string(raw) == "true" || string(raw) == "false" {
		return nil
	}
	invalid := errors.New("priority must be a signed 64-bit integer")
	var text string
	if json.Unmarshal(raw, &text) == nil {
		text = strings.TrimSpace(text)
		if !decisionIntegerString.MatchString(text) {
			return invalid
		}
		text, _, _ = strings.Cut(text, ".")
		text = strings.ReplaceAll(text, "_", "")
	} else {
		text = string(raw)
		if strings.ContainsAny(text, ".eE") {
			value, err := strconv.ParseFloat(text, 64)
			if err != nil || value < -0x1p63 || value >= 0x1p63 || value != math.Trunc(value) {
				return invalid
			}
			return nil
		}
	}
	if _, err := strconv.ParseInt(text, 10, 64); err != nil {
		return invalid
	}
	return nil
}

// JSON objects carry question and option order; a Go map would lose it.
type decisionField struct {
	key   string
	value json.RawMessage
}

func parseDecisionObject(data []byte) ([]decisionField, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return nil, errors.New("must be an object")
	}
	var fields []decisionField
	seen := map[string]bool{}
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		name := key.(string)
		if seen[name] {
			return nil, fmt.Errorf("duplicate key %q", name)
		}
		seen[name] = true
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}
		fields = append(fields, decisionField{key: name, value: value})
	}
	if _, err := decoder.Token(); err != nil {
		return nil, err
	}
	return fields, nil
}

func decodeDecisionStrict(data []byte, value any) error {
	strictErrors, err := kjson.UnmarshalStrict(data, value, kjson.DisallowUnknownFields)
	if err != nil {
		return err
	}
	return errors.Join(strictErrors...)
}

func decisionText(raw json.RawMessage) string {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	return string(raw)
}

// vLLM System One exposes Python str(level), including repr() of nested values.
func decisionPythonString(raw json.RawMessage) (string, error) {
	var text *string
	if json.Unmarshal(raw, &text) == nil && text != nil {
		return *text, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	return decisionPythonRepr(decoder)
}

func decisionPythonRepr(decoder *json.Decoder) (string, error) {
	token, err := decoder.Token()
	if err != nil {
		return "", err
	}
	switch value := token.(type) {
	case nil:
		return "None", nil
	case bool:
		if value {
			return "True", nil
		}
		return "False", nil
	case string:
		return decisionPythonQuote(value), nil
	case json.Number:
		text := value.String()
		if !strings.ContainsAny(text, ".eE") {
			if text == "-0" {
				text = "0"
			}
			return text, nil
		}
		number, err := value.Float64()
		if err != nil && !errors.Is(err, strconv.ErrRange) {
			return "", err
		}
		if math.IsInf(number, 0) {
			if number < 0 {
				return "-inf", nil
			}
			return "inf", nil
		}
		format := byte('f')
		if number != 0 && (math.Abs(number) < 1e-4 || math.Abs(number) >= 1e16) {
			format = 'e'
		}
		text = strconv.FormatFloat(number, format, -1, 64)
		if !strings.ContainsAny(text, ".e") {
			text += ".0"
		}
		return text, nil
	case json.Delim:
		var parts []string
		keys := map[string]int{}
		for decoder.More() {
			key := ""
			if value == '{' {
				token, err := decoder.Token()
				if err != nil {
					return "", err
				}
				key = token.(string)
			}
			part, err := decisionPythonRepr(decoder)
			if err != nil {
				return "", err
			}
			if value == '{' {
				part = decisionPythonQuote(key) + ": " + part
				if index, ok := keys[key]; ok {
					parts[index] = part
					continue
				}
				keys[key] = len(parts)
			}
			parts = append(parts, part)
		}
		if _, err := decoder.Token(); err != nil {
			return "", err
		}
		if value == '{' {
			return "{" + strings.Join(parts, ", ") + "}", nil
		}
		return "[" + strings.Join(parts, ", ") + "]", nil
	default:
		return "", errors.New("invalid score level")
	}
}

func decisionPythonQuote(text string) string {
	quote := '\''
	if strings.ContainsRune(text, '\'') && !strings.ContainsRune(text, '"') {
		quote = '"'
	}
	var result strings.Builder
	result.WriteRune(quote)
	for _, r := range text {
		switch {
		case r == quote || r == '\\':
			result.WriteByte('\\')
			result.WriteRune(r)
		case r == '\n':
			result.WriteString(`\n`)
		case r == '\r':
			result.WriteString(`\r`)
		case r == '\t':
			result.WriteString(`\t`)
		case unicode.IsPrint(r):
			result.WriteRune(r)
		case r <= 0xff:
			fmt.Fprintf(&result, `\x%02x`, r)
		case r <= 0xffff:
			fmt.Fprintf(&result, `\u%04x`, r)
		default:
			fmt.Fprintf(&result, `\U%08x`, r)
		}
	}
	result.WriteRune(quote)
	return result.String()
}
