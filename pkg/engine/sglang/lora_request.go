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

package sglang

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/valyala/fasthttp"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

// loraUpdateOutput is the body sglang answers a load or unload request with.
type loraUpdateOutput struct {
	Rid           *string `json:"rid"`
	HTTPWorkerIPC *string `json:"http_worker_ipc"`
	Success       bool    `json:"success"`
	ErrorMessage  *string `json:"error_message"`
	// LoadedAdapters maps each loaded adapter to its path. It is null when the
	// request failed before reaching the adapters.
	LoadedAdapters map[string]string `json:"loaded_adapters"`
}

func failedUpdate(message string) loraUpdateOutput {
	return loraUpdateOutput{ErrorMessage: &message}
}

// failedUpdateLocked is a failure the adapters themselves report, so it lists
// them. Caller must hold mu.
func (r *loraRegistry) failedUpdateLocked(message string) loraUpdateOutput {
	return loraUpdateOutput{ErrorMessage: &message, LoadedAdapters: r.loadedAdaptersLocked()}
}

// succeededUpdateLocked is a success, listing the adapters. Caller must hold mu.
func (r *loraRegistry) succeededUpdateLocked() loraUpdateOutput {
	message := ""
	return loraUpdateOutput{Success: true, ErrorMessage: &message, LoadedAdapters: r.loadedAdaptersLocked()}
}

type bodyFieldKind int

const (
	requiredString bodyFieldKind = iota
	optionalString
	optionalBool
)

type bodyField struct {
	name string
	kind bodyFieldKind
}

// The fields of sglang's LoadLoRAAdapterReqInput and UnloadLoRAAdapterReqInput,
// in the order pydantic validates them.
var (
	loadLoraFields = []bodyField{
		{"rid", optionalString}, {"http_worker_ipc", optionalString},
		{"lora_name", requiredString}, {"lora_path", requiredString},
		{"pinned", optionalBool}, {"lora_id", optionalString},
	}
	unloadLoraFields = []bodyField{
		{"rid", optionalString}, {"http_worker_ipc", optionalString},
		{"lora_name", requiredString}, {"lora_id", optionalString},
	}
)

// pyDict is a JSON object decoded in key order, as Python's json module keeps it.
type pyDict []pyItem

type pyItem struct {
	key   string
	value any
}

func (d pyDict) get(key string) (any, bool) {
	for _, item := range d {
		if item.key == key {
			return item.value, true
		}
	}
	return nil, false
}

func (d pyDict) set(key string, value any) {
	for i := range d {
		if d[i].key == key {
			d[i].value = value
		}
	}
}

// pyTuple is a pydantic error location, which Python prints as a tuple.
type pyTuple []any

// validationError is one pydantic validation error, as FastAPI reports it.
type validationError struct {
	errorType string
	loc       pyTuple
	msg       string
	input     any
	// ctxError is the "error" entry of the error's ctx, "" for none
	ctxError string
}

func (e validationError) repr() string {
	fields := pyDict{{"type", e.errorType}, {"loc", e.loc}, {"msg", e.msg}, {"input", e.input}}
	if e.ctxError != "" {
		fields = append(fields, pyItem{"ctx", pyDict{{"error", e.ctxError}}})
	}
	return pyRepr(fields)
}

// decodeLoraRequest validates a load or unload request body as FastAPI and
// pydantic's lax mode do, returning its fields with booleans coerced. Unknown
// fields are ignored. A body that fails is answered the way sglang's
// RequestValidationError handler answers it: a 400 whose message is
// str(exc) and str(exc.errors()), without the endpoint's source location.
func decodeLoraRequest(body []byte, fields []bodyField) (pyDict, *api.Error) {
	if len(body) == 0 {
		return nil, validationFailure([]validationError{
			{errorType: "missing", loc: pyTuple{"body"}, msg: "Field required"},
		})
	}
	var syntaxCheck any
	if err := json.Unmarshal(body, &syntaxCheck); err != nil {
		offset := int64(len(body))
		var syntaxErr *json.SyntaxError
		if errors.As(err, &syntaxErr) {
			offset = syntaxErr.Offset
		}
		return nil, validationFailure([]validationError{{errorType: "json_invalid", loc: pyTuple{"body", offset},
			msg: "JSON decode error", input: pyDict{}, ctxError: err.Error()}})
	}
	decoded, err := decodeOrdered(body)
	if err != nil {
		apiErr := api.NewError("failed to decode the request body, "+err.Error(), fasthttp.StatusBadRequest, nil)
		return nil, &apiErr
	}
	values, ok := decoded.(pyDict)
	if !ok {
		return nil, validationFailure([]validationError{
			{errorType: "dict_type", loc: pyTuple{"body"}, msg: "Input should be a valid dictionary", input: decoded},
		})
	}

	var errs []validationError
	for _, field := range fields {
		loc := pyTuple{"body", field.name}
		value, present := values.get(field.name)
		switch {
		case !present:
			if field.kind == requiredString {
				errs = append(errs, validationError{errorType: "missing", loc: loc, msg: "Field required", input: values})
			}
		case field.kind == optionalBool:
			coerced, err := laxBool(value, loc)
			if err != nil {
				errs = append(errs, *err)
			} else {
				values.set(field.name, coerced)
			}
		case value == nil && field.kind == optionalString:
		default:
			if _, isString := value.(string); !isString {
				errs = append(errs, validationError{errorType: "string_type", loc: loc,
					msg: "Input should be a valid string", input: value})
			}
		}
	}
	if errs != nil {
		return nil, validationFailure(errs)
	}
	return values, nil
}

// invalidLoraPath is the error pydantic reports for a lora_path that is
// neither a string nor a list of strings and nulls: one error for each member
// of its Union[List[Optional[str]], str, None] type.
func invalidLoraPath(raw json.RawMessage) *api.Error {
	value, err := decodeOrdered(raw)
	if err != nil {
		apiErr := api.NewError("failed to decode lora_path, "+err.Error(), fasthttp.StatusBadRequest, nil)
		return &apiErr
	}
	const listMember, stringMember = "list[nullable[str]]", "str"
	var errs []validationError
	if list, isList := value.([]any); isList {
		for i, item := range list {
			if _, isString := item.(string); item != nil && !isString {
				errs = append(errs, validationError{errorType: "string_type",
					loc: pyTuple{"body", "lora_path", listMember, int64(i)}, msg: "Input should be a valid string",
					input: item})
			}
		}
	} else {
		errs = append(errs, validationError{errorType: "list_type", loc: pyTuple{"body", "lora_path", listMember},
			msg: "Input should be a valid list", input: value})
	}
	errs = append(errs, validationError{errorType: "string_type", loc: pyTuple{"body", "lora_path", stringMember},
		msg: "Input should be a valid string", input: value})
	return validationFailure(errs)
}

func validationFailure(errs []validationError) *api.Error {
	reprs := make([]string, len(errs))
	for i, e := range errs {
		reprs[i] = e.repr()
	}
	plural := "s"
	if len(errs) == 1 {
		plural = ""
	}
	message := fmt.Sprintf("%d validation error%s:\n  %s", len(errs), plural, strings.Join(reprs, "\n  ")) +
		" [" + strings.Join(reprs, ", ") + "]"
	err := api.NewError(message, fasthttp.StatusBadRequest, nil)
	err.Type = "Bad Request"
	return &err
}

// laxBool coerces value to a boolean the way pydantic's lax mode does.
func laxBool(value any, loc pyTuple) (bool, *validationError) {
	unparsable := &validationError{errorType: "bool_parsing", loc: loc,
		msg: "Input should be a valid boolean, unable to interpret input", input: value}
	switch v := value.(type) {
	case bool:
		return v, nil
	case json.Number:
		switch f, _ := v.Float64(); f {
		case 0:
			return false, nil
		case 1:
			return true, nil
		}
		return false, unparsable
	case string:
		switch strings.ToLower(v) {
		case "0", "off", "f", "false", "n", "no":
			return false, nil
		case "1", "on", "t", "true", "y", "yes":
			return true, nil
		}
		return false, unparsable
	}
	return false, &validationError{errorType: "bool_type", loc: loc, msg: "Input should be a valid boolean",
		input: value}
}

// decodeOrdered decodes valid JSON into nil, bool, json.Number, string, []any
// and pyDict values.
func decodeOrdered(data []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	return decodeOrderedValue(decoder)
}

func decodeOrderedValue(decoder *json.Decoder) (any, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	switch token {
	case json.Delim('{'):
		dict := pyDict{}
		for decoder.More() {
			key, err := decoder.Token()
			if err != nil {
				return nil, err
			}
			value, err := decodeOrderedValue(decoder)
			if err != nil {
				return nil, err
			}
			name, _ := key.(string)
			if _, duplicate := dict.get(name); duplicate {
				dict.set(name, value)
			} else {
				dict = append(dict, pyItem{name, value})
			}
		}
		_, err := decoder.Token()
		return dict, err
	case json.Delim('['):
		list := []any{}
		for decoder.More() {
			value, err := decodeOrderedValue(decoder)
			if err != nil {
				return nil, err
			}
			list = append(list, value)
		}
		_, err := decoder.Token()
		return list, err
	}
	if token == nil {
		return nil, nil
	}
	if _, isDelim := token.(json.Delim); isDelim {
		return nil, io.ErrUnexpectedEOF
	}
	return token, nil
}

// pyRepr renders a decoded value the way Python's repr() does.
func pyRepr(value any) string {
	switch v := value.(type) {
	case nil:
		return "None"
	case bool:
		if v {
			return "True"
		}
		return "False"
	case json.Number:
		return pyNumber(v)
	case int64:
		return strconv.FormatInt(v, 10)
	case string:
		return pyString(v)
	case []any:
		return "[" + joinReprs(v) + "]"
	case pyTuple:
		if len(v) == 1 {
			return "(" + pyRepr(v[0]) + ",)"
		}
		return "(" + joinReprs(v) + ")"
	case pyDict:
		items := make([]string, len(v))
		for i, item := range v {
			items[i] = pyString(item.key) + ": " + pyRepr(item.value)
		}
		return "{" + strings.Join(items, ", ") + "}"
	}
	return fmt.Sprint(value)
}

func joinReprs(values []any) string {
	reprs := make([]string, len(values))
	for i, value := range values {
		reprs[i] = pyRepr(value)
	}
	return strings.Join(reprs, ", ")
}

// pyNumber renders a JSON number as Python decodes it: an int when it has no
// fraction or exponent, a float otherwise.
func pyNumber(n json.Number) string {
	literal := n.String()
	if !strings.ContainsAny(literal, ".eE") {
		return literal
	}
	f, err := n.Float64()
	if err != nil {
		return literal
	}
	repr := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(repr, ".eIN") {
		repr += ".0"
	}
	return repr
}

// pyString quotes s the way Python's repr() does: in single quotes unless s
// holds a single quote and no double quote.
func pyString(s string) string {
	quote := byte('\'')
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		quote = '"'
	}
	var b strings.Builder
	b.WriteByte(quote)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == rune(quote):
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte(quote)
	return b.String()
}
