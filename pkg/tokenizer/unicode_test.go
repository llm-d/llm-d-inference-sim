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

package tokenizer

import (
	"reflect"
	"strings"
	"testing"
)

func TestSimpleTokenizerUnicodeRoundTrip(t *testing.T) {
	for _, input := range []string{
		"\u4f60\u597d\uff0c\u4eff\u771f\u5668",
		"Hello \u4e16\u754c! \U0001f680",
		"caf\u00e9 e\u0301 \u0645\u0631\u062d\u0628\u0627",
		"\U0001f469\u200d\U0001f4bb\uff01\n",
		"use `go test` | grep",
		" leading spaces",
		"\t\n\v\f\r  ",
		"\u00a0\u0085\u2003\u2028\u2029text",
		"a\x00b\x7fc",
	} {
		t.Run(input, func(t *testing.T) {
			st := NewSimpleTokenizer()
			ids, parts, err := st.RenderText(input)
			if err != nil || len(ids) == 0 || len(ids) != len(parts) {
				t.Fatalf("invalid tokenization: ids=%v parts=%q err=%v", ids, parts, err)
			}
			if got := strings.Join(parts, ""); got != input {
				t.Fatalf("tokenization lost text: got %q, want %q", got, input)
			}
			got, err := st.Detokenize(ids)
			if err != nil || got != input {
				t.Fatalf("round trip: got %q, err %v, want %q", got, err, input)
			}
		})
	}
}

func TestSimpleTokenizerWordBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input string
		want  []string
	}{
		{"accented", "na\u00efve caf\u00e9", []string{"na\u00efve ", "caf\u00e9"}},
		{"cyrillic", "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440", []string{"\u041f\u0440\u0438\u0432\u0435\u0442 ", "\u043c\u0438\u0440"}},
		{"combining", "cafe\u0301 e\u0301", []string{"cafe\u0301 ", "e\u0301"}},
		{"chinese", "\u4f60\u597d\uff0c\u4e16\u754c", []string{"\u4f60\u597d", "\uff0c", "\u4e16\u754c"}},
		{"ascii_symbols", "use `go test` | grep", []string{"use ", "`", "go ", "test", "` ", "| ", "grep"}},
		{"underscores", "_foo foo_bar __", []string{"_", "foo ", "foo_bar ", "_", "_"}},
		{"leading_whitespace", " \tleading spaces", []string{" \t", "leading ", "spaces"}},
		{"whitespace_only", "\t\n\v\f\r  ", []string{"\t\n\v\f\r  "}},
		{"unicode_whitespace", "a\u00a0\u0085\u2003\u2028\u2029b", []string{"a\u00a0\u0085\u2003\u2028\u2029", "b"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, parts, err := NewSimpleTokenizer().RenderText(tc.input)
			if err != nil || !reflect.DeepEqual(parts, tc.want) {
				t.Fatalf("got %q, err %v, want %q", parts, err, tc.want)
			}
		})
	}
}

func TestTokenizerUnicodeFixedTokenCount(t *testing.T) {
	input := " \u4f60\u597d\uff0c caf\u00e9 | \U0001f680\n"
	bt := newBaseTokenizer()
	for _, count := range []int{1, 3, 20} {
		parts := bt.splitIntoTokens(input, count)
		if len(parts) != count || strings.Join(parts, "") != input {
			t.Fatalf("count %d: got %q", count, parts)
		}
	}
}

func TestSimpleTokenizerASCIIBoundaries(t *testing.T) {
	_, parts, err := NewSimpleTokenizer().RenderText("Hello, world! 42\n")
	want := []string{"Hello", ", ", "world", "! ", "42\n"}
	if err != nil || !reflect.DeepEqual(parts, want) {
		t.Fatalf("got %q, err %v, want %q", parts, err, want)
	}
}
