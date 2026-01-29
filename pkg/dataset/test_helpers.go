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

package dataset

import (
	"strconv"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

// IsValidText validates that the given text could be generated from the predefined list of sentences
// used in tests
func IsValidText(text string) bool {
	charsTested := 0

	for charsTested < len(text) {
		textToCheck := text[charsTested:]
		found := false

		for _, fakeSentence := range completionFakeResponses {
			if len(textToCheck) <= len(fakeSentence) {
				if strings.HasPrefix(fakeSentence, textToCheck) {
					found = true
					charsTested = len(text)
					break
				}
			} else {
				if strings.HasPrefix(textToCheck, fakeSentence) {
					charsTested += len(fakeSentence)
					found = true
					break
				}
			}
		}

		if !found {
			return false
		}
	}

	return true
}

func maxTokensToStr(maxTokens *int64) string {
	if maxTokens != nil {
		return strconv.Itoa(int(*maxTokens))
	}
	return "nil"
}

func initTokenizer(model string, modelExists bool, cacheDir string) (tokenizer.Tokenizer, error) {
	var err error
	var tknzer tokenizer.Tokenizer
	if modelExists {
		tknzer, err = tokenizer.NewHFTokenizer(common.Configuration{Model: model, TokenizersCacheDir: cacheDir})
	} else {
		tknzer = tokenizer.NewSimpleTokenizer()
	}
	return tknzer, err
}
