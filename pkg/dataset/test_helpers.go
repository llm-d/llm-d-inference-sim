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

import "strings"

// IsValidText validates that the given text could be generated from the predefined list of sentences
// used in tests
func IsValidText(text string) bool {
	charsTested := 0

	for charsTested < len(text) {
		textToCheck := text[charsTested:]
		found := false

		for _, fakeSentence := range chatCompletionFakeResponses {
			if len(textToCheck) <= len(fakeSentence) {
				if strings.HasPrefix(fakeSentence, textToCheck) {
					found = true
					charsTested = len(text)
					break
				}
			} else {
				if strings.HasPrefix(textToCheck, fakeSentence) {
					charsTested += len(fakeSentence)
					// during generation sentences are connected by space, skip it
					// additional space at the end of the string is invalid
					if text[charsTested] == ' ' && charsTested < len(text)-1 {
						charsTested += 1
						found = true
					}
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
