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
	"fmt"
	"math/rand"
	"regexp"
	"strings"
	"time"
)

// list of responses to use in random mode for comepltion requests
var chatCompletionFakeResponses = []string{
	`Testing@, #testing 1$ ,2%,3^, [4&*5], 6~, 7-_ + (8 : 9) / \ < > .`,
	`Testing, testing 1,2,3.`,
	`I am fine, how are you today?`,
	`I am your AI assistant, how can I help you today?`,
	`Today is a nice sunny day.`,
	`The temperature here is twenty-five degrees centigrade.`,
	`Today it is partially cloudy and raining.`,
	`To be or not to be that is the question.`,
	`Alas, poor Yorick! I knew him, Horatio: A fellow of infinite jest`,
	`The rest is silence. `,
	`Give a man a fish and you feed him for a day; teach a man to fish and you feed him for a lifetime`,
}

// returns the max tokens or error if incorrect
func getMaxTokens(maxCompletionTokens *int64, maxTokens *int64) (*int64, error) {
	var typeToken string
	var tokens *int64
	// if both arguments are passed,
	// use maxCompletionTokens
	// as in the real vllm
	if maxCompletionTokens != nil {
		tokens = maxCompletionTokens
		typeToken = "max_completion_tokens"
	} else if maxTokens != nil {
		tokens = maxTokens
		typeToken = "max_tokens"
	}
	if tokens != nil && *tokens < 1 {
		return nil, fmt.Errorf("%s must be at least 1, got %d", typeToken, *tokens)
	}
	return tokens, nil
}

// getRandomResponseText returns random response text from the pre-defined list of responses
// considering max completion tokens if it is not nil, and a finish reason (stop or length)
func getRandomResponseText(maxCompletionTokens *int64) (string, string) {
	index := randomInt(0, len(chatCompletionFakeResponses)-1)
	text := chatCompletionFakeResponses[index]

	return getResponseText(maxCompletionTokens, text)
}

// getResponseText returns response text, from a given text
// considering max completion tokens if it is not nil, and a finish reason (stop or length)
func getResponseText(maxCompletionTokens *int64, text string) (string, string) {
	// should not happen
	if maxCompletionTokens != nil && *maxCompletionTokens <= 0 {
		return "", stopFinishReason
	}

	// no max completion tokens, return entire text
	if maxCompletionTokens == nil {
		return text, stopFinishReason
	}
	// create tokens from text, splitting by spaces
	tokens := strings.Fields(text)

	// return entire text
	if *maxCompletionTokens >= int64(len(tokens)) {
		return text, stopFinishReason
	}
	// return truncated text
	return strings.Join(tokens[0:*maxCompletionTokens], " "), lengthFinishReason
}

// Given a partial string, access the full string
func getFullTextFromPartialString(partial string) string {
	for _, str := range chatCompletionFakeResponses {
		if strings.Contains(str, partial) {
			return str
		}
	}
	return ""
}

func randomNumericString(length int) string {
	digits := "0123456789"
	result := make([]byte, length)
	for i := 0; i < length; i++ {
		num := randomInt(0, 9)
		result[i] = digits[num]
	}
	return string(result)
}

// Returns an integer between min and max (included)
func randomInt(min int, max int) int {
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)
	return r.Intn(max-min+1) + min
}

// Returns true or false randomly
func flipCoin() bool {
	return randomInt(0, 1) != 0
}

// Returns a random float64 in the range [min, max)
func randomFloat(min float64, max float64) float64 {
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)
	return r.Float64()*(max-min) + min
}

// Regular expression for the response tokenization
var re *regexp.Regexp

func init() {
	re = regexp.MustCompile(`(\{|\}|:|,|-|\.|\?|\!|;|@|#|\$|%|\^|&|\*|\(|\)|\+|\-|_|~|/|\\|>|<|\[|\]|=|"|\w+)(\s*)`)
}

func tokenize(text string) []string {
	return re.FindAllString(text, -1)
}
