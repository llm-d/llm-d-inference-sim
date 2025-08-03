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

package common

import (
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"strings"
	"sync"

	"github.com/google/uuid"
)

const (
	ResponseLenMax              = 128
	responseLenMean             = 40
	responseLenStddev           = 20
	stopFinishReasonProbability = 0.8

	StopFinishReason         = "stop"
	LengthFinishReason       = "length"
	ToolsFinishReason        = "tool_calls"
	RemoteDecodeFinishReason = "remote_decode"
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
func GetMaxTokens(maxCompletionTokens *int64, maxTokens *int64) (*int64, error) {
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

// ValidateContextWindow checks if the request fits within the model's context window
// Returns validation result, actual completion tokens, and total tokens
func ValidateContextWindow(promptTokens int, maxCompletionTokens *int64, maxModelLen int) (bool, int64, int64) {
	completionTokens := int64(0)
	if maxCompletionTokens != nil {
		completionTokens = *maxCompletionTokens
	}

	totalTokens := int64(promptTokens) + completionTokens
	isValid := totalTokens <= int64(maxModelLen)

	return isValid, completionTokens, totalTokens
}

// GetRandomResponseLen returns int in range [1, responseLenMax]
// numbers are chosen according a gaussian distribution with mean responseLenMean, and standard deviation responseLenStddev
func GetRandomResponseLen() int {
	for {
		val := rand.NormFloat64()*responseLenStddev + responseLenMean
		if val >= 1 && val <= ResponseLenMax {
			return int(math.Round(val))
		}
		// else reject and resample
	}
}

// GetRandomFinishReason returns finish reason with the probability for 'stop' as defined by stopFinishReasonProbability
func GetRandomFinishReason() string {
	if rand.Float64() < stopFinishReasonProbability {
		return StopFinishReason
	}
	return LengthFinishReason
}

// GetRandomText generates random text for the required number of tokens,
// select randomly a sentence from chatCompletionFakeResponses,
// if number of tokens is lower than required - select another sentence,
// continue until the required number of tokens is achieved
func GetRandomText(numOfTokens int) string {
	allTokens := make([]string, 0)

	for len(allTokens) < numOfTokens {
		index := RandomInt(0, len(chatCompletionFakeResponses)-1)
		// create tokens from text, splitting by spaces and special characters
		tokens := Tokenize(chatCompletionFakeResponses[index])
		remaining := numOfTokens - len(allTokens)

		if len(tokens) > remaining {
			// there is too many tokens, append only the relevant part
			tokens = tokens[:remaining]
		}

		if len(allTokens) > 0 {
			// for not first sentences add space to the first token to separate between sentences without adding an additional token
			tokens[0] = " " + tokens[0]
		}

		allTokens = append(allTokens, tokens...)
	}

	// return all tokens as text
	return strings.Join(allTokens, "")
}

// GetRandomResponseText generates text to be returned in a response, and the finish reason (stop or length)
// if maxCompletionTokens is defined
// - currently, the generated number of words in the text will be equal to it value
// - in future - need to find statistics about generated tokens distribution and return less tokens in part os requests
// - finish reason will be chosen randomly from the collection (stop, length) with 80% for stop and 20% for length
// if maxCompletionTokens is nil
// - the response text's length is randomly chosen from the range [1, responseLenMax] according additional parameters
// - finish reason is stop
func GetRandomResponseText(maxCompletionTokens *int64) (string, string) {
	numOfTokens := 0
	finishReason := StopFinishReason

	// no max completion tokens, return text with random length
	if maxCompletionTokens == nil {
		numOfTokens = GetRandomResponseLen()
	} else {
		numOfTokens = int(*maxCompletionTokens)
		finishReason = GetRandomFinishReason()
	}

	text := GetRandomText(numOfTokens)
	return text, finishReason
}

// GetResponseText returns response text, from a given text
// considering max completion tokens if it is not nil, and a finish reason (stop or length)
func GetResponseText(maxCompletionTokens *int64, text string) (string, string) {
	// no max completion tokens, return entire text
	if maxCompletionTokens == nil {
		return text, StopFinishReason
	}

	// create tokens from text, splitting by spaces
	tokens := Tokenize(text)

	// return entire text
	if *maxCompletionTokens >= int64(len(tokens)) {
		return text, StopFinishReason
	}
	// return truncated text
	return strings.Join(tokens[0:*maxCompletionTokens], " "), LengthFinishReason
}

func RandomNumericString(length int) string {
	digits := "0123456789"
	result := make([]byte, length)
	for i := 0; i < length; i++ {
		num := RandomInt(0, 9)
		result[i] = digits[num]
	}
	return string(result)
}

var randomGenerator *rand.Rand
var randMutex sync.Mutex

func InitRandom(seed int64) {
	src := rand.NewSource(seed)
	randomGenerator = rand.New(src)
	uuid.SetRand(randomGenerator)
}

// Returns an integer between min and max (included)
func RandomInt(min int, max int) int {
	randMutex.Lock()
	defer randMutex.Unlock()
	return randomGenerator.Intn(max-min+1) + min
}

// Returns true or false randomly
func FlipCoin() bool {
	return RandomInt(0, 1) != 0
}

// probability is an integer between 0 and 100
func RandomBool(probability int) bool {
	randMutex.Lock()
	defer randMutex.Unlock()
	return randomGenerator.Float64() < float64(probability)/100
}

// Returns a random float64 in the range [min, max)
func RandomFloat(min float64, max float64) float64 {
	randMutex.Lock()
	defer randMutex.Unlock()
	return randomGenerator.Float64()*(max-min) + min
}

// Returns a normally distributed float64
// If the generated value differs by more than 70% from mean, the returned
// value will be 70% of mean
func RandomNorm(mean float64, stddev float64) float64 {
	if stddev == 0 {
		return mean
	}
	randMutex.Lock()
	defer randMutex.Unlock()
	value := randomGenerator.NormFloat64()*stddev + mean
	if value < 0.3*mean {
		value = 0.3 * mean
	} else if value > 1.7*mean {
		value = 1.7 * mean
	}
	return value
}

// Regular expression for the response tokenization
var re *regexp.Regexp

func init() {
	re = regexp.MustCompile(`(\{|\}|:|,|-|\.|\?|\!|;|@|#|\$|%|\^|&|\*|\(|\)|\+|\-|_|~|/|\\|>|<|\[|\]|=|"|\w+)(\s*)`)
}

func Tokenize(text string) []string {
	return re.FindAllString(text, -1)
}
