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
	"math/rand"
	"regexp"
	"sync"

	"github.com/google/uuid"
)

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

// Returns a normally distributed int
// If the generated value differs by more than 70% from mean, the returned
// value will be 70% of mean
func RandomNorm(mean int, stddev int) int {
	if stddev == 0 {
		return mean
	}
	randMutex.Lock()
	defer randMutex.Unlock()
	mean_ := float64(mean)
	stddev_ := float64(stddev)
	value := randomGenerator.NormFloat64()*stddev_ + mean_
	if value < 0.3*mean_ {
		value = 0.3 * mean_
	} else if value > 1.7*mean_ {
		value = 1.7 * mean_
	}
	return int(value)
}

// GenerateUUIDString generates a UUID string under a lock
func GenerateUUIDString() string {
	randMutex.Lock()
	defer randMutex.Unlock()
	return uuid.NewString()
}

// Regular expression for the response tokenization
var re *regexp.Regexp

func init() {
	re = regexp.MustCompile(`(\{|\}|:|,|-|\.|\?|\!|;|@|#|\$|%|\^|&|\*|\(|\)|\+|\-|_|~|/|\\|>|<|\[|\]|=|"|\w+)(\s*)`)
}

func Tokenize(text string) []string {
	return re.FindAllString(text, -1)
}
