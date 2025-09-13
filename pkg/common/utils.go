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
	"time"

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

// this array defines the probabilities for the buckets to be used for the generation of number of tokens in response
var respLenBucketsProbabilities = [...]float64{0.2, 0.3, 0.2, 0.05, 0.1, 0.15}
var cumulativeBucketsProbabilities []float64

const (
	flexBucketIndex    = 3
	maxFixedBucketSize = 20
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

func init() {
	cumulativeBucketsProbabilities = make([]float64, len(respLenBucketsProbabilities))
	sum := 0.0

	for i, val := range respLenBucketsProbabilities {
		sum += val
		cumulativeBucketsProbabilities[i] = sum
	}
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
// if ignore_eos is true - the response will be generated with exactly maxCompletionTokens tokens
// - request was validated so that when ignore_eos is true, maxCompletionTokens must be defined
func GetRandomResponseText(maxCompletionTokens *int64, ignore_eos bool) (string, string) {
	numOfTokens := 0
	finishReason := StopFinishReason

	// no max completion tokens, return text with random length
	if maxCompletionTokens == nil {
		numOfTokens = GetRandomResponseLen()
	} else {
		maxTokens := int(*maxCompletionTokens)
		if ignore_eos {
			numOfTokens = maxTokens
			finishReason = LengthFinishReason
		} else {
			// max tokens is defined - generate real length of the response based on it
			numOfTokens = getResponseLengthByHistogram(maxTokens)
			if numOfTokens == maxTokens {
				// if response should be create with maximum number of tokens - finish reason will be 'length'
				finishReason = LengthFinishReason
			}
		}
	}

	text := GetRandomText(numOfTokens)
	return text, finishReason
}

// getResponseLengthByHistogram calculates the number of tokens to be returned in a response based on the max tokens value and the pre-defined buckets.
// The response length is distributed according to the probabilities, defined in respLenBucketsProbabilities.
// The histogram contains equally sized buckets and the last special bucket, which contains only the maxTokens value.
// The last element of respLenBucketsProbabilities defines the probability of a reposnse with maxToken tokens.
// Other values define probabilities for the equally sized buckets.
// If maxToken is small (smaller than number of buckets) - the response length is randomly selected from the range [1, maxTokens]
func getResponseLengthByHistogram(maxTokens int) int {
	if maxTokens <= 1 {
		return maxTokens
	}
	// maxTokens is small - no need to use the histogram of probabilities, just select a random value in the range [1, maxTokens]
	if maxTokens <= len(cumulativeBucketsProbabilities) {
		res := RandomInt(1, maxTokens)
		return res
	}

	r := RandomFloat(0, 1)

	// check if r is in the last bucket, then maxTokens should be returned
	if r > cumulativeBucketsProbabilities[len(cumulativeBucketsProbabilities)-2] {
		return maxTokens
	}

	// determine which bucket to use, the bucket with a cumulative probability larger than r is the bucket to use
	// initialize bucketIndex with the last bucket to handle the case (which should not happen) when the probabilities sum is less than 1
	bucketIndex := len(cumulativeBucketsProbabilities) - 1
	for i, c := range cumulativeBucketsProbabilities {
		if r <= c {
			bucketIndex = i
			break
		}
	}

	// calculate the size of all of the buckets (except the special last bucket)
	start, end := calcBucketBoundaries(maxTokens, bucketIndex)

	// pick uniformly within the bucket’s range
	return RandomInt(start, end)
}

// calcBucketBoundaries calculates boundaries of a bucket with the given index.
// Maximum size for equally sized buckets is defined by maxFixedBucketSize.
// [maxFixedBucketSize*(number-of-buckets-1)+1] is the value of maxTokens for which
// division to equally size buckets will give buckets with size maxFixedBucketSize.
// If maxTokens is [maxFixedBucketSize*(number-of-buckets-1)+1] or less,
// all buckets will be of equal size, except the last bucket, which contains only one value.
// If maxTokens is higher than [maxFixedBucketSize*(number-of-buckets-1)+1],
// and flexBucketIndex is valid (between 0 and number of buckets - 1) the buckets sizes will not be equal.
// In this case, all buckets except the one at flexBucketIndex index will have size 20 (and the last is with size 1),
// and the bucket at flexBucketIndex index will 'stretch' to cover the remaining range.
func calcBucketBoundaries(maxTokens int, bucketIndex int) (start int, end int) {
	maxEquallyBucketsSz := maxFixedBucketSize*(len(cumulativeBucketsProbabilities)-1) + 1

	if maxTokens <= maxEquallyBucketsSz || flexBucketIndex < 0 || flexBucketIndex >= len(cumulativeBucketsProbabilities)-1 {
		// create equally size buckets
		// calculate the size of all of the buckets (except the special last bucket)
		bucketSize := float64(maxTokens-1) / float64(len(cumulativeBucketsProbabilities)-1)
		start = int(bucketSize*float64(bucketIndex)) + 1
		end = int(bucketSize * float64(bucketIndex+1))
	} else {
		// create non-equally sized buckets and find boundaries of the required bucket
		if bucketIndex < flexBucketIndex {
			// the relevant bucket is before the flex bucket, all buckets are of the same size (maxFixedBucketSize)
			// start is the minimum number in the required bucket
			start = maxFixedBucketSize*bucketIndex + 1
			end = maxFixedBucketSize * (bucketIndex + 1)
		} else {
			flexBucketSize := maxTokens - (maxFixedBucketSize * (len(cumulativeBucketsProbabilities) - 2))

			if bucketIndex == flexBucketIndex {
				// the relevant bucket is the flex bucket
				start = int(maxFixedBucketSize*float64(bucketIndex)) + 1
				end = maxFixedBucketSize*bucketIndex + flexBucketSize
			} else {
				// the relevant bucket is one of buckets after the flex bucket
				start = int(maxFixedBucketSize*float64(bucketIndex-1)) + flexBucketSize + 1
				end = maxFixedBucketSize*bucketIndex + flexBucketSize
			}
		}
	}

	// sometimes end could be maxTokens because of rounding, change the value to maxToken-1
	if end >= maxTokens {
		end = maxTokens - 1
	}

	return start, end
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

// SleepSec sleep seconds, the only param, duration, is the second(s) to sleep
func SleepSec(duration int) {
	time.Sleep(time.Duration(duration) * time.Second)
}

// SleepMilliSec sleep milliseconds, the only param, duration, is the millisecond(s) to sleep
func SleepMilliSec(duration int) {
	time.Sleep(time.Duration(duration) * time.Millisecond)
}

// SleepMicroSec sleep microseconds, the only param, duration, is the microsecond(s) to sleep
func SleepMicroSec(duration int) {
	time.Sleep(time.Duration(duration) * time.Microsecond)
}
