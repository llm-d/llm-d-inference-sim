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
	"math/rand"
	"regexp"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/pkg/errors"
)

// Definition of buckets for time-to-first-token and time-per-output-token metrics, each value is an upper boundary of a bucket
var TTFTBucketsBoundaries = []float64{0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
	0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
	2560.0}
var TPOTBucketsBoundaries = []float64{0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
	1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0}

var RequestLatencyBucketsBoundaries = []float64{0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0,
	20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0}

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

type Random struct {
	randomGenerator *rand.Rand
	randMutex       sync.Mutex
	uuidNamespace   uuid.UUID
	uuidName        string
	uuidCount       int
}

func NewRandom(seed int64, port int) *Random {
	src := rand.NewSource(seed)
	randomGenerator := rand.New(src)
	return &Random{randomGenerator: randomGenerator,
		uuidNamespace: uuid.NameSpaceURL,
		uuidName:      fmt.Sprintf("%d-%d", seed, port),
	}
}

// Returns an integer between min and max (included)
func (r *Random) RandomInt(min int, max int) int {
	r.randMutex.Lock()
	defer r.randMutex.Unlock()

	return r.randomGenerator.Intn(max-min+1) + min
}

// Returns true or false randomly
func (r *Random) FlipCoin() bool {
	return r.RandomInt(0, 1) != 0
}

// probability is an integer between 0 and 100
func (r *Random) RandomBool(probability int) bool {
	r.randMutex.Lock()
	defer r.randMutex.Unlock()

	return r.randomGenerator.Float64() < float64(probability)/100
}

// Returns a random float64 in the range [min, max)
func (r *Random) RandomFloat(min float64, max float64) float64 {
	r.randMutex.Lock()
	defer r.randMutex.Unlock()

	return r.randomGenerator.Float64()*(max-min) + min
}

// Returns a normally distributed float64
func (r *Random) RandomNorm(mean int, stddev int) float64 {
	if stddev == 0 {
		return float64(mean)
	}
	r.randMutex.Lock()
	defer r.randMutex.Unlock()

	mean_ := float64(mean)
	stddev_ := float64(stddev)
	return r.randomGenerator.NormFloat64()*stddev_ + mean_
}

// Returns a normally distributed int
// If the generated value differs by more than 70% from mean, the returned
// value will be 70% of mean
func (r *Random) RandomNormTruncated(mean int, stddev int) int {
	value := r.RandomNorm(mean, stddev)
	mean_ := float64(mean)
	if value < 0.3*mean_ {
		value = 0.3 * mean_
	} else if value > 1.7*mean_ {
		value = 1.7 * mean_
	}
	return int(value)
}

func (r *Random) RandomNormDuration(mean, stddev time.Duration) time.Duration {
	meanMilliseconds := mean.Milliseconds()
	stddevMilliseconds := stddev.Milliseconds()
	value := r.RandomNorm(int(meanMilliseconds), int(stddevMilliseconds))
	mean_ := float64(meanMilliseconds)
	if value < 0.3*mean_ {
		value = 0.3 * mean_
	} else if value > 1.7*mean_ {
		value = 1.7 * mean_
	}

	return time.Millisecond * time.Duration(value)
}

// GenerateUUIDString generates a UUID string under a lock
func (r *Random) GenerateUUIDString() string {
	r.randMutex.Lock()
	defer r.randMutex.Unlock()
	name := fmt.Sprintf("%s-%d", r.uuidName, r.uuidCount)
	r.uuidCount++
	return uuid.NewSHA1(r.uuidNamespace, []byte(name)).String()
}

func (r *Random) RandomNumericString(length int) string {
	digits := "0123456789"
	result := make([]byte, length)
	for i := 0; i < length; i++ {
		num := r.RandomInt(0, 9)
		result[i] = digits[num]
	}
	return string(result)
}

// Regular expression for the response tokenization
var re *regexp.Regexp

func init() {
	re = regexp.MustCompile(`(\{|\}|:|,|-|\.|\?|\!|;|@|#|\$|%|\^|&|\*|\(|\)|\+|\-|_|~|/|\\|>|<|\[|\]|=|"|\w+)(\s*)`)
}

func Tokenize(text string) []string {
	return re.FindAllString(text, -1)
}

func WriteToChannel[T any](channel chan T, object T, logger logr.Logger, channelName string) {
	select {
	case channel <- object:
	default:
		logger.V(logging.WARN).Info("failed to write to", "channel", channelName)
	}
}

// MaxIntSlice receives a slice of ints, returns the maximum value in the slice if not empty,
// and error if the slice is empty
func MaxIntSlice(numbers []int) (int, error) {
	if len(numbers) == 0 {
		return 0, errors.New("cannot return maximum of an empty slice")
	}
	max := numbers[0]
	for _, num := range numbers[1:] {
		if num > max {
			max = num
		}
	}
	return max, nil
}
