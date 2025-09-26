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
	"context"
	"errors"
	"math"
	"math/rand"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	_ "github.com/mattn/go-sqlite3"
)

const (
	RoleAssistant = "assistant"
	RoleUser      = "user"
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

// list of responses to use in random mode for completion requests
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

type Dataset interface {
	// Init initializes the dataset using configs
	Init(ctx context.Context, logger logr.Logger, path string, url string, useInMemory bool) error
	// Close closes the dataset
	Close() error
	// GetTokens returns tokens for the given request and mode (echo or random)
	GetTokens(req openaiserverapi.CompletionRequest, mode string) ([]string, string, error)
}

func init() {
	cumulativeBucketsProbabilities = make([]float64, len(respLenBucketsProbabilities))
	sum := 0.0

	for i, val := range respLenBucketsProbabilities {
		sum += val
		cumulativeBucketsProbabilities[i] = sum
	}
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

// GenPresetRandomTokens generates random tokens for the required number of tokens,
// select randomly a sentence from chatCompletionFakeResponses,
// if number of tokens is lower than required - select another sentence,
// continue until the required number of tokens is achieved
func GenPresetRandomTokens(numOfTokens int) []string {
	allTokens := make([]string, 0)

	for len(allTokens) < numOfTokens {
		index := common.RandomInt(0, len(chatCompletionFakeResponses)-1)
		// create tokens from text, splitting by spaces and special characters
		tokens := common.Tokenize(chatCompletionFakeResponses[index])
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

	return allTokens
}

// howManyTokensToGen generates the number of tokens to be returned in a response, and the finish reason (see constants)
// if maxCompletionTokens is defined
// - currently, the generated number of words in the text will be equal to it value
// - in future - need to find statistics about generated tokens distribution and return less tokens in part os requests
// - finish reason will be chosen randomly from the collection (stop, length) with 80% for stop and 20% for length
// if maxCompletionTokens is nil
// - the response text's length is randomly chosen from the range [1, responseLenMax] according additional parameters
// - finish reason is stop
// if ignore_eos is true - the response will be generated with exactly maxCompletionTokens tokens
// - request was validated so that when ignore_eos is true, maxCompletionTokens must be defined
func howManyTokensToGen(maxCompletionTokens *int64, ignore_eos bool) (int, string) {
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

	return numOfTokens, finishReason
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
		res := common.RandomInt(1, maxTokens)
		return res
	}

	r := common.RandomFloat(0, 1)

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
	return common.RandomInt(start, end)
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

// EchoResponseTokens returns needed tokens, from a given text
// considering max completion tokens if it is not nil, and a finish reason (stop or length)
func EchoResponseTokens(maxCompletionTokens *int64, text string) ([]string, string) {
	tokens := common.Tokenize(text)
	// no max completion tokens, return entire text
	if maxCompletionTokens == nil {
		return tokens, StopFinishReason
	}

	if *maxCompletionTokens >= int64(len(tokens)) {
		return tokens, StopFinishReason
	}
	// return truncated text
	return tokens[0:*maxCompletionTokens], LengthFinishReason
}

type BaseDataset struct {
	logger logr.Logger
}

func (d *BaseDataset) Init(ctx context.Context, logger logr.Logger, path string, url string, useInMemory bool) error {
	d.logger = logger
	return nil
}

func (d *BaseDataset) Close() error {
	return nil
}

func (d *BaseDataset) echo(req openaiserverapi.CompletionRequest) ([]string, string, error) {
	nMaxTokens := d.extractMaxTokens(req)
	prompt, err := d.extractPrompt(req)
	if err != nil {
		return nil, "", err
	}
	tokens, finishReason := EchoResponseTokens(nMaxTokens, prompt)
	return tokens, finishReason, nil
}

// GetTokens returns tokens and finishReason for the given request and mode (echo or random)
func (d *BaseDataset) GetTokens(req openaiserverapi.CompletionRequest, mode string) ([]string, string, error) {
	if mode == common.ModeEcho {
		return d.echo(req)
	}
	nTokensToGen, finishReason := howManyTokensToGen(d.extractMaxTokens(req), req.GetIgnoreEOS())
	return GenPresetRandomTokens(nTokensToGen), finishReason, nil
}

// extractMaxTokens extracts the max tokens from the request
// for chat completion - max_completion_tokens field is used
// for text completion - max_tokens field is used
func (d *BaseDataset) extractMaxTokens(req openaiserverapi.CompletionRequest) *int64 {
	if chatReq, ok := req.(*openaiserverapi.ChatCompletionRequest); ok {
		return chatReq.GetMaxCompletionTokens()
	} else if textReq, ok := req.(*openaiserverapi.TextCompletionRequest); ok {
		return textReq.MaxTokens
	}
	return nil
}

// extractPrompt extracts the prompt from the request
// for chat completion - the last user message is used as the prompt
// for text completion - the prompt field is used
func (d *BaseDataset) extractPrompt(req openaiserverapi.CompletionRequest) (string, error) {
	if chatReq, ok := req.(*openaiserverapi.ChatCompletionRequest); ok {
		return chatReq.GetLastUserMsg(), nil
	} else if textReq, ok := req.(*openaiserverapi.TextCompletionRequest); ok {
		return textReq.GetPrompt(), nil
	}
	return "", errors.New("unknown request type")
}
