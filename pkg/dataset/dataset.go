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
	"math"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	_ "github.com/mattn/go-sqlite3"
)

// list of responses to use in random mode for completion requests
var completionFakeResponses = []string{
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
	// Close closes the dataset
	Close() error
	// GetTokens returns tokens for the given request and mode (echo or random)
	GetTokens(req openaiserverapi.CompletionRequest, mode string) ([]string, string, error)
}

type BaseDataset struct {
	logger          logr.Logger
	maxModelLen     int
	random          *common.Random
	histogramHelper *histogramHelper
}

func (d *BaseDataset) Init(ctx context.Context, logger logr.Logger, random *common.Random, maxModelLen int) error {
	d.logger = logger
	d.maxModelLen = maxModelLen
	d.random = random
	d.histogramHelper = newHistogramHelper(d.random)

	return nil
}

func (d *BaseDataset) Close() error {
	return nil
}

// GetTokens returns tokens and finishReason for the given request and mode (echo or random)
func (d *BaseDataset) GetTokens(req openaiserverapi.CompletionRequest, mode string) ([]string, string, error) {
	if mode == common.ModeEcho {
		return d.getTokensInEchoMode(req)
	}

	maxRespTokens, isMaxTokensInReq := d.calculateResponseMaxLen(req)

	numOfRespTokens := 0
	finishReason := common.StopFinishReason

	switch {
	case req.GetIgnoreEOS():
		// ignore_eos is true - response must have the maximum number of tokens
		numOfRespTokens = maxRespTokens
	case isMaxTokensInReq:
		// max tokens is defined in the request - generate number of tokens in the response based on the histogram
		numOfRespTokens = d.histogramHelper.getResponseLengthByHistogram(maxRespTokens)
	default:
		// no tokens limitation in the request - use gaussian with the mean (currently hard-coded)
		numOfRespTokens = d.getRandomResponseLenByGaussian(maxRespTokens)
	}

	if numOfRespTokens == maxRespTokens {
		// if response should be create with maximum number of tokens - finish reason will be 'length'
		finishReason = common.LengthFinishReason
	}

	return d.generatePresetRandomTokens(numOfRespTokens), finishReason, nil
}

func (d *BaseDataset) getTokensInEchoMode(req openaiserverapi.CompletionRequest) ([]string, string, error) {
	tokens := common.Tokenize(req.ExtractPrompt())
	maxTokens := req.GetMaxCompletionTokens()
	finishReason := common.StopFinishReason

	if maxTokens != nil && len(tokens) >= int(*maxTokens) {
		finishReason = common.LengthFinishReason
	}

	return tokens, finishReason, nil
}

// calculateResponseMaxLen - calculates maximum length of a response to be randomly chosen from the dataset
// for the given request and the simulator configuration.
// If max-tokens/max-completion-tokens is defined - use it,
// otherwise use <model content window size> - <number of input tokens>
// boolean returned value defines whether max tokens number was passed in the request
func (d *BaseDataset) calculateResponseMaxLen(req openaiserverapi.CompletionRequest) (int, bool) {
	maxTokens := req.GetMaxCompletionTokens()

	if maxTokens != nil {
		return int(*maxTokens), true
	}

	return d.maxModelLen - len(common.Tokenize(req.GetFullPrompt())), false
}

// getRandomResponseLenByDistribution returns int in range [1, responseLenMax]
// numbers are chosen according a gaussian distribution with mean responseLenMean, and standard deviation responseLenStddev
func (d *BaseDataset) getRandomResponseLenByGaussian(maxLen int) int {
	for {
		val := d.random.RandomNorm(responseLenMean, responseLenStddev)
		if val >= 1 && val <= float64(maxLen) {
			return int(math.Round(val))
		}
		// else reject and resample
	}
}

// generatePresetRandomTokens generates random tokens for the required number of tokens,
// select randomly a sentence from chatCompletionFakeResponses,
// if number of tokens is lower than required - select another sentence,
// continue until the required number of tokens is achieved,
// returned exactly <numOfTokens> tokens
func (d BaseDataset) generatePresetRandomTokens(numOfTokens int) []string {
	allTokens := make([]string, 0)

	for len(allTokens) < numOfTokens {
		index := d.random.RandomInt(0, len(completionFakeResponses)-1)
		tokens := common.Tokenize(completionFakeResponses[index])
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
