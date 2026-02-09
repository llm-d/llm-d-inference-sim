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
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	_ "github.com/mattn/go-sqlite3"
)

// list of responses to use in random mode for completion requests
var completionFakeResponses = []string{
	`Testing@, #testing 1$ ,2%,3^, [4&*5], 6~, 7-_ + (8 : 9) / \ < > . `,
	`Testing, testing 1,2,3. `,
	`I am fine, how are you today? `,
	`I am your AI assistant, how can I help you today? `,
	`Today is a nice sunny day. `,
	`The temperature here is twenty-five degrees centigrade. `,
	`Today it is partially cloudy and raining. `,
	`To be or not to be that is the question. `,
	`Alas, poor Yorick! I knew him, Horatio: A fellow of infinite jest `,
	`The rest is silence. `,
	`Give a man a fish and you feed him for a day; teach a man to fish and you feed him for a lifetime `,
}

type Dataset interface {
	// Close closes the dataset
	Close() error
	// GetResponseTokens returns ressponse tokens for the given request
	GetResponseTokens(req openaiserverapi.Request) (*openaiserverapi.Tokenized, string, error)
}

type EchoDataset struct{}

// GetResponseTokens returns response tokens when simulator is in echo mode
// for /completion request the prompt is returned
// for /chat/completion request the last user message is returned (if there is no user messages, last message is used)
// if max-tokens is defined in the request and response's length is >= it value, finish reason is set to LENGTH,
// otherwise finish reason is STOP
func (ed *EchoDataset) GetResponseTokens(req openaiserverapi.Request) (*openaiserverapi.Tokenized, string, error) {
	tokens := req.TokenizedPrompt()
	maxTokens := req.GetMaxCompletionTokens()
	return tokens, common.FinishReason(maxTokens, len(tokens.Tokens)), nil
}

func (ed *EchoDataset) Close() error {
	return nil
}

type DefaultDataset struct {
	logger             logr.Logger
	maxModelLen        int
	random             *common.Random
	histogramHelper    *histogramHelper
	tokenizedResponses []openaiserverapi.Tokenized
}

func (d *DefaultDataset) Init(ctx context.Context, logger logr.Logger, random *common.Random, maxModelLen int,
	tokenizer tokenizer.Tokenizer) error {
	d.logger = logger
	d.maxModelLen = maxModelLen
	d.random = random
	d.histogramHelper = newHistogramHelper(d.random)

	d.tokenizedResponses = make([]openaiserverapi.Tokenized, len(completionFakeResponses))
	for i, text := range completionFakeResponses {
		tokens, textTokens, err := tokenizer.Encode(text, "")
		if err != nil {
			logger.Error(err, "failed to tokenize")
			return err
		}
		d.tokenizedResponses[i] = openaiserverapi.Tokenized{
			Tokens:  tokens,
			Strings: textTokens,
		}
	}

	return nil
}

func (d *DefaultDataset) Close() error {
	return nil
}

// GetResponseTokens returns response tokens and finishReason for the given request
func (d *DefaultDataset) GetResponseTokens(req openaiserverapi.Request) (*openaiserverapi.Tokenized, string, error) {
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
		// if response should be created with maximum number of tokens - finish reason will be 'length'
		finishReason = common.LengthFinishReason
	}

	respTokens := d.generatePresetRandomTokens(numOfRespTokens)
	return &respTokens, finishReason, nil
}

// calculateResponseMaxLen - calculates maximum length of a response to be randomly chosen from the dataset
// for the given request and the simulator configuration.
// If max-tokens/max-completion-tokens is defined - use it,
// otherwise use <model content window size> - <number of input tokens>
// boolean returned value defines whether max tokens number was passed in the request
func (d *DefaultDataset) calculateResponseMaxLen(req openaiserverapi.Request) (int, bool) {
	maxTokens := req.GetMaxCompletionTokens()

	if maxTokens != nil {
		return int(*maxTokens), true
	}

	return d.maxModelLen - req.TokenizedPrompt().Length(), false
}

// getRandomResponseLenByDistribution returns int in range [1, responseLenMax]
// numbers are chosen according a gaussian distribution with mean responseLenMean, and standard deviation responseLenStddev
func (d *DefaultDataset) getRandomResponseLenByGaussian(maxLen int) int {
	for {
		val := d.random.RandomNorm(responseLenMean, responseLenStddev)
		if val >= 1 && val <= float64(maxLen) {
			return int(math.Round(val))
		}
		// else reject and resample
	}
}

// generatePresetRandomTokens generates random tokens for the required number of tokens,
// select randomly a sentence from completionFakeResponses,
// if number of tokens is lower than required - select another sentence,
// continue until the required number of tokens is achieved,
// returned exactly <numOfTokens> tokens
func (d DefaultDataset) generatePresetRandomTokens(numOfTokens int) openaiserverapi.Tokenized {
	result := openaiserverapi.Tokenized{
		Tokens:  make([]uint32, 0),
		Strings: make([]string, 0),
	}

	for len(result.Tokens) < numOfTokens {
		index := d.random.RandomInt(0, len(completionFakeResponses)-1)
		tokens := d.tokenizedResponses[index].Tokens
		strTokens := d.tokenizedResponses[index].Strings
		remaining := numOfTokens - len(result.Tokens)

		if len(tokens) > remaining {
			// there is too many tokens, append only the relevant part
			tokens = tokens[:remaining]
			strTokens = strTokens[:remaining]
		}

		result.Tokens = append(result.Tokens, tokens...)
		result.Strings = append(result.Strings, strTokens...)
	}

	return result
}
