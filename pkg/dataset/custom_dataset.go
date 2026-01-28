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
	"crypto/sha256"
	"encoding/hex"
	"errors"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

type CustomDataset struct {
	DefaultDataset
	sqliteHelper *sqliteHelper
}

func (d *CustomDataset) Init(ctx context.Context, logger logr.Logger, random *common.Random,
	path string, useInMemory bool, maxModelLen int) error {
	if err := d.DefaultDataset.Init(ctx, logger, random, maxModelLen); err != nil {
		return err
	}
	if path == "" {
		return errors.New("no dataset path provided")
	}

	d.sqliteHelper = newSqliteHelper(logger)
	d.logger.V(logging.INFO).Info("Using dataset from", "path", path)
	return d.sqliteHelper.connectToDB(path, useInMemory)
}

func (d *CustomDataset) getPromptHash(req openaiserverapi.Request) []byte {
	hashArray := sha256.Sum256([]byte(req.GetFullPrompt()))
	return hashArray[:]
}

func (d *CustomDataset) getPromptHashHex(hashBytes []byte) string {
	return hex.EncodeToString(hashBytes)
}

// categorizeResponses receives list of responses tokens and maximum response length
// categorize responses to three collections:
// - shorter or equal length to maxLen
// - exact maxLen length
// - longer than maxLen
func (d *CustomDataset) categorizeResponses(responses [][]string, maxLen int) (shorterOrEqLen [][]string, equalLen [][]string, longerLen [][]string) {
	for _, respTokens := range responses {
		switch {
		case len(respTokens) == maxLen:
			shorterOrEqLen = append(shorterOrEqLen, respTokens)
			equalLen = append(equalLen, respTokens)
		case len(respTokens) < maxLen:
			shorterOrEqLen = append(shorterOrEqLen, respTokens)
		default:
			longerLen = append(longerLen, respTokens)
		}
	}
	return
}

// getRandomResponse returns a randomly selected element from the given array, array is not empty
func (d *CustomDataset) getRandomResponse(responses [][]string) []string {
	return responses[d.random.RandomInt(0, len(responses)-1)]
}

// GetTokens returns tokens and finishReason for the given request:
// Calculate maximum length of response (basedon max-tokens or max-completions-tokens or model-len)
// If dataset contains responses for the given prompt, and there are responses with length <=
// max response length - use random one from the list,
// otherwise select random one from the longer responses and trim it as required
// If no responses were found in the dataset for the given prompt,
// get random record fromn the dataset with response length equal or lower than max response length,
// if there is no records shorter/equal to max length - get random response from the dataset
// and trim it to the required length
// if ignore_eos=true the response always will have the max response len tokens, missing tokens
// are randomly selected from the hard-coded collection
func (d *CustomDataset) GetTokens(req openaiserverapi.Request) ([]string, string, error) {
	maxResponseLen, _ := d.calculateResponseMaxLen(req)
	responseTokens := []string{}

	// get all records for the hashes prompt
	promptHash := d.getPromptHash(req)
	promptHashHex := d.getPromptHashHex(promptHash)
	responses, err := d.sqliteHelper.getResponsesForPrompt(promptHashHex)
	if err != nil {
		return responseTokens, "", err
	}

	if len(responses) > 0 {
		// has responses for the given request
		d.logger.V(logging.TRACE).Info("Reponses were found in the dataset for the request's prompt")
		shorterOrEqLenResponses, equalLenResponses, longerLenResponses := d.categorizeResponses(responses, maxResponseLen)

		if req.GetIgnoreEOS() {
			// must return response with exactly calculated max length
			switch {
			case len(equalLenResponses) > 0:
				// has responses with required length - return randomly selected response
				responseTokens = d.getRandomResponse(equalLenResponses)
			case len(longerLenResponses) > 0:
				// has responses longer than required - return randomly selected trimmed response
				responseTokens = d.getRandomResponse(longerLenResponses)[:maxResponseLen]
			default:
				// all responses are shorter than required, select randomly and pad with random tokens
				responseTokens = d.getRandomResponse(shorterOrEqLenResponses)
				responseTokens = append(responseTokens, d.generatePresetRandomTokens(maxResponseLen-len(responseTokens))...)
			}
		} else {
			// has responses for the request, return response shorter or equal to the maxReponsesLen
			// finishReason = common.LengthFinishReason
			if len(shorterOrEqLenResponses) > 0 {
				// has responses shorter or equal length than required - return randomly selected response
				responseTokens = d.getRandomResponse(shorterOrEqLenResponses)
			} else {
				// all responses are longer than required, use randomly selected trimmed response
				responseTokens = d.getRandomResponse(longerLenResponses)[:maxResponseLen]
			}
		}
	} else {
		// no responses for the given request
		d.logger.V(logging.TRACE).Info("No responses in the dataset for the request's prompt")
		// try to find a random response with number of tokens <= tokens limit
		randomResponses, err := d.sqliteHelper.getResponsesForLen(maxResponseLen, req.GetIgnoreEOS())
		if err != nil {
			return responseTokens, "", err
		}
		if len(randomResponses) == 0 {
			// failed to get response with number of tokens <= tokensLimit, get response with any number of tokens
			randomResponses, err = d.sqliteHelper.getRandomResponse()
			if err != nil {
				return responseTokens, "", err
			}
			if len(randomResponses) == 0 {
				// shouldn't happen
				return responseTokens, "", errors.New("Dataset is empty")
			}
		}
		// if response has too much tokens, trim it
		if len(randomResponses[0]) > maxResponseLen {
			responseTokens = randomResponses[0][:maxResponseLen]
		} else {
			responseTokens = randomResponses[0]
			if req.GetIgnoreEOS() {
				responseTokens = append(responseTokens, d.generatePresetRandomTokens(maxResponseLen-len(responseTokens))...)
			}
		}
	}

	finishReason := common.StopFinishReason
	if len(responseTokens) == maxResponseLen {
		finishReason = common.LengthFinishReason
	}

	return responseTokens, finishReason, nil
}
