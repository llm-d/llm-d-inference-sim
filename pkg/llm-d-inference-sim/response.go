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
	"sync"

	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

type responseInfo struct {
	tokens   *openaiserverapi.Tokenized
	respCtx  responseContext
	err      *openaiserverapi.Error
	toolCall *openaiserverapi.ToolCall
}

type responseContext interface {
	createResponse(*openaiserverapi.Tokenized) openaiserverapi.CompletionResponse
	createUsageChunk() openaiserverapi.CompletionRespChunk
	createCompletionChunk(tokens []string, tool *openaiserverapi.ToolCall,
		role string, finishReason *string) openaiserverapi.CompletionRespChunk
	createFirstCompletionChunk() openaiserverapi.CompletionRespChunk
	requestContext() requestContext
	requestID() string
	usageData() *openaiserverapi.Usage
	displayModel() string
	doRemotePrefill() bool
	doRemoteDecode() bool
	numberCachedPromptTokens() int
	responseTokens() *openaiserverapi.Tokenized
	finishReason() *string
	sendUsageData() bool
	toolCalls() []openaiserverapi.ToolCall
	setCreationTime(int64)
	setWG(*sync.WaitGroup)
	done()
}

type baseResponseContext struct {
	// the corresponding request
	reqCtx requestContext
	// the ID of the request
	id string
	// creation time of the response
	creationTime int64
	// indicates whether do_remote_prefill field is true in the request
	remotePrefill bool
	// indicates whether do_remote_decode field is true in the request
	remoteDecode bool
	// the number of prompt tokens that are in the local KV Cache
	nCachedPromptTokens int
	// tokenized content to be sent in the response
	respTokens *openaiserverapi.Tokenized
	// display model name returned to the client and used in metrics. It is either the first alias
	// from --served-model-name (for a base-model request) or the LoRA adapter name (for a LoRA request)
	displayModelName string
	// a pointer to a string that represents finish reason, can be nil or stop or length, ...
	finishReasonPtr *string
	// usage (tokens statistics) for this response
	usage *openaiserverapi.Usage
	// indicates whether to send usage data in this response
	sendUsage bool
	// number of logprob options to include or nil if no logprobs needed
	logprobs *int
	// wait group of this response
	wg *sync.WaitGroup
}

func newBaseResponseContext(reqCtx requestContext, displayModel string, responseTokens *openaiserverapi.Tokenized,
	finishReason *string, usageData *openaiserverapi.Usage, sendUsageData bool, logprobs *int, id string,
	doRemotePrefill bool, doRemoteDecode bool, nCachedPromptTokens int) baseResponseContext {
	return baseResponseContext{
		reqCtx:              reqCtx,
		respTokens:          responseTokens,
		displayModelName:    displayModel,
		finishReasonPtr:     finishReason,
		usage:               usageData,
		sendUsage:           sendUsageData,
		logprobs:            logprobs,
		id:                  id,
		remotePrefill:       doRemotePrefill,
		remoteDecode:        doRemoteDecode,
		nCachedPromptTokens: nCachedPromptTokens,
	}
}

func (respCtx *baseResponseContext) requestContext() requestContext {
	return respCtx.reqCtx
}
func (respCtx *baseResponseContext) usageData() *openaiserverapi.Usage {
	return respCtx.usage
}
func (respCtx *baseResponseContext) displayModel() string {
	return respCtx.displayModelName
}
func (respCtx *baseResponseContext) requestID() string {
	return respCtx.id
}
func (respCtx *baseResponseContext) doRemotePrefill() bool {
	return respCtx.remotePrefill
}
func (respCtx *baseResponseContext) doRemoteDecode() bool {
	return respCtx.remoteDecode
}
func (respCtx *baseResponseContext) numberCachedPromptTokens() int {
	return respCtx.nCachedPromptTokens
}
func (respCtx *baseResponseContext) responseTokens() *openaiserverapi.Tokenized {
	return respCtx.respTokens
}
func (respCtx *baseResponseContext) finishReason() *string {
	return respCtx.finishReasonPtr
}
func (respCtx *baseResponseContext) sendUsageData() bool {
	return respCtx.sendUsage
}
func (respCtx *baseResponseContext) setCreationTime(creationTime int64) {
	respCtx.creationTime = creationTime
}

func (b *baseResponseContext) done() {
	b.wg.Done()
}
func (b *baseResponseContext) setWG(wg *sync.WaitGroup) {
	b.wg = wg
}
