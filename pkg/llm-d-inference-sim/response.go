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

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

const (
	ResponseStatusCreated = "created"
	ResponseEndOfTokens   = "eot"
)

type ResponseInfo struct {
	Tokens    *api.Tokenized
	RespCtx   ResponseContext
	Err       *api.Error
	ToolCall  *api.ToolCall
	Status    string
	ChoiceIdx int
}

type ResponseContext interface {
	RequestContext() requestContext
	RequestID() string
	UsageData() *api.Usage
	DisplayModel() string
	doRemotePrefill() bool
	DoRemoteDecode() bool
	NumberCachedPromptTokens() int
	responseTokens() *api.Tokenized
	FinishReason() *string
	SendUsageData() bool
	ToolCalls() []api.ToolCall
	Instructions() *string
	CreationTime() int64
	SetCreationTime(int64)
	TopLogprobs() *int
	ECTransferParams() map[string]api.ECTransferParams
	SendImage() bool
	setWG(*sync.WaitGroup)
	Done()
}

type baseResponseContext struct {
	// the corresponding request
	reqCtx requestContext
	// the ID of the request
	id string
	// creation time of the response
	created int64
	// indicates whether do_remote_prefill field is true in the request
	remotePrefill bool
	// indicates whether do_remote_decode field is true in the request
	remoteDecode bool
	// the number of prompt tokens that are in the local KV Cache
	nCachedPromptTokens int
	// tokenized content to be sent in the response
	respTokens *api.Tokenized
	// display model name returned to the client and used in metrics. It is either the first alias
	// from --served-model-name (for a base-model request) or the LoRA adapter name (for a LoRA request)
	displayModelName string
	// a pointer to a string that represents finish reason, can be nil or stop or length, ...
	finishReasonPtr *string
	// usage (tokens statistics) for this response
	usage *api.Usage
	// indicates whether to send usage data in this response
	sendUsage bool
	// number of logprob options to include or nil if no logprobs needed
	nLogprobs *int
	// wait group of this response
	wg *sync.WaitGroup
}

func newBaseResponseContext(reqCtx requestContext, displayModel string, responseTokens *api.Tokenized,
	finishReason *string, usageData *api.Usage, sendUsageData bool, logprobs *int, id string,
	doRemotePrefill bool, doRemoteDecode bool, nCachedPromptTokens int) baseResponseContext {
	return baseResponseContext{
		reqCtx:              reqCtx,
		respTokens:          responseTokens,
		displayModelName:    displayModel,
		finishReasonPtr:     finishReason,
		usage:               usageData,
		sendUsage:           sendUsageData,
		nLogprobs:           logprobs,
		id:                  id,
		remotePrefill:       doRemotePrefill,
		remoteDecode:        doRemoteDecode,
		nCachedPromptTokens: nCachedPromptTokens,
	}
}

func (respCtx *baseResponseContext) RequestContext() requestContext {
	return respCtx.reqCtx
}
func (respCtx *baseResponseContext) UsageData() *api.Usage {
	return respCtx.usage
}
func (respCtx *baseResponseContext) DisplayModel() string {
	return respCtx.displayModelName
}
func (respCtx *baseResponseContext) RequestID() string {
	return respCtx.id
}
func (respCtx *baseResponseContext) doRemotePrefill() bool {
	return respCtx.remotePrefill
}
func (respCtx *baseResponseContext) DoRemoteDecode() bool {
	return respCtx.remoteDecode
}
func (respCtx *baseResponseContext) NumberCachedPromptTokens() int {
	return respCtx.nCachedPromptTokens
}
func (respCtx *baseResponseContext) responseTokens() *api.Tokenized {
	return respCtx.respTokens
}
func (respCtx *baseResponseContext) FinishReason() *string {
	return respCtx.finishReasonPtr
}
func (respCtx *baseResponseContext) SendUsageData() bool {
	return respCtx.sendUsage
}
func (respCtx *baseResponseContext) SetCreationTime(creationTime int64) {
	respCtx.created = creationTime
}
func (respCtx *baseResponseContext) CreationTime() int64 {
	return respCtx.created
}
func (respCtx *baseResponseContext) TopLogprobs() *int {
	return respCtx.nLogprobs
}

func (respCtx *baseResponseContext) Instructions() *string {
	return nil
}

func (respCtx *baseResponseContext) ECTransferParams() map[string]api.ECTransferParams {
	return nil
}

func (b *baseResponseContext) Done() {
	b.wg.Done()
}
func (b *baseResponseContext) setWG(wg *sync.WaitGroup) {
	b.wg = wg
}
func (b *baseResponseContext) SendImage() bool {
	return b.reqCtx.request().SendImage()
}

func respIsEmpty(respCtx ResponseContext) bool {
	tokens := respCtx.responseTokens()
	if tokens == nil {
		return respCtx.ToolCalls() == nil
	}
	return len(tokens.Tokens) == 0
}
