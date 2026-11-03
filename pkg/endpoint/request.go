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

package endpoint

import (
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	"github.com/valyala/fasthttp"
)

type requestBuilder interface {
	Unmarshal(data []byte) error
	Validate(toolsValidator *ToolsValidator) *api.Error
	BuildRequestContext(runtime Runtime, channel common.Channel[*ResponseInfo], choiceIdx int, doneFn func()) RequestContext
	AsString() string
	createResponseContext(reqCtx RequestContext, displayModel string, responseTokens *api.Tokenized,
		finishReason *string, usageData *api.Usage, sendUsageData bool, logprobs *int,
		toolCalls []api.ToolCall, mmEncoderOnlyMode bool) ResponseContext
	// Split returns one or more processing-form Requests, each carrying a single
	// prompt with a unique RequestID. For request types whose wire form is always
	// a single prompt (chat, generation, responses, post-split text completions),
	// the implementation is trivial: return the receiver wrapped in a one-element
	// slice. Only TextCompletionsParsedRequest does real work here.
	Split() []Request
}

// RenderableRequest is implemented by the request types reachable from the
// /v1/{chat/,}completions/render, /v1/responses/render, /v1/messages/render,
// and /v1/messages/count_tokens endpoints (ChatCompletionsRequest,
// TextCompletionsParsedRequest, ResponsesRenderRequest, and MessagesRequest).
// It lets the HTTP layer parse + render
// without going through the worker pipeline.
type RenderableRequest interface {
	Request
	// ValidateBody checks that the unmarshalled body matches the endpoint's
	// expected shape.
	ValidateBody() *api.Error
	// Render tokenizes the request and returns the tokens (one slice per
	// prompt; chat completions always returns a single slice) and any
	// mm_features produced by the tokenizer. mmPayload is passed to
	// Tokenizer.RenderMessages.
	Render(t tokenizer.Tokenizer, mmPayload string) ([][]uint32, *api.RenderMMFeatures, error)
}

type Request interface {
	requestBuilder
	api.Request
}

// RequestContext is a request as it moves through worker processing.
type RequestContext interface {
	Request() Request
	StartProcessingTime() time.Time
	tokenize() *api.Error
	kvCacheOnRequestStart(minHitRate float64) (stats metrics.PrefixCacheQueried, started bool, serverError *api.Error)
	KVCacheOnRequestEnd()
	createToolCalls() ([]api.ToolCall, int, string, error)
	createStructuredOutput() (*api.Tokenized, string, error)
	HandleRequest() (ResponseContext, *api.Error)
	ResponseChannel() common.Channel[*ResponseInfo]
	tokenizedPromptForEcho() (*api.Tokenized, error)
	encode() ([]uint32, []string, *api.RenderMMFeatures, error)
	ChoiceIndex() int
	SignalDone()
}

type baseRequestContext struct {
	RequestContext
	runtime         Runtime
	logger          logr.Logger
	startProcessing time.Time
	respChannel     common.Channel[*ResponseInfo]
	idx             int
	doneFn          func()
}

func newBaseRequestContext(runtime Runtime, channel common.Channel[*ResponseInfo], choiceIdx int, doneFn func()) baseRequestContext {
	return baseRequestContext{
		runtime:         runtime,
		logger:          runtime.Logger(),
		startProcessing: time.Now(),
		respChannel:     channel,
		idx:             choiceIdx,
		doneFn:          doneFn,
	}
}

func (b *baseRequestContext) ResponseChannel() common.Channel[*ResponseInfo] {
	return b.respChannel
}

func (b *baseRequestContext) ChoiceIndex() int {
	return b.idx
}

func (b *baseRequestContext) SignalDone() {
	if b.doneFn != nil {
		b.doneFn()
	}
}

func (b *baseRequestContext) StartProcessingTime() time.Time {
	return b.startProcessing
}

func (b *baseRequestContext) tokenize() *api.Error {
	req := b.Request()

	if tokens := req.TokenizedPrompt(); tokens == nil {
		// the prompt is still not tokenized - tokenize now
		tokens, textTokens, mmFeatures, err := b.encode()
		if err != nil {
			b.logger.Error(err, "failed to tokenize")
			serverErr := api.NewError("Failed to tokenize, "+err.Error(), fasthttp.StatusInternalServerError, nil)
			return &serverErr
		}

		req.SetTokenizedPrompt(&api.Tokenized{
			Tokens:  tokens,
			Strings: textTokens,
		})

		req.SetMMFeatures(mmFeatures)
	}

	if b.runtime.Config().Mode == common.ModeEcho {
		// in echo mode need to calculate which part of input will be sent back,
		// e.g. in /chat/completions we send back only the last message's content
		echoTokenized, err := b.tokenizedPromptForEcho()
		if err != nil {
			b.logger.Error(err, "failed to tokenize prompt part for echo mode")
			serverErr := api.NewError("Failed to tokenize prompt part for echo mode, "+err.Error(), fasthttp.StatusInternalServerError, nil)
			return &serverErr
		}

		req.SetTokenizedPromptForEcho(echoTokenized)
	}

	return nil
}

// validate context window and other token-count-dependent request constraints,
// which can only be checked once the prompt is tokenized
func (b *baseRequestContext) validateTokenizedRequest() (string, int) {
	req := b.Request()
	promptTokens := getNumberOfPromptTokens(req)
	maxModelLen := b.runtime.Config().MaxModelLen

	if !common.ValidateContextWindow(promptTokens, maxModelLen) {
		message := fmt.Sprintf("This model's maximum context length is %d tokens. However, you requested %d tokens in the messages. Please reduce the length of the messages",
			maxModelLen, promptTokens)
		return message, fasthttp.StatusBadRequest
	}

	effectiveMaxTokens, _ := api.ResponseMaxLen(req, maxModelLen)
	if minTokens := req.GetMinTokens(); minTokens != nil && *minTokens > int64(effectiveMaxTokens) {
		message := fmt.Sprintf("min_tokens must be less than or equal to max_tokens=%d, got %d.",
			effectiveMaxTokens, *minTokens)
		return message, fasthttp.StatusBadRequest
	}

	return "", fasthttp.StatusOK
}

func (reqCtx *baseRequestContext) HandleRequest() (ResponseContext, *api.Error) {
	req := reqCtx.Request()
	dispModel := req.GetDisplayedModel()

	reqCtx.runtime.RequestStarted(req)

	if err := reqCtx.tokenize(); err != nil {
		return nil, err
	}

	if errMsg, errCode := reqCtx.validateTokenizedRequest(); errMsg != "" {
		oaiServerError := api.NewError(errMsg, errCode, nil)
		return nil, &oaiServerError
	}

	// A request below its cache hit threshold is not added to the KV cache, so it
	// does not raise the hit rate of the other choices of its prompt.
	prefixCacheStats, inKVCache, oaiServerError := reqCtx.kvCacheOnRequestStart(reqCtx.cacheHitThreshold(req))
	if oaiServerError != nil {
		return nil, oaiServerError
	}

	var finishReason string
	// The cache threshold finish reason header forces a cache_threshold finish reason
	if req.CacheThresholdFinishReason() || !inKVCache {
		finishReason = common.CacheThresholdFinishReason

		numOfInputTokens := getNumberOfPromptTokens(req)
		usageData := api.Usage{
			PromptTokens:     numOfInputTokens,
			CompletionTokens: 0,
			TotalTokens:      numOfInputTokens,
		}
		var logprobs *int
		if !req.IsStream() {
			logprobs = req.GetLogprobs()
		}
		sendUsageData := !req.IsStream() || req.IncludeUsage()
		respCtx := req.createResponseContext(reqCtx, dispModel, &api.Tokenized{},
			&finishReason, &usageData, sendUsageData, logprobs, nil, reqCtx.runtime.Config().MMEncoderOnly)
		return respCtx, nil
	}

	var responseTokens *api.Tokenized
	toolCalls, completionTokens, finishReason, err := reqCtx.createToolCalls()
	if toolCalls == nil && err == nil {
		// Either no tool calls were defined, or we randomly chose not to create tool calls,
		// so we generate a response text.
		responseTokens, finishReason, err = reqCtx.createStructuredOutput()
		if responseTokens == nil && err == nil {
			responseTokens, finishReason, err = reqCtx.runtime.GetResponseTokens(req)
		}
		if err == nil {
			completionTokens += responseTokens.Length()
		}
	}
	if err != nil {
		prefix := "failed to create response for " + req.AsString() + " "
		reqCtx.logger.Error(err, prefix)
		oaiServerError := api.NewError(prefix+err.Error(), fasthttp.StatusInternalServerError, nil)
		return nil, &oaiServerError
	}

	numOfInputTokens := getNumberOfPromptTokens(req)
	usageData := api.Usage{
		PromptTokens:     numOfInputTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      numOfInputTokens + completionTokens,
		PromptTokensDetails: &api.PromptTokensDetails{
			CachedTokens: prefixCacheStats.CachedPromptTokens,
		},
	}

	// Extract logprob data from request (unified approach)
	var logprobs *int
	if toolCalls == nil {
		logprobs = req.GetLogprobs()
	}

	sendUsageData := true
	if req.IsStream() {
		sendUsageData = req.IncludeUsage()
	} else if req.IsDoRemoteDecode() {
		// in case this is prefill pod processing, return special finish reason
		finishReason = common.RemoteDecodeFinishReason
	}

	respCtx := req.createResponseContext(reqCtx, dispModel, responseTokens, &finishReason,
		&usageData, sendUsageData, logprobs, toolCalls, reqCtx.runtime.Config().MMEncoderOnly)

	return respCtx, nil
}

// cacheHitThreshold returns the minimum fraction of req's prompt that must be in
// the KV cache: the request's cache_hit_threshold, or the global one when the
// request has none. 0 means no threshold.
func (reqCtx *baseRequestContext) cacheHitThreshold(req api.Request) float64 {
	if reqThreshold := req.GetCacheHitThreshold(); reqThreshold != nil && *reqThreshold >= 0 && *reqThreshold <= 1 {
		return *reqThreshold
	}
	return reqCtx.runtime.Config().GlobalCacheHitThreshold
}

// kvCacheRequest keys a sub-request in the KV cache by its choice: the n
// choices of one request share its request ID but hold their blocks separately.
type kvCacheRequest struct {
	api.Request
	id string
}

func (r kvCacheRequest) GetRequestID() string { return r.id }

func (reqCtx *baseRequestContext) kvCacheID() string {
	return fmt.Sprintf("%s-%d", reqCtx.Request().GetRequestID(), reqCtx.idx)
}

func (reqCtx *baseRequestContext) kvCacheOnRequestStart(minHitRate float64) (stat metrics.PrefixCacheQueried,
	started bool, oaiServerError *api.Error) {
	return reqCtx.runtime.KVCacheOnRequestStart(kvCacheRequest{Request: reqCtx.Request(), id: reqCtx.kvCacheID()},
		minHitRate)
}

func (reqCtx *baseRequestContext) KVCacheOnRequestEnd() {
	reqCtx.runtime.KVCacheOnRequestEnd(reqCtx.kvCacheID())
}
