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
	"fmt"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

type requestBuilder interface {
	Unmarshal(data []byte) error
	validate(toolsValidator *toolsValidator) (string, int)
	buildRequestContext(simCtx *SimContext, channel chan *ResponseInfo) requestContext
	AsString() string
	createResponseContext(reqCtx requestContext, displayModel string, responseTokens *openaiserverapi.Tokenized, finishReason *string,
		usageData *openaiserverapi.Usage, sendUsageData bool, logprobs *int, toolCalls []openaiserverapi.ToolCall) ResponseContext
}

type Request interface {
	requestBuilder
	openaiserverapi.Request
}

type requestContext interface {
	request() Request
	startProcessingTime() time.Time
	tokenize() *openaiserverapi.Error
	kvCacheOnRequestStart() (hitRate float64, serverError *openaiserverapi.Error)
	kvCacheOnRequestEnd()
	createToolCalls() ([]openaiserverapi.ToolCall, int, string, error)
	handleRequest() (ResponseContext, *openaiserverapi.Error)
	responseChannel() chan *ResponseInfo
	tokenizedPromptForEcho() (*openaiserverapi.Tokenized, error)
}

type baseRequestContext struct {
	requestContext
	sim             *SimContext
	startProcessing time.Time
	respChannel     chan *ResponseInfo
}

func newBaseRequestContext(simCtx *SimContext, channel chan *ResponseInfo) baseRequestContext {
	return baseRequestContext{
		sim:             simCtx,
		startProcessing: time.Now(),
		respChannel:     channel,
	}
}

func (b *baseRequestContext) responseChannel() chan *ResponseInfo {
	return b.respChannel
}

func (b *baseRequestContext) startProcessingTime() time.Time {
	return b.startProcessing
}

func (b *baseRequestContext) tokenize() *openaiserverapi.Error {
	req := b.request()

	if tokens := req.TokenizedPrompt(); tokens == nil {
		prompt := req.GetPrompt()
		tokens, textTokens, err := b.sim.Tokenizer.Encode(prompt, "")
		if err != nil {
			b.sim.logger.Error(err, "failed to tokenize")
			serverErr := openaiserverapi.NewError("Failed to tokenize, "+err.Error(), fasthttp.StatusInternalServerError, nil)
			return &serverErr
		}

		req.SetTokenizedPrompt(&openaiserverapi.Tokenized{
			Tokens:  tokens,
			Strings: textTokens,
		})
	}

	if b.sim.Config.Mode == common.ModeEcho {
		// in echo mode need to calculate which part of input will be sent back,
		// e.g. in /chat/completions we send back only the last message content
		echoTokenized, err := b.tokenizedPromptForEcho()
		if err != nil {
			b.sim.logger.Error(err, "failed to tokenize prompt part for echo mode")
			serverErr := openaiserverapi.NewError("Failed to tokenize prompt part for echo mode, "+err.Error(), fasthttp.StatusInternalServerError, nil)
			return &serverErr
		}

		req.SetTokenizedPromptForEcho(echoTokenized)
	}

	return nil
}

// validate context window constraints
func (b *baseRequestContext) validateContextWindow() (string, int) {
	promptTokens := getNumberOfPromptTokens(b.request())
	completionTokens := b.request().GetMaxCompletionTokens()
	isValid, actualCompletionTokens, totalTokens := common.ValidateContextWindow(promptTokens, completionTokens,
		b.sim.Config.MaxModelLen)
	if !isValid {
		message := fmt.Sprintf("This model's maximum context length is %d tokens. However, you requested %d tokens (%d in the messages, %d in the completion). Please reduce the length of the messages or completion",
			b.sim.Config.MaxModelLen, totalTokens, promptTokens, actualCompletionTokens)
		return message, fasthttp.StatusBadRequest
	}
	return "", fasthttp.StatusOK
}

func (reqCtx *baseRequestContext) handleRequest() (ResponseContext, *openaiserverapi.Error) {
	req := reqCtx.request()
	model := req.GetModel()

	// increment running requests count
	common.WriteToChannel(reqCtx.sim.metrics.runReqChan, 1, reqCtx.sim.logger, "metrics.runReqChan")

	if reqCtx.sim.isLora(model) {
		// update loraInfo metric to reflect that
		// the request has changed its status from waiting to running
		common.WriteToChannel(reqCtx.sim.metrics.lorasChan, loraUsage{model, runningUsageState}, reqCtx.sim.logger,
			"metrics.lorasChan")
	}

	if err := reqCtx.tokenize(); err != nil {
		return nil, err
	}

	if errMsg, errCode := reqCtx.validateContextWindow(); errMsg != "" {
		oaiServerError := openaiserverapi.NewError(errMsg, errCode, nil)
		return nil, &oaiServerError
	}

	hitRate, oaiServerError := reqCtx.kvCacheOnRequestStart()
	if oaiServerError != nil {
		return nil, oaiServerError
	}

	var finishReason string
	if reqCtx.shouldReturnCacheThresholdFinishReason(req, hitRate) {
		finishReason = common.CacheThresholdFinishReason

		numOfInputTokens := getNumberOfPromptTokens(req)
		usageData := openaiserverapi.Usage{
			PromptTokens:     numOfInputTokens,
			CompletionTokens: 0,
			TotalTokens:      numOfInputTokens,
		}
		var logprobs *int
		if !req.IsStream() {
			logprobs = req.GetLogprobs()
		}
		sendUsageData := !req.IsStream() || req.IncludeUsage()
		respCtx := req.createResponseContext(reqCtx, reqCtx.sim.getDisplayedModelName(model), &openaiserverapi.Tokenized{},
			&finishReason, &usageData, sendUsageData, logprobs, nil)
		return respCtx, nil
	}

	var responseTokens *openaiserverapi.Tokenized
	toolCalls, completionTokens, finishReason, err := reqCtx.createToolCalls()
	if toolCalls == nil && err == nil {
		// Either no tool calls were defined, or we randomly chose not to create tool calls,
		// so we generate a response text.
		responseTokens, finishReason, err = reqCtx.sim.dataset.GetResponseTokens(req)
		completionTokens += responseTokens.Length()
	}
	if err != nil {
		prefix := "failed to create response for " + req.AsString() + " "
		reqCtx.sim.logger.Error(err, prefix)
		oaiServerError := openaiserverapi.NewError(prefix+err.Error(), fasthttp.StatusInternalServerError, nil)
		return nil, &oaiServerError
	}

	numOfInputTokens := getNumberOfPromptTokens(req)
	usageData := openaiserverapi.Usage{
		PromptTokens:     numOfInputTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      numOfInputTokens + completionTokens,
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

	respCtx := req.createResponseContext(reqCtx, reqCtx.sim.getDisplayedModelName(model), responseTokens, &finishReason,
		&usageData, sendUsageData, logprobs, toolCalls)

	return respCtx, nil
}

func (reqCtx *baseRequestContext) shouldReturnCacheThresholdFinishReason(req openaiserverapi.Request, hitRate float64) bool {
	// Check for cache threshold finish reason header - this forces a cache_threshold finish reason
	if req.CacheThresholdFinishReason() {
		return true
	}
	// Check cache hit threshold if specified and KV cache is enabled
	// First, get cache hit info without modifying cache state
	if reqCtx.sim.Config.EnableKVCache {
		// Get cacheHitThreshold from request first, fall back to global cacheHitThreshold if not set
		var cacheHitThreshold *float64
		if reqThreshold := req.GetCacheHitThreshold(); reqThreshold != nil && *reqThreshold >= 0 && *reqThreshold <= 1 {
			cacheHitThreshold = reqThreshold
		} else if reqCtx.sim.Config.GlobalCacheHitThreshold > 0 {
			cacheHitThreshold = &reqCtx.sim.Config.GlobalCacheHitThreshold
		}

		if cacheHitThreshold != nil {
			// If hit rate is below threshold, return cache_threshold finish reason
			if hitRate < *cacheHitThreshold {
				return true
			}
		}
	}

	return false
}
