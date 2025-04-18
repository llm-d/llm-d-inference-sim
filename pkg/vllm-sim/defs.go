/*
Copyright 2025 The vLLM-Sim Authors.

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

// Definitions of all sturctures used by vLLM simultor
// Contains the main simulator class and all definitions related to request/response for all supported APIs
package vllmsim

import (
	"sync"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
)

const (
	modeRandom        = "random"
	modeEcho          = "echo"
	chatComplIdPrefix = "chatcmpl-"
	stopFinishReason  = "stop"
	roleAssistant     = "assistant"
	roleUser          = "user"
)

// VllmSimulator simulates vLLM server supporting OpenAI API
type VllmSimulator struct {
	// logger is used for information and errors logging
	logger logr.Logger
	// timeToFirstToken time before the first token will be returned, in milliseconds
	timeToFirstToken int
	// interTokenLatency time between generated tokens, in milliseconds
	interTokenLatency int
	// port defines on which port the simulator runs
	port int
	// mode defenes the simulator response generation mode, valid values: echo, random
	mode string
	// model defines the current base model name
	model string
	// loraAdaptors contains list of LoRA available adaptors
	loraAdaptors sync.Map
	// maxLoras defines maximum number of loaded loras
	maxLoras int
	// maxLoras defines maximum number of loras to store in CPU memory
	maxCpuLoras int
	// runningLoras is a collection of running loras, key of lora's name, value is number of requests using this lora
	runningLoras sync.Map
	// waitingLoras will represent collection of loras defined in requests in the queue - Not implemented yet
	//waitingLoras sync.Map
	nRunningReqs int64
	// loraInfo is prometheus gauge
	loraInfo *prometheus.GaugeVec
	// runningRequests is prometheus gauge
	runningRequests *prometheus.GaugeVec
	// waitingRequests is prometheus gauge for number of queued requests
	waitingRequests *prometheus.GaugeVec
	// kvCacheUsagePercentage is prometheus gauge
	kvCacheUsagePercentage *prometheus.GaugeVec
}

// baseResponseChoice contains base completion response's choice related information
type baseResponseChoice struct {
	// Index defines completion response choise Index
	Index int `json:"index"`
	// FinishReason defines finish reason for response or for chunks, for not last chinks is defined as null
	FinishReason *string `json:"finish_reason"`
}

// baseCompletionResponse contains base completion response related information
type baseCompletionResponse struct {
	// ID defines the response ID
	ID string `json:"id"`
	// Created defines the response creation timestamp
	Created int64 `json:"created"`
	// Model defines the Model name for current request
	Model string `json:"model"`
}

// completionResponse interface representing both completion response types (text and chat)
type completionResponse interface {
}

// baseCompletionRequest contains base completion request related information
type baseCompletionRequest struct {
	// Stream is a boolean value, defines whether response should be sent as a Stream
	Stream bool `json:"stream"`
	// Model defines Model name to use for "inferense", could be base Model name or one of available LoRA adapters
	Model string `json:"model"`
}

func (b *baseCompletionRequest) isStream() bool {
	return b.Stream
}

func (b *baseCompletionRequest) getModel() string {
	return b.Model
}

// completionRequest interface representing both completion request types (text and chat)
type completionRequest interface {
	// createResponseText creates and returns response payload based on this request
	createResponseText(mode string) string
	// isStream returns boolean that defines is response should be streamed
	isStream() bool
	// getModel returns model name as defined in the request
	getModel() string
}

// v1/chat/completion
// message defines vLLM chat completion message
type message struct {
	// Role is the message Role, optional values are 'user', 'assistant', ...
	Role string `json:"role,omitempty"`
	// Content defines text of this message
	Content string `json:"content,omitempty"`
}

// chatCompletionRequest defines structure of /chat/completion request
type chatCompletionRequest struct {
	baseCompletionRequest
	// Messages list of request's Messages
	Messages []message `json:"messages"`
}

// chatCompletionResponse defines structure of /chat/completion response
type chatCompletionResponse struct {
	baseCompletionResponse
	// Choices list of Choices of the response, according of OpenAI API
	Choices []chatRespChoice `json:"choices"`
}

// chatRespChoice represents a single chat completion response choise
type chatRespChoice struct {
	baseResponseChoice
	// Message contains choice's Message
	Message message `json:"message"`
}

// v1/completion
// textCompletionRequest defines structure of /completion request
type textCompletionRequest struct {
	baseCompletionRequest
	// Prompt defines request's content
	Prompt string `json:"prompt"`
	// TODO - do we want to support max tokens?
	// MaxTokens is a maximum number of tokens in response
	MaxTokens int `json:"max_tokens"`
}

// textCompletionResponse defines structure of /completion response
type textCompletionResponse struct {
	baseCompletionResponse
	// Choices list of Choices of the response, according of OpenAI API
	Choices []textRespChoice `json:"choices"`
}

// textRespChoice represents a single text completion response choise
type textRespChoice struct {
	baseResponseChoice
	// Text defines request's content
	Text string `json:"text"`
}

// completionRespChunk is an interface that defines a single response chunk
type completionRespChunk interface{}

// chatCompletionRespChunk is a single chat completion response chunk
type chatCompletionRespChunk struct {
	baseCompletionResponse
	// Choices list of Choices of the response, according of OpenAI API
	Choices []chatRespChunkChoice `json:"choices"`
}

// chatRespChunkChoice represents a single chat completion response choise in case of streaming
type chatRespChunkChoice struct {
	baseResponseChoice
	// Delta is a content of the chunk
	Delta message `json:"delta"`
}

// createResponseText creates response text for the given chat completion request and mode
func (req chatCompletionRequest) createResponseText(mode string) string {
	if mode == modeEcho {
		return req.getLastUserMsg()
	}
	return getRandomResponseText()
}

// createResponseText creates response text for the given text completion request and mode
func (req textCompletionRequest) createResponseText(mode string) string {
	if mode == modeEcho {
		return req.Prompt
	} else {
		return getRandomResponseText()
	}
}

// getLastUserMsg returns last message from this request's messages with user role,
// if does not exist - returns an empty string
func (r *chatCompletionRequest) getLastUserMsg() string {
	for i := len(r.Messages) - 1; i >= 0; i-- {
		if r.Messages[i].Role == roleUser {
			return r.Messages[i].Content
		}
	}

	return ""
}

// completionError defines structure of error returned by completion request
type completionError struct {
	// Object is a type of this Object, "error"
	Object string `json:"object"`
	// Message is an error Message
	Message string `json:"message"`
	// Type is a type of the error
	Type string `json:"type"`
	// Params is the error's parameters
	Param *string `json:"param"`
	// Code is http status Code
	Code int `json:"code"`
}

type loadLoraRequest struct {
	LoraName string `json:"lora_name"`
	LoraPath string `json:"lora_path"`
}

type unloadLoraRequest struct {
	LoraName string `json:"lora_name"`
}

func (s *VllmSimulator) getLoras() []string {

	loras := make([]string, 0)

	s.loraAdaptors.Range(func(key, _ any) bool {
		if lora, ok := key.(string); ok {
			loras = append(loras, lora)
		}
		return true
	})

	return loras
}

func (s *VllmSimulator) addLora(lora string) {
	s.loraAdaptors.Store(lora, "")
}

func (s *VllmSimulator) removeLora(lora string) {
	s.loraAdaptors.Delete(lora)
}
