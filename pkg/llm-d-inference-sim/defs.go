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

// Definitions of structures and constants used by vLLM simulator
// Contains the main simulator class and the constants
package llmdinferencesim

import (
	"sync"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
)

const (
	modeRandom                = "random"
	modeEcho                  = "echo"
	chatComplIDPrefix         = "chatcmpl-"
	stopFinishReason          = "stop"
	lengthFinishReason        = "length"
	toolsFinishReason         = "tool_calls"
	roleAssistant             = "assistant"
	roleUser                  = "user"
	textCompletionObject      = "text_completion"
	chatCompletionObject      = "chat.completion"
	chatCompletionChunkObject = "chat.completion.chunk"
	toolChoiceNone            = "none"
	toolChoiceAuto            = "auto"
	toolChoiceRequired        = "required"
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
	// mode defines the simulator response generation mode, valid values: echo, random
	mode string
	// model defines the current base model name
	model string
	// loraAdaptors contains list of LoRA available adaptors
	loraAdaptors sync.Map
	// maxLoras defines maximum number of loaded loras
	maxLoras int
	// maxCPULoras defines maximum number of loras to store in CPU memory
	maxCPULoras int
	// runningLoras is a collection of running loras, key of lora's name, value is number of requests using this lora
	runningLoras sync.Map
	// waitingLoras will represent collection of loras defined in requests in the queue - Not implemented yet
	waitingLoras sync.Map
	// maxRunningReqs defines the maximum number of inference requests that could be processed at the same time
	maxRunningReqs int64
	// nRunningReqs is the number of inference requests that are currently being processed
	nRunningReqs int64
	// nWaitingReqs is the number of inference requests that are waiting to be processed
	nWaitingReqs int64
	// loraInfo is prometheus gauge
	loraInfo *prometheus.GaugeVec
	// runningRequests is prometheus gauge
	runningRequests *prometheus.GaugeVec
	// waitingRequests is prometheus gauge for number of queued requests
	waitingRequests *prometheus.GaugeVec
	// kvCacheUsagePercentage is prometheus gauge
	kvCacheUsagePercentage *prometheus.GaugeVec
	// channel for requeasts to be passed to workers
	reqChan chan *completionReqCtx
	// schema validator for tools parameters
	toolsValidator *validator
}
