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

package common

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"gopkg.in/yaml.v3"
)

const (
	ModeRandom = "random"
	ModeEcho   = "echo"

	// Failure type constants
	FailureTypeRateLimit      = "rate_limit"
	FailureTypeInvalidAPIKey  = "invalid_api_key"
	FailureTypeContextLength  = "context_length"
	FailureTypeServerError    = "server_error"
	FailureTypeInvalidRequest = "invalid_request"
	FailureTypeModelNotFound  = "model_not_found"

	StopFinishReason           = "stop"
	LengthFinishReason         = "length"
	ToolsFinishReason          = "tool_calls"
	RemoteDecodeFinishReason   = "remote_decode"
	CacheThresholdFinishReason = "cache_threshold"

	ChatCmplToolIDPrefix = "chatcmpl-tool-"
	MessagesToolIDPrefix = "toolu_"

	podIPEnv = "POD_IP"

	DefaultLatencyCalculator        = ""
	ConstantLatencyCalculator       = "constant"
	PerPromptTokenLatencyCalculator = "per-token"

	DefaultDSTableName = "llmd"
)

var (
	requiredFinishReasons = []string{
		StopFinishReason,
		LengthFinishReason,
		ToolsFinishReason,
		RemoteDecodeFinishReason,
		CacheThresholdFinishReason,
	}

	validFinishReasons = map[string]struct{}{
		StopFinishReason:           {},
		LengthFinishReason:         {},
		ToolsFinishReason:          {},
		RemoteDecodeFinishReason:   {},
		CacheThresholdFinishReason: {},
	}
)

type Configuration struct {
	// IP defines on which IP the simulator runs, loaded from env
	IP string
	// Port defines on which port the simulator runs
	Port int `yaml:"port" json:"port"`
	// Model defines the current base model name
	Model string `yaml:"model" json:"model"`
	// DisplayModelName defines the model name that will be shown in API responses
	// If ServedModelNames are not set, it defaults to the value of Model
	DisplayModelName string
	// ServedModelNames is one or many model names exposed by the API
	ServedModelNames []string `yaml:"served-model-name" json:"served-model-name"`
	// MaxLoras defines maximum number of loaded LoRAs
	MaxLoras int `yaml:"max-loras" json:"max-loras"`
	// MaxCPULoras defines maximum number of LoRAs to store in CPU memory
	MaxCPULoras int `yaml:"max-cpu-loras" json:"max-cpu-loras"`
	// MaxNumSeqs is maximum number of sequences per iteration (the maximum
	// number of inference requests that could be processed at the same time)
	MaxNumSeqs int `yaml:"max-num-seqs" json:"max-num-seqs"`
	// MaxWaitingQueueLength defines maximum size of waiting requests queue
	MaxWaitingQueueLength int `yaml:"max-waiting-queue-length" json:"max-waiting-queue-length"`
	// MaxModelLen is the model's context window, the maximum number of tokens
	// in a single request including input and output. Default value is 1024.
	MaxModelLen int `yaml:"max-model-len" json:"max-model-len"`
	// LoraModulesString is a list of LoRA adapters as strings (YAML parse helper; omitted from external output)
	LoraModulesString []string `yaml:"lora-modules" json:"-"`
	// LoraModules is a list of LoRA adapters
	LoraModules []LoraModule `json:"lora-modules"`

	// PodNameSpace specifies the Kubernetes namespace in which the simulator pod is running.
	// Useful for multi-namespace deployments and resource scoping.
	// Set by env variable POD_NAMESPACE
	PodNameSpace string
	// PodName specifies the name of the pod running the simulator instance.
	// Used for identification in Kubernetes environments.
	// Set by env variable POD_NAME
	PodName string
	// VllmDevMode enables development mode for the vLLM simulator
	// Allowing for additional debugging features during local development and testing.
	// Set by env variable VLLM_SERVER_DEV_MODE
	VllmDevMode bool

	// Latencies groups the request-latency simulation parameters. YAML and JSON
	// both nest it under "latencies"; both a YAML config file (via load's
	// foldLegacyKeys) and a POST /admin/config body (via Update's
	// foldFlatLatencies) still accept the legacy flat top-level keys too,
	// folded into "latencies" before unmarshalling into a Configuration.
	Latencies LatenciesConfig `yaml:"latencies" json:"latencies"`

	// LatencyCalculator is the name of the latency calculator to use in the simulation of the response latencies.
	// The default calculation is based on the current load of the simulator and on the configured latency
	// parameters, e.g., time-to-first-token and prefill-time-per-token. It is a top-level flag, not part of
	// LatenciesConfig, since it selects a calculation strategy rather than a latency value.
	LatencyCalculator string `yaml:"latency-calculator" json:"latency-calculator" admin:"configurable" rebuild:"latency"`

	// Mode defines the simulator response generation mode, valid values: echo, random
	Mode string `yaml:"mode" json:"mode"`
	// Seed defines random seed for operations
	Seed int64 `yaml:"seed" json:"seed"`

	// MaxToolCallIntegerParam defines the maximum possible value of integer parameters in a tool call,
	// optional, defaults to 100
	MaxToolCallIntegerParam int `yaml:"max-tool-call-integer-param" json:"max-tool-call-integer-param"`
	// MinToolCallIntegerParam defines the minimum possible value of integer parameters in a tool call,
	// optional, defaults to 0
	MinToolCallIntegerParam int `yaml:"min-tool-call-integer-param" json:"min-tool-call-integer-param"`
	// MaxToolCallNumberParam defines the maximum possible value of number (float) parameters in a tool call,
	// optional, defaults to 100
	MaxToolCallNumberParam float64 `yaml:"max-tool-call-number-param" json:"max-tool-call-number-param"`
	// MinToolCallNumberParam defines the minimum possible value of number (float) parameters in a tool call,
	// optional, defaults to 0
	MinToolCallNumberParam float64 `yaml:"min-tool-call-number-param" json:"min-tool-call-number-param"`

	// MaxToolCallArrayParamLength defines the maximum possible length of array parameters in a tool call,
	// optional, defaults to 5
	MaxToolCallArrayParamLength int `yaml:"max-tool-call-array-param-length" json:"max-tool-call-array-param-length"`
	// MinToolCallArrayParamLength defines the minimum possible length of array parameters in a tool call,
	// optional, defaults to 1
	MinToolCallArrayParamLength int `yaml:"min-tool-call-array-param-length" json:"min-tool-call-array-param-length"`

	// ToolCallNotRequiredParamProbability is the probability to add a parameter, that is not required,
	// in a tool call, optional, defaults to 50
	ToolCallNotRequiredParamProbability int `yaml:"tool-call-not-required-param-probability" json:"tool-call-not-required-param-probability"`
	// ObjectToolCallNotRequiredParamProbability is the probability to add a field, that is not required,
	// in an object in a tool call, optional, defaults to 50
	ObjectToolCallNotRequiredParamProbability int `yaml:"object-tool-call-not-required-field-probability" json:"object-tool-call-not-required-field-probability"`
	// SkipToolValidation disables the built-in meta-validation of incoming tool schemas.
	// Real vLLM forwards tool schemas to the model verbatim, so schemas using fields outside
	// the simulator's whitelist are rejected here but accepted upstream. Optional, defaults to false.
	SkipToolValidation bool `yaml:"skip-tool-validation" json:"skip-tool-validation"`
	// ToolCallExtraCallProbability is the probability (0-100) to make one additional tool call beyond the
	// minimum. Rolls repeat until a roll fails or len(availableTools) is reached, so the number of calls
	// follows a truncated geometric distribution that almost always equals the minimum but can reach
	// the total number of available tools. A value of 0 always produces the minimum number of calls;
	// a value of 100 always produces len(availableTools) calls. Optional, defaults to 45.
	ToolCallExtraCallProbability int `yaml:"tool-call-extra-call-probability" json:"tool-call-extra-call-probability"`

	// GlobalCacheHitThreshold is the default cache hit threshold (0-1] for all requests.
	// If a request specifies cache_hit_threshold, it takes precedence over this global value.
	GlobalCacheHitThreshold float64 `yaml:"global-cache-hit-threshold" json:"global-cache-hit-threshold"`

	// KVCache groups KV-cache sizing, hashing, and ZMQ event settings. KV-cache
	// transfer latencies and the global cache-hit threshold are configured separately.
	KVCache KVCacheConfig `yaml:"kvcache" json:"kvcache"`

	// FakeMetrics is a set of metrics to send to Prometheus instead of the real data
	FakeMetrics *FakeMetrics `yaml:"fake-metrics" json:"fake-metrics" admin:"configurable"`

	// FakeMetricsRefreshInterval defines how often function-based fake metrics are recalculated, defaults to 100ms
	FakeMetricsRefreshInterval time.Duration `yaml:"fake-metrics-refresh-interval" json:"fake-metrics-refresh-interval"`

	// FailureInjectionRate is the probability (0-100) of injecting failures
	FailureInjectionRate int `yaml:"failure-injection-rate" json:"failure-injection-rate" admin:"configurable"`
	// FailureTypes is a list of specific failure types to inject (empty means all types)
	FailureTypes []string `yaml:"failure-types" json:"failure-types" admin:"configurable"`

	// DPSize is data parallel size - a number of ranks to run, minimum is 1, maximum is 8, default is 1
	DPSize int `yaml:"data-parallel-size" json:"data-parallel-size"`

	// Rank specifies the rank of this instance. Only used when running Data Parallel
	// ranks as separate processes. If set, data-parallel-size is ignored
	Rank int `yaml:"data-parallel-rank" json:"data-parallel-rank"`

	// SSLCertFile is the path to the SSL certificate file for HTTPS
	SSLCertFile string `yaml:"ssl-certfile" json:"ssl-certfile"`
	// SSLKeyFile is the path to the SSL private key file for HTTPS
	SSLKeyFile string `yaml:"ssl-keyfile" json:"ssl-keyfile"`
	// SelfSignedCerts enables automatic generation of self-signed certificates for HTTPS
	SelfSignedCerts bool `yaml:"self-signed-certs" json:"self-signed-certs"`

	// DatasetPath Optional local file path to the SQLite database file used for generating responses from a dataset.
	//   - If not set, hardcoded preset responses will be used.
	//   - If set but the file does not exist the `dataset-url` will be used to download the database to the path specified by `dataset-path`.
	//   - If the file exists but is currently occupied by another process, responses will be randomly generated from preset text (the same behavior as if the path were not set).
	//   - Responses are retrieved from the dataset by the hash of the conversation history, with a fallback to a random dataset response, constrained by the maximum output tokens and EoS token handling, if no matching history is found.
	//   - Refer to [llm-d converted ShareGPT](https://huggingface.co/datasets/hf07397/inference-sim-datasets/blob/0b7ac1a4daf0aace1556326964bd75633372299e/README.md) for detailed information on the expected format of the SQLite database file.
	DatasetPath string `yaml:"dataset-path" json:"dataset-path"`
	// DatasetURL Optional URL for downloading the SQLite database file used for response generation.
	//   - This parameter is only used if the `dataset-path` is also set and the file does not exist at that path.
	//   - If the file needs to be downloaded, it will be saved to the location specified by `dataset-path`.
	//   - If the file already exists at the `dataset-path`, it will not be downloaded again
	//   - Example URL `https://huggingface.co/datasets/hf07397/inference-sim-datasets/resolve/91ffa7aafdfd6b3b1af228a517edc1e8f22cd274/huggingface/ShareGPT_Vicuna_unfiltered/conversations.sqlite3`
	DatasetURL string `yaml:"dataset-url" json:"dataset-url"`
	// DatasetInMemory defines whether to load the entire dataset into memory for faster access.
	DatasetInMemory bool `yaml:"dataset-in-memory" json:"dataset-in-memory"`
	// DatasetTableName defines custom SQLite dataset table name
	DatasetTableName string `yaml:"dataset-table-name" json:"dataset-table-name"`

	// RenderURL is the URL of the tokenizer render service. When set, the
	// simulator uses a HuggingFace tokenizer served over HTTP. When empty,
	// the simulator uses the in-process simulated tokenizer.
	RenderURL string `yaml:"render-url" json:"render-url"`
	// RenderTimeout is the timeout for tokenizer render requests
	RenderTimeout time.Duration `yaml:"render-timeout" json:"render-timeout"`
	// MMRenderTimeout is the timeout for multi-modal tokenizer render requests
	MMRenderTimeout time.Duration `yaml:"mm-render-timeout" json:"mm-render-timeout"`
	// ForceDummyTokenizer forces the use of the dummy tokenizer even if a real model name is provided.
	// This flag is retained for backward compatibility; omit --render-url to use the simulated tokenizer.
	ForceDummyTokenizer bool `yaml:"force-dummy-tokenizer" json:"force-dummy-tokenizer"`

	// StartupDuration defines how long /health/ready returns 503 to simulate GPU model loading.
	// After this duration from startup, /health/ready returns 200. Default is 0 (immediately ready).
	StartupDuration time.Duration `yaml:"startup-duration" json:"startup-duration"`

	// EnableSleepMode enables sleep mode
	EnableSleepMode bool `yaml:"enable-sleep-mode" json:"enable-sleep-mode"`

	// EnableRequestIDHeaders enables including X-Request-Id header in responses
	EnableRequestIDHeaders bool `yaml:"enable-request-id-headers" json:"enable-request-id-headers"`

	// LogHTTP logs full HTTP request and response details (method, URI, headers, bodies where buffered, status) for each request.
	LogHTTP bool `yaml:"log-http" json:"log-http"`

	// DefaultEmbeddingDimensions is the default size of embedding vectors when the request does not specify dimensions.
	// Used by the /v1/embeddings endpoint. Default is 384.
	DefaultEmbeddingDimensions int `yaml:"default-embedding-dimensions" json:"default-embedding-dimensions"`

	// MMEncoderOnly defines whether to skip the language component of the model.
	MMEncoderOnly bool `yaml:"mm-encoder-only" json:"mm-encoder-only"`

	// Omni enables omni mode: the simulator will emit a synthetic image chunk
	// after the token stream when the X-Send-Image request header is present,
	// or randomly based on ImageEmissionRate.
	Omni bool `yaml:"omni" json:"omni"`

	// ImageEmissionRate is the probability (0-100) of emitting a synthetic image
	// chunk per chat completion request when omni mode is enabled. 0 means never
	// emit via the rate mechanism, 100 means always emit. The X-Send-Image header
	// can still trigger emission independently.
	ImageEmissionRate int `yaml:"image-emission-rate" json:"image-emission-rate" admin:"configurable"`

	// Ignored parameters:
	// MMProcessorKWArgs defines arguments to be forwarded to the model's processor for multi-modal data.
	// Ignored in the simulator.
	MMProcessorKWArgs string `yaml:"mm-processor-kwargs" json:"mm-processor-kwargs"`
	// ECTransferConfig defines the configurations for distributed EC cache transfer.
	// Ignored in the simulator.
	ECTransferConfig string `yaml:"ec-transfer-config" json:"ec-transfer-config"`
	// EnforceEager defines whether to always use eager-mode PyTorch.
	// Ignored in the simulator.
	EnforceEager bool `yaml:"enforce-eager" json:"enforce-eager"`
	// EnablePrefixCaching defines whether to enable prefix caching.
	// Ignored in the simulator.
	EnablePrefixCaching bool `yaml:"enable-prefix-caching" json:"enable-prefix-caching"`
	// TPSize defines the number of tensor parallel replicas.
	// Ignored in the simulator.
	TPSize int `yaml:"tensor-parallel-size" json:"tensor-parallel-size"`
	// MaxRequestBodySizeMB sets the maximum allowed request body size in megabytes for the HTTP server.
	// Default is 4 (matching the fasthttp built-in default). Must be between 1 and 512.
	MaxRequestBodySizeMB int `yaml:"max-request-body-size-mb" json:"max-request-body-size-mb"`

	// EngineName is the inference engine backend being simulated. Currently only "vllm" is supported.
	EngineName string `yaml:"engine" json:"engine"`
}

type LoraModule struct {
	// Name is the LoRA's name
	Name string `json:"name"`
	// Path is the LoRA's path
	Path string `json:"path"`
	// BaseModelName is the LoRA's base model
	BaseModelName string `json:"base_model_name"`
}

// KVCacheConfig groups the KV-cache sizing, hashing, and ZMQ event settings.
// When EnableKVCache is false, every other field is reset to its zero value,
// since the rest of the struct is unused while the cache is disabled.
type KVCacheConfig struct {
	// EnableKVCache defines if kv cache feature will be enabled
	EnableKVCache bool `yaml:"enable-kvcache" json:"enable-kvcache"`

	//  KVCacheSize is the maximum number of token blocks in kv cache, the default value is 1024
	KVCacheSize int `yaml:"kv-cache-size" json:"kv-cache-size"`

	// TokenBlockSize is token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128, defaults to 16
	TokenBlockSize int `yaml:"block-size" json:"block-size"`

	// HashSeed is the seed for hash generation. Effective value follows configuration precedence in the docs (command-line --hash-seed, else PYTHONHASHSEED, else YAML, else default).
	HashSeed string `yaml:"hash-seed" json:"hash-seed"`

	// ZMQEndpoint is the ZMQ address to publish events, the default value is tcp://localhost:5557
	ZMQEndpoint string `yaml:"zmq-endpoint" json:"zmq-endpoint"`

	// KVEventsReplayEndpoint is the ZMQ ROUTER address to bind for receiving KV events replay requests.
	// Empty (default) disables the replay listener. Example: "tcp://*:5558"
	KVEventsReplayEndpoint string `yaml:"kv-events-replay-endpoint" json:"kv-events-replay-endpoint"`

	// KVEventsReplayQueueSize is the max number of event batches held in the replay queue; oldest dropped when full. Defaults to 1024.
	KVEventsReplayQueueSize int `yaml:"kv-events-replay-queue-size" json:"kv-events-replay-queue-size"`

	// EventBatchSize is the maximum number of kv-cache events to be sent together, defaults to 16
	EventBatchSize int `yaml:"event-batch-size" json:"event-batch-size"`

	// UseVllmMapEventFormat encodes KV cache events as msgpack maps with named fields (vLLM PR #42892 format)
	// instead of the legacy positional array format. Default is false (legacy array format).
	UseVllmMapEventFormat bool `yaml:"use-vllm-map-event-format" json:"use-vllm-map-event-format"`
}

// LatenciesConfig groups the request-latency simulation parameters.
// NOTE: For all duration fields please use duration strings, e.g., "100ms", "1.5s"
type LatenciesConfig struct {
	// TimeToFirstToken time before the first token will be returned
	TimeToFirstToken time.Duration `yaml:"time-to-first-token" json:"time-to-first-token" admin:"configurable" rebuild:"latency"`
	// TimeToFirstTokenStdDev standard deviation for time before the first token will be returned
	// optional, default is 0, can't be more than 30% of TimeToFirstToken, will not
	// cause the actual time to first token to differ by more than 70% from TimeToFirstToken
	TimeToFirstTokenStdDev time.Duration `yaml:"time-to-first-token-std-dev" json:"time-to-first-token-std-dev" admin:"configurable" rebuild:"latency"`

	// InterTokenLatency time between generated tokens
	InterTokenLatency time.Duration `yaml:"inter-token-latency" json:"inter-token-latency" admin:"configurable" rebuild:"latency"`
	// InterTokenLatencyStdDev standard deviation for time between generated tokens
	// optional, default is 0, can't be more than 30% of InterTokenLatency, will not cause the actual
	// inter token latency to differ by more than 70% from InterTokenLatency
	InterTokenLatencyStdDev time.Duration `yaml:"inter-token-latency-std-dev" json:"inter-token-latency-std-dev" admin:"configurable" rebuild:"latency"`
	// KVCacheTransferLatency time to "transfer" kv-cache from another vLLM instance in case P/D is activated,
	KVCacheTransferLatency time.Duration `yaml:"kv-cache-transfer-latency" json:"kv-cache-transfer-latency" admin:"configurable" rebuild:"latency"`
	// KVCacheTransferLatencyStdDev standard deviation for time to "transfer" kv-cache from another
	// vLLM instance in case P/D is activated, can't be more than 30% of KVCacheTransferLatency, will not
	// cause the actual latency to differ by more than 70% from KVCacheTransferLatency
	KVCacheTransferLatencyStdDev time.Duration `yaml:"kv-cache-transfer-latency-std-dev" json:"kv-cache-transfer-latency-std-dev" admin:"configurable" rebuild:"latency"`

	// $Total Prefill Time = PrefillOverhead + n * PrefillTimePerToken$
	// the assumption is that n is less than k, where k is the number of prallelism units of GPU
	// PrefillOverhead time taken to prefill the context
	PrefillOverhead     time.Duration `yaml:"prefill-overhead" json:"prefill-overhead" admin:"configurable" rebuild:"latency"`
	PrefillTimePerToken time.Duration `yaml:"prefill-time-per-token" json:"prefill-time-per-token" admin:"configurable" rebuild:"latency"`
	// PrefillOverheadStdDev similar to TimeToFirstTokenStdDev
	PrefillTimeStdDev time.Duration `yaml:"prefill-time-std-dev" json:"prefill-time-std-dev" admin:"configurable" rebuild:"latency"`
	// $Total KV Cache Transfer Time = n * KVCacheTransferTimePerToken$
	// the assumption is that the cache blocks are all missed at the remote pod
	// KVCacheTransfer overhead time taken to transfer kv-cache from another vLLM instance in case P/D is activated
	KVCacheTransferTimePerToken time.Duration `yaml:"kv-cache-transfer-time-per-token" json:"kv-cache-transfer-time-per-token" admin:"configurable" rebuild:"latency"`
	// KVCacheTransferOverheadStdDev similar to TimeToFirstTokenStdDev
	KVCacheTransferTimeStdDev time.Duration `yaml:"kv-cache-transfer-time-std-dev" json:"kv-cache-transfer-time-std-dev" admin:"configurable" rebuild:"latency"`

	// TimeToGenerateImage is the simulated time to generate an image in omni mode.
	// When an image is going to be emitted in a chat completion, the simulator
	// sleeps for this duration before sending the image chunk.
	TimeToGenerateImage time.Duration `yaml:"time-to-generate-image" json:"time-to-generate-image" admin:"configurable" rebuild:"latency"`
	// TimeToGenerateImageStdDev standard deviation for time to generate an image.
	// Optional, default is 0, can't be more than 30% of TimeToGenerateImage.
	TimeToGenerateImageStdDev time.Duration `yaml:"time-to-generate-image-std-dev" json:"time-to-generate-image-std-dev" admin:"configurable" rebuild:"latency"`

	// TimeFactorUnderLoad is a multiplicative factor that affects the overall time taken for requests when parallel
	// requests are being processed.
	// The value of this factor must be >= 1.0, with a default of 1.0.
	// - If this factor is 1.0, no extra time is added.
	// - When the factor is x (where x > 1.0) and there are MaxNumSeqs requests, the total time will be multiplied by x.
	// - The extra time then decreases multiplicatively to 1.0 when the number of requests is less than MaxNumSeqs.
	TimeFactorUnderLoad float64 `yaml:"time-factor-under-load" json:"time-factor-under-load" admin:"configurable" rebuild:"latency"`
}

// NewConfig returns a Configuration populated with its documented defaults.
func NewConfig() *Configuration {
	return &Configuration{
		EngineName:                          "vllm",
		IP:                                  os.Getenv(podIPEnv),
		Port:                                8000,
		MaxLoras:                            1,
		MaxNumSeqs:                          5,
		MaxWaitingQueueLength:               1000,
		MaxModelLen:                         1024,
		Mode:                                ModeRandom,
		Seed:                                time.Now().UnixNano(),
		Latencies:                           LatenciesConfig{TimeFactorUnderLoad: 1.0},
		MaxToolCallIntegerParam:             100,
		MaxToolCallNumberParam:              100,
		MaxToolCallArrayParamLength:         5,
		MinToolCallArrayParamLength:         1,
		ToolCallNotRequiredParamProbability: 50,
		ObjectToolCallNotRequiredParamProbability: 50,
		ToolCallExtraCallProbability:              45,
		KVCache: KVCacheConfig{
			KVCacheSize:             1024,
			TokenBlockSize:          16,
			ZMQEndpoint:             "tcp://127.0.0.1:5557",
			KVEventsReplayQueueSize: 1024,
			EventBatchSize:          16,
		},
		DPSize:                     1,
		Rank:                       -1,
		DatasetTableName:           DefaultDSTableName,
		DefaultEmbeddingDimensions: 384,
		FakeMetricsRefreshInterval: 100 * time.Millisecond,
		MaxRequestBodySizeMB:       4,
		RenderURL:                  "",
		RenderTimeout:              30 * time.Second,
		MMRenderTimeout:            60 * time.Second,
	}
}

func (c *Configuration) load(configFile string) error {
	configBytes, err := os.ReadFile(configFile)
	if err != nil {
		return fmt.Errorf("failed to read configuration file: %s", err)
	}

	var raw map[string]any
	if err := yaml.Unmarshal(configBytes, &raw); err != nil {
		return fmt.Errorf("failed to unmarshal configuration: %s", err)
	}
	if err := foldLegacyKeys(raw, "kvcache", kvCacheYAMLKeys); err != nil {
		return err
	}
	if err := foldLegacyKeys(raw, "latencies", latenciesYAMLKeys); err != nil {
		return err
	}

	mergedBytes, err := yaml.Marshal(raw)
	if err != nil {
		return fmt.Errorf("failed to re-marshal configuration: %s", err)
	}

	if err := yaml.Unmarshal(mergedBytes, c); err != nil {
		return fmt.Errorf("failed to unmarshal configuration: %s", err)
	}

	return nil
}

// foldLegacyKeys moves top-level keys in flatKeys (the flat layout used
// before a group of settings was nested under nestedKey) into the nested
// block in place, so config files using either layout load the same way. It
// returns an error if a config file mixes the two layouts, i.e. sets any of
// flatKeys at the top level while the nested block is also present.
func foldLegacyKeys(raw map[string]any, nestedKey string, flatKeys []string) error {
	nested, _ := raw[nestedKey].(map[string]any)

	var setFlatKeys []string
	for _, key := range flatKeys {
		if _, ok := raw[key]; ok {
			setFlatKeys = append(setFlatKeys, key)
		}
	}
	if len(setFlatKeys) > 0 && len(nested) > 0 {
		return fmt.Errorf("%s settings mix the legacy flat layout (%s) with the nested %s block; use only one",
			nestedKey, strings.Join(setFlatKeys, ", "), nestedKey)
	}

	if nested == nil {
		nested = map[string]any{}
	}
	for _, key := range setFlatKeys {
		nested[key] = raw[key]
		delete(raw, key)
	}
	if len(nested) > 0 {
		raw[nestedKey] = nested
	}
	return nil
}

func (c *Configuration) validate() error {
	if c.Model == "" {
		return errors.New("model parameter is empty")
	}
	// Upstream vLLM behaviour: when --served-model-name is not provided,
	// it falls back to using the value of --model as the single public name
	// returned by the API and exposed in Prometheus metrics.
	if len(c.ServedModelNames) == 0 {
		c.ServedModelNames = []string{c.Model}
	}

	// set display model name
	c.DisplayModelName = c.ServedModelNames[0]

	if c.Mode != ModeEcho && c.Mode != ModeRandom {
		return fmt.Errorf("invalid mode '%s', valid values are 'random' and 'echo'", c.Mode)
	}
	if c.Port <= 0 {
		return fmt.Errorf("invalid port '%d'", c.Port)
	}
	if c.Latencies.InterTokenLatency < 0 {
		return errors.New("inter token latency cannot be negative")
	}
	if c.Latencies.InterTokenLatencyStdDev < 0 {
		return errors.New("inter token latency standard deviation cannot be negative")
	}
	if float32(c.Latencies.InterTokenLatencyStdDev) > 0.3*float32(c.Latencies.InterTokenLatency) {
		return errors.New("inter token latency standard deviation cannot be more than 30% of inter token latency")
	}
	if c.Latencies.TimeToFirstToken < 0 {
		return errors.New("time to first token cannot be negative")
	}
	if c.Latencies.TimeToFirstTokenStdDev < 0 {
		return errors.New("time to first token standard deviation cannot be negative")
	}
	if float32(c.Latencies.TimeToFirstTokenStdDev) > 0.3*float32(c.Latencies.TimeToFirstToken) {
		return errors.New("time to first token standard deviation cannot be more than 30% of time to first token")
	}

	if c.Latencies.TimeToGenerateImage < 0 {
		return errors.New("time to generate image cannot be negative")
	}
	if c.Latencies.TimeToGenerateImageStdDev < 0 {
		return errors.New("time to generate image standard deviation cannot be negative")
	}
	if float32(c.Latencies.TimeToGenerateImageStdDev) > 0.3*float32(c.Latencies.TimeToGenerateImage) {
		return errors.New("time to generate image standard deviation cannot be more than 30% of time to generate image")
	}

	if c.Latencies.PrefillOverhead < 0 {
		return errors.New("prefill overhead cannot be negative")
	}
	if c.Latencies.PrefillTimePerToken < 0 {
		return errors.New("prefill time per token cannot be negative")
	}
	if c.Latencies.PrefillTimeStdDev < 0 {
		return errors.New("prefill time standard deviation cannot be negative")
	}
	// No upper-bound check on PrefillTimeStdDev: it is applied to the total prefill time
	// (prefill-overhead + n × prefill-time-per-token), which depends on the prompt length n
	// and is unknown at config time. Sampled durations are clamped at runtime by
	// RandomNormDuration to [0.3, 1.7] × mean, so an oversized std-dev cannot produce
	// nonsensical values.

	if c.Latencies.TimeFactorUnderLoad < 1.0 {
		return errors.New("time factor under load cannot be less than 1.0")
	}

	if c.MaxModelLen < 1 {
		return errors.New("max model len cannot be less than 1")
	}

	if c.MaxNumSeqs < 1 {
		return errors.New("max num seqs cannot be less than 1")
	}

	if c.MaxWaitingQueueLength < 0 {
		return errors.New("max waiting queue size cannot be less than 0")
	}

	if c.MaxToolCallIntegerParam < c.MinToolCallIntegerParam {
		return errors.New("MaxToolCallIntegerParam cannot be less than MinToolCallIntegerParam")
	}
	if c.MaxToolCallNumberParam < c.MinToolCallNumberParam {
		return errors.New("MaxToolCallNumberParam cannot be less than MinToolCallNumberParam")
	}
	if c.MaxToolCallArrayParamLength < c.MinToolCallArrayParamLength {
		return errors.New("MaxToolCallArrayParamLength cannot be less than MinToolCallArrayParamLength")
	}
	if c.MinToolCallArrayParamLength < 0 {
		return errors.New("MinToolCallArrayParamLength cannot be negative")
	}
	if c.ToolCallNotRequiredParamProbability < 0 || c.ToolCallNotRequiredParamProbability > 100 {
		return errors.New("ToolCallNotRequiredParamProbability should be between 0 and 100")
	}
	if c.ObjectToolCallNotRequiredParamProbability < 0 || c.ObjectToolCallNotRequiredParamProbability > 100 {
		return errors.New("ObjectToolCallNotRequiredParamProbability should be between 0 and 100")
	}
	if c.ToolCallExtraCallProbability < 0 || c.ToolCallExtraCallProbability > 100 {
		return errors.New("ToolCallExtraCallProbability should be between 0 and 100")
	}

	if c.FailureInjectionRate < 0 || c.FailureInjectionRate > 100 {
		return errors.New("failure injection rate should be between 0 and 100")
	}

	if c.ImageEmissionRate < 0 || c.ImageEmissionRate > 100 {
		return errors.New("image emission rate should be between 0 and 100")
	}

	validFailureTypes := map[string]bool{
		FailureTypeRateLimit:      true,
		FailureTypeInvalidAPIKey:  true,
		FailureTypeContextLength:  true,
		FailureTypeServerError:    true,
		FailureTypeInvalidRequest: true,
		FailureTypeModelNotFound:  true,
	}
	for _, ft := range c.FailureTypes {
		if !validFailureTypes[ft] {
			return fmt.Errorf("invalid failure type '%s', valid types are: %s, %s, %s, %s, %s, %s", ft,
				FailureTypeRateLimit, FailureTypeInvalidAPIKey, FailureTypeContextLength,
				FailureTypeServerError, FailureTypeInvalidRequest, FailureTypeModelNotFound)
		}
	}

	if c.DPSize < 1 || c.DPSize > 8 {
		return errors.New("data parallel size must be between 1 and 8")
	}

	if c.Rank > 7 {
		return errors.New("data parallel rank must be between 0 and 7")
	}

	if (c.SSLCertFile == "") != (c.SSLKeyFile == "") {
		return errors.New("both ssl-certfile and ssl-keyfile must be provided together")
	}

	if c.SelfSignedCerts && (c.SSLCertFile != "" || c.SSLKeyFile != "") {
		return errors.New("cannot use both self-signed-certs and explicit ssl-certfile/ssl-keyfile")
	}

	if c.DatasetPath == "" && c.DatasetURL != "" {
		return errors.New("dataset-path is required when dataset-url is set")
	}

	if c.Mode == ModeEcho && (c.DatasetPath != "" || c.DatasetURL != "") {
		return errors.New("dataset cannot be defined in echo mode")
	}

	if c.LatencyCalculator != DefaultLatencyCalculator && c.LatencyCalculator != ConstantLatencyCalculator &&
		c.LatencyCalculator != PerPromptTokenLatencyCalculator {
		return fmt.Errorf("unknown latency-calculator %s, supported calculators are: %s and %s",
			c.LatencyCalculator, ConstantLatencyCalculator, PerPromptTokenLatencyCalculator)
	}

	if c.DefaultEmbeddingDimensions < 1 {
		return errors.New("default embedding dimensions must be at least 1")
	}

	if c.MaxRequestBodySizeMB < 1 || c.MaxRequestBodySizeMB > 512 {
		return fmt.Errorf("max-request-body-size-mb must be between 1 MB and 512 MB, got %d", c.MaxRequestBodySizeMB)
	}

	if c.EngineName != "vllm" {
		return fmt.Errorf("invalid engine '%s', currently only 'vllm' is supported", c.EngineName)
	}

	return nil
}

// SSLEnabled returns true if SSL is enabled either via certificate files or self-signed certificates
func (c *Configuration) SSLEnabled() bool {
	return (c.SSLCertFile != "" && c.SSLKeyFile != "") || c.SelfSignedCerts
}

// durationFields holds the JSON key names of all time.Duration fields in
// Configuration and LatenciesConfig.
// configurableFields maps each admin-configurable JSON field key to its rebuild tag
// (value of the rebuild struct tag, e.g. "latency"), or "" for fields with no rebuild tag.
// kvCacheYAMLKeys and latenciesYAMLKeys hold the YAML key names of every KVCacheConfig
// and LatenciesConfig field respectively. Since those names are identical to the fields'
// JSON key names, load() also reuses them to fold legacy flat top-level YAML keys into
// the nested "kvcache"/"latencies" blocks, and Update's foldFlatLatencies reuses
// latenciesYAMLKeys the same way for the legacy flat POST /admin/config body shape.
// latenciesYAMLKeySet is the same set as latenciesYAMLKeys, for membership checks;
// unfoldNestedLatencies uses it to reject fields that are admin-configurable but not
// part of LatenciesConfig (e.g. latency-calculator) inside the nested "latencies" object.
// All are populated once at init via reflection so there is no static list to keep in
// sync with the structs.
var (
	durationFields      map[string]bool
	configurableFields  map[string]string
	kvCacheYAMLKeys     []string
	latenciesYAMLKeys   []string
	latenciesYAMLKeySet map[string]bool
)

func init() {
	durationFields = make(map[string]bool)
	configurableFields = make(map[string]string)
	// Configuration's Latencies and KVCache fields are named, non-anonymous
	// fields, so a field walk over Configuration does not descend into them;
	// their own admin-configurable and duration fields are collected via a
	// separate walk over LatenciesConfig.
	collectFieldMeta(reflect.TypeOf(Configuration{}))
	collectFieldMeta(reflect.TypeOf(LatenciesConfig{}))

	kvCacheYAMLKeys = yamlKeysOf(reflect.TypeOf(KVCacheConfig{}))
	latenciesYAMLKeys = yamlKeysOf(reflect.TypeOf(LatenciesConfig{}))
	latenciesYAMLKeySet = make(map[string]bool, len(latenciesYAMLKeys))
	for _, key := range latenciesYAMLKeys {
		latenciesYAMLKeySet[key] = true
	}
}

// collectFieldMeta walks t's direct fields, adding each duration field's JSON
// key to durationFields and each admin-configurable field's JSON key (with
// its rebuild tag) to configurableFields.
func collectFieldMeta(t reflect.Type) {
	durationType := reflect.TypeOf(time.Duration(0))
	for i := range t.NumField() {
		f := t.Field(i)
		jsonKey := strings.SplitN(f.Tag.Get("json"), ",", 2)[0]
		if jsonKey == "" || jsonKey == "-" {
			continue
		}
		if f.Type == durationType {
			durationFields[jsonKey] = true
		}
		if f.Tag.Get("admin") == "configurable" {
			configurableFields[jsonKey] = f.Tag.Get("rebuild")
		}
	}
}

// yamlKeysOf returns the YAML key names of every direct field of t.
func yamlKeysOf(t reflect.Type) []string {
	var keys []string
	for i := range t.NumField() {
		f := t.Field(i)
		yamlKey := strings.SplitN(f.Tag.Get("yaml"), ",", 2)[0]
		if yamlKey == "" || yamlKey == "-" {
			continue
		}
		keys = append(keys, yamlKey)
	}
	return keys
}

// normalizeDurationStrings converts duration string values (e.g. "1s") in raw
// to nanosecond integers in-place, so subsequent json.Unmarshal into
// time.Duration fields works correctly. Non-string values are left unchanged.
func normalizeDurationStrings(raw map[string]json.RawMessage) error {
	for key, val := range raw {
		if !durationFields[key] || len(val) < 2 || val[0] != '"' {
			continue
		}
		var s string
		if err := json.Unmarshal(val, &s); err != nil {
			return fmt.Errorf("field %q: invalid duration string: %w", key, err)
		}
		d, err := time.ParseDuration(s)
		if err != nil {
			return fmt.Errorf("field %q: %w", key, err)
		}
		ns, err := json.Marshal(int64(d))
		if err != nil {
			return fmt.Errorf("field %q: failed to marshal nanoseconds: %w", key, err)
		}
		raw[key] = ns
	}
	return nil
}

// unfoldNestedLatencies expands an optional top-level "latencies" object in
// an admin-config JSON body into flat keys in place, so POST /admin/config
// accepts either shape, matching the flat/nested flexibility YAML config
// files already have via foldLegacyKeys. Every key inside the nested object
// must be one of LatenciesConfig's own fields: latency-calculator is a
// top-level-only field (see Configuration.LatencyCalculator) and is rejected
// here even though it is otherwise admin-configurable. It also returns an
// error if a field is set both at the top level and inside the nested
// "latencies" object.
func unfoldNestedLatencies(raw map[string]json.RawMessage) error {
	nestedRaw, ok := raw["latencies"]
	if !ok {
		return nil
	}

	var nested map[string]json.RawMessage
	if err := json.Unmarshal(nestedRaw, &nested); err != nil {
		return fmt.Errorf(`field "latencies": %w`, err)
	}

	var conflicts []string
	for key := range nested {
		if !latenciesYAMLKeySet[key] {
			return fmt.Errorf("field '%s' is not a latencies field", key)
		}
		if _, exists := raw[key]; exists {
			conflicts = append(conflicts, key)
		}
	}
	if len(conflicts) > 0 {
		return fmt.Errorf("latencies settings mix the flat layout (%s) with the nested latencies object; use only one",
			strings.Join(conflicts, ", "))
	}

	for key, val := range nested {
		raw[key] = val
	}
	delete(raw, "latencies")
	return nil
}

// foldFlatLatencies moves the legacy flat top-level latency keys in raw into
// a nested "latencies" object, the reverse of unfoldNestedLatencies. Update
// calls this after validating raw's keys against configurableFields (which
// uses the flat key names), so that the subsequent json.Unmarshal into a
// Configuration - whose Latencies field is nested under "latencies" -
// populates correctly regardless of which shape the caller originally sent.
func foldFlatLatencies(raw map[string]json.RawMessage) error {
	nested := make(map[string]json.RawMessage)
	for _, key := range latenciesYAMLKeys {
		if v, ok := raw[key]; ok {
			nested[key] = v
			delete(raw, key)
		}
	}
	if len(nested) == 0 {
		return nil
	}
	data, err := json.Marshal(nested)
	if err != nil {
		return fmt.Errorf("failed to marshal latencies: %w", err)
	}
	raw["latencies"] = data
	return nil
}

// Update validates a partial JSON update and returns:
//   - next: a deep copy of the receiver with the body's changes applied.
//     Ready to be atomically swapped in by the caller.
//   - update: a fresh Configuration populated only with the fields that
//     appeared in the body.
//   - latencyChanged: true if the body touched any latency-related field, so
//     the caller knows it must rebuild the latency calculator.
//
// "field absent" and "field set to null" both decode to a nil pointer or nil
// slice; explicit null no longer clears a metric. To clear a slice/map
// metric, send an empty value (`[]` or `{}`).
func (c *Configuration) Update(body []byte) (*Configuration, *Configuration, bool, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, nil, false, fmt.Errorf("failed to unmarshal payload: %w", err)
	}

	if err := unfoldNestedLatencies(raw); err != nil {
		return nil, nil, false, err
	}

	// convert any duration-string values (e.g. "1s") to nanosecond integers
	if err := normalizeDurationStrings(raw); err != nil {
		return nil, nil, false, err
	}

	latencyChanged := false
	for key := range raw {
		rebuildTag, isConfigurable := configurableFields[key]
		if !isConfigurable {
			return nil, nil, false, fmt.Errorf("field '%s' is not admin-configurable", key)
		}
		if rebuildTag == "latency" {
			latencyChanged = true
		}
	}

	if err := foldFlatLatencies(raw); err != nil {
		return nil, nil, false, err
	}
	// re-marshal after normalization and folding so subsequent Unmarshal calls
	// get integers and see Latencies nested under "latencies"
	var err error
	body, err = json.Marshal(raw)
	if err != nil {
		return nil, nil, false, fmt.Errorf("failed to re-marshal normalized payload: %w", err)
	}

	// update is a fresh struct populated only with the body's fields; the
	// caller reads update.FakeMetrics to decide whether to apply Prometheus
	// side effects.
	update := &Configuration{}
	if err := json.Unmarshal(body, update); err != nil {
		return nil, nil, false, fmt.Errorf("failed to unmarshal payload: %w", err)
	}

	// next is a deep copy of c; unmarshalling body on top merges body fields
	// into next, including overlaying the fake-metrics partial onto next's
	// (deep-copied) FakeMetrics. validate() then sees the fully merged state.
	next, err := c.Copy()
	if err != nil {
		return nil, nil, false, fmt.Errorf("failed to copy configuration: %w", err)
	}
	if err := json.Unmarshal(body, next); err != nil {
		return nil, nil, false, fmt.Errorf("failed to unmarshal payload: %w", err)
	}

	if err := next.validate(); err != nil {
		return nil, nil, false, err
	}
	return next, update, latencyChanged, nil
}

// Copy returns a deep copy of c.
func (c *Configuration) Copy() (*Configuration, error) {
	var dst Configuration
	data, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &dst)
	return &dst, err
}

// cleanedMap returns the configuration as a JSON-friendly map with internal
// fields removed/renamed for external display (logs, /admin/config GET).
// Latencies' json tag already nests the latency fields under "latencies",
// the same way KVCacheConfig is nested under "kvcache".
func (c *Configuration) cleanedMap() (map[string]any, error) {
	cfgJSON, err := json.Marshal(c)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal configuration to JSON: %w", err)
	}

	var m map[string]any
	if err := json.Unmarshal(cfgJSON, &m); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON to map: %w", err)
	}
	if c.DPSize > 1 {
		// in DP mode, the per-rank port is not meaningful externally
		delete(m, "port")
	}
	formatDurationFields(m)
	if latencies, ok := m["latencies"].(map[string]any); ok {
		formatDurationFields(latencies)
	}
	return m, nil
}

// formatDurationFields rewrites, in place, every key in m that names a
// time.Duration field of Configuration from the nanosecond count
// json.Unmarshal produced (as a float64) into a Go duration string (e.g.
// "250ms"). Used for both the top-level map and the nested "latencies" map,
// since durationFields holds flat key names shared by both.
func formatDurationFields(m map[string]any) {
	for key := range durationFields {
		if v, ok := m[key]; ok {
			if ns, ok := v.(float64); ok {
				m[key] = time.Duration(int64(ns)).String()
			}
		}
	}
}

// MarshalCleaned returns the configuration as JSON suitable for external
// display (e.g. /admin/config GET), with internal fields removed.
func (c *Configuration) MarshalCleaned() ([]byte, error) {
	m, err := c.cleanedMap()
	if err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

func (c *Configuration) Show(logger logr.Logger) error {
	m, err := c.cleanedMap()
	if err != nil {
		return err
	}
	cfgJSON, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal configuration to JSON: %w", err)
	}
	logger.V(logging.INFO).Info("Configuration:", "", string(cfgJSON))
	return nil
}
