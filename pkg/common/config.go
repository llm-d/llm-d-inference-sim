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
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"gopkg.in/yaml.v3"
)

const (
	vLLMDefaultPort = 8000
	ModeRandom      = "random"
	ModeEcho        = "echo"

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
	// LoraModulesString is a list of LoRA adapters as strings
	LoraModulesString []string `yaml:"lora-modules" json:"lora-modules"`
	// LoraModules is a list of LoRA adapters
	LoraModules []LoraModule

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

	// --- Duration Configuration ---
	// NOTE: For all duration fields listed below, providing a raw integer (milliseconds) is DEPRECATED
	// and support will be removed in the next version.
	// Please use duration strings instead (e.g., "100ms", "1.5s").

	// TimeToFirstToken time before the first token will be returned
	TimeToFirstToken Duration `yaml:"time-to-first-token" json:"time-to-first-token"`
	// TimeToFirstTokenStdDev standard deviation for time before the first token will be returned
	// optional, default is 0, can't be more than 30% of TimeToFirstToken, will not
	// cause the actual time to first token to differ by more than 70% from TimeToFirstToken
	TimeToFirstTokenStdDev Duration `yaml:"time-to-first-token-std-dev" json:"time-to-first-token-std-dev"`

	// InterTokenLatency time between generated tokens
	InterTokenLatency Duration `yaml:"inter-token-latency" json:"inter-token-latency"`
	// InterTokenLatencyStdDev standard deviation for time between generated tokens
	// optional, default is 0, can't be more than 30% of InterTokenLatency, will not cause the actual
	// inter token latency to differ by more than 70% from InterTokenLatency
	InterTokenLatencyStdDev Duration `yaml:"inter-token-latency-std-dev" json:"inter-token-latency-std-dev"`
	// KVCacheTransferLatency time to "transfer" kv-cache from another vLLM instance in case P/D is activated,
	KVCacheTransferLatency Duration `yaml:"kv-cache-transfer-latency" json:"kv-cache-transfer-latency"`
	// KVCacheTransferLatencyStdDev standard deviation for time to "transfer" kv-cache from another
	// vLLM instance in case P/D is activated, can't be more than 30% of KVCacheTransferLatency, will not
	// cause the actual latency to differ by more than 70% from KVCacheTransferLatency
	KVCacheTransferLatencyStdDev Duration `yaml:"kv-cache-transfer-latency-std-dev" json:"kv-cache-transfer-latency-std-dev"`

	// $Total Prefill Time = PrefillOverhead + n * PrefillTimePerToken$
	// the assumption is that n is less than k, where k is the number of prallelism units of GPU
	// PrefillOverhead time taken to prefill the context
	PrefillOverhead     Duration `yaml:"prefill-overhead" json:"prefill-overhead"`
	PrefillTimePerToken Duration `yaml:"prefill-time-per-token" json:"prefill-time-per-token"`
	// PrefillOverheadStdDev similar to TimeToFirstTokenStdDev
	PrefillTimeStdDev Duration `yaml:"prefill-time-std-dev" json:"prefill-time-std-dev"`
	// $Total KV Cache Transfer Time = n * KVCacheTransferTimePerToken$
	// the assumption is that the cache blocks are all missed at the remote pod
	// KVCacheTransfer overhead time taken to transfer kv-cache from another vLLM instance in case P/D is activated
	KVCacheTransferTimePerToken Duration `yaml:"kv-cache-transfer-time-per-token" json:"kv-cache-transfer-time-per-token"`
	// KVCacheTransferOverheadStdDev similar to TimeToFirstTokenStdDev
	KVCacheTransferTimeStdDev Duration `yaml:"kv-cache-transfer-time-std-dev" json:"kv-cache-transfer-time-std-dev"`

	// TimeFactorUnderLoad is a multiplicative factor that affects the overall time taken for requests when parallel
	// requests are being processed.
	// The value of this factor must be >= 1.0, with a default of 1.0.
	// - If this factor is 1.0, no extra time is added.
	// - When the factor is x (where x > 1.0) and there are MaxNumSeqs requests, the total time will be multiplied by x.
	// - The extra time then decreases multiplicatively to 1.0 when the number of requests is less than MaxNumSeqs.
	TimeFactorUnderLoad float64 `yaml:"time-factor-under-load" json:"time-factor-under-load"`

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

	// EnableKVCache defines if kv cache feature will be enabled
	EnableKVCache bool `yaml:"enable-kvcache" json:"enable-kvcache"`
	//  KVCacheSize is the maximum number of token blocks in kv cache, the default value is 1024
	KVCacheSize int `yaml:"kv-cache-size" json:"kv-cache-size"`
	// GlobalCacheHitThreshold is the default cache hit threshold (0-1] for all requests.
	// If a request specifies cache_hit_threshold, it takes precedence over this global value.
	GlobalCacheHitThreshold float64 `yaml:"global-cache-hit-threshold" json:"global-cache-hit-threshold"`

	// TokenBlockSize is token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128, defaults to 16
	TokenBlockSize int `yaml:"block-size" json:"block-size"`
	// HashSeed is the seed for hash generation. Effective value follows configuration precedence in the docs (command-line --hash-seed, else PYTHONHASHSEED, else YAML, else default).
	HashSeed string `yaml:"hash-seed" json:"hash-seed"`

	// ZMQEndpoint is the ZMQ address to publish events, the default value is tcp://localhost:5557
	ZMQEndpoint string `yaml:"zmq-endpoint" json:"zmq-endpoint"`

	// EventBatchSize is the maximum number of kv-cache events to be sent together, defaults to 16
	EventBatchSize int `yaml:"event-batch-size" json:"event-batch-size"`

	// FakeMetrics is a set of metrics to send to Prometheus instead of the real data
	FakeMetrics *FakeMetrics `yaml:"fake-metrics" json:"fake-metrics"`

	// FakeMetricsRefreshInterval defines how often function-based fake metrics are recalculated, defaults to 100ms
	FakeMetricsRefreshInterval time.Duration `yaml:"fake-metrics-refresh-interval" json:"fake-metrics-refresh-interval"`

	// FailureInjectionRate is the probability (0-100) of injecting failures
	FailureInjectionRate int `yaml:"failure-injection-rate" json:"failure-injection-rate"`
	// FailureTypes is a list of specific failure types to inject (empty means all types)
	FailureTypes []string `yaml:"failure-types" json:"failure-types"`

	// DPSize is data parallel size - a number of ranks to run, minimum is 1, maximum is 8, default is 1
	DPSize int `yaml:"data-parallel-size" json:"data-parallel-size"`

	// Rank is the vLLM parameter used to specify the rank of this instance. Here only
	// used when running Data Parallel ranks as separate processes. If set, data-parallel-size is ignored
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

	// RenderURL is the URL of the tokenizer render service
	RenderURL string `yaml:"render-url" json:"render-url"`
	// RenderTimeout is the timeout for tokenizer render requests
	RenderTimeout time.Duration `yaml:"render-timeout" json:"render-timeout"`
	// MMRenderTimeout is the timeout for multi-modal tokenizer render requests
	MMRenderTimeout time.Duration `yaml:"mm-render-timeout" json:"mm-render-timeout"`

	// StartupDuration defines how long /health/ready returns 503 to simulate GPU model loading.
	// After this duration from startup, /health/ready returns 200. Default is 0 (immediately ready).
	StartupDuration time.Duration `yaml:"startup-duration" json:"startup-duration"`

	// EnableSleepMode enables sleep mode
	EnableSleepMode bool `yaml:"enable-sleep-mode" json:"enable-sleep-mode"`

	// EnableRequestIDHeaders enables including X-Request-Id header in responses
	EnableRequestIDHeaders bool `yaml:"enable-request-id-headers" json:"enable-request-id-headers"`

	// LogHTTP logs full HTTP request and response details (method, URI, headers, bodies where buffered, status) for each request.
	LogHTTP bool `yaml:"log-http" json:"log-http"`

	// LatencyCalculator is the name of the latency calculator to use in the simulation of the response latencies.
	// The default calculation is based on the current load of the simulator and on the configured latency
	// parameters, e.g., time-to-first-token and prefill-time-per-token.
	LatencyCalculator string `yaml:"latency-calculator" json:"latency-calculator"`

	// DefaultEmbeddingDimensions is the default size of embedding vectors when the request does not specify dimensions.
	// Used by the /v1/embeddings endpoint. Default is 384.
	DefaultEmbeddingDimensions int `yaml:"default-embedding-dimensions" json:"default-embedding-dimensions"`

	// MMEncoderOnly defines whether to skip the language component of the model.
	MMEncoderOnly bool `yaml:"mm-encoder-only" json:"mm-encoder-only"`

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
}

type LoraModule struct {
	// Name is the LoRA's name
	Name string `json:"name"`
	// Path is the LoRA's path
	Path string `json:"path"`
	// BaseModelName is the LoRA's base model
	BaseModelName string `json:"base_model_name"`
}

func newConfig() *Configuration {
	return &Configuration{
		IP:                                  os.Getenv(podIPEnv),
		Port:                                vLLMDefaultPort,
		MaxLoras:                            1,
		MaxNumSeqs:                          5,
		MaxWaitingQueueLength:               1000,
		MaxModelLen:                         1024,
		Mode:                                ModeRandom,
		Seed:                                time.Now().UnixNano(),
		TimeFactorUnderLoad:                 1.0,
		MaxToolCallIntegerParam:             100,
		MaxToolCallNumberParam:              100,
		MaxToolCallArrayParamLength:         5,
		MinToolCallArrayParamLength:         1,
		ToolCallNotRequiredParamProbability: 50,
		ObjectToolCallNotRequiredParamProbability: 50,
		KVCacheSize:                1024,
		TokenBlockSize:             16,
		ZMQEndpoint:                "tcp://127.0.0.1:5557",
		EventBatchSize:             16,
		DPSize:                     1,
		Rank:                       -1,
		DatasetTableName:           DefaultDSTableName,
		DefaultEmbeddingDimensions: 384,
		FakeMetricsRefreshInterval: 100 * time.Millisecond,
		RenderURL:                  "http://localhost:8082",
		RenderTimeout:              30 * time.Second,
		MMRenderTimeout:            60 * time.Second,
	}
}

func (c *Configuration) load(configFile string) error {
	configBytes, err := os.ReadFile(configFile)
	if err != nil {
		return fmt.Errorf("failed to read configuration file: %s", err)
	}

	if err := yaml.Unmarshal(configBytes, &c); err != nil {
		return fmt.Errorf("failed to unmarshal configuration: %s", err)
	}

	if err := c.unmarshalLoras(); err != nil {
		return err
	}
	if err := c.unmarshalLoraFakeMetrics(); err != nil {
		return err
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
	if c.InterTokenLatency < 0 {
		return errors.New("inter token latency cannot be negative")
	}
	if c.InterTokenLatencyStdDev < 0 {
		return errors.New("inter token latency standard deviation cannot be negative")
	}
	if float32(c.InterTokenLatencyStdDev) > 0.3*float32(c.InterTokenLatency) {
		return errors.New("inter token latency standard deviation cannot be more than 30% of inter token latency")
	}
	if c.TimeToFirstToken < 0 {
		return errors.New("time to first token cannot be negative")
	}
	if c.TimeToFirstTokenStdDev < 0 {
		return errors.New("time to first token standard deviation cannot be negative")
	}
	if float32(c.TimeToFirstTokenStdDev) > 0.3*float32(c.TimeToFirstToken) {
		return errors.New("time to first token standard deviation cannot be more than 30% of time to first token")
	}

	if c.PrefillOverhead < 0 {
		return errors.New("prefill overhead cannot be negative")
	}
	if c.PrefillTimePerToken < 0 {
		return errors.New("prefill time per token cannot be negative")
	}
	if c.PrefillTimeStdDev < 0 {
		return errors.New("prefill time standard deviation cannot be negative")
	}
	if float32(c.PrefillTimeStdDev) > 0.3*float32(c.PrefillTimePerToken) {
		return errors.New("prefill time standard deviation cannot be more than 30% of prefill time per token")
	}

	if c.KVCacheTransferTimePerToken < 0 {
		return errors.New("kv-cache tranfer time per token cannot be negative")
	}
	if c.KVCacheTransferTimeStdDev < 0 {
		return errors.New("kv-cache tranfer time standard deviation cannot be negative")
	}
	if float32(c.KVCacheTransferTimeStdDev) > 0.3*float32(c.KVCacheTransferTimePerToken) {
		return errors.New("kv-cache tranfer time standard deviation cannot be more than 30% of kv-cache tranfer time")
	}

	if c.KVCacheTransferLatency < 0 {
		return errors.New("kv-cache tranfer time cannot be negative")
	}
	if c.KVCacheTransferLatencyStdDev < 0 {
		return errors.New("kv-cache tranfer time standard deviation cannot be negative")
	}
	if float32(c.KVCacheTransferLatencyStdDev) > 0.3*float32(c.KVCacheTransferLatency) {
		return errors.New("kv-cache tranfer standard deviation cannot be more than 30% of kv-cache tranfer")
	}

	if c.TimeFactorUnderLoad < 1.0 {
		return errors.New("time factor under load cannot be less than 1.0")
	}

	if c.MaxLoras < 1 {
		return errors.New("max LoRAs cannot be less than 1")
	}
	if c.MaxCPULoras == 0 {
		// max CPU LoRAs by default is same as max LoRAs
		c.MaxCPULoras = c.MaxLoras
	}
	if c.MaxCPULoras < c.MaxLoras {
		return errors.New("max CPU LoRAs cannot be less than max LoRAs")
	}
	if c.MaxModelLen < 1 {
		return errors.New("max model len cannot be less than 1")
	}

	if c.MaxNumSeqs < 1 {
		return errors.New("max num seqs cannot be less than 1")
	}

	if c.MaxWaitingQueueLength < 1 {
		return errors.New("max waiting queue size cannot be less than 1")
	}

	for _, lora := range c.LoraModules {
		if lora.Name == "" {
			return errors.New("empty LoRA name")
		}
		if lora.BaseModelName != "" && lora.BaseModelName != c.Model {
			return fmt.Errorf("unknown base model '%s' for LoRA '%s'", lora.BaseModelName, lora.Name)
		}
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

	if c.TokenBlockSize != 8 && c.TokenBlockSize != 16 && c.TokenBlockSize != 32 &&
		c.TokenBlockSize != 64 && c.TokenBlockSize != 128 {
		return errors.New("token block size should be one of the following: 8, 16, 32, 64, 128")
	}

	if c.KVCacheSize < 0 {
		return errors.New("KV cache size cannot be negative")
	}
	if c.EventBatchSize < 1 {
		return errors.New("event batch size cannot less than 1")
	}

	if c.FailureInjectionRate < 0 || c.FailureInjectionRate > 100 {
		return errors.New("failure injection rate should be between 0 and 100")
	}

	validFailureTypes := map[string]bool{
		FailureTypeRateLimit:      true,
		FailureTypeInvalidAPIKey:  true,
		FailureTypeContextLength:  true,
		FailureTypeServerError:    true,
		FailureTypeInvalidRequest: true,
		FailureTypeModelNotFound:  true,
	}
	for _, failureType := range c.FailureTypes {
		if !validFailureTypes[failureType] {
			return fmt.Errorf("invalid failure type '%s', valid types are: %s, %s, %s, %s, %s, %s", failureType,
				FailureTypeRateLimit, FailureTypeInvalidAPIKey, FailureTypeContextLength,
				FailureTypeServerError, FailureTypeInvalidRequest, FailureTypeModelNotFound)
		}
	}

	if c.FakeMetrics != nil {
		if err := c.FakeMetrics.validate(); err != nil {
			return err
		}
		if c.FakeMetricsRefreshInterval <= 0 {
			return errors.New("fake metrics refresh interval must be positive")
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

	if c.GlobalCacheHitThreshold < 0 || c.GlobalCacheHitThreshold > 1 {
		return errors.New("global cache hit threshold must be between in range [0, 1]")
	}

	if c.DefaultEmbeddingDimensions < 1 {
		return errors.New("default embedding dimensions must be at least 1")
	}

	return nil
}

// SSLEnabled returns true if SSL is enabled either via certificate files or self-signed certificates
func (c *Configuration) SSLEnabled() bool {
	return (c.SSLCertFile != "" && c.SSLKeyFile != "") || c.SelfSignedCerts
}

func (c *Configuration) Copy() (*Configuration, error) {
	var dst Configuration
	data, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &dst)
	return &dst, err
}

func (c *Configuration) Show(logger logr.Logger) error {
	cfgJSON, err := json.Marshal(c)
	if err != nil {
		return fmt.Errorf("failed to marshal configuration to JSON: %w", err)
	}

	var m map[string]interface{}
	err = json.Unmarshal(cfgJSON, &m)
	if err != nil {
		return fmt.Errorf("failed to unmarshal JSON to map: %w", err)
	}
	if c.DPSize > 1 {
		// remove the port
		delete(m, "port")
	}
	// clean LoraModulesString field
	m["lora-modules"] = m["LoraModules"]
	delete(m, "LoraModules")
	delete(m, "LoraModulesString")

	// clean fake-metrics field
	if field, ok := m["fake-metrics"].(map[string]interface{}); ok {
		delete(field, "LorasString")
	}

	// show in JSON
	cfgJSON, err = json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal configuration to JSON: %w", err)
	}
	logger.V(logging.INFO).Info("Configuration:", "", string(cfgJSON))
	return nil
}
