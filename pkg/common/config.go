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
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/pflag"
	"gopkg.in/yaml.v3"
	"k8s.io/klog/v2"
)

const (
	vLLMDefaultPort = 8000
	ModeRandom      = "random"
	ModeEcho        = "echo"
	dummy           = "dummy"

	// Failure type constants
	FailureTypeRateLimit      = "rate_limit"
	FailureTypeInvalidAPIKey  = "invalid_api_key"
	FailureTypeContextLength  = "context_length"
	FailureTypeServerError    = "server_error"
	FailureTypeInvalidRequest = "invalid_request"
	FailureTypeModelNotFound  = "model_not_found"
)

type Configuration struct {
	// Port defines on which port the simulator runs
	Port int `yaml:"port" json:"port"`
	// Model defines the current base model name
	Model string `yaml:"model" json:"model"`
	// ServedModelNames is one or many model names exposed by the API
	ServedModelNames []string `yaml:"served-model-name" json:"served-model-name"`
	// MaxLoras defines maximum number of loaded LoRAs
	MaxLoras int `yaml:"max-loras" json:"max-loras"`
	// MaxCPULoras defines maximum number of LoRAs to store in CPU memory
	MaxCPULoras int `yaml:"max-cpu-loras" json:"max-cpu-loras"`
	// MaxNumSeqs is maximum number of sequences per iteration (the maximum
	// number of inference requests that could be processed at the same time)
	MaxNumSeqs int `yaml:"max-num-seqs" json:"max-num-seqs"`
	// MaxModelLen is the model's context window, the maximum number of tokens
	// in a single request including input and output. Default value is 1024.
	MaxModelLen int `yaml:"max-model-len" json:"max-model-len"`
	// LoraModulesString is a list of LoRA adapters as strings
	LoraModulesString []string `yaml:"lora-modules" json:"lora-modules"`
	// LoraModules is a list of LoRA adapters
	LoraModules []LoraModule

	// TimeToFirstToken time before the first token will be returned, in milliseconds
	TimeToFirstToken int `yaml:"time-to-first-token" json:"time-to-first-token"`
	// TimeToFirstTokenStdDev standard deviation for time before the first token will be returned,
	// in milliseconds, optional, default is 0, can't be more than 30% of TimeToFirstToken, will not
	// cause the actual time to first token to differ by more than 70% from TimeToFirstToken
	TimeToFirstTokenStdDev int `yaml:"time-to-first-token-std-dev" json:"time-to-first-token-std-dev"`

	// InterTokenLatency time between generated tokens, in milliseconds
	InterTokenLatency int `yaml:"inter-token-latency" json:"inter-token-latency"`
	// InterTokenLatencyStdDev standard deviation for time between generated tokens, in milliseconds,
	// optional, default is 0, can't be more than 30% of InterTokenLatency, will not cause the actual
	// inter token latency to differ by more than 70% from InterTokenLatency
	InterTokenLatencyStdDev int `yaml:"inter-token-latency-std-dev" json:"inter-token-latency-std-dev"`
	// KVCacheTransferLatency time to "transfer" kv-cache from another vLLM instance in case P/D is activated,
	// in milliseconds
	KVCacheTransferLatency int `yaml:"kv-cache-transfer-latency" json:"kv-cache-transfer-latency"`
	// KVCacheTransferLatencyStdDev standard deviation for time to "transfer" kv-cache from another
	// vLLM instance in case P/D is activated, in milliseconds, optional, default is 0, can't be more
	// than 30% of KVCacheTransferLatency, will not cause the actual latency to differ by more than 70% from
	// KVCacheTransferLatency
	KVCacheTransferLatencyStdDev int `yaml:"kv-cache-transfer-latency-std-dev" json:"kv-cache-transfer-latency-std-dev"`

	// $Total Prefill Time = PrefillOverhead + n * PrefillTimePerToken$
	// the assumption is that n is less than k, where k is the number of prallelism units of GPU
	// PrefillOverhead time taken to prefill the context, in milliseconds
	PrefillOverhead     int `yaml:"prefill-overhead" json:"prefill-overhead"`
	PrefillTimePerToken int `yaml:"prefill-time-per-token" json:"prefill-time-per-token"`
	// PrefillOverheadStdDev similar to TimeToFirstTokenStdDev
	PrefillTimeStdDev int `yaml:"prefill-time-std-dev" json:"prefill-time-std-dev"`
	// $Total KV Cache Transfer Time = n * KVCacheTransferTimePerToken$
	// the assumption is that the cache blocks are all missed at the remote pod
	// KVCacheTransfer overhead time taken to transfer kv-cache from another vLLM instance in case P/D is activated,
	// in milliseconds.
	KVCacheTransferTimePerToken int `yaml:"kv-cache-transfer-time-per-token" json:"kv-cache-transfer-time-per-token"`
	// KVCacheTransferOverheadStdDev similar to TimeToFirstTokenStdDev
	KVCacheTransferTimeStdDev int `yaml:"kv-cache-transfer-time-std-dev" json:"kv-cache-transfer-time-std-dev"`

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

	// TokenizersCacheDir is the directory for caching tokenizers
	TokenizersCacheDir string `yaml:"tokenizers-cache-dir" json:"tokenizers-cache-dir"`
	// TokenBlockSize is token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128, defaults to 16
	TokenBlockSize int `yaml:"block-size" json:"block-size"`
	// HashSeed is the seed for hash generation (if not set, is read from PYTHONHASHSEED environment variable)
	HashSeed string `yaml:"hash-seed" json:"hash-seed"`

	// ZMQEndpoint is the ZMQ address to publish events, the default value is tcp://localhost:5557
	ZMQEndpoint string `yaml:"zmq-endpoint" json:"zmq-endpoint"`
	// ZMQMaxConnectAttempts defines the maximum number (10) of retries when ZMQ connection fails
	ZMQMaxConnectAttempts uint `yaml:"zmq-max-connect-attempts" json:"zmq-max-connect-attempts"`

	// EventBatchSize is the maximum number of kv-cache events to be sent together, defaults to 16
	EventBatchSize int `yaml:"event-batch-size" json:"event-batch-size"`

	// FakeMetrics is a set of metrics to send to Prometheus instead of the real data
	FakeMetrics *Metrics `yaml:"fake-metrics" json:"fake-metrics"`

	// FailureInjectionRate is the probability (0-100) of injecting failures
	FailureInjectionRate int `yaml:"failure-injection-rate" json:"failure-injection-rate"`
	// FailureTypes is a list of specific failure types to inject (empty means all types)
	FailureTypes []string `yaml:"failure-types" json:"failure-types"`

	// DPSize is data parallel size - a number of ranks to run, minimum is 1, maximum is 8, default is 1
	DPSize int `yaml:"data-parallel-size" json:"data-parallel-size"`

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
}

type Metrics struct {
	// LoraMetrics
	LoraMetrics []LorasMetrics `json:"loras"`
	LorasString []string       `yaml:"loras"`
	// RunningRequests is the number of inference requests that are currently being processed
	RunningRequests int64 `yaml:"running-requests" json:"running-requests"`
	// WaitingRequests is the number of inference requests that are waiting to be processed
	WaitingRequests int64 `yaml:"waiting-requests" json:"waiting-requests"`
	// KVCacheUsagePercentage  is the fraction of KV-cache blocks currently in use (from 0 to 1)
	KVCacheUsagePercentage float32 `yaml:"kv-cache-usage" json:"kv-cache-usage"`
}

type LorasMetrics struct {
	// RunningLoras is a comma separated list of running LoRAs
	RunningLoras string `json:"running"`
	// WaitingLoras is a comma separated list of waiting LoRAs
	WaitingLoras string `json:"waiting"`
	// Timestamp is the timestamp of the metric
	Timestamp float64 `json:"timestamp"`
}

type LoraModule struct {
	// Name is the LoRA's name
	Name string `json:"name"`
	// Path is the LoRA's path
	Path string `json:"path"`
	// BaseModelName is the LoRA's base model
	BaseModelName string `json:"base_model_name"`
}

// Needed to parse values that contain multiple strings
type multiString struct {
	values []string
}

func (l *multiString) String() string {
	return strings.Join(l.values, " ")
}

func (l *multiString) Set(val string) error {
	l.values = append(l.values, val)
	return nil
}

func (l *multiString) Type() string {
	return "strings"
}

func (c *Configuration) unmarshalLoras() error {
	c.LoraModules = make([]LoraModule, 0)
	for _, jsonStr := range c.LoraModulesString {
		var lora LoraModule
		if err := json.Unmarshal([]byte(jsonStr), &lora); err != nil {
			return err
		}
		c.LoraModules = append(c.LoraModules, lora)
	}
	return nil
}

func (c *Configuration) unmarshalFakeMetrics(fakeMetricsString string) error {
	var metrics *Metrics
	if err := json.Unmarshal([]byte(fakeMetricsString), &metrics); err != nil {
		return err
	}
	c.FakeMetrics = metrics
	return nil
}

func (c *Configuration) unmarshalLoraFakeMetrics() error {
	if c.FakeMetrics != nil {
		c.FakeMetrics.LoraMetrics = make([]LorasMetrics, 0)
		for _, jsonStr := range c.FakeMetrics.LorasString {
			var lora LorasMetrics
			if err := json.Unmarshal([]byte(jsonStr), &lora); err != nil {
				return err
			}
			c.FakeMetrics.LoraMetrics = append(c.FakeMetrics.LoraMetrics, lora)
		}
	}
	return nil
}

func newConfig() *Configuration {
	return &Configuration{
		Port:                                vLLMDefaultPort,
		MaxLoras:                            1,
		MaxNumSeqs:                          5,
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
		KVCacheSize:    1024,
		TokenBlockSize: 16,
		ZMQEndpoint:    "tcp://localhost:5557",
		EventBatchSize: 16,
		DPSize:         1,
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

	if c.ZMQMaxConnectAttempts > 10 {
		return errors.New("zmq retries times cannot be more than 10")
	}

	if c.FakeMetrics != nil {
		if c.FakeMetrics.RunningRequests < 0 || c.FakeMetrics.WaitingRequests < 0 {
			return errors.New("fake metrics request counters cannot be negative")
		}
		if c.FakeMetrics.KVCacheUsagePercentage < 0 || c.FakeMetrics.KVCacheUsagePercentage > 1 {
			return errors.New("fake metrics KV cache usage must be between 0 ans 1")
		}
	}

	if c.DPSize < 1 || c.DPSize > 8 {
		return errors.New("data parallel size must be between 1 ans 8")
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

// ParseCommandParamsAndLoadConfig loads configuration, parses command line parameters, merges the values
// (command line values overwrite the config file ones), and validates the configuration
func ParseCommandParamsAndLoadConfig() (*Configuration, error) {
	config := newConfig()

	configFileValues := getParamValueFromArgs("config")
	if len(configFileValues) == 1 {
		if err := config.load(configFileValues[0]); err != nil {
			return nil, err
		}
	}

	servedModelNames := getParamValueFromArgs("served-model-name")
	loraModuleNames := getParamValueFromArgs("lora-modules")
	fakeMetrics := getParamValueFromArgs("fake-metrics")

	f := pflag.NewFlagSet("llm-d-inference-sim flags", pflag.ContinueOnError)

	f.IntVar(&config.Port, "port", config.Port, "Port")
	f.StringVar(&config.Model, "model", config.Model, "Currently 'loaded' model")
	f.IntVar(&config.MaxNumSeqs, "max-num-seqs", config.MaxNumSeqs, "Maximum number of inference requests that could be processed at the same time (parameter to simulate requests waiting queue)")
	f.IntVar(&config.MaxLoras, "max-loras", config.MaxLoras, "Maximum number of LoRAs in a single batch")
	f.IntVar(&config.MaxCPULoras, "max-cpu-loras", config.MaxCPULoras, "Maximum number of LoRAs to store in CPU memory")
	f.IntVar(&config.MaxModelLen, "max-model-len", config.MaxModelLen, "Model's context window, maximum number of tokens in a single request including input and output")

	f.StringVar(&config.Mode, "mode", config.Mode, "Simulator mode: echo - returns the same text that was sent in the request, for chat completion returns the last message; random - returns random sentence from a bank of pre-defined sentences")
	f.IntVar(&config.InterTokenLatency, "inter-token-latency", config.InterTokenLatency, "Time to generate one token (in milliseconds)")
	f.IntVar(&config.TimeToFirstToken, "time-to-first-token", config.TimeToFirstToken, "Time to first token (in milliseconds)")

	f.IntVar(&config.PrefillOverhead, "prefill-overhead", config.PrefillOverhead, "Time to prefill in milliseconds. This argument is ignored if <time-to-first-token> is not 0.")
	f.IntVar(&config.PrefillTimePerToken, "prefill-time-per-token", config.PrefillTimePerToken, "Time to prefill per token (in milliseconds)")
	f.IntVar(&config.PrefillTimeStdDev, "prefill-time-std-dev", config.PrefillTimeStdDev, "Standard deviation for time to prefill (in milliseconds)")
	f.IntVar(&config.KVCacheTransferTimePerToken, "kv-cache-transfer-time-per-token", config.KVCacheTransferTimePerToken, "Time for KV-cache transfer per token from a remote vLLM (in milliseconds)")
	f.IntVar(&config.KVCacheTransferTimeStdDev, "kv-cache-transfer-time-std-dev", config.KVCacheTransferTimeStdDev, "Standard deviation for time for KV-cache transfer per token from a remote vLLM (in milliseconds)")

	f.IntVar(&config.KVCacheTransferLatency, "kv-cache-transfer-latency", config.KVCacheTransferLatency, "Time for KV-cache transfer from a remote vLLM (in milliseconds)")
	f.IntVar(&config.InterTokenLatencyStdDev, "inter-token-latency-std-dev", config.InterTokenLatencyStdDev, "Standard deviation for time between generated tokens (in milliseconds)")
	f.IntVar(&config.TimeToFirstTokenStdDev, "time-to-first-token-std-dev", config.TimeToFirstTokenStdDev, "Standard deviation for time before the first token will be returned (in milliseconds)")
	f.IntVar(&config.KVCacheTransferLatencyStdDev, "kv-cache-transfer-latency-std-dev", config.KVCacheTransferLatencyStdDev, "Standard deviation for time for KV-cache transfer from a remote vLLM (in milliseconds)")
	f.Int64Var(&config.Seed, "seed", config.Seed, "Random seed for operations (if not set, current Unix time in nanoseconds is used)")
	f.Float64Var(&config.TimeFactorUnderLoad, "time-factor-under-load", config.TimeFactorUnderLoad, "Time factor under load (must be >= 1.0)")

	f.IntVar(&config.MaxToolCallIntegerParam, "max-tool-call-integer-param", config.MaxToolCallIntegerParam, "Maximum possible value of integer parameters in a tool call")
	f.IntVar(&config.MinToolCallIntegerParam, "min-tool-call-integer-param", config.MinToolCallIntegerParam, "Minimum possible value of integer parameters in a tool call")
	f.Float64Var(&config.MaxToolCallNumberParam, "max-tool-call-number-param", config.MaxToolCallNumberParam, "Maximum possible value of number (float) parameters in a tool call")
	f.Float64Var(&config.MinToolCallNumberParam, "min-tool-call-number-param", config.MinToolCallNumberParam, "Minimum possible value of number (float) parameters in a tool call")
	f.IntVar(&config.MaxToolCallArrayParamLength, "max-tool-call-array-param-length", config.MaxToolCallArrayParamLength, "Maximum possible length of array parameters in a tool call")
	f.IntVar(&config.MinToolCallArrayParamLength, "min-tool-call-array-param-length", config.MinToolCallArrayParamLength, "Minimum possible length of array parameters in a tool call")
	f.IntVar(&config.ToolCallNotRequiredParamProbability, "tool-call-not-required-param-probability", config.ToolCallNotRequiredParamProbability, "Probability to add a parameter, that is not required, in a tool call")
	f.IntVar(&config.ObjectToolCallNotRequiredParamProbability, "object-tool-call-not-required-field-probability", config.ObjectToolCallNotRequiredParamProbability, "Probability to add a field, that is not required, in an object in a tool call")

	f.BoolVar(&config.EnableKVCache, "enable-kvcache", config.EnableKVCache, "Defines if KV cache feature is enabled")
	f.IntVar(&config.KVCacheSize, "kv-cache-size", config.KVCacheSize, "Maximum number of token blocks in kv cache")
	f.IntVar(&config.TokenBlockSize, "block-size", config.TokenBlockSize, "Token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128")
	f.StringVar(&config.TokenizersCacheDir, "tokenizers-cache-dir", config.TokenizersCacheDir, "Directory for caching tokenizers")
	f.StringVar(&config.HashSeed, "hash-seed", config.HashSeed, "Seed for hash generation (if not set, is read from PYTHONHASHSEED environment variable)")
	f.StringVar(&config.ZMQEndpoint, "zmq-endpoint", config.ZMQEndpoint, "ZMQ address to publish events")
	f.UintVar(&config.ZMQMaxConnectAttempts, "zmq-max-connect-attempts", config.ZMQMaxConnectAttempts, "Maximum number of times to try ZMQ connect")
	f.IntVar(&config.EventBatchSize, "event-batch-size", config.EventBatchSize, "Maximum number of kv-cache events to be sent together")
	f.IntVar(&config.DPSize, "data-parallel-size", config.DPSize, "Number of ranks to run")

	f.StringVar(&config.DatasetPath, "dataset-path", config.DatasetPath, "Local path to the sqlite db file for response generation from a dataset")
	f.StringVar(&config.DatasetURL, "dataset-url", config.DatasetURL, "URL to download the sqlite db file for response generation from a dataset")
	f.BoolVar(&config.DatasetInMemory, "dataset-in-memory", config.DatasetInMemory, "Load the entire dataset into memory for faster access")

	f.IntVar(&config.FailureInjectionRate, "failure-injection-rate", config.FailureInjectionRate, "Probability (0-100) of injecting failures")
	failureTypes := getParamValueFromArgs("failure-types")
	var dummyFailureTypes multiString
	failureTypesDescription := fmt.Sprintf("List of specific failure types to inject (%s, %s, %s, %s, %s, %s)",
		FailureTypeRateLimit, FailureTypeInvalidAPIKey, FailureTypeContextLength, FailureTypeServerError, FailureTypeInvalidRequest,
		FailureTypeModelNotFound)
	f.Var(&dummyFailureTypes, "failure-types", failureTypesDescription)
	f.Lookup("failure-types").NoOptDefVal = dummy

	f.StringVar(&config.SSLCertFile, "ssl-certfile", config.SSLCertFile, "Path to SSL certificate file for HTTPS (optional)")
	f.StringVar(&config.SSLKeyFile, "ssl-keyfile", config.SSLKeyFile, "Path to SSL private key file for HTTPS (optional)")
	f.BoolVar(&config.SelfSignedCerts, "self-signed-certs", config.SelfSignedCerts, "Enable automatic generation of self-signed certificates for HTTPS")

	// These values were manually parsed above in getParamValueFromArgs, we leave this in order to get these flags in --help
	var dummyString string
	f.StringVar(&dummyString, "config", "", "The path to a yaml configuration file. The command line values overwrite the configuration file values")
	var dummyMultiString multiString
	f.Var(&dummyMultiString, "served-model-name", "Model names exposed by the API (a list of space-separated strings)")
	f.Var(&dummyMultiString, "lora-modules", "List of LoRA adapters (a list of space-separated JSON strings)")
	f.Var(&dummyMultiString, "fake-metrics", "A set of metrics to report to Prometheus instead of the real metrics")
	// In order to allow empty arguments, we set a dummy NoOptDefVal for these flags
	f.Lookup("served-model-name").NoOptDefVal = dummy
	f.Lookup("lora-modules").NoOptDefVal = dummy
	f.Lookup("fake-metrics").NoOptDefVal = dummy

	flagSet := flag.NewFlagSet("simFlagSet", flag.ExitOnError)
	klog.InitFlags(flagSet)
	f.AddGoFlagSet(flagSet)

	if err := f.Parse(os.Args[1:]); err != nil {
		if err == pflag.ErrHelp {
			// --help - exit without printing an error message
			os.Exit(0)
		}
		return nil, err
	}

	// Need to read in a variable to avoid merging the values with the config file ones
	if loraModuleNames != nil {
		config.LoraModulesString = loraModuleNames
		if err := config.unmarshalLoras(); err != nil {
			return nil, err
		}
	}
	if fakeMetrics != nil {
		if err := config.unmarshalFakeMetrics(fakeMetrics[0]); err != nil {
			return nil, err
		}
	}
	if servedModelNames != nil {
		config.ServedModelNames = servedModelNames
	}
	if failureTypes != nil {
		config.FailureTypes = failureTypes
	}

	if config.HashSeed == "" {
		hashSeed := os.Getenv("PYTHONHASHSEED")
		if hashSeed != "" {
			config.HashSeed = hashSeed
		}
	}

	if err := config.validate(); err != nil {
		return nil, err
	}

	return config, nil
}

func getParamValueFromArgs(param string) []string {
	var values []string
	var readValues bool
	for _, arg := range os.Args[1:] {
		if readValues {
			if strings.HasPrefix(arg, "--") {
				break
			}
			if arg != "" {
				values = append(values, arg)
			}
		} else {
			if arg == "--"+param {
				readValues = true
				values = make([]string, 0)
			} else if strings.HasPrefix(arg, "--"+param+"=") {
				// Handle --param=value
				values = append(values, strings.TrimPrefix(arg, "--"+param+"="))
				break
			}
		}
	}
	return values
}
