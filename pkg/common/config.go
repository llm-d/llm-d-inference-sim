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
	ModeFailure     = "failure"
)

type Configuration struct {
	// Port defines on which port the simulator runs
	Port int `yaml:"port"`
	// Model defines the current base model name
	Model string `yaml:"model"`
	// ServedModelNames is one or many model names exposed by the API
	ServedModelNames []string `yaml:"served-model-name"`
	// MaxLoras defines maximum number of loaded LoRAs
	MaxLoras int `yaml:"max-loras"`
	// MaxCPULoras defines maximum number of LoRAs to store in CPU memory
	MaxCPULoras int `yaml:"max-cpu-loras"`
	// MaxNumSeqs is maximum number of sequences per iteration (the maximum
	// number of inference requests that could be processed at the same time)
	MaxNumSeqs int `yaml:"max-num-seqs"`
	// MaxModelLen is the model's context window, the maximum number of tokens
	// in a single request including input and output. Default value is 1024.
	MaxModelLen int `yaml:"max-model-len"`
	// LoraModulesString is a list of LoRA adapters as strings
	LoraModulesString []string `yaml:"lora-modules"`
	// LoraModules is a list of LoRA adapters
	LoraModules []LoraModule

	// TimeToFirstToken time before the first token will be returned, in milliseconds
	TimeToFirstToken int `yaml:"time-to-first-token"`
	// TimeToFirstTokenStdDev standard deviation for time before the first token will be returned,
	// in milliseconds, optional, default is 0, can't be more than 30% of TimeToFirstToken, will not
	// cause the actual time to first token to differ by more than 70% from TimeToFirstToken
	TimeToFirstTokenStdDev int `yaml:"time-to-first-token-std-dev"`
	// InterTokenLatency time between generated tokens, in milliseconds
	InterTokenLatency int `yaml:"inter-token-latency"`
	// InterTokenLatencyStdDev standard deviation for time between generated tokens, in milliseconds,
	// optional, default is 0, can't be more than 30% of InterTokenLatency, will not cause the actual
	// inter token latency to differ by more than 70% from InterTokenLatency
	InterTokenLatencyStdDev int `yaml:"inter-token-latency-std-dev"`
	// KVCacheTransferLatency time to "transfer" kv-cache from another vLLM instance in case P/D is activated,
	// in milliseconds
	KVCacheTransferLatency int `yaml:"kv-cache-transfer-latency"`
	// KVCacheTransferLatencyStdDev standard deviation for time to "transfer" kv-cache from another
	// vLLM instance in case P/D is activated, in milliseconds, optional, default is 0, can't be more
	// than 30% of KVCacheTransferLatency, will not cause the actual latency to differ by more than 70% from
	// KVCacheTransferLatency
	KVCacheTransferLatencyStdDev int `yaml:"kv-cache-transfer-latency-std-dev"`

	// Mode defines the simulator response generation mode, valid values: echo, random
	Mode string `yaml:"mode"`
	// Seed defines random seed for operations
	Seed int64 `yaml:"seed"`

	// MaxToolCallIntegerParam defines the maximum possible value of integer parameters in a tool call,
	// optional, defaults to 100
	MaxToolCallIntegerParam int `yaml:"max-tool-call-integer-param"`
	// MinToolCallIntegerParam defines the minimum possible value of integer parameters in a tool call,
	// optional, defaults to 0
	MinToolCallIntegerParam int `yaml:"min-tool-call-integer-param"`
	// MaxToolCallNumberParam defines the maximum possible value of number (float) parameters in a tool call,
	// optional, defaults to 100
	MaxToolCallNumberParam float64 `yaml:"max-tool-call-number-param"`
	// MinToolCallNumberParam defines the minimum possible value of number (float) parameters in a tool call,
	// optional, defaults to 0
	MinToolCallNumberParam float64 `yaml:"min-tool-call-number-param"`

	// MaxToolCallArrayParamLength defines the maximum possible length of array parameters in a tool call,
	// optional, defaults to 5
	MaxToolCallArrayParamLength int `yaml:"max-tool-call-array-param-length"`
	// MinToolCallArrayParamLength defines the minimum possible length of array parameters in a tool call,
	// optional, defaults to 1
	MinToolCallArrayParamLength int `yaml:"min-tool-call-array-param-length"`

	// ToolCallNotRequiredParamProbability is the probability to add a parameter, that is not required,
	// in a tool call, optional, defaults to 50
	ToolCallNotRequiredParamProbability int `yaml:"tool-call-not-required-param-probability"`
	// ObjectToolCallNotRequiredParamProbability is the probability to add a field, that is not required,
	// in an object in a tool call, optional, defaults to 50
	ObjectToolCallNotRequiredParamProbability int `yaml:"object-tool-call-not-required-field-probability"`

	// EnableKVCache defines if kv cache feature will be enabled
	EnableKVCache bool `yaml:"enable-kvcache"`
	//  KVCacheSize is the maximum number of token blocks in kv cache, the default value is 1024
	KVCacheSize int `yaml:"kv-cache-size"`

	// TokenizersCacheDir is the directory for caching tokenizers
	TokenizersCacheDir string `yaml:"tokenizers-cache-dir"`
	// TokenBlockSize is token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128, defaults to 16
	TokenBlockSize int `yaml:"block-size"`
	// HashSeed is the seed for hash generation (if not set, is read from PYTHONHASHSEED environment variable)
	HashSeed string `yaml:"hash-seed"`

	// ZMQEndpoint is the ZMQ address to publish events, the default value is tcp://localhost:5557
	ZMQEndpoint string `yaml:"zmq-endpoint"`
	// EventBatchSize is the maximum number of kv-cache events to be sent together, defaults to 16
	EventBatchSize int `yaml:"event-batch-size"`

	// FailureInjectionRate is the probability (0-100) of injecting failures when in failure mode
	FailureInjectionRate int `yaml:"failure-injection-rate"`
	// FailureTypes is a list of specific failure types to inject (empty means all types)
	FailureTypes []string `yaml:"failure-types"`
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

func newConfig() *Configuration {
	return &Configuration{
		Port:                                vLLMDefaultPort,
		MaxLoras:                            1,
		MaxNumSeqs:                          5,
		MaxModelLen:                         1024,
		Mode:                                ModeRandom,
		Seed:                                time.Now().UnixNano(),
		MaxToolCallIntegerParam:             100,
		MaxToolCallNumberParam:              100,
		MaxToolCallArrayParamLength:         5,
		MinToolCallArrayParamLength:         1,
		ToolCallNotRequiredParamProbability: 50,
		ObjectToolCallNotRequiredParamProbability: 50,
		KVCacheSize:        1024,
		TokenBlockSize:     16,
		ZMQEndpoint:        "tcp://localhost:5557",
		EventBatchSize:     16,
		FailureInjectionRate: 10,
		FailureTypes:       []string{},
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

	return c.unmarshalLoras()
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

	if c.Mode != ModeEcho && c.Mode != ModeRandom && c.Mode != ModeFailure {
		return fmt.Errorf("invalid mode '%s', valid values are 'random', 'echo', and 'failure'", c.Mode)
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
	if c.KVCacheTransferLatency < 0 {
		return errors.New("kv-cache tranfer time cannot be negative")
	}
	if c.KVCacheTransferLatencyStdDev < 0 {
		return errors.New("kv-cache tranfer time standard deviation cannot be negative")
	}
	if float32(c.KVCacheTransferLatencyStdDev) > 0.3*float32(c.KVCacheTransferLatency) {
		return errors.New("kv-cache tranfer standard deviation cannot be more than 30% of kv-cache tranfer")
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
		"rate_limit":        true,
		"invalid_api_key":   true,
		"context_length":    true,
		"server_error":      true,
		"invalid_request":   true,
		"model_not_found":   true,
	}
	for _, failureType := range c.FailureTypes {
		if !validFailureTypes[failureType] {
			return fmt.Errorf("invalid failure type '%s', valid types are: rate_limit, invalid_api_key, context_length, server_error, invalid_request, model_not_found", failureType)
		}
	}

	return nil
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

	f := pflag.NewFlagSet("llm-d-inference-sim flags", pflag.ContinueOnError)

	f.IntVar(&config.Port, "port", config.Port, "Port")
	f.StringVar(&config.Model, "model", config.Model, "Currently 'loaded' model")
	f.IntVar(&config.MaxNumSeqs, "max-num-seqs", config.MaxNumSeqs, "Maximum number of inference requests that could be processed at the same time (parameter to simulate requests waiting queue)")
	f.IntVar(&config.MaxLoras, "max-loras", config.MaxLoras, "Maximum number of LoRAs in a single batch")
	f.IntVar(&config.MaxCPULoras, "max-cpu-loras", config.MaxCPULoras, "Maximum number of LoRAs to store in CPU memory")
	f.IntVar(&config.MaxModelLen, "max-model-len", config.MaxModelLen, "Model's context window, maximum number of tokens in a single request including input and output")

	f.StringVar(&config.Mode, "mode", config.Mode, "Simulator mode: echo - returns the same text that was sent in the request, for chat completion returns the last message; random - returns random sentence from a bank of pre-defined sentences; failure - randomly injects API errors")
	f.IntVar(&config.InterTokenLatency, "inter-token-latency", config.InterTokenLatency, "Time to generate one token (in milliseconds)")
	f.IntVar(&config.TimeToFirstToken, "time-to-first-token", config.TimeToFirstToken, "Time to first token (in milliseconds)")
	f.IntVar(&config.KVCacheTransferLatency, "kv-cache-transfer-latency", config.KVCacheTransferLatency, "Time for KV-cache transfer from a remote vLLM (in milliseconds)")
	f.IntVar(&config.InterTokenLatencyStdDev, "inter-token-latency-std-dev", config.InterTokenLatencyStdDev, "Standard deviation for time between generated tokens (in milliseconds)")
	f.IntVar(&config.TimeToFirstTokenStdDev, "time-to-first-token-std-dev", config.TimeToFirstTokenStdDev, "Standard deviation for time before the first token will be returned (in milliseconds)")
	f.IntVar(&config.KVCacheTransferLatencyStdDev, "kv-cache-transfer-latency-std-dev", config.KVCacheTransferLatencyStdDev, "Standard deviation for time for KV-cache transfer from a remote vLLM (in milliseconds)")
	f.Int64Var(&config.Seed, "seed", config.Seed, "Random seed for operations (if not set, current Unix time in nanoseconds is used)")

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
	f.IntVar(&config.EventBatchSize, "event-batch-size", config.EventBatchSize, "Maximum number of kv-cache events to be sent together")
	
	f.IntVar(&config.FailureInjectionRate, "failure-injection-rate", config.FailureInjectionRate, "Probability (0-100) of injecting failures when in failure mode")

	failureTypes := getParamValueFromArgs("failure-types")
	var dummyFailureTypes multiString
	f.Var(&dummyFailureTypes, "failure-types", "List of specific failure types to inject (rate_limit, invalid_api_key, context_length, server_error, invalid_request, model_not_found)")
	f.Lookup("failure-types").NoOptDefVal = "dummy"

	// These values were manually parsed above in getParamValueFromArgs, we leave this in order to get these flags in --help
	var dummyString string
	f.StringVar(&dummyString, "config", "", "The path to a yaml configuration file. The command line values overwrite the configuration file values")
	var dummyMultiString multiString
	f.Var(&dummyMultiString, "served-model-name", "Model names exposed by the API (a list of space-separated strings)")
	f.Var(&dummyMultiString, "lora-modules", "List of LoRA adapters (a list of space-separated JSON strings)")
	// In order to allow empty arguments, we set a dummy NoOptDefVal for these flags
	f.Lookup("served-model-name").NoOptDefVal = "dummy"
	f.Lookup("lora-modules").NoOptDefVal = "dummy"

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
