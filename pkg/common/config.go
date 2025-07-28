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
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	vLLMDefaultPort = 8000
	ModeRandom      = "random"
	ModeEcho        = "echo"
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

	// EnableKVCache defines is kv cache feature will be enabled
	EnableKVCache bool `yaml:"enable-kvcache"`
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
type MultiString struct {
	Values []string
}

func (l *MultiString) String() string {
	return strings.Join(l.Values, " ")
}

func (l *MultiString) Set(val string) error {
	l.Values = append(l.Values, val)
	return nil
}

func (l *MultiString) Type() string {
	return "strings"
}

func (c *Configuration) UnmarshalLoras() error {
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

func NewConfig() *Configuration {
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
		EnableKVCache: false,
	}
}

func (c *Configuration) Load(configFile string) error {
	configBytes, err := os.ReadFile(configFile)
	if err != nil {
		return fmt.Errorf("failed to read configuration file: %s", err)
	}

	if err := yaml.Unmarshal(configBytes, &c); err != nil {
		return fmt.Errorf("failed to unmarshal configuration: %s", err)
	}

	return c.UnmarshalLoras()
}

func (c *Configuration) Validate() error {
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
	return nil
}
