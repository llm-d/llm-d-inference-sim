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
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type configuration struct {
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
	// LoraModules is a list of LoRA adapters
	LoraModules loraModulesValue `yaml:"lora-modules"`

	// TimeToFirstToken time before the first token will be returned, in milliseconds
	TimeToFirstToken int `yaml:"time-to-first-token"`
	// InterTokenLatency time between generated tokens, in milliseconds
	InterTokenLatency int `yaml:"inter-token-latency"`
	// Mode defines the simulator response generation mode, valid values: echo, random
	Mode string `yaml:"mode"`
}

type loraModule struct {
	// Name is the LoRA's name
	Name string `yaml:"name"`
	// Path is the LoRA's path
	Path string `yaml:"path"`
	// BaseModelName is the LoRA's base model
	BaseModelName string `yaml:"base_model_name"`
}

type loraModulesValue []loraModule

func (l *loraModulesValue) String() string {
	b, _ := json.Marshal(l)
	return string(b)
}

func (l *loraModulesValue) Set(val string) error {
	return json.Unmarshal([]byte(val), l)
}

func (l *loraModulesValue) Type() string {
	return "loras"
}

// Implement custom YAML unmarshaling for just this type
func (l *loraModulesValue) UnmarshalYAML(unmarshal func(interface{}) error) error {
	// Try parsing as an array of loraModule
	var arr []loraModule
	if err := unmarshal(&arr); err == nil {
		*l = arr
		return nil
	}
	// Try parsing as a JSON string
	var str string
	if err := unmarshal(&str); err == nil {
		return json.Unmarshal([]byte(str), l)
	}
	return errors.New("lora-modules: invalid format")
}

func newConfig() *configuration {
	return &configuration{
		Port:       vLLMDefaultPort,
		MaxLoras:   1,
		MaxNumSeqs: 5,
		Mode:       modeRandom,
	}
}

func (c *configuration) load(configFile string) error {
	configBytes, err := os.ReadFile(configFile)
	if err != nil {
		return fmt.Errorf("failed to read configuration file: %s", err)
	}

	if err := yaml.Unmarshal(configBytes, &c); err != nil {
		return fmt.Errorf("failed to unmarshal configuration: %s", err)
	}
	return nil
}

func (c *configuration) validate() error {
	if c.Model == "" {
		return errors.New("model parameter is empty")
	}
	// Upstream vLLM behaviour: when --served-model-name is not provided,
	// it falls back to using the value of --model as the single public name
	// returned by the API and exposed in Prometheus metrics.
	if len(c.ServedModelNames) == 0 {
		c.ServedModelNames = []string{c.Model}
	}

	if c.Mode != modeEcho && c.Mode != modeRandom {
		return fmt.Errorf("invalid mode '%s', valid values are 'random' and 'echo'", c.Mode)
	}
	if c.Port <= 0 {
		return fmt.Errorf("invalid port '%d'", c.Port)
	}
	if c.InterTokenLatency < 0 {
		return errors.New("inter token latency cannot be negative")
	}
	if c.TimeToFirstToken < 0 {
		return errors.New("time to first token cannot be negative")
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

	for _, lora := range c.LoraModules {
		if lora.Name == "" {
			return errors.New("empty LoRA name")
		}
		if lora.BaseModelName != "" && lora.BaseModelName != c.Model {
			return fmt.Errorf("unknown base model '%s' for LoRA '%s'", lora.BaseModelName, lora.Name)
		}
	}

	return nil
}
