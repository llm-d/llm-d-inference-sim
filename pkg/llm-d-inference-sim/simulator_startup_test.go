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
	"os"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

const (
	qwenModelName = "Qwen/Qwen2-0.5B"
)

func createSimConfig(args []string) (*common.Configuration, error) {
	oldArgs := os.Args
	defer func() {
		os.Args = oldArgs
	}()
	os.Args = args

	s, err := New(klog.Background())
	if err != nil {
		return nil, err
	}
	if err := s.parseCommandParamsAndLoadConfig(); err != nil {
		return nil, err
	}
	return s.config, nil
}

func createDefaultConfig(model string) *common.Configuration {
	c := common.NewConfig()

	c.Model = model
	c.ServedModelNames = []string{c.Model}
	c.MaxNumSeqs = 5
	c.MaxLoras = 2
	c.MaxCPULoras = 5
	c.TimeToFirstToken = 2000
	c.InterTokenLatency = 1000
	c.KVCacheTransferLatency = 100
	c.Seed = 100100100
	c.LoraModules = []common.LoraModule{}

	return c
}

type testCase struct {
	name           string
	args           []string
	expectedConfig *common.Configuration
}

var _ = Describe("Simulator configuration", func() {
	tests := make([]testCase, 0)

	// Simple config with a few parameters
	c := common.NewConfig()
	c.Model = model
	c.ServedModelNames = []string{c.Model}
	c.MaxCPULoras = 1
	c.Seed = 100
	test := testCase{
		name:           "simple",
		args:           []string{"cmd", "--model", model, "--mode", common.ModeRandom, "--seed", "100"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file
	c = createDefaultConfig(qwenModelName)
	c.Port = 8001
	c.ServedModelNames = []string{"model1", "model2"}
	c.LoraModules = []common.LoraModule{{Name: "lora1", Path: "/path/to/lora1"}, {Name: "lora2", Path: "/path/to/lora2"}}
	test = testCase{
		name:           "config file",
		args:           []string{"cmd", "--config", "../../manifests/config.yaml"},
		expectedConfig: c,
	}
	c.LoraModulesString = []string{
		"{\"name\":\"lora1\",\"path\":\"/path/to/lora1\"}",
		"{\"name\":\"lora2\",\"path\":\"/path/to/lora2\"}",
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args
	c = createDefaultConfig(model)
	c.Port = 8002
	c.ServedModelNames = []string{"alias1", "alias2"}
	c.Seed = 100
	c.LoraModules = []common.LoraModule{{Name: "lora3", Path: "/path/to/lora3"}, {Name: "lora4", Path: "/path/to/lora4"}}
	c.LoraModulesString = []string{
		"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
		"{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}",
	}
	test = testCase{
		name: "config file with command line args",
		args: []string{"cmd", "--model", model, "--config", "../../manifests/config.yaml", "--port", "8002",
			"--served-model-name", "alias1", "alias2", "--seed", "100",
			"--lora-modules", "{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}", "{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with different format
	c = createDefaultConfig(model)
	c.Port = 8002
	c.LoraModules = []common.LoraModule{{Name: "lora3", Path: "/path/to/lora3"}}
	c.LoraModulesString = []string{
		"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
	}
	test = testCase{
		name: "config file with command line args with different format",
		args: []string{"cmd", "--model", model, "--config", "../../manifests/config.yaml", "--port", "8002",
			"--served-model-name",
			"--lora-modules={\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with empty string
	c = createDefaultConfig(model)
	c.Port = 8002
	c.LoraModules = []common.LoraModule{{Name: "lora3", Path: "/path/to/lora3"}}
	c.LoraModulesString = []string{
		"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
	}
	test = testCase{
		name: "config file with command line args with empty string",
		args: []string{"cmd", "--model", model, "--config", "../../manifests/config.yaml", "--port", "8002",
			"--served-model-name", "",
			"--lora-modules", "{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with empty string for loras
	c = createDefaultConfig(qwenModelName)
	c.Port = 8001
	c.ServedModelNames = []string{"model1", "model2"}
	c.LoraModulesString = []string{}
	test = testCase{
		name:           "config file with command line args with empty string for loras",
		args:           []string{"cmd", "--config", "../../manifests/config.yaml", "--lora-modules", ""},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with empty parameter for loras
	c = createDefaultConfig(qwenModelName)
	c.Port = 8001
	c.ServedModelNames = []string{"model1", "model2"}
	c.LoraModulesString = []string{}
	test = testCase{
		name:           "config file with command line args with empty parameter for loras",
		args:           []string{"cmd", "--config", "../../manifests/config.yaml", "--lora-modules"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with time to copy cache
	c = createDefaultConfig(qwenModelName)
	c.Port = 8001
	// basic config file does not contain properties related to lora
	c.MaxLoras = 1
	c.MaxCPULoras = 1
	c.KVCacheTransferLatency = 50
	test = testCase{
		name:           "config file with command line args with time to transfer kv-cache",
		args:           []string{"cmd", "--config", "../../manifests/basic-config.yaml", "--kv-cache-transfer-latency", "50"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	for _, test := range tests {
		When(test.name, func() {
			It("should create correct configuration", func() {
				config, err := createSimConfig(test.args)
				Expect(err).NotTo(HaveOccurred())
				Expect(config).To(Equal(test.expectedConfig))
			})
		})
	}

	// Invalid configurations
	invalidTests := []testCase{
		{
			name: "invalid model",
			args: []string{"cmd", "--model", "", "--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid port",
			args: []string{"cmd", "--port", "-50", "--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid max-loras",
			args: []string{"cmd", "--max-loras", "15", "--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid mode",
			args: []string{"cmd", "--mode", "hello", "--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid lora",
			args: []string{"cmd", "--config", "../../manifests/config.yaml",
				"--lora-modules", "[{\"path\":\"/path/to/lora15\"}]"},
		},
		{
			name: "invalid max-model-len",
			args: []string{"cmd", "--max-model-len", "0", "--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid tool-call-not-required-param-probability",
			args: []string{"cmd", "--tool-call-not-required-param-probability", "-10", "--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid max-tool-call-number-param",
			args: []string{"cmd", "--max-tool-call-number-param", "-10", "--min-tool-call-number-param", "0",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid max-tool-call-integer-param",
			args: []string{"cmd", "--max-tool-call-integer-param", "-10", "--min-tool-call-integer-param", "0",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid max-tool-call-array-param-length",
			args: []string{"cmd", "--max-tool-call-array-param-length", "-10", "--min-tool-call-array-param-length", "0",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid tool-call-not-required-param-probability",
			args: []string{"cmd", "--tool-call-not-required-param-probability", "-10",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid object-tool-call-not-required-field-probability",
			args: []string{"cmd", "--object-tool-call-not-required-field-probability", "1210",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid time-to-first-token-std-dev",
			args: []string{"cmd", "--time-to-first-token-std-dev", "3000",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid (negative) time-to-first-token-std-dev",
			args: []string{"cmd", "--time-to-first-token-std-dev", "10", "--time-to-first-token-std-dev", "-1",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid inter-token-latency-std-dev",
			args: []string{"cmd", "--inter-token-latency", " 1000", "--inter-token-latency-std-dev", "301",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid (negative) inter-token-latency-std-dev",
			args: []string{"cmd", "--inter-token-latency", " 1000", "--inter-token-latency-std-dev", "-1",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid kv-cache-transfer-latency-std-dev",
			args: []string{"cmd", "--kv-cache-transfer-latency", "70", "--kv-cache-transfer-latency-std-dev", "35",
				"--config", "../../manifests/config.yaml"},
		},
		{
			name: "invalid (negative) kv-cache-transfer-latency-std-dev",
			args: []string{"cmd", "--kv-cache-transfer-latency-std-dev", "-35",
				"--config", "../../manifests/config.yaml"},
		},
	}

	for _, test := range invalidTests {
		When(test.name, func() {
			It("should fail for invalid configuration", func() {
				_, err := createSimConfig(test.args)
				Expect(err).To(HaveOccurred())
			})
		})
	}
})
