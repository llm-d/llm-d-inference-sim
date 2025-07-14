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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

const (
	qwenModelName = "Qwen/Qwen2-0.5B"
)

func createSimConfig(args []string) (*configuration, error) {
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

func createDefaultBasicConfig(model string) *configuration {
	c := newConfig()

	c.Model = model
	c.ServedModelNames = []string{c.Model}
	c.MaxNumSeqs = 5
	c.MaxLoras = 1
	c.MaxCPULoras = 1
	c.TimeToFirstToken = 2000
	c.InterTokenLatency = 1000
	c.KVCacheTransferLatency = 100
	c.Seed = 100100100
	c.LoraModules = []loraModule{}

	return c
}

func createDefaultConfig(model string) *configuration {
	c := createDefaultBasicConfig(model)

	// parameters special to config.yaml
	c.MaxLoras = 2
	c.MaxCPULoras = 5

	return c
}

type testCase struct {
	name           string
	args           []string
	expectedConfig *configuration
}

var _ = Describe("Simulator configuration", func() {
	tests := make([]testCase, 0)

	// Simple config with a few parameters
	c := newConfig()
	c.Model = model
	c.ServedModelNames = []string{c.Model}
	c.MaxCPULoras = 1
	c.Seed = 100
	test := testCase{
		name:           "simple",
		args:           []string{"cmd", "--model", model, "--mode", modeRandom, "--seed", "100"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file
	c = createDefaultConfig(qwenModelName)
	c.Port = 8001
	c.ServedModelNames = []string{"model1", "model2"}
	c.LoraModules = []loraModule{{Name: "lora1", Path: "/path/to/lora1"}, {Name: "lora2", Path: "/path/to/lora2"}}
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
	c.LoraModules = []loraModule{{Name: "lora3", Path: "/path/to/lora3"}, {Name: "lora4", Path: "/path/to/lora4"}}
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
	c.LoraModules = []loraModule{{Name: "lora3", Path: "/path/to/lora3"}}
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
	c.LoraModules = []loraModule{{Name: "lora3", Path: "/path/to/lora3"}}
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
	c = createDefaultBasicConfig(qwenModelName)
	c.Port = 8001
	c.KVCacheTransferLatency = 50
	test = testCase{
		name:           "config file with command line args with time to transfer kv-cache",
		args:           []string{"cmd", "--config", "../../manifests/basic-config.yaml", "--kv-cache-transfer-latency", "50"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Invalid configurations
	test = testCase{
		name: "invalid model",
		args: []string{"cmd", "--model", "", "--config", "../../manifests/config.yaml"},
	}
	tests = append(tests, test)

	test = testCase{
		name: "invalid port",
		args: []string{"cmd", "--port", "-50", "--config", "../../manifests/config.yaml"},
	}
	tests = append(tests, test)

	test = testCase{
		name: "invalid max-loras",
		args: []string{"cmd", "--max-loras", "15", "--config", "../../manifests/config.yaml"},
	}
	tests = append(tests, test)

	test = testCase{
		name: "invalid mode",
		args: []string{"cmd", "--mode", "hello", "--config", "../../manifests/config.yaml"},
	}
	tests = append(tests, test)

	test = testCase{
		name: "invalid lora",
		args: []string{"cmd", "--config", "../../manifests/config.yaml",
			"--lora-modules", "[{\"path\":\"/path/to/lora15\"}]"},
	}
	tests = append(tests, test)

	test = testCase{
		name: "invalid max-model-len",
		args: []string{"cmd", "--max-model-len", "0", "--config", "../../manifests/config.yaml"},
	}
	tests = append(tests, test)

	DescribeTable("check configurations",
		func(args []string, expectedConfig *configuration) {
			config, err := createSimConfig(args)
			Expect(err).NotTo(HaveOccurred())
			Expect(config).To(Equal(expectedConfig))
		},
		Entry(tests[0].name, tests[0].args, tests[0].expectedConfig),
		Entry(tests[1].name, tests[1].args, tests[1].expectedConfig),
		Entry(tests[2].name, tests[2].args, tests[2].expectedConfig),
		Entry(tests[3].name, tests[3].args, tests[3].expectedConfig),
		Entry(tests[4].name, tests[4].args, tests[4].expectedConfig),
		Entry(tests[5].name, tests[5].args, tests[5].expectedConfig),
		Entry(tests[6].name, tests[6].args, tests[6].expectedConfig),
		Entry(tests[7].name, tests[7].args, tests[7].expectedConfig),
	)

	DescribeTable("invalid configurations",
		func(args []string) {
			_, err := createSimConfig(args)
			Expect(err).To(HaveOccurred())
		},
		Entry(tests[8].name, tests[8].args),
		Entry(tests[9].name, tests[9].args),
		Entry(tests[10].name, tests[10].args),
		Entry(tests[11].name, tests[11].args),
		Entry(tests[12].name, tests[12].args),
	)
})
