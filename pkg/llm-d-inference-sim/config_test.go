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

type testCase struct {
	name           string
	args           []string
	expectedConfig *configuration
}

var _ = Describe("Simulator configuration", func() {
	tests := make([]testCase, 0)

	// Simple config with only model name set
	c := newConfig()
	c.Model = model
	test := testCase{
		name:           "simple",
		args:           []string{"cmd", "--model", model, "--mode", modeRandom},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file
	c = newConfig()
	c.Port = 8001
	c.Model = "Qwen/Qwen2-0.5B"
	c.MaxLoras = 2
	c.MaxCPULoras = 5
	c.MaxNumSeqs = 5
	c.TimeToFirstToken = 2
	c.InterTokenLatency = 1
	c.LoraModules = []loraModule{{Name: "lora1", Path: "/path/to/lora1"}, {Name: "lora2", Path: "/path/to/lora2"}}
	test = testCase{
		name:           "config file",
		args:           []string{"cmd", "--config", "../../manifests/config.yaml"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args
	c = newConfig()
	c.Port = 8002
	c.Model = model
	c.MaxLoras = 2
	c.MaxCPULoras = 5
	c.MaxNumSeqs = 5
	c.TimeToFirstToken = 2
	c.InterTokenLatency = 1
	c.LoraModules = []loraModule{{Name: "lora1", Path: "/path/to/lora1"}, {Name: "lora2", Path: "/path/to/lora2"}}
	test = testCase{
		name:           "config file with command line args",
		args:           []string{"cmd", "--model", model, "--config", "../../manifests/config.yaml", "--port", "8002"},
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

	DescribeTable("check configurations",
		func(args []string, expectedConfig *configuration) {
			config, err := createSimConfig(args)
			Expect(err).NotTo(HaveOccurred())
			Expect(config).To(Equal(expectedConfig))
		},
		Entry(tests[0].name, tests[0].args, tests[0].expectedConfig),
		Entry(tests[1].name, tests[1].args, tests[1].expectedConfig),
		Entry(tests[2].name, tests[2].args, tests[2].expectedConfig),
	)

	DescribeTable("invalid configurations",
		func(args []string) {
			_, err := createSimConfig(args)
			Expect(err).To(HaveOccurred())
		},
		Entry(tests[3].name, tests[3].args),
		Entry(tests[4].name, tests[4].args),
		Entry(tests[5].name, tests[5].args),
		Entry(tests[6].name, tests[6].args),
		Entry(tests[7].name, tests[7].args),
	)
})
