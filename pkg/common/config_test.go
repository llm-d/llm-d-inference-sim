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
	"os"
	"reflect"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func createSimConfig(args []string) (*Configuration, error) {
	oldArgs := os.Args
	defer func() {
		os.Args = oldArgs
	}()
	os.Args = args

	return ParseCommandParamsAndLoadConfig()
}

func createConfigWithModel(model string, servedModelNames []string) *Configuration {
	c := newConfig()

	c.Model = model
	if len(servedModelNames) > 0 {
		c.ServedModelNames = servedModelNames
	} else {
		c.ServedModelNames = []string{c.Model}
	}

	c.DisplayModelName = c.ServedModelNames[0]

	return c
}

func createDefaultConfig(model string, servedModelNames []string) *Configuration {
	c := createConfigWithModel(model, servedModelNames)

	c.MaxNumSeqs = 5
	c.MaxLoras = 2
	c.MaxCPULoras = 5
	c.TimeToFirstToken = 2000 * time.Millisecond
	c.InterTokenLatency = 1000 * time.Millisecond
	c.KVCacheTransferLatency = 100 * time.Millisecond
	c.Seed = 100100100
	c.LoraModules = []LoraModule{}
	return c
}

type testCase struct {
	name           string
	args           []string
	expectedError  string
	expectedConfig *Configuration
}

var _ = Describe("Simulator configuration", func() {
	//nolint:prealloc
	tests := make([]testCase, 0)

	// Simple config with a few parameters
	c := createConfigWithModel(TestModelName, nil)
	c.MaxCPULoras = 1
	c.Seed = 100
	test := testCase{
		name:           "simple",
		args:           []string{"cmd", "--model", TestModelName, "--mode", ModeRandom, "--seed", "100"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file
	c = createDefaultConfig(QwenModelName, []string{"model1", "model2"})
	c.Port = 8001
	c.LoraModules = []LoraModule{{Name: "lora1", Path: "/path/to/lora1"}, {Name: "lora2", Path: "/path/to/lora2"}}
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
	c = createDefaultConfig(TestModelName, []string{"alias1", "alias2"})
	c.Port = 8002
	c.Seed = 100
	c.LoraModules = []LoraModule{{Name: "lora3", Path: "/path/to/lora3"}, {Name: "lora4", Path: "/path/to/lora4"}}
	c.LoraModulesString = []string{
		"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
		"{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}",
	}
	c.EventBatchSize = 5
	test = testCase{
		name: "config file with command line args",
		args: []string{"cmd", "--model", TestModelName, "--config", "../../manifests/config.yaml", "--port", "8002",
			"--served-model-name", "alias1", "alias2", "--seed", "100",
			"--lora-modules", "{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}", "{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}",
			"--event-batch-size", "5",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with different format
	c = createDefaultConfig(TestModelName, nil)
	c.Port = 8002
	c.LoraModules = []LoraModule{{Name: "lora3", Path: "/path/to/lora3"}}
	c.LoraModulesString = []string{
		"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
	}
	test = testCase{
		name: "config file with command line args with different format",
		args: []string{"cmd", "--model", TestModelName, "--config", "../../manifests/config.yaml", "--port", "8002",
			"--served-model-name",
			"--lora-modules={\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with empty string
	c = createDefaultConfig(TestModelName, nil)
	c.Port = 8002
	c.LoraModules = []LoraModule{{Name: "lora3", Path: "/path/to/lora3"}}
	c.LoraModulesString = []string{
		"{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
	}
	test = testCase{
		name: "config file with command line args with empty string",
		args: []string{"cmd", "--model", TestModelName, "--config", "../../manifests/config.yaml", "--port", "8002",
			"--served-model-name", "",
			"--lora-modules", "{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with empty string for loras
	c = createDefaultConfig(QwenModelName, []string{"model1", "model2"})
	c.Port = 8001
	c.LoraModulesString = []string{}
	test = testCase{
		name:           "config file with command line args with empty string for loras",
		args:           []string{"cmd", "--config", "../../manifests/config.yaml", "--lora-modules", ""},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with empty parameter for loras
	c = createDefaultConfig(QwenModelName, []string{"model1", "model2"})
	c.Port = 8001
	c.LoraModulesString = []string{}
	test = testCase{
		name:           "config file with command line args with empty parameter for loras",
		args:           []string{"cmd", "--config", "../../manifests/config.yaml", "--lora-modules"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config_with_duration_latency.yaml file plus command line args with empty parameter for loras
	c = createDefaultConfig(QwenModelName, []string{"model1", "model2"})
	c.Port = 8001
	c.LoraModulesString = []string{}
	c.TimeToFirstToken = 4 * time.Second
	c.InterTokenLatency = 2 * time.Second
	c.KVCacheTransferLatency = time.Second
	test = testCase{
		name:           "config file with command line args with empty parameter for loras",
		args:           []string{"cmd", "--config", "../../manifests/config_with_duration_latency.yaml", "--lora-modules"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from basic-config.yaml file plus command line args with time to copy cache
	c = createDefaultConfig(QwenModelName, nil)
	c.Port = 8001
	// basic config file does not contain properties related to lora
	c.MaxLoras = 1
	c.MaxCPULoras = 1
	c.KVCacheTransferLatency = 50 * time.Millisecond
	test = testCase{
		name:           "basic config file with command line args with time to transfer kv-cache",
		args:           []string{"cmd", "--config", "../../manifests/basic-config.yaml", "--kv-cache-transfer-latency", "50ms"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config_with_fake.yaml file
	c = createDefaultConfig(QwenModelName, nil)
	c.FakeMetrics = &FakeMetrics{
		RunningRequests: &FakeMetricWithFunction{FixedValue: 16},
		WaitingRequests: &FakeMetricWithFunction{
			FixedValue: 0,
			IsFunction: true,
			Function: &FunctionInfo{
				Name:   OscillateFuncName,
				Start:  0,
				End:    5,
				Period: time.Second,
			},
		},
		KVCacheUsagePercentage: &FakeMetricWithFunction{FixedValue: 0.3},
		LoraMetrics: []LorasMetrics{
			{RunningLoras: "lora1,lora2", WaitingLoras: "lora3", Timestamp: 1257894567},
			{RunningLoras: "lora1,lora3", WaitingLoras: "", Timestamp: 1257894569},
		},
		LorasString: []string{
			"{\"running\":\"lora1,lora2\",\"waiting\":\"lora3\",\"timestamp\":1257894567}",
			"{\"running\":\"lora1,lora3\",\"waiting\":\"\",\"timestamp\":1257894569}",
		},
		TTFTBucketValues:           []int{10, 20, 30, 10},
		TPOTBucketValues:           []int{0, 0, 10, 20, 30},
		RequestPromptTokens:        []int{10, 20, 30, 15},
		RequestGenerationTokens:    []int{50, 60, 40},
		RequestParamsMaxTokens:     []int{128, 256, 512},
		RequestMaxGenerationTokens: []int{0, 0, 10, 20},
		RequestSuccessTotal: map[string]int64{
			StopFinishReason:           20,
			LengthFinishReason:         0,
			ToolsFinishReason:          0,
			RemoteDecodeFinishReason:   0,
			CacheThresholdFinishReason: 0,
		},
	}
	test = testCase{
		name:           "config with fake metrics file",
		args:           []string{"cmd", "--config", "../../manifests/config_with_fake.yaml"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Fake metrics from command line
	c = createConfigWithModel(TestModelName, nil)
	c.MaxCPULoras = 1
	c.Seed = 100
	c.FakeMetrics = &FakeMetrics{
		RunningRequests: &FakeMetricWithFunction{
			FixedValue: 0,
			IsFunction: true,
			Function: &FunctionInfo{
				Name:   RampFuncName,
				Start:  10,
				End:    35,
				Period: 10 * time.Second,
			},
		},
		WaitingRequests:        &FakeMetricWithFunction{FixedValue: 30},
		KVCacheUsagePercentage: &FakeMetricWithFunction{FixedValue: 0.4},
		LoraMetrics: []LorasMetrics{
			{RunningLoras: "lora4,lora2", WaitingLoras: "lora3", Timestamp: 1257894567},
			{RunningLoras: "lora4,lora3", WaitingLoras: "", Timestamp: 1257894569},
		},
		LorasString: nil,
	}
	test = testCase{
		name: "metrics from command line",
		args: []string{"cmd", "--model", TestModelName, "--seed", "100",
			"--fake-metrics",
			"{\"running-requests\":\"ramp:10:35:10s\",\"waiting-requests\":30,\"kv-cache-usage\":0.4,\"loras\":[{\"running\":\"lora4,lora2\",\"waiting\":\"lora3\",\"timestamp\":1257894567},{\"running\":\"lora4,lora3\",\"waiting\":\"\",\"timestamp\":1257894569}]}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Fake metrics from both the config file and command line
	c = createDefaultConfig(QwenModelName, nil)
	c.FakeMetrics = &FakeMetrics{
		RunningRequests:        &FakeMetricWithFunction{FixedValue: 10},
		WaitingRequests:        &FakeMetricWithFunction{FixedValue: 30},
		KVCacheUsagePercentage: &FakeMetricWithFunction{FixedValue: 0.4},
		LoraMetrics: []LorasMetrics{
			{RunningLoras: "lora4,lora2", WaitingLoras: "lora3", Timestamp: 1257894567},
			{RunningLoras: "lora4,lora3", WaitingLoras: "", Timestamp: 1257894569},
		},
		LorasString: nil,
	}
	test = testCase{
		name: "metrics from config file and command line",
		args: []string{"cmd", "--config", "../../manifests/config_with_fake.yaml",
			"--fake-metrics",
			"{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":0.4,\"loras\":[{\"running\":\"lora4,lora2\",\"waiting\":\"lora3\",\"timestamp\":1257894567},{\"running\":\"lora4,lora3\",\"waiting\":\"\",\"timestamp\":1257894569}]}",
		},
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
			name:          "invalid model",
			args:          []string{"cmd", "--model", "", "--config", "../../manifests/config.yaml"},
			expectedError: "model parameter is empty",
		},
		{
			name:          "invalid port",
			args:          []string{"cmd", "--port", "-50", "--config", "../../manifests/config.yaml"},
			expectedError: "invalid port",
		},
		{
			name:          "invalid max-loras",
			args:          []string{"cmd", "--max-loras", "15", "--config", "../../manifests/config.yaml"},
			expectedError: "max CPU LoRAs cannot be less than max LoRAs",
		},
		{
			name:          "invalid mode",
			args:          []string{"cmd", "--mode", "hello", "--config", "../../manifests/config.yaml"},
			expectedError: "invalid mode ",
		},
		{
			name: "invalid lora",
			args: []string{"cmd", "--config", "../../manifests/config.yaml",
				"--lora-modules", "{\"path\":\"/path/to/lora15\"}"},
			expectedError: "empty LoRA name",
		},
		{
			name:          "invalid max-model-len",
			args:          []string{"cmd", "--max-model-len", "0", "--config", "../../manifests/config.yaml"},
			expectedError: "max model len cannot be less than 1",
		},
		{
			name:          "invalid tool-call-not-required-param-probability",
			args:          []string{"cmd", "--tool-call-not-required-param-probability", "-10", "--config", "../../manifests/config.yaml"},
			expectedError: "ToolCallNotRequiredParamProbability should be between 0 and 100",
		},
		{
			name: "invalid max-tool-call-number-param",
			args: []string{"cmd", "--max-tool-call-number-param", "-10", "--min-tool-call-number-param", "0",
				"--config", "../../manifests/config.yaml"},
			expectedError: "MaxToolCallNumberParam cannot be less than MinToolCallNumberParam",
		},
		{
			name: "invalid max-tool-call-integer-param",
			args: []string{"cmd", "--max-tool-call-integer-param", "-10", "--min-tool-call-integer-param", "0",
				"--config", "../../manifests/config.yaml"},
			expectedError: "MaxToolCallIntegerParam cannot be less than MinToolCallIntegerParam",
		},
		{
			name: "invalid max-tool-call-array-param-length",
			args: []string{"cmd", "--max-tool-call-array-param-length", "-10", "--min-tool-call-array-param-length", "0",
				"--config", "../../manifests/config.yaml"},
			expectedError: "MaxToolCallArrayParamLength cannot be less than MinToolCallArrayParamLength",
		},
		{
			name: "invalid tool-call-not-required-param-probability",
			args: []string{"cmd", "--tool-call-not-required-param-probability", "-10",
				"--config", "../../manifests/config.yaml"},
			expectedError: "ToolCallNotRequiredParamProbability should be between 0 and 100",
		},
		{
			name: "invalid object-tool-call-not-required-field-probability",
			args: []string{"cmd", "--object-tool-call-not-required-field-probability", "1210",
				"--config", "../../manifests/config.yaml"},
			expectedError: "ObjectToolCallNotRequiredParamProbability should be between 0 and 100",
		},
		{
			name: "invalid time-to-first-token-std-dev",
			args: []string{"cmd", "--time-to-first-token-std-dev", "3000ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "time to first token standard deviation cannot be more than 30%",
		},
		{
			name: "invalid (negative) time-to-first-token-std-dev",
			args: []string{"cmd", "--time-to-first-token-std-dev", "10ms", "--time-to-first-token-std-dev", "-1ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "time to first token standard deviation cannot be negative",
		},
		{
			name: "invalid inter-token-latency-std-dev",
			args: []string{"cmd", "--inter-token-latency", "1000ms", "--inter-token-latency-std-dev", "301ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "inter token latency standard deviation cannot be more than 30%",
		},
		{
			name: "invalid (negative) inter-token-latency-std-dev",
			args: []string{"cmd", "--inter-token-latency", "1000ms", "--inter-token-latency-std-dev", "-1s",
				"--config", "../../manifests/config.yaml"},
			expectedError: "inter token latency standard deviation cannot be negative",
		},
		{
			name: "invalid kv-cache-transfer-latency-std-dev",
			args: []string{"cmd", "--kv-cache-transfer-latency", "70ms", "--kv-cache-transfer-latency-std-dev", "35ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "kv-cache tranfer standard deviation cannot be more than 30% of kv-cache tranfer",
		},
		{
			name: "invalid (negative) kv-cache-transfer-latency-std-dev",
			args: []string{"cmd", "--kv-cache-transfer-latency-std-dev", "-35ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "kv-cache tranfer time standard deviation cannot be negative",
		},
		{
			name: "invalid (negative) kv-cache-size",
			args: []string{"cmd", "--kv-cache-size", "-35",
				"--config", "../../manifests/config.yaml"},
			expectedError: "KV cache size cannot be negative",
		},
		{
			name: "invalid block-size",
			args: []string{"cmd", "--block-size", "35",
				"--config", "../../manifests/config.yaml"},
			expectedError: "token block size should be one of the following",
		},
		{
			name: "invalid (negative) event-batch-size",
			args: []string{"cmd", "--event-batch-size", "-35",
				"--config", "../../manifests/config.yaml"},
			expectedError: "event batch size cannot less than 1",
		},
		{
			name:          "invalid failure injection rate > 100",
			args:          []string{"cmd", "--model", TestModelName, "--failure-injection-rate", "150"},
			expectedError: "failure injection rate should be between 0 and 100",
		},
		{
			name:          "invalid failure injection rate < 0",
			args:          []string{"cmd", "--model", TestModelName, "--failure-injection-rate", "-10"},
			expectedError: "failure injection rate should be between 0 and 100",
		},
		{
			name: "invalid failure type",
			args: []string{"cmd", "--model", TestModelName, "--failure-injection-rate", "50",
				"--failure-types", "invalid_type"},
			expectedError: "invalid failure type",
		},
		{
			name: "invalid fake metrics: negative running requests",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":-10,\"waiting-requests\":30,\"kv-cache-usage\":0.4}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics request counters cannot be negative",
		},
		{
			name: "invalid fake metrics: invalid running requests function",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":\"foo:0:8:10s\",\"waiting-requests\":30,\"kv-cache-usage\":0.4}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "invalid fake metrics generation function foo",
		},
		{
			name: "invalid fake metrics: invalid function parameter period",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":19,\"waiting-requests\":\"squarewave:0:8:170\",\"kv-cache-usage\":0.4}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "unknown format in fake metric generation function: time: missing unit in duration",
		},
		{
			name: "invalid fake metrics: invalid function parameter period, can't be 0",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":19,\"waiting-requests\":\"squarewave:0:8:0s\",\"kv-cache-usage\":0.4}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "invalid fake metrics generation parameter: period must be positive",
		},
		{
			name: "invalid fake metrics: incomplete waiting requests function parameters",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":19,\"waiting-requests\":\"rampreset:0:8\",\"kv-cache-usage\":0.4}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "need func:start:end:period in fake metric generation function",
		},
		{
			name: "invalid fake metrics: kv cache usage",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":40}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics KV cache usage must be between 0 and 1",
		},
		{
			name: "invalid fake metrics: negative kv cache usage function parameters",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":\"ramp:0:-8:10s\"}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "invalid fake metrics generation parameter: start and end must not be negative",
		},
		{
			name: "invalid fake metrics: invalid kv cache usage function parameters",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":\"ramp:0:5:10s\"}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics KV cache usage start and end must be between 0 and 1",
		},
		{
			name: "invalid fake metrics refresh period",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":\"ramp:0:1:10s\"}",
				"--fake-metrics-refresh-interval", "-20s",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics refresh interval must be positive",
		},
		{
			name: "invalid (negative) prefill-overhead",
			args: []string{"cmd", "--prefill-overhead", "-1ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "prefill overhead cannot be negative",
		},
		{
			name: "invalid (negative) prefill-time-per-token",
			args: []string{"cmd", "--prefill-time-per-token", "-1ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "prefill time per token cannot be negative",
		},
		{
			name: "invalid (negative) prefill-time-std-dev",
			args: []string{"cmd", "--prefill-time-std-dev", "-1ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "prefill time standard deviation cannot be negative",
		},
		{
			name: "invalid (negative) kv-cache-transfer-time-per-token",
			args: []string{"cmd", "--kv-cache-transfer-time-per-token", "-1ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "kv-cache tranfer time per token cannot be negative",
		},
		{
			name: "invalid (negative) kv-cache-transfer-time-std-dev",
			args: []string{"cmd", "--kv-cache-transfer-time-std-dev", "-1ms",
				"--config", "../../manifests/config.yaml"},
			expectedError: "kv-cache tranfer time standard deviation cannot be negative",
		},
		{
			name: "invalid data-parallel-size",
			args: []string{"cmd", "--data-parallel-size", "15",
				"--config", "../../manifests/config.yaml"},
			expectedError: "data parallel size must be between 1 and 8",
		},
		{
			name: "invalid data-parallel-rank",
			args: []string{"cmd", "--data-parallel-rank", "15",
				"--config", "../../manifests/config.yaml"},
			expectedError: "data parallel rank must be between 0 and 7",
		},
		{
			name: "invalid max-num-seqs",
			args: []string{"cmd", "--max-num-seqs", "0",
				"--config", "../../manifests/config.yaml"},
			expectedError: "max num seqs cannot be less than 1",
		},
		{
			name: "invalid max-num-seqs",
			args: []string{"cmd", "--max-num-seqs", "-1",
				"--config", "../../manifests/config.yaml"},
			expectedError: "max num seqs cannot be less than 1",
		},
		{
			name: "invalid max-waiting-queue-length",
			args: []string{"cmd", "--max-waiting-queue-length", "0",
				"--config", "../../manifests/config.yaml"},
			expectedError: "max waiting queue size cannot be less than 1",
		},
		{
			name: "invalid max-waiting-queue-length",
			args: []string{"cmd", "--max-waiting-queue-length", "-1",
				"--config", "../../manifests/config.yaml"},
			expectedError: "max waiting queue size cannot be less than 1",
		},
		{
			name: "invalid time-factor-under-load",
			args: []string{"cmd", "--time-factor-under-load", "0",
				"--config", "../../manifests/config.yaml"},
			expectedError: "time factor under load cannot be less than 1.0",
		},
		{
			name: "invalid time-factor-under-load",
			args: []string{"cmd", "--time-factor-under-load", "-1",
				"--config", "../../manifests/config.yaml"},
			expectedError: "time factor under load cannot be less than 1.0",
		},
		{
			name: "invalid ttft",
			args: []string{"cmd", "--fake-metrics", "{\"ttft-buckets-values\":[1, 2, -10, 1]}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "time-to-first-token fake metrics should contain only non-negative values",
		},
		{
			name: "invalid tpot",
			args: []string{"cmd", "--fake-metrics", "{\"tpot-buckets-values\":[1, 2, -10, 1]}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "time-per-output-token fake metrics should contain only non-negative values",
		},
		{
			name: "invalid request-max-generation-tokens",
			args: []string{"cmd", "--fake-metrics", "{\"request-max-generation-tokens\": [1, -1, 2]}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics request-max-generation-tokens cannot contain negative values",
		},
		{
			name: "invalid fake metrics: negative prefix-cache-hits",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-hits\":-5,\"prefix-cache-queries\":10}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-hits cannot be negative",
		},
		{
			name: "invalid fake metrics: negative prefix-cache-queries",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-hits\":0,\"prefix-cache-queries\":-1}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-queries cannot be negative",
		},
		{
			name: "invalid fake metrics: prefix-cache-hits without prefix-cache-queries",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-hits\":100}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-hits and prefix-cache-queries must be specified together",
		},
		{
			name: "invalid fake metrics: prefix-cache-queries without prefix-cache-hits",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-queries\":100}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-hits and prefix-cache-queries must be specified together",
		},
		{
			name: "invalid fake metrics: prefix-cache-hits exceeds prefix-cache-queries",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-hits\":100,\"prefix-cache-queries\":50}",
				"--config", "../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-hits cannot exceed prefix-cache-queries",
		},
		{
			name: "invalid echo mode with dataset",
			args: []string{"cmd", "--model", TestModelName, "--dataset-path", "my/path",
				"--mode", "echo"},
			expectedError: "dataset cannot be defined in echo mode",
		},
		{
			name:          "invalid latency calculator",
			args:          []string{"cmd", "--config", "../../manifests/config.yaml", "--latency-calculator", "hello"},
			expectedError: "unknown latency-calculator",
		},
	}

	for _, test := range invalidTests {
		When(test.name, func() {
			It("should fail for invalid configuration", func() {
				_, err := createSimConfig(test.args)
				// ensure that error occurred
				Expect(err).To(HaveOccurred())
				// ensure that an expected error occurred
				Expect(err.Error()).To(ContainSubstring(test.expectedError))
			})
		})
	}
})

var _ = Describe("ApplyAdminUpdate", func() {
	var base *Configuration

	BeforeEach(func() {
		base = createDefaultConfig("model", nil)
		base.FailureInjectionRate = 10
		base.FailureTypes = []string{FailureTypeRateLimit}
	})

	It("updates failure-injection-rate and returns a new Configuration", func() {
		next, update, latencyChanged, err := base.Update([]byte(`{"failure-injection-rate": 42}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(latencyChanged).To(BeFalse())
		Expect(update.FakeMetrics).To(BeNil())
		Expect(next).ToNot(BeIdenticalTo(base))
		Expect(next.FailureInjectionRate).To(Equal(42))
		Expect(next.FailureTypes).To(Equal([]string{FailureTypeRateLimit}))
		// original is unchanged
		Expect(base.FailureInjectionRate).To(Equal(10))
	})

	It("updates failure-types", func() {
		next, _, _, err := base.Update([]byte(`{"failure-types": ["server_error", "model_not_found"]}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(next.FailureTypes).To(Equal([]string{FailureTypeServerError, FailureTypeModelNotFound}))
		Expect(next.FailureInjectionRate).To(Equal(10))
		Expect(base.FailureTypes).To(Equal([]string{FailureTypeRateLimit}))
	})

	It("updates both fields at once", func() {
		next, _, _, err := base.Update([]byte(`{"failure-injection-rate": 5, "failure-types": ["invalid_request"]}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(next.FailureInjectionRate).To(Equal(5))
		Expect(next.FailureTypes).To(Equal([]string{FailureTypeInvalidRequest}))
	})

	It("returns the parsed fake-metrics partial via update.FakeMetrics", func() {
		next, update, _, err := base.Update([]byte(
			`{"failure-injection-rate": 50, "fake-metrics": {"running-requests": 7}}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(next.FailureInjectionRate).To(Equal(50))
		Expect(update.FakeMetrics).ToNot(BeNil())
		Expect(update.FakeMetrics.RunningRequests).ToNot(BeNil())
		Expect(update.FakeMetrics.RunningRequests.FixedValue).To(Equal(float64(7)))
		// Fields not in the body are nil on the fake-metrics partial.
		Expect(update.FakeMetrics.WaitingRequests).To(BeNil())
	})

	DescribeTable("flags latencyChanged according to the body keys",
		func(body string, expected bool) {
			_, _, latencyChanged, err := base.Update([]byte(body))
			Expect(err).ToNot(HaveOccurred())
			Expect(latencyChanged).To(Equal(expected))
		},
		Entry("only failure-injection-rate -> false",
			`{"failure-injection-rate": 0}`, false),
		Entry("only failure-types -> false",
			`{"failure-types": ["rate_limit"]}`, false),
		Entry("time-to-first-token -> true",
			`{"time-to-first-token": "250ms"}`, true),
		Entry("inter-token-latency -> true",
			`{"inter-token-latency": "1ms"}`, true),
		Entry("time-factor-under-load -> true",
			`{"time-factor-under-load": 1.5}`, true),
		Entry("latency-calculator -> true",
			`{"latency-calculator": "constant"}`, true),
		Entry("std-dev field -> true",
			`{"time-to-first-token": "1s", "time-to-first-token-std-dev": "100ms"}`, true),
		Entry("mixed latency + non-latency -> true",
			`{"failure-injection-rate": 0, "prefill-overhead": "1ms"}`, true),
	)

	It("returns latencyChanged=false when validation fails on a latency body", func() {
		// The 30% std-dev rule trips, but we still expect a clean error path
		// that does not claim latencyChanged.
		_, _, latencyChanged, err := base.Update([]byte(
			`{"time-to-first-token": "1ms", "time-to-first-token-std-dev": "0.5ms"}`))
		Expect(err).To(HaveOccurred())
		Expect(latencyChanged).To(BeFalse())
	})

	It("rejects an invalid duration string", func() {
		_, _, _, err := base.Update([]byte(`{"time-to-first-token": "notaduration"}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("time-to-first-token"))
	})

	It("rejects fields that are not admin-configurable", func() {
		_, _, _, err := base.Update([]byte(`{"port": 9000}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("not admin-configurable"))
	})

	It("rejects an out-of-range failure-injection-rate", func() {
		_, _, _, err := base.Update([]byte(`{"failure-injection-rate": 150}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failure injection rate"))
	})

	It("rejects an unknown failure type", func() {
		_, _, _, err := base.Update([]byte(`{"failure-types": ["bogus"]}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("invalid failure type"))
	})

	It("rejects malformed JSON", func() {
		_, _, _, err := base.Update([]byte(`not json`))
		Expect(err).To(HaveOccurred())
	})
})

var _ = Describe("Model environment variable", func() {
	BeforeEach(func() {
		Expect(os.Unsetenv(ModelEnv)).To(Succeed())
	})
	AfterEach(func() {
		Expect(os.Unsetenv(ModelEnv)).To(Succeed())
	})

	It("does not override --model when the flag is passed", func() {
		Expect(os.Setenv(ModelEnv, "from-env")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--model", TestModelName, "--mode", ModeRandom, "--seed", "100"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.Model).To(Equal(TestModelName))
	})

	It("overrides model from config file when --model is omitted", func() {
		Expect(os.Setenv(ModelEnv, "env-override-model")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--config", "../../manifests/config.yaml"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.Model).To(Equal("env-override-model"))
	})

	It("does not change model when unset and --model is passed", func() {
		config, err := createSimConfig([]string{"cmd", "--model", TestModelName, "--mode", ModeRandom, "--seed", "100"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.Model).To(Equal(TestModelName))
	})
})

var _ = Describe("PYTHONHASHSEED environment variable", func() {
	BeforeEach(func() {
		Expect(os.Unsetenv(PythonHashSeedEnv)).To(Succeed())
	})
	AfterEach(func() {
		Expect(os.Unsetenv(PythonHashSeedEnv)).To(Succeed())
	})

	It("does not override --hash-seed when the flag is passed", func() {
		Expect(os.Setenv(PythonHashSeedEnv, "from-env")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--model", TestModelName, "--hash-seed", "from-flag", "--mode", ModeRandom, "--seed", "100"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.HashSeed).To(Equal("from-flag"))
	})

	It("applies when --hash-seed is omitted", func() {
		Expect(os.Setenv(PythonHashSeedEnv, "env-seed")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--model", TestModelName, "--mode", ModeRandom, "--seed", "100"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.HashSeed).To(Equal("env-seed"))
	})
})

var _ = Describe("Configuration.Copy", func() {
	It("should round-trip a non-nil FakeMetrics with a fixed-value metric", func() {
		c := &Configuration{
			FakeMetrics: &FakeMetrics{
				RunningRequests: &FakeMetricWithFunction{FixedValue: 5},
			},
		}

		got, err := c.Copy()
		Expect(err).NotTo(HaveOccurred())
		Expect(got.FakeMetrics).NotTo(BeNil())
		Expect(got.FakeMetrics.RunningRequests).NotTo(BeNil())
		Expect(got.FakeMetrics.RunningRequests.IsFunction).To(BeFalse())
		Expect(got.FakeMetrics.RunningRequests.FixedValue).To(Equal(5.0))
	})

	It("should round-trip a non-nil FakeMetrics with a function-valued metric", func() {
		c := &Configuration{
			FakeMetrics: &FakeMetrics{
				WaitingRequests: &FakeMetricWithFunction{
					IsFunction: true,
					Function: &FunctionInfo{
						Name:   OscillateFuncName,
						Start:  0,
						End:    10,
						Period: 5 * time.Second,
					},
				},
			},
		}

		got, err := c.Copy()
		Expect(err).NotTo(HaveOccurred())
		Expect(got.FakeMetrics).NotTo(BeNil())
		Expect(got.FakeMetrics.WaitingRequests).NotTo(BeNil())
		Expect(got.FakeMetrics.WaitingRequests.IsFunction).To(BeTrue())
		Expect(got.FakeMetrics.WaitingRequests.Function).NotTo(BeNil())
		Expect(got.FakeMetrics.WaitingRequests.Function.Name).To(Equal(OscillateFuncName))
		Expect(got.FakeMetrics.WaitingRequests.Function.Start).To(Equal(0.0))
		Expect(got.FakeMetrics.WaitingRequests.Function.End).To(Equal(10.0))
		Expect(got.FakeMetrics.WaitingRequests.Function.Period).To(Equal(5 * time.Second))
	})

	It("should round-trip an explicit-zero metric (non-nil pointer to zero-value struct)", func() {
		c := &Configuration{
			FakeMetrics: &FakeMetrics{
				RunningRequests: &FakeMetricWithFunction{},
			},
		}

		got, err := c.Copy()
		Expect(err).NotTo(HaveOccurred())
		Expect(got.FakeMetrics).NotTo(BeNil())
		Expect(got.FakeMetrics.RunningRequests).NotTo(BeNil())
		Expect(got.FakeMetrics.RunningRequests.IsFunction).To(BeFalse())
		Expect(got.FakeMetrics.RunningRequests.FixedValue).To(Equal(0.0))
	})
})

var _ = Describe("admin struct tags", func() {
	It("has no unrecognized tag values", func() {
		t := reflect.TypeOf(Configuration{})
		for i := range t.NumField() {
			f := t.Field(i)
			Expect(f.Tag.Get("admin")).To(BeElementOf("", "configurable"),
				"field %s has unexpected admin tag %q", f.Name, f.Tag.Get("admin"))
			Expect(f.Tag.Get("rebuild")).To(BeElementOf("", "latency"),
				"field %s has unexpected rebuild tag %q", f.Name, f.Tag.Get("rebuild"))
			if f.Tag.Get("rebuild") == "latency" {
				Expect(f.Tag.Get("admin")).To(Equal("configurable"),
					"field %s has rebuild:\"latency\" but missing admin:\"configurable\"", f.Name)
			}
		}
	})

	It("configurableFields contains exactly the expected entries with their rebuild tags", func() {
		Expect(configurableFields).To(Equal(map[string]string{
			"time-to-first-token":               "latency",
			"time-to-first-token-std-dev":       "latency",
			"inter-token-latency":               "latency",
			"inter-token-latency-std-dev":       "latency",
			"kv-cache-transfer-latency":         "latency",
			"kv-cache-transfer-latency-std-dev": "latency",
			"prefill-overhead":                  "latency",
			"prefill-time-per-token":            "latency",
			"prefill-time-std-dev":              "latency",
			"kv-cache-transfer-time-per-token":  "latency",
			"kv-cache-transfer-time-std-dev":    "latency",
			"time-factor-under-load":            "latency",
			"latency-calculator":                "latency",
			"failure-injection-rate":            "",
			"failure-types":                     "",
			"fake-metrics":                      "",
		}))
	})

})
