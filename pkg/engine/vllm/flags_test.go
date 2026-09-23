/*
Copyright 2026 The llm-d-inference-sim Authors.

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

package vllm

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

func createSimConfig(args []string) (*common.Configuration, error) {
	oldArgs := os.Args
	defer func() {
		os.Args = oldArgs
	}()
	os.Args = args

	eng := New()
	return common.ParseCommandParamsAndLoadConfig(eng)
}

// defaultConfig returns a Configuration carrying common's defaults plus this
// engine's own, the same pairing ParseCommandParamsAndLoadConfig starts from.
func defaultConfig() *common.Configuration {
	c := common.NewConfig()
	New().ApplyDefaults(c)
	return c
}

func createConfigWithModel(model string, servedModelNames []string) *common.Configuration {
	c := defaultConfig()
	// KV cache is disabled by default, and a disabled cache reports its
	// sizing/hashing/eventing fields as all-zero; tests that enable it
	// restore defaultConfig().KVCache explicitly.
	c.KVCache = common.KVCacheConfig{}

	c.Model = model
	if len(servedModelNames) > 0 {
		c.ServedModelNames = servedModelNames
	} else {
		c.ServedModelNames = []string{c.Model}
	}

	c.DisplayModelName = c.ServedModelNames[0]

	return c
}

func createDefaultConfig(model string, servedModelNames []string) *common.Configuration {
	c := createConfigWithModel(model, servedModelNames)

	c.MaxNumSeqs = 5
	c.Lora.MaxLoras = 2
	c.Lora.MaxCPULoras = 5
	c.Latencies.TimeToFirstToken = 2000 * time.Millisecond
	c.Latencies.InterTokenLatency = 1000 * time.Millisecond
	c.Latencies.KVCacheTransferLatency = 100 * time.Millisecond
	c.Seed = 100100100
	c.Lora.LoraModules = []common.LoraModule{}
	return c
}

type testCase struct {
	name           string
	args           []string
	expectedError  string
	expectedConfig *common.Configuration
}

var _ = Describe("Simulator configuration", func() {
	//nolint:prealloc
	tests := make([]testCase, 0)

	// Simple config with a few parameters
	c := createConfigWithModel(common.TestModelName, nil)
	c.Lora.MaxCPULoras = 1
	c.Seed = 100
	test := testCase{
		name:           "simple",
		args:           []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--seed", "100"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file
	c = createDefaultConfig(common.QwenModelName, []string{"model1", "model2"})
	c.Port = 8001
	c.Lora.LoraModules = []common.LoraModule{{Name: "lora1", Path: "/path/to/lora1"}, {Name: "lora2", Path: "/path/to/lora2"}}
	test = testCase{
		name:           "config file",
		args:           []string{"cmd", "--config", "../../../manifests/config.yaml"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args
	c = createDefaultConfig(common.TestModelName, []string{"alias1", "alias2"})
	c.Port = 8002
	c.Seed = 100
	c.Lora.LoraModules = []common.LoraModule{{Name: "lora3", Path: "/path/to/lora3"}, {Name: "lora4", Path: "/path/to/lora4"}}
	c.KVCache = defaultConfig().KVCache
	c.KVCache.EnableKVCache = true
	c.KVCache.EventBatchSize = 5
	test = testCase{
		name: "config file with command line args",
		args: []string{"cmd", "--model", common.TestModelName, "--config", "../../../manifests/config.yaml", "--port", "8002",
			"--served-model-name", "alias1", "alias2", "--seed", "100",
			"--lora-modules", "{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}", "{\"name\":\"lora4\",\"path\":\"/path/to/lora4\"}",
			"--enable-kvcache", "--event-batch-size", "5",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with different format
	c = createDefaultConfig(common.TestModelName, nil)
	c.Port = 8002
	c.Lora.LoraModules = []common.LoraModule{{Name: "lora3", Path: "/path/to/lora3"}}
	test = testCase{
		name: "config file with command line args with different format",
		args: []string{"cmd", "--model", common.TestModelName, "--config", "../../../manifests/config.yaml", "--port", "8002",
			"--served-model-name",
			"--lora-modules={\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with empty string
	c = createDefaultConfig(common.TestModelName, nil)
	c.Port = 8002
	c.Lora.LoraModules = []common.LoraModule{{Name: "lora3", Path: "/path/to/lora3"}}
	test = testCase{
		name: "config file with command line args with empty string",
		args: []string{"cmd", "--model", common.TestModelName, "--config", "../../../manifests/config.yaml", "--port", "8002",
			"--served-model-name", "",
			"--lora-modules", "{\"name\":\"lora3\",\"path\":\"/path/to/lora3\"}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with empty string for loras
	c = createDefaultConfig(common.QwenModelName, []string{"model1", "model2"})
	c.Port = 8001
	test = testCase{
		name:           "config file with command line args with empty string for loras",
		args:           []string{"cmd", "--config", "../../../manifests/config.yaml", "--lora-modules", ""},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config.yaml file plus command line args with empty parameter for loras
	c = createDefaultConfig(common.QwenModelName, []string{"model1", "model2"})
	c.Port = 8001
	test = testCase{
		name:           "config file with command line args with empty parameter for loras",
		args:           []string{"cmd", "--config", "../../../manifests/config.yaml", "--lora-modules"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config_with_duration_latency.yaml file plus command line args with empty parameter for loras
	c = createDefaultConfig(common.QwenModelName, []string{"model1", "model2"})
	c.Port = 8001
	c.Latencies.TimeToFirstToken = 4 * time.Second
	c.Latencies.InterTokenLatency = 2 * time.Second
	c.Latencies.KVCacheTransferLatency = time.Second
	test = testCase{
		name:           "config file with command line args with empty parameter for loras",
		args:           []string{"cmd", "--config", "../../../manifests/config_with_duration_latency.yaml", "--lora-modules"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from basic-config.yaml file plus command line args with time to copy cache
	c = createDefaultConfig(common.QwenModelName, nil)
	c.Port = 8001
	// basic config file does not contain properties related to lora
	c.Lora.MaxLoras = 1
	c.Lora.MaxCPULoras = 1
	c.Latencies.KVCacheTransferLatency = 50 * time.Millisecond
	test = testCase{
		name:           "basic config file with command line args with time to transfer kv-cache",
		args:           []string{"cmd", "--config", "../../../manifests/basic-config.yaml", "--kv-cache-transfer-latency", "50ms"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config with image generation latencies
	c = createDefaultConfig(common.QwenModelName, nil)
	c.Port = 8001
	c.Lora.MaxLoras = 1
	c.Lora.MaxCPULoras = 1
	c.Latencies.TimeToGenerateImage = 500 * time.Millisecond
	c.Latencies.TimeToGenerateImageStdDev = 50 * time.Millisecond
	test = testCase{
		name: "basic config file with image generation latencies",
		args: []string{"cmd", "--config", "../../../manifests/basic-config.yaml",
			"--time-to-generate-image", "500ms",
			"--time-to-generate-image-std-dev", "50ms",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Config from config_with_fake.yaml file
	c = createDefaultConfig(common.QwenModelName, nil)
	c.FakeMetrics = &VLLMFakeMetrics{
		RunningRequests: &common.FakeMetricWithFunction{FixedValue: 16},
		WaitingRequests: &common.FakeMetricWithFunction{
			FixedValue: 0,
			IsFunction: true,
			Function: &common.FunctionInfo{
				Name:   common.OscillateFuncName,
				Start:  0,
				End:    5,
				Period: time.Second,
			},
		},
		KVCacheUsagePercentage: &common.FakeMetricWithFunction{FixedValue: 0.3},
		LoraMetrics: []common.LorasMetrics{
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
			common.StopFinishReason:           20,
			common.LengthFinishReason:         0,
			common.ToolsFinishReason:          0,
			common.RemoteDecodeFinishReason:   0,
			common.CacheThresholdFinishReason: 0,
		},
	}
	test = testCase{
		name:           "config with fake metrics file",
		args:           []string{"cmd", "--config", "../../../manifests/config_with_fake.yaml"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Fake metrics from command line
	c = createConfigWithModel(common.TestModelName, nil)
	c.Lora.MaxCPULoras = 1
	c.Seed = 100
	c.FakeMetrics = &VLLMFakeMetrics{
		RunningRequests: &common.FakeMetricWithFunction{
			FixedValue: 0,
			IsFunction: true,
			Function: &common.FunctionInfo{
				Name:   common.RampFuncName,
				Start:  10,
				End:    35,
				Period: 10 * time.Second,
			},
		},
		WaitingRequests:        &common.FakeMetricWithFunction{FixedValue: 30},
		KVCacheUsagePercentage: &common.FakeMetricWithFunction{FixedValue: 0.4},
		LoraMetrics: []common.LorasMetrics{
			{RunningLoras: "lora4,lora2", WaitingLoras: "lora3", Timestamp: 1257894567},
			{RunningLoras: "lora4,lora3", WaitingLoras: "", Timestamp: 1257894569},
		},
		LorasString: nil,
	}
	test = testCase{
		name: "metrics from command line",
		args: []string{"cmd", "--model", common.TestModelName, "--seed", "100",
			"--fake-metrics",
			"{\"running-requests\":\"ramp:10:35:10s\",\"waiting-requests\":30,\"kv-cache-usage\":0.4,\"loras\":[{\"running\":\"lora4,lora2\",\"waiting\":\"lora3\",\"timestamp\":1257894567},{\"running\":\"lora4,lora3\",\"waiting\":\"\",\"timestamp\":1257894569}]}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// Fake metrics from both the config file and command line
	c = createDefaultConfig(common.QwenModelName, nil)
	c.FakeMetrics = &VLLMFakeMetrics{
		RunningRequests:        &common.FakeMetricWithFunction{FixedValue: 10},
		WaitingRequests:        &common.FakeMetricWithFunction{FixedValue: 30},
		KVCacheUsagePercentage: &common.FakeMetricWithFunction{FixedValue: 0.4},
		LoraMetrics: []common.LorasMetrics{
			{RunningLoras: "lora4,lora2", WaitingLoras: "lora3", Timestamp: 1257894567},
			{RunningLoras: "lora4,lora3", WaitingLoras: "", Timestamp: 1257894569},
		},
		LorasString: nil,
	}
	test = testCase{
		name: "metrics from config file and command line",
		args: []string{"cmd", "--config", "../../../manifests/config_with_fake.yaml",
			"--fake-metrics",
			"{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":0.4,\"loras\":[{\"running\":\"lora4,lora2\",\"waiting\":\"lora3\",\"timestamp\":1257894567},{\"running\":\"lora4,lora3\",\"waiting\":\"\",\"timestamp\":1257894569}]}",
		},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// max-request-body-size-mb set to exactly 1 MB (lower boundary)
	c = createConfigWithModel(common.TestModelName, nil)
	c.Lora.MaxCPULoras = 1
	c.Seed = 100
	c.MaxRequestBodySizeMB = 1
	test = testCase{
		name:           "valid max-request-body-size-mb (1 MB boundary)",
		args:           []string{"cmd", "--model", common.TestModelName, "--seed", "100", "--max-request-body-size-mb", "1"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// kv-events-replay-endpoint set via CLI flag
	c = createConfigWithModel(common.TestModelName, nil)
	c.Lora.MaxCPULoras = 1
	c.Seed = 100
	c.KVCache = defaultConfig().KVCache
	c.KVCache.EnableKVCache = true
	c.KVCache.KVEventsReplayEndpoint = "tcp://*:5558"
	test = testCase{
		name: "kv-events-replay-endpoint via CLI",
		args: []string{"cmd", "--model", common.TestModelName, "--seed", "100", "--enable-kvcache",
			"--kv-events-replay-endpoint", "tcp://*:5558"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// kv-events-replay-endpoint not set — defaults to empty (disabled)
	c = createConfigWithModel(common.TestModelName, nil)
	c.Lora.MaxCPULoras = 1
	c.Seed = 100
	test = testCase{
		name:           "kv-events-replay-endpoint disabled by default",
		args:           []string{"cmd", "--model", common.TestModelName, "--seed", "100"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// kv-cache-only flags without --enable-kvcache are inert: the whole
	// KVCache block reports all-zero rather than the flag values.
	c = createConfigWithModel(common.TestModelName, nil)
	c.Lora.MaxCPULoras = 1
	c.Seed = 100
	test = testCase{
		name: "kv-cache flags without --enable-kvcache report an all-zero KVCache block",
		args: []string{"cmd", "--model", common.TestModelName, "--seed", "100",
			"--kv-cache-size", "2048", "--block-size", "32", "--zmq-endpoint", "tcp://127.0.0.1:5559",
			"--event-batch-size", "8"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// tensor-parallel-size is accepted for vLLM command line compatibility and ignored:
	// it has no Configuration field, so the resulting config is unaffected.
	c = createConfigWithModel(common.TestModelName, nil)
	c.Lora.MaxCPULoras = 1
	c.Seed = 100
	test = testCase{
		name:           "tensor-parallel-size",
		args:           []string{"cmd", "--model", common.TestModelName, "--seed", "100", "--tensor-parallel-size", "2"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// zmq-endpoint and kv-events-replay-endpoint ports far enough apart that
	// they don't collide even once each rank's offset (0..data-parallel-size-1) is applied
	c = createConfigWithModel(common.TestModelName, nil)
	c.Lora.MaxCPULoras = 1
	c.Seed = 100
	c.DPSize = 3
	c.KVCache = defaultConfig().KVCache
	c.KVCache.EnableKVCache = true
	c.KVCache.ZMQEndpoint = "tcp://127.0.0.1:5557"
	c.KVCache.KVEventsReplayEndpoint = "tcp://*:5600"
	test = testCase{
		name: "zmq-endpoint and kv-events-replay-endpoint ports don't collide with data-parallel-size",
		args: []string{"cmd", "--model", common.TestModelName, "--seed", "100", "--data-parallel-size", "3",
			"--enable-kvcache", "--zmq-endpoint", "tcp://127.0.0.1:5557", "--kv-events-replay-endpoint", "tcp://*:5600"},
		expectedConfig: c,
	}
	tests = append(tests, test)

	// data-parallel-rank is set to a single fixed value for this process, but the
	// collision check still spans the full data-parallel-size range: the other
	// ranks of the cluster are still out there running with their own fixed rank
	// and the same base endpoints, so the check is unaffected by data-parallel-rank
	// being set here. These ports (range [5557,5559] vs [5600,5602]) don't collide
	// either way.
	c = createConfigWithModel(common.TestModelName, nil)
	c.Lora.MaxCPULoras = 1
	c.Seed = 100
	c.DPSize = 3
	c.Rank = 2
	c.KVCache = defaultConfig().KVCache
	c.KVCache.EnableKVCache = true
	c.KVCache.ZMQEndpoint = "tcp://127.0.0.1:5557"
	c.KVCache.KVEventsReplayEndpoint = "tcp://*:5600"
	test = testCase{
		name: "zmq-endpoint and kv-events-replay-endpoint ports don't collide when data-parallel-rank is set",
		args: []string{"cmd", "--model", common.TestModelName, "--seed", "100", "--data-parallel-size", "3",
			"--data-parallel-rank", "2", "--enable-kvcache",
			"--zmq-endpoint", "tcp://127.0.0.1:5557", "--kv-events-replay-endpoint", "tcp://*:5600"},
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
			args:          []string{"cmd", "--model", "", "--config", "../../../manifests/config.yaml"},
			expectedError: "model parameter is empty",
		},
		{
			name:          "invalid port",
			args:          []string{"cmd", "--port", "-50", "--config", "../../../manifests/config.yaml"},
			expectedError: "invalid port",
		},
		{
			name:          "invalid max-loras",
			args:          []string{"cmd", "--max-loras", "15", "--config", "../../../manifests/config.yaml"},
			expectedError: "max CPU LoRAs cannot be less than max LoRAs",
		},
		{
			name:          "invalid mode",
			args:          []string{"cmd", "--mode", "hello", "--config", "../../../manifests/config.yaml"},
			expectedError: "invalid mode ",
		},
		{
			name: "invalid lora",
			args: []string{"cmd", "--config", "../../../manifests/config.yaml",
				"--lora-modules", "{\"path\":\"/path/to/lora15\"}"},
			expectedError: "empty LoRA name",
		},
		{
			name:          "invalid max-model-len",
			args:          []string{"cmd", "--max-model-len", "0", "--config", "../../../manifests/config.yaml"},
			expectedError: "max model len cannot be less than 1",
		},
		{
			name:          "invalid tool-call-not-required-param-probability",
			args:          []string{"cmd", "--tool-call-not-required-param-probability", "-10", "--config", "../../../manifests/config.yaml"},
			expectedError: "ToolCallNotRequiredParamProbability should be between 0 and 100",
		},
		{
			name: "invalid max-tool-call-number-param",
			args: []string{"cmd", "--max-tool-call-number-param", "-10", "--min-tool-call-number-param", "0",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "MaxToolCallNumberParam cannot be less than MinToolCallNumberParam",
		},
		{
			name: "invalid max-tool-call-integer-param",
			args: []string{"cmd", "--max-tool-call-integer-param", "-10", "--min-tool-call-integer-param", "0",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "MaxToolCallIntegerParam cannot be less than MinToolCallIntegerParam",
		},
		{
			name: "invalid max-tool-call-array-param-length",
			args: []string{"cmd", "--max-tool-call-array-param-length", "-10", "--min-tool-call-array-param-length", "0",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "MaxToolCallArrayParamLength cannot be less than MinToolCallArrayParamLength",
		},
		{
			name: "invalid tool-call-not-required-param-probability",
			args: []string{"cmd", "--tool-call-not-required-param-probability", "-10",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "ToolCallNotRequiredParamProbability should be between 0 and 100",
		},
		{
			name: "invalid object-tool-call-not-required-field-probability",
			args: []string{"cmd", "--object-tool-call-not-required-field-probability", "1210",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "ObjectToolCallNotRequiredParamProbability should be between 0 and 100",
		},
		{
			name: "invalid tool-call-extra-call-probability",
			args: []string{"cmd", "--tool-call-extra-call-probability", "-1",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "ToolCallExtraCallProbability should be between 0 and 100",
		},
		{
			name: "invalid time-to-first-token-std-dev",
			args: []string{"cmd", "--time-to-first-token-std-dev", "3000ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "time to first token standard deviation cannot be more than 30%",
		},
		{
			name: "invalid (negative) time-to-first-token-std-dev",
			args: []string{"cmd", "--time-to-first-token-std-dev", "10ms", "--time-to-first-token-std-dev", "-1ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "time to first token standard deviation cannot be negative",
		},
		{
			name: "invalid inter-token-latency-std-dev",
			args: []string{"cmd", "--inter-token-latency", "1000ms", "--inter-token-latency-std-dev", "301ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "inter token latency standard deviation cannot be more than 30%",
		},
		{
			name: "invalid (negative) inter-token-latency-std-dev",
			args: []string{"cmd", "--inter-token-latency", "1000ms", "--inter-token-latency-std-dev", "-1s",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "inter token latency standard deviation cannot be negative",
		},
		{
			name: "invalid kv-cache-transfer-latency-std-dev",
			args: []string{"cmd", "--kv-cache-transfer-latency", "70ms", "--kv-cache-transfer-latency-std-dev", "35ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "kv-cache transfer standard deviation cannot be more than 30% of kv-cache transfer",
		},
		{
			name: "invalid (negative) kv-cache-transfer-latency-std-dev",
			args: []string{"cmd", "--kv-cache-transfer-latency-std-dev", "-35ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "kv-cache transfer time standard deviation cannot be negative",
		},
		{
			name: "invalid (negative) kv-cache-size",
			args: []string{"cmd", "--enable-kvcache", "--kv-cache-size", "-35",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "KV cache size cannot be negative",
		},
		{
			name: "invalid block-size",
			args: []string{"cmd", "--enable-kvcache", "--block-size", "35",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "token block size should be one of the following",
		},
		{
			name: "invalid (negative) event-batch-size",
			args: []string{"cmd", "--enable-kvcache", "--event-batch-size", "-35",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "event batch size cannot less than 1",
		},
		{
			name:          "invalid failure injection rate > 100",
			args:          []string{"cmd", "--model", common.TestModelName, "--failure-injection-rate", "150"},
			expectedError: "failure injection rate should be between 0 and 100",
		},
		{
			name:          "invalid failure injection rate < 0",
			args:          []string{"cmd", "--model", common.TestModelName, "--failure-injection-rate", "-10"},
			expectedError: "failure injection rate should be between 0 and 100",
		},
		{
			name: "invalid failure type",
			args: []string{"cmd", "--model", common.TestModelName, "--failure-injection-rate", "50",
				"--failure-types", "invalid_type"},
			expectedError: "invalid failure type",
		},
		{
			name: "invalid fake metrics: negative running requests",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":-10,\"waiting-requests\":30,\"kv-cache-usage\":0.4}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics request counters cannot be negative",
		},
		{
			name: "invalid fake metrics: invalid running requests function",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":\"foo:0:8:10s\",\"waiting-requests\":30,\"kv-cache-usage\":0.4}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "invalid fake metrics generation function foo",
		},
		{
			name: "invalid fake metrics: invalid function parameter period",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":19,\"waiting-requests\":\"squarewave:0:8:170\",\"kv-cache-usage\":0.4}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "unknown format in fake metric generation function: time: missing unit in duration",
		},
		{
			name: "invalid fake metrics: invalid function parameter period, can't be 0",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":19,\"waiting-requests\":\"squarewave:0:8:0s\",\"kv-cache-usage\":0.4}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "invalid fake metrics generation parameter: period must be positive",
		},
		{
			name: "invalid fake metrics: incomplete waiting requests function parameters",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":19,\"waiting-requests\":\"rampreset:0:8\",\"kv-cache-usage\":0.4}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "need func:start:end:period in fake metric generation function",
		},
		{
			name: "invalid fake metrics: kv cache usage",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":40}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics KV cache usage must be between 0 and 1",
		},
		{
			name: "invalid fake metrics: negative kv cache usage function parameters",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":\"ramp:0:-8:10s\"}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "invalid fake metrics generation parameter: start and end must not be negative",
		},
		{
			name: "invalid fake metrics: invalid kv cache usage function parameters",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":\"ramp:0:5:10s\"}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics KV cache usage start and end must be between 0 and 1",
		},
		{
			name: "invalid fake metrics refresh period",
			args: []string{"cmd", "--fake-metrics", "{\"running-requests\":10,\"waiting-requests\":30,\"kv-cache-usage\":\"ramp:0:1:10s\"}",
				"--fake-metrics-refresh-interval", "-20s",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics refresh interval must be positive",
		},
		{
			name: "invalid (negative) prefill-overhead",
			args: []string{"cmd", "--prefill-overhead", "-1ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "prefill overhead cannot be negative",
		},
		{
			name: "invalid (negative) prefill-time-per-token",
			args: []string{"cmd", "--prefill-time-per-token", "-1ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "prefill time per token cannot be negative",
		},
		{
			name: "invalid (negative) prefill-time-std-dev",
			args: []string{"cmd", "--prefill-time-std-dev", "-1ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "prefill time standard deviation cannot be negative",
		},
		{
			name: "invalid (negative) kv-cache-transfer-time-per-token",
			args: []string{"cmd", "--kv-cache-transfer-time-per-token", "-1ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "kv-cache transfer time per token cannot be negative",
		},
		{
			name: "invalid (negative) kv-cache-transfer-time-std-dev",
			args: []string{"cmd", "--kv-cache-transfer-time-std-dev", "-1ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "kv-cache transfer time standard deviation cannot be negative",
		},
		{
			name: "invalid (negative) time-to-generate-image",
			args: []string{"cmd", "--time-to-generate-image", "-1ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "time to generate image cannot be negative",
		},
		{
			name: "invalid (negative) time-to-generate-image-std-dev",
			args: []string{"cmd", "--time-to-generate-image-std-dev", "-1ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "time to generate image standard deviation cannot be negative",
		},
		{
			name: "invalid time-to-generate-image-std-dev exceeds 30%",
			args: []string{"cmd", "--time-to-generate-image", "500ms", "--time-to-generate-image-std-dev", "200ms",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "time to generate image standard deviation cannot be more than 30% of time to generate image",
		},
		{
			name: "invalid data-parallel-size",
			args: []string{"cmd", "--data-parallel-size", "15",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "data parallel size must be between 1 and 8",
		},
		{
			name: "invalid data-parallel-rank",
			args: []string{"cmd", "--data-parallel-rank", "15",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "data parallel rank must be between 0 and 7",
		},
		{
			name: "invalid zmq-endpoint and kv-events-replay-endpoint on the same port",
			args: []string{"cmd", "--enable-kvcache", "--zmq-endpoint", "tcp://127.0.0.1:5557",
				"--kv-events-replay-endpoint", "tcp://127.0.0.1:5557",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "zmq-endpoint (tcp://127.0.0.1:5557) and kv-events-replay-endpoint (tcp://127.0.0.1:5557) ports collide",
		},
		{
			name: "invalid zmq-endpoint and kv-events-replay-endpoint colliding once offset by data-parallel-size",
			args: []string{"cmd", "--enable-kvcache", "--data-parallel-size", "3",
				"--zmq-endpoint", "tcp://127.0.0.1:5557",
				"--kv-events-replay-endpoint", "tcp://127.0.0.1:5558",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "zmq-endpoint (tcp://127.0.0.1:5557) and kv-events-replay-endpoint (tcp://127.0.0.1:5558) ports collide",
		},
		{
			name: "invalid zmq-endpoint and kv-events-replay-endpoint on the same port with data-parallel-rank set",
			args: []string{"cmd", "--enable-kvcache", "--data-parallel-size", "3", "--data-parallel-rank", "2",
				"--zmq-endpoint", "tcp://127.0.0.1:5557",
				"--kv-events-replay-endpoint", "tcp://127.0.0.1:5557",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "zmq-endpoint (tcp://127.0.0.1:5557) and kv-events-replay-endpoint (tcp://127.0.0.1:5557) ports collide",
		},
		{
			// data-parallel-rank is fixed to 2 for this process, but the check still
			// spans the full data-parallel-size range: rank 2's zmq port (5559) would
			// collide with rank 0's replay port (5559) elsewhere in the same cluster,
			// even though this process's own zmq (5559) and replay (5561) ports don't
			// collide with each other.
			name: "invalid zmq-endpoint and kv-events-replay-endpoint colliding with another rank's port when data-parallel-rank is set",
			args: []string{"cmd", "--enable-kvcache", "--data-parallel-size", "3", "--data-parallel-rank", "2",
				"--zmq-endpoint", "tcp://127.0.0.1:5557",
				"--kv-events-replay-endpoint", "tcp://127.0.0.1:5559",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "zmq-endpoint (tcp://127.0.0.1:5557) and kv-events-replay-endpoint (tcp://127.0.0.1:5559) ports collide",
		},
		{
			name: "invalid kv-events-replay-queue-size",
			args: []string{"cmd", "--enable-kvcache", "--kv-events-replay-endpoint", "tcp://*:5558",
				"--kv-events-replay-queue-size", "0",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "kv-events-replay-queue-size cannot be less than 1",
		},
		{
			name: "invalid max-num-seqs",
			args: []string{"cmd", "--max-num-seqs", "0",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "max num seqs cannot be less than 1",
		},
		{
			name: "invalid max-num-seqs",
			args: []string{"cmd", "--max-num-seqs", "-1",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "max num seqs cannot be less than 1",
		},
		{
			name: "invalid max-waiting-queue-length",
			args: []string{"cmd", "--max-waiting-queue-length", "-1",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "max waiting queue size cannot be less than 0",
		},
		{
			name: "invalid time-factor-under-load",
			args: []string{"cmd", "--time-factor-under-load", "0",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "time factor under load cannot be less than 1.0",
		},
		{
			name: "invalid time-factor-under-load",
			args: []string{"cmd", "--time-factor-under-load", "-1",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "time factor under load cannot be less than 1.0",
		},
		{
			name: "invalid ttft",
			args: []string{"cmd", "--fake-metrics", "{\"ttft-buckets-values\":[1, 2, -10, 1]}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "time-to-first-token fake metrics should contain only non-negative values",
		},
		{
			name: "invalid tpot",
			args: []string{"cmd", "--fake-metrics", "{\"tpot-buckets-values\":[1, 2, -10, 1]}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "time-per-output-token fake metrics should contain only non-negative values",
		},
		{
			name: "invalid request-max-generation-tokens",
			args: []string{"cmd", "--fake-metrics", "{\"request-max-generation-tokens\": [1, -1, 2]}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics request-max-generation-tokens cannot contain negative values",
		},
		{
			name: "invalid fake metrics: negative prefix-cache-hits",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-hits\":-5,\"prefix-cache-queries\":10}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-hits cannot be negative",
		},
		{
			name: "invalid fake metrics: negative prefix-cache-queries",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-hits\":0,\"prefix-cache-queries\":-1}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-queries cannot be negative",
		},
		{
			name: "invalid fake metrics: prefix-cache-hits without prefix-cache-queries",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-hits\":100}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-hits and prefix-cache-queries must be specified together",
		},
		{
			name: "invalid fake metrics: prefix-cache-queries without prefix-cache-hits",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-queries\":100}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-hits and prefix-cache-queries must be specified together",
		},
		{
			name: "invalid fake metrics: prefix-cache-hits exceeds prefix-cache-queries",
			args: []string{"cmd", "--fake-metrics", "{\"prefix-cache-hits\":100,\"prefix-cache-queries\":50}",
				"--config", "../../../manifests/config.yaml"},
			expectedError: "fake metrics prefix-cache-hits cannot exceed prefix-cache-queries",
		},
		{
			name: "invalid echo mode with dataset",
			args: []string{"cmd", "--model", common.TestModelName, "--dataset-path", "my/path",
				"--mode", "echo"},
			expectedError: "dataset cannot be defined in echo mode",
		},
		{
			name:          "invalid latency calculator",
			args:          []string{"cmd", "--config", "../../../manifests/config.yaml", "--latency-calculator", "hello"},
			expectedError: "unknown latency-calculator",
		},
		{
			name:          "invalid max-request-body-size-mb (too small)",
			args:          []string{"cmd", "--config", "../../../manifests/config.yaml", "--max-request-body-size-mb", "-1"},
			expectedError: "max-request-body-size-mb must be between 1 MB and 512 MB",
		},
		{
			name:          "invalid max-request-body-size-mb (too large)",
			args:          []string{"cmd", "--config", "../../../manifests/config.yaml", "--max-request-body-size-mb", "513"},
			expectedError: "max-request-body-size-mb must be between 1 MB and 512 MB",
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

var _ = Describe("Model environment variable", func() {
	BeforeEach(func() {
		Expect(os.Unsetenv(common.ModelEnv)).To(Succeed())
	})
	AfterEach(func() {
		Expect(os.Unsetenv(common.ModelEnv)).To(Succeed())
	})

	It("does not override --model when the flag is passed", func() {
		Expect(os.Setenv(common.ModelEnv, "from-env")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--seed", "100"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.Model).To(Equal(common.TestModelName))
	})

	It("overrides model from config file when --model is omitted", func() {
		Expect(os.Setenv(common.ModelEnv, "env-override-model")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--config", "../../../manifests/config.yaml"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.Model).To(Equal("env-override-model"))
	})

	It("does not change model when unset and --model is passed", func() {
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom, "--seed", "100"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.Model).To(Equal(common.TestModelName))
	})
})

var _ = Describe("PYTHONHASHSEED environment variable", func() {
	BeforeEach(func() {
		Expect(os.Unsetenv(pythonHashSeedEnv)).To(Succeed())
	})
	AfterEach(func() {
		Expect(os.Unsetenv(pythonHashSeedEnv)).To(Succeed())
	})

	It("does not override --hash-seed when the flag is passed", func() {
		Expect(os.Setenv(pythonHashSeedEnv, "from-env")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName, "--enable-kvcache", "--hash-seed", "from-flag", "--mode", common.ModeRandom, "--seed", "100"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.KVCache.HashSeed).To(Equal("from-flag"))
	})

	It("applies when --hash-seed is omitted", func() {
		Expect(os.Setenv(pythonHashSeedEnv, "env-seed")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName, "--enable-kvcache", "--mode", common.ModeRandom, "--seed", "100"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.KVCache.HashSeed).To(Equal("env-seed"))
	})
})

var _ = Describe("VLLM_SERVER_DEV_MODE environment variable", func() {
	BeforeEach(func() {
		Expect(os.Unsetenv(devModeEnv)).To(Succeed())
	})
	AfterEach(func() {
		Expect(os.Unsetenv(devModeEnv)).To(Succeed())
	})

	It("enables dev mode when set to 1", func() {
		Expect(os.Setenv(devModeEnv, "1")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.DevMode).To(BeTrue())
	})

	It("leaves dev mode off for any other value", func() {
		Expect(os.Setenv(devModeEnv, "true")).To(Succeed())
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.DevMode).To(BeFalse())
	})

	It("leaves dev mode off when unset", func() {
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.DevMode).To(BeFalse())
	})
})

var _ = Describe("lora YAML folding", func() {
	writeConfig := func(contents string) string {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "config.yaml")
		Expect(os.WriteFile(path, []byte(contents), 0o644)).To(Succeed())
		return path
	}

	It("populates Lora from the nested lora block", func() {
		config, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
lora:
  max-loras: 4
  max-cpu-loras: 8
`)})
		Expect(err).NotTo(HaveOccurred())

		Expect(config.Lora.MaxLoras).To(Equal(4))
		Expect(config.Lora.MaxCPULoras).To(Equal(8))
	})

	It("populates Lora from legacy flat top-level keys", func() {
		config, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
max-loras: 4
max-cpu-loras: 8
`)})
		Expect(err).NotTo(HaveOccurred())

		Expect(config.Lora.MaxLoras).To(Equal(4))
		Expect(config.Lora.MaxCPULoras).To(Equal(8))
	})

	It("errors when lora settings mix the flat and nested layouts", func() {
		_, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
max-loras: 4
lora:
  max-loras: 8
`)})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("lora"))
	})

	It("errors when a flat lora key is set alongside an unrelated nested key", func() {
		_, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
max-loras: 4
lora:
  max-cpu-loras: 8
`)})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("lora"))
	})
})

var _ = Describe("kv-cache YAML folding", func() {
	writeConfig := func(contents string) string {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "config.yaml")
		Expect(os.WriteFile(path, []byte(contents), 0o644)).To(Succeed())
		return path
	}

	It("populates KVCache from the nested kvcache block", func() {
		config, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
kvcache:
  enable-kvcache: true
  kv-cache-size: 2048
  kv-cache-dtype: turboquant_4bit_nc
  block-size: 32
`)})
		Expect(err).NotTo(HaveOccurred())

		Expect(config.KVCache.EnableKVCache).To(BeTrue())
		Expect(config.KVCache.KVCacheSize).To(Equal(2048))
		Expect(config.KVCache.KVCacheDType).To(Equal("turboquant_4bit_nc"))
		Expect(config.KVCache.TokenBlockSize).To(Equal(32))
		// Settings the block omits keep the defaults ApplyDefaults applied.
		Expect(config.KVCache.EventBatchSize).To(Equal(16))
		Expect(config.KVCache.ZMQEndpoint).To(Equal("tcp://127.0.0.1:5557"))
	})

	It("lets a flag override the kvcache block, which overrides the engine default", func() {
		config, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
kvcache:
  enable-kvcache: true
  kv-cache-size: 2048
  block-size: 32
`), "--block-size", "64"})
		Expect(err).NotTo(HaveOccurred())

		// flag wins over the config file
		Expect(config.KVCache.TokenBlockSize).To(Equal(64))
		// config file wins over ApplyDefaults' 1024
		Expect(config.KVCache.KVCacheSize).To(Equal(2048))
		// ApplyDefaults stands where neither sets a value
		Expect(config.KVCache.EventBatchSize).To(Equal(16))
	})

	It("populates KVCache from legacy flat top-level keys", func() {
		config, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
enable-kvcache: true
kv-cache-size: 2048
kv-cache-dtype: turboquant_4bit_nc
block-size: 32
`)})
		Expect(err).NotTo(HaveOccurred())

		Expect(config.KVCache.EnableKVCache).To(BeTrue())
		Expect(config.KVCache.KVCacheSize).To(Equal(2048))
		Expect(config.KVCache.KVCacheDType).To(Equal("turboquant_4bit_nc"))
		Expect(config.KVCache.TokenBlockSize).To(Equal(32))
	})

	It("defaults zmq-topic to empty so the generated topic is used", func() {
		config, err := createSimConfig([]string{"cmd", "--model", "test-model", "--enable-kvcache"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.KVCache.ZMQTopic).To(BeEmpty())
	})

	It("populates zmq-topic from the nested kvcache block and lets a flag override it", func() {
		config, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
kvcache:
  enable-kvcache: true
  zmq-topic: from-yaml
`)})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.KVCache.ZMQTopic).To(Equal("from-yaml"))

		config, err = createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
kvcache:
  enable-kvcache: true
  zmq-topic: from-yaml
`), "--zmq-topic", "from-flag"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.KVCache.ZMQTopic).To(Equal("from-flag"))
	})

	It("errors when kv-cache settings mix the flat and nested layouts", func() {
		_, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
kv-cache-size: 111
kvcache:
  kv-cache-size: 222
`)})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("kvcache"))
	})

	It("errors when a flat kv-cache key is set alongside an unrelated nested key", func() {
		_, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
kv-cache-size: 111
kvcache:
  block-size: 32
`)})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("kvcache"))
	})
})

var _ = Describe("legacy flat YAML key lists", func() {
	// The legacy key lists are what let a config file set these settings at the
	// top level instead of inside their nested block. Nothing else ties them to
	// the structs they mirror, so a field added to loraYAML/kvCacheYAML without
	// a matching list entry would silently stop being accepted at the top
	// level. Derive the expected lists from the structs' own yaml tags.
	yamlTagsOf := func(v any) []string {
		t := reflect.TypeOf(v)
		keys := make([]string, 0, t.NumField())
		for i := range t.NumField() {
			key := strings.SplitN(t.Field(i).Tag.Get("yaml"), ",", 2)[0]
			if key == "" || key == "-" {
				continue
			}
			keys = append(keys, key)
		}
		return keys
	}

	It("loraLegacyFlatKeys covers every loraYAML field", func() {
		Expect(loraLegacyFlatKeys).To(ConsistOf(yamlTagsOf(loraYAML{})))
	})

	It("kvCacheLegacyFlatKeys covers every kvCacheYAML field", func() {
		Expect(kvCacheLegacyFlatKeys).To(ConsistOf(yamlTagsOf(kvCacheYAML{})))
	})
})

var _ = Describe("fake-metrics and lora edge values", func() {
	writeConfig := func(contents string) string {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "config.yaml")
		Expect(os.WriteFile(path, []byte(contents), 0o644)).To(Succeed())
		return path
	}

	It("leaves fake metrics unset when the YAML block is present but empty", func() {
		// Every setting commented out is a common real-world state. Reporting
		// fake metrics here would silently freeze the whole metrics surface at
		// zero, since fake and real metrics never coexist.
		config, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
fake-metrics:
  # running-requests: 5
`)})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.FakeMetrics).To(BeNil())
	})

	It("leaves fake metrics unset when the flag value is JSON null", func() {
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName, "--fake-metrics", "null"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.FakeMetrics).To(BeNil())
	})

	It("rejects an explicit max-loras of 0 from YAML", func() {
		_, err := createSimConfig([]string{"cmd", "--config", writeConfig(`
model: test-model
lora:
  max-loras: 0
`)})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("max LoRAs"))
	})
})
