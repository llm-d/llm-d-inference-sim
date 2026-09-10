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
	"encoding/json"

	"github.com/spf13/pflag"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const dummy = " "

// BindFlags registers the vLLM-specific CLI flags on f and reconciles the
// lora-modules/fake-metrics values (which may have come from a YAML config
// file already loaded into cfg, or from the command line) into their
// structured fields. Must be called before f.Parse.
func (Engine) BindFlags(f *pflag.FlagSet, cfg *common.Configuration) error {
	f.IntVar(&cfg.MaxLoras, "max-loras", cfg.MaxLoras, "Maximum number of LoRAs in a single batch")
	f.IntVar(&cfg.MaxCPULoras, "max-cpu-loras", cfg.MaxCPULoras, "Maximum number of LoRAs to store in CPU memory")

	f.DurationVar(&cfg.Latencies.KVCacheTransferTimePerToken, "kv-cache-transfer-time-per-token", cfg.Latencies.KVCacheTransferTimePerToken, "Time for KV-cache transfer per token from a remote vLLM, e.g. 100ms")
	f.DurationVar(&cfg.Latencies.KVCacheTransferTimeStdDev, "kv-cache-transfer-time-std-dev", cfg.Latencies.KVCacheTransferTimeStdDev, "Standard deviation for time for KV-cache transfer per token from a remote vLLM, e.g. 100ms")
	f.DurationVar(&cfg.Latencies.KVCacheTransferLatency, "kv-cache-transfer-latency", cfg.Latencies.KVCacheTransferLatency, "Time for KV-cache transfer from a remote vLLM, e.g. 100ms")
	f.DurationVar(&cfg.Latencies.KVCacheTransferLatencyStdDev, "kv-cache-transfer-latency-std-dev", cfg.Latencies.KVCacheTransferLatencyStdDev, "Standard deviation for time for KV-cache transfer from a remote vLLM, e.g. 100ms")

	f.BoolVar(&cfg.KVCache.EnableKVCache, "enable-kvcache", cfg.KVCache.EnableKVCache, "Defines if KV cache feature is enabled")
	f.IntVar(&cfg.KVCache.KVCacheSize, "kv-cache-size", cfg.KVCache.KVCacheSize, "Maximum number of token blocks in kv cache")
	f.Float64Var(&cfg.GlobalCacheHitThreshold, "global-cache-hit-threshold", cfg.GlobalCacheHitThreshold, "Default cache hit threshold [0, 1] for all requests. If a request specifies cache_hit_threshold, it takes precedence")
	f.IntVar(&cfg.KVCache.TokenBlockSize, "block-size", cfg.KVCache.TokenBlockSize, "Token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128")
	f.StringVar(&cfg.KVCache.HashSeed, "hash-seed", cfg.KVCache.HashSeed,
		"Seed for hash generation (if omitted on the command line, "+common.PythonHashSeedEnv+" may set it; see docs)")
	f.StringVar(&cfg.KVCache.ZMQEndpoint, "zmq-endpoint", cfg.KVCache.ZMQEndpoint, "ZMQ address to publish events")
	f.StringVar(&cfg.KVCache.KVEventsReplayEndpoint, "kv-events-replay-endpoint", cfg.KVCache.KVEventsReplayEndpoint, "ZMQ ROUTER address to bind for receiving KV events replay requests (empty disables)")
	f.IntVar(&cfg.KVCache.KVEventsReplayQueueSize, "kv-events-replay-queue-size", cfg.KVCache.KVEventsReplayQueueSize, "Max number of event batches held in the replay queue; oldest dropped when full")
	f.IntVar(&cfg.KVCache.EventBatchSize, "event-batch-size", cfg.KVCache.EventBatchSize, "Maximum number of kv-cache events to be sent together")
	f.BoolVar(&cfg.KVCache.UseVllmMapEventFormat, "use-vllm-map-event-format", cfg.KVCache.UseVllmMapEventFormat, "Encode KV cache events as msgpack maps with named fields (vLLM PR #42892 format) instead of positional arrays")

	common.AddToggle(f, &cfg.EnableSleepMode, "enable-sleep-mode", "Enable sleep mode", "Disable sleep mode")

	f.DurationVar(&cfg.FakeMetricsRefreshInterval, "fake-metrics-refresh-interval", cfg.FakeMetricsRefreshInterval,
		"Defines how often function-based fake metrics are recalculated, defaults to 100ms")

	common.AddToggle(f, &cfg.MMEncoderOnly,
		"mm-encoder-only", "Skip the language component of the model", "Don't skip the language component of the model")
	f.StringVar(&cfg.MMProcessorKWArgs, "mm-processor-kwargs", cfg.MMProcessorKWArgs, "Arguments to be forwarded to the model's processor for multi-modal data, ignored")
	f.StringVar(&cfg.ECTransferConfig, "ec-transfer-config", cfg.ECTransferConfig, "Configuration for distributed EC cache transfer, ignored")
	common.AddToggle(f, &cfg.EnforceEager,
		"enforce-eager", "Always use eager-mode PyTorch, ignored", "Don't always use eager-mode PyTorch, ignored")
	common.AddToggle(f, &cfg.EnablePrefixCaching,
		"enable-prefix-caching", "Enable prefix caching, ignored", "Disable prefix caching, ignored")
	f.IntVar(&cfg.TPSize, "tensor-parallel-size", cfg.TPSize, "Number of tensor parallel replicas, ignored")

	// lora-modules and fake-metrics take multiple space-separated JSON strings,
	// which pflag cannot bind directly; pre-scanned from os.Args like
	// common.GetParamValueFromArgs's other callers, and registered below only
	// so they show up in --help.
	loraModuleNames := common.GetParamValueFromArgs("lora-modules")
	fakeMetricsStrings := common.GetParamValueFromArgs("fake-metrics")
	// A YAML config file, if given, was already loaded into cfg by the time
	// BindFlags runs; a lora-modules/fake-metrics list there needs the same
	// string-to-struct reconciliation CLI values need.
	configFileGiven := len(common.GetParamValueFromArgs("config")) == 1

	var dummyMultiString multiString
	f.Var(&dummyMultiString, "lora-modules", "List of LoRA adapters (a list of space-separated JSON strings)")
	f.Lookup("lora-modules").NoOptDefVal = dummy
	f.Lookup("lora-modules").DefValue = ""
	f.Var(&dummyMultiString, "fake-metrics", "A set of metrics to report to Prometheus instead of the real metrics")
	f.Lookup("fake-metrics").NoOptDefVal = dummy
	f.Lookup("fake-metrics").DefValue = ""

	if configFileGiven {
		if err := unmarshalLoraFakeMetrics(cfg); err != nil {
			return err
		}
	}
	if fakeMetricsStrings != nil {
		// A --fake-metrics flag replaces the whole FakeMetrics struct (its JSON
		// "loras" key maps straight onto LoraMetrics), so any YAML-driven
		// reconciliation above is superseded, not merged.
		if err := unmarshalFakeMetrics(cfg, fakeMetricsStrings[0]); err != nil {
			return err
		}
	}

	if configFileGiven {
		if err := unmarshalLoras(cfg); err != nil {
			return err
		}
	}
	if loraModuleNames != nil {
		cfg.LoraModulesString = loraModuleNames
		if err := unmarshalLoras(cfg); err != nil {
			return err
		}
	}

	return nil
}

// multiString collects the repeated space-separated values of a flag that
// takes multiple JSON strings. The actual values are pre-scanned from
// os.Args via common.GetParamValueFromArgs; this only registers the flag so
// it appears in --help.
type multiString struct {
	values []string
}

func (l *multiString) String() string {
	return ""
}

func (l *multiString) Set(val string) error {
	l.values = append(l.values, val)
	return nil
}

func (l *multiString) Type() string {
	return "strings"
}

// unmarshalLoras reconciles cfg.LoraModulesString (raw JSON strings, from a
// YAML config file or the --lora-modules flag) into cfg.LoraModules.
func unmarshalLoras(cfg *common.Configuration) error {
	cfg.LoraModules = make([]common.LoraModule, 0)
	for _, jsonStr := range cfg.LoraModulesString {
		var lora common.LoraModule
		if err := json.Unmarshal([]byte(jsonStr), &lora); err != nil {
			return err
		}
		cfg.LoraModules = append(cfg.LoraModules, lora)
	}
	return nil
}

// unmarshalFakeMetrics parses the --fake-metrics flag's JSON string into cfg.FakeMetrics.
func unmarshalFakeMetrics(cfg *common.Configuration, fakeMetricsString string) error {
	var metrics *common.FakeMetrics
	if err := json.Unmarshal([]byte(fakeMetricsString), &metrics); err != nil {
		return err
	}
	cfg.FakeMetrics = metrics
	return nil
}

// unmarshalLoraFakeMetrics reconciles cfg.FakeMetrics.LorasString (raw JSON
// strings, from a YAML config file) into cfg.FakeMetrics.LoraMetrics.
func unmarshalLoraFakeMetrics(cfg *common.Configuration) error {
	if cfg.FakeMetrics != nil {
		cfg.FakeMetrics.LoraMetrics = make([]common.LorasMetrics, 0)
		for _, jsonStr := range cfg.FakeMetrics.LorasString {
			var lora common.LorasMetrics
			if err := json.Unmarshal([]byte(jsonStr), &lora); err != nil {
				return err
			}
			cfg.FakeMetrics.LoraMetrics = append(cfg.FakeMetrics.LoraMetrics, lora)
		}
	}
	return nil
}
