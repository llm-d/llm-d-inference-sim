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
	"github.com/llm-d/llm-d-inference-sim/pkg/engine/vllm/fakemetrics"
)

const dummy = " "

// loraYAML mirrors the "lora" block of a YAML config file, in vLLM's own
// wire format: LoraModules is a list of JSON-encoded common.LoraModule
// strings, matching the --lora-modules CLI flag's format.
type loraYAML struct {
	MaxLoras    int      `yaml:"max-loras"`
	MaxCPULoras int      `yaml:"max-cpu-loras"`
	LoraModules []string `yaml:"lora-modules"`
}

// loraLegacyFlatKeys are the top-level YAML keys accepted in place of a
// nested "lora" block, for backward compatibility.
var loraLegacyFlatKeys = []string{"max-loras", "max-cpu-loras", "lora-modules"}

// kvCacheYAML mirrors the "kvcache" block of a YAML config file, in vLLM's
// own wire format. Its field sequence must match common.KVCacheConfig's so
// the two are convertible (struct tags are ignored for conversion, so this
// is only a tag change, not a copy).
type kvCacheYAML struct {
	EnableKVCache           bool   `yaml:"enable-kvcache"`
	KVCacheSize             int    `yaml:"kv-cache-size"`
	KVCacheDType            string `yaml:"kv-cache-dtype"`
	TokenBlockSize          int    `yaml:"block-size"`
	HashSeed                string `yaml:"hash-seed"`
	ZMQEndpoint             string `yaml:"zmq-endpoint"`
	KVEventsReplayEndpoint  string `yaml:"kv-events-replay-endpoint"`
	KVEventsReplayQueueSize int    `yaml:"kv-events-replay-queue-size"`
	EventBatchSize          int    `yaml:"event-batch-size"`
	UseVllmMapEventFormat   bool   `yaml:"use-vllm-map-event-format"`
}

// kvCacheLegacyFlatKeys are the top-level YAML keys accepted in place of a
// nested "kvcache" block, for backward compatibility.
var kvCacheLegacyFlatKeys = []string{
	"enable-kvcache", "kv-cache-size", "kv-cache-dtype", "block-size", "hash-seed",
	"zmq-endpoint", "kv-events-replay-endpoint", "kv-events-replay-queue-size",
	"event-batch-size", "use-vllm-map-event-format",
}

// BindFlags registers the vLLM-specific CLI flags on f and builds the config
// groups whose wire format this engine owns: it folds and unmarshals rawYAML's
// "lora", "kvcache", and "fake-metrics" blocks, then reconciles the
// lora-modules and fake-metrics values from either source into their
// structured fields. The YAML groups are read before the flags that bind to
// them are registered, so each flag's pflag default reflects the config file.
// Must be called before f.Parse.
func (Engine) BindFlags(f *pflag.FlagSet, cfg *common.Configuration, rawYAML map[string]any) error {
	loraModuleNames := common.GetParamValueFromArgs("lora-modules")

	// A YAML config file's "lora" block uses vLLM's own wire format, folded
	// and unmarshaled here (rather than by common.Configuration.load) since
	// a different engine could shape LoRA config differently.
	var yamlLoraModules []string
	if rawYAML != nil {
		if err := common.FoldLegacyKeys(rawYAML, "lora", loraLegacyFlatKeys); err != nil {
			return err
		}
		// Seeded from the current values so that keys the block omits keep
		// their defaults while an explicit zero still comes through (and is
		// then rejected by ValidateConfig, rather than silently ignored).
		// loraYAML is not convertible from common.LoraConfig the way
		// kvCacheYAML is, since LoraModules differs in type.
		ly := loraYAML{MaxLoras: cfg.Lora.MaxLoras, MaxCPULoras: cfg.Lora.MaxCPULoras}
		if err := common.UnmarshalYAMLKey(rawYAML, "lora", &ly); err != nil {
			return err
		}
		cfg.Lora.MaxLoras = ly.MaxLoras
		cfg.Lora.MaxCPULoras = ly.MaxCPULoras
		yamlLoraModules = ly.LoraModules
	}

	// A YAML config file's "kvcache" block uses vLLM's own wire format,
	// folded and unmarshaled here (rather than by common.Configuration.load)
	// since a different engine could shape KV-cache config differently.
	if rawYAML != nil {
		if err := common.FoldLegacyKeys(rawYAML, "kvcache", kvCacheLegacyFlatKeys); err != nil {
			return err
		}
		kv := kvCacheYAML(cfg.KVCache)
		if err := common.UnmarshalYAMLKey(rawYAML, "kvcache", &kv); err != nil {
			return err
		}
		cfg.KVCache = common.KVCacheConfig(kv)
	}

	// A YAML config file's "fake-metrics" block is unmarshaled directly into
	// vLLM's own concrete type here (rather than by common.Configuration.load)
	// since Configuration.FakeMetrics is an engine-owned interface that
	// encoding/yaml cannot allocate a concrete value into on its own. There is
	// no legacy flat layout to fold: fake-metrics has always been nested only.
	// A present but empty block (every setting commented out) carries a nil
	// value and must leave fake metrics unset: reporting them suppresses every
	// real metric, so allocating here would silently freeze the whole metrics
	// surface at zero.
	var yamlFakeMetrics *fakemetrics.Config
	if rawYAML != nil {
		if v, ok := rawYAML["fake-metrics"]; ok && v != nil {
			yamlFakeMetrics = &fakemetrics.Config{}
			if err := common.UnmarshalYAMLKey(rawYAML, "fake-metrics", yamlFakeMetrics); err != nil {
				return err
			}
		}
	}

	f.IntVar(&cfg.Lora.MaxLoras, "max-loras", cfg.Lora.MaxLoras, "Maximum number of LoRAs in a single batch")
	f.IntVar(&cfg.Lora.MaxCPULoras, "max-cpu-loras", cfg.Lora.MaxCPULoras, "Maximum number of LoRAs to store in CPU memory")

	f.DurationVar(&cfg.Latencies.KVCacheTransferTimePerToken, "kv-cache-transfer-time-per-token", cfg.Latencies.KVCacheTransferTimePerToken, "Time for KV-cache transfer per token from a remote vLLM, e.g. 100ms")
	f.DurationVar(&cfg.Latencies.KVCacheTransferTimeStdDev, "kv-cache-transfer-time-std-dev", cfg.Latencies.KVCacheTransferTimeStdDev, "Standard deviation for time for KV-cache transfer per token from a remote vLLM, e.g. 100ms")
	f.DurationVar(&cfg.Latencies.KVCacheTransferLatency, "kv-cache-transfer-latency", cfg.Latencies.KVCacheTransferLatency, "Time for KV-cache transfer from a remote vLLM, e.g. 100ms")
	f.DurationVar(&cfg.Latencies.KVCacheTransferLatencyStdDev, "kv-cache-transfer-latency-std-dev", cfg.Latencies.KVCacheTransferLatencyStdDev, "Standard deviation for time for KV-cache transfer from a remote vLLM, e.g. 100ms")

	f.BoolVar(&cfg.KVCache.EnableKVCache, "enable-kvcache", cfg.KVCache.EnableKVCache, "Defines if KV cache feature is enabled")
	f.IntVar(&cfg.KVCache.KVCacheSize, "kv-cache-size", cfg.KVCache.KVCacheSize, "Maximum number of token blocks in kv cache")
	f.StringVar(&cfg.KVCache.KVCacheDType, "kv-cache-dtype", cfg.KVCache.KVCacheDType, "KV cache dtype reported in vLLM-compatible metrics")
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

	// These vLLM CLI flags are accepted for command-line compatibility and
	// ignored: nothing in the simulator ever reads them back, so unlike the
	// rest of this engine's flags they have no Configuration field to bind
	// to, just a local variable to satisfy pflag's API.
	var mmProcessorKWArgs, ecTransferConfig string
	var enforceEager, enablePrefixCaching bool
	var tpSize int
	f.StringVar(&mmProcessorKWArgs, "mm-processor-kwargs", "", "Arguments to be forwarded to the model's processor for multi-modal data, ignored")
	f.StringVar(&ecTransferConfig, "ec-transfer-config", "", "Configuration for distributed EC cache transfer, ignored")
	common.AddToggle(f, &enforceEager,
		"enforce-eager", "Always use eager-mode PyTorch, ignored", "Don't always use eager-mode PyTorch, ignored")
	common.AddToggle(f, &enablePrefixCaching,
		"enable-prefix-caching", "Enable prefix caching, ignored", "Disable prefix caching, ignored")
	f.IntVar(&tpSize, "tensor-parallel-size", 0, "Number of tensor parallel replicas, ignored")

	// lora-modules and fake-metrics take multiple space-separated JSON strings,
	// which pflag cannot bind directly; pre-scanned from os.Args like
	// common.GetParamValueFromArgs's other callers, and registered below only
	// so they show up in --help.
	fakeMetricsStrings := common.GetParamValueFromArgs("fake-metrics")

	var dummyMultiString multiString
	f.Var(&dummyMultiString, "lora-modules", "List of LoRA adapters (a list of space-separated JSON strings)")
	f.Lookup("lora-modules").NoOptDefVal = dummy
	f.Lookup("lora-modules").DefValue = ""
	f.Var(&dummyMultiString, "fake-metrics", "A set of metrics to report to Prometheus instead of the real metrics")
	f.Lookup("fake-metrics").NoOptDefVal = dummy
	f.Lookup("fake-metrics").DefValue = ""

	if yamlFakeMetrics != nil {
		if err := unmarshalLoraFakeMetrics(yamlFakeMetrics); err != nil {
			return err
		}
		cfg.FakeMetrics = yamlFakeMetrics
	}
	if fakeMetricsStrings != nil {
		// A --fake-metrics flag replaces the whole FakeMetrics struct (its JSON
		// "loras" key maps straight onto LoraMetrics), so any YAML-driven
		// reconciliation above is superseded, not merged.
		if err := unmarshalFakeMetrics(cfg, fakeMetricsStrings[0]); err != nil {
			return err
		}
	}

	if rawYAML != nil {
		if err := unmarshalLoras(cfg, yamlLoraModules); err != nil {
			return err
		}
	}
	if loraModuleNames != nil {
		if err := unmarshalLoras(cfg, loraModuleNames); err != nil {
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

// unmarshalLoras reconciles loraModuleNames (raw JSON strings, from a YAML
// config file's "lora" block or the --lora-modules flag) into cfg.Lora.LoraModules.
func unmarshalLoras(cfg *common.Configuration, loraModuleNames []string) error {
	cfg.Lora.LoraModules = make([]common.LoraModule, 0)
	for _, jsonStr := range loraModuleNames {
		var lora common.LoraModule
		if err := json.Unmarshal([]byte(jsonStr), &lora); err != nil {
			return err
		}
		cfg.Lora.LoraModules = append(cfg.Lora.LoraModules, lora)
	}
	return nil
}

// unmarshalFakeMetrics parses the --fake-metrics flag's JSON string into cfg.FakeMetrics.
func unmarshalFakeMetrics(cfg *common.Configuration, fakeMetricsString string) error {
	var metrics *fakemetrics.Config
	if err := json.Unmarshal([]byte(fakeMetricsString), &metrics); err != nil {
		return err
	}
	if metrics == nil {
		// A JSON null decodes to a nil pointer, which would make the
		// FakeMetrics interface itself non-nil and every "are fake metrics
		// configured" check pass while holding nothing to read.
		return nil
	}
	cfg.FakeMetrics = metrics
	return nil
}

// unmarshalLoraFakeMetrics reconciles fm.LorasString (raw JSON strings, from
// a YAML config file) into fm.LoraMetrics.
func unmarshalLoraFakeMetrics(fm *fakemetrics.Config) error {
	fm.LoraMetrics = make([]common.LorasMetrics, 0)
	for _, jsonStr := range fm.LorasString {
		var lora common.LorasMetrics
		if err := json.Unmarshal([]byte(jsonStr), &lora); err != nil {
			return err
		}
		fm.LoraMetrics = append(fm.LoraMetrics, lora)
	}
	return nil
}
