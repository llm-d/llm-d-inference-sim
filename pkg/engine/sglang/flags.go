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

package sglang

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"

	"github.com/spf13/pflag"
	"gopkg.in/yaml.v3"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// notYet marks the usage string of a flag this engine accepts but does not act
// on yet.
const notYet = " (not simulated by the sglang engine yet)"

// loraYAML mirrors the "lora" block of a YAML config file, in sglang's own
// wire format. LoraPaths takes the same item forms as the --lora-paths flag.
type loraYAML struct {
	EnableLora       *bool    `yaml:"enable-lora"`
	LoraPaths        []string `yaml:"lora-paths"`
	MaxLorasPerBatch int      `yaml:"max-loras-per-batch"`
	MaxLoadedLoras   int      `yaml:"max-loaded-loras"`
	MaxLoraRank      int      `yaml:"max-lora-rank"`
	TargetModules    []string `yaml:"lora-target-modules"`
}

// BindFlags registers this engine's own CLI flags and builds the config groups
// whose wire format it owns. Must be called before f.Parse.
//
// What it registers is this engine's name for each of the three core fields the
// engines disagree on the name of, its LoRA flags, plus the toggles of two
// features this engine does not simulate yet, bound to the configuration fields
// they will eventually fill. Asking for one of those two is therefore parsed and
// then rejected by ValidateConfig, which names the feature, instead of failing
// as an unknown flag: these are features the engine is expected to grow, not
// mistakes. Both are spelled the same whatever the engine, which is why this
// engine can name them already.
func (Engine) BindFlags(f *pflag.FlagSet, cfg *common.Configuration, rawYAML map[string]any) error {
	// lora-paths and lora-target-modules take multiple space-separated values,
	// which pflag cannot bind directly, so their values are pre-scanned from
	// os.Args.
	loraPaths := common.GetParamValueFromArgs("lora-paths")
	targetModules := common.GetParamValueFromArgs("lora-target-modules")

	if err := claimYAMLKeys(cfg, rawYAML); err != nil {
		return err
	}
	if err := claimLoraYAML(cfg, rawYAML); err != nil {
		return err
	}

	if err := declareFields(f, cfg, rawYAML); err != nil {
		return err
	}

	registerLoraFlags(f, cfg)
	if loraPaths != nil {
		if err := parseLoraPaths(cfg, loraPaths); err != nil {
			return err
		}
	}
	if targetModules != nil {
		cfg.Lora.TargetModules = targetModules
	}

	// Registered after the config file has been read, so the flag's default is
	// the file's value and an explicit flag still overrides it.
	common.AddToggle(f, &cfg.KVCache.EnableKVCache,
		"enable-kvcache", "Enable KV cache simulation"+notYet, "Disable KV cache simulation")

	// fake-metrics takes multiple space-separated JSON strings, which pflag
	// cannot bind directly, so its presence is read from os.Args instead. The
	// value is deliberately left unparsed: ValidateConfig rejects the feature
	// either way, and parsing first would report a malformed value rather than
	// the reason the setting cannot be honored.
	if common.GetParamValueFromArgs("fake-metrics") != nil {
		cfg.FakeMetrics = fakeMetrics{}
	}
	var fakeMetricsValues []string
	f.StringArrayVar(&fakeMetricsValues, "fake-metrics", nil,
		"A set of metrics to report to Prometheus instead of the real metrics"+notYet)
	f.Lookup("fake-metrics").NoOptDefVal = " "
	f.Lookup("fake-metrics").DefValue = ""

	return nil
}

// declareFields names the three core fields the engines disagree on the name
// of, under sglang's names: the field declarations in sglang's arg_groups/fields
// are context_length in model.py, and max_running_requests and
// max_queued_requests in schedule.py.
func declareFields(f *pflag.FlagSet, cfg *common.Configuration, rawYAML map[string]any) error {
	if err := common.DeclareConfigIntField(f, rawYAML, cfg, common.FieldContextWindow,
		&cfg.MaxModelLen, "context-length",
		"Model's context window, maximum number of tokens in a single request including input and output"); err != nil {
		return err
	}
	if err := common.DeclareConfigIntField(f, rawYAML, cfg, common.FieldConcurrency,
		&cfg.MaxNumSeqs, "max-running-requests",
		"Maximum number of inference requests that could be processed at the same time"); err != nil {
		return err
	}
	return common.DeclareConfigIntField(f, rawYAML, cfg, common.FieldQueueLength,
		&cfg.MaxWaitingQueueLength, "max-queued-requests",
		"Maximum length of inference requests waiting queue")
}

// claimYAMLKeys reads the config-file keys of the two unimplemented features out
// of a config file's raw tree, so that a file asking for one is refused by
// ValidateConfig naming the feature rather than reported as an unrecognized key.
// rawYAML is nil when no --config file was given; indexing and deleting a nil
// map are both no-ops.
//
// Only the flat spelling of enable-kvcache is claimed. A nested kvcache block
// carries the rest of the group, whose names this engine cannot commit to yet,
// so leaving the block unclaimed reports it, which is the honest outcome.
func claimYAMLKeys(cfg *common.Configuration, rawYAML map[string]any) error {
	if v, ok := rawYAML["enable-kvcache"]; ok {
		// A valueless "enable-kvcache:" leaves the default, the way unmarshaling
		// a null into a bool does for the engines that read the key that way.
		if v != nil {
			enabled, isBool := v.(bool)
			if !isBool {
				return fmt.Errorf("enable-kvcache must be a boolean, got '%v'", v)
			}
			cfg.KVCache.EnableKVCache = enabled
		}
		delete(rawYAML, "enable-kvcache")
	}

	if v, ok := rawYAML["fake-metrics"]; ok {
		// A present but empty block leaves fake metrics unset, as it does for
		// every engine: reporting fake metrics suppresses every real metric, so
		// a block with everything commented out must not turn them on.
		if v != nil {
			cfg.FakeMetrics = fakeMetrics{}
		}
		delete(rawYAML, "fake-metrics")
	}

	return nil
}

// registerLoraFlags declares this engine's LoRA flags on f, defaulting each to
// the value cfg already holds so a config file's setting survives an unset flag.
func registerLoraFlags(f *pflag.FlagSet, cfg *common.Configuration) {
	addOptionalToggle(f, &cfg.Lora.EnableLora, "enable-lora",
		"Enable LoRA support, implied by --lora-paths", "Disable LoRA support, ignoring --lora-paths")
	f.IntVar(&cfg.Lora.MaxLoras, "max-loras-per-batch", cfg.Lora.MaxLoras,
		"Maximum number of adapters for a running batch")
	f.IntVar(&cfg.Lora.MaxCPULoras, "max-loaded-loras", cfg.Lora.MaxCPULoras,
		"Maximum number of LoRA adapters loaded at a time, 0 for no limit, must be >= max-loras-per-batch")
	f.IntVar(&cfg.Lora.MaxLoraRank, "max-lora-rank", cfg.Lora.MaxLoraRank, "Maximum rank of LoRA adapters")

	// Declared only so they appear in --help; BindFlags pre-scans their values.
	var multiValues []string
	f.StringArrayVar(&multiValues, "lora-paths", nil,
		"List of LoRA adapters (space-separated), each <PATH>, <NAME>=<PATH>, or JSON {\"lora_name\":str,\"lora_path\":str,\"pinned\":bool}")
	f.Lookup("lora-paths").NoOptDefVal = " "
	f.Lookup("lora-paths").DefValue = ""
	f.StringArrayVar(&multiValues, "lora-target-modules", nil,
		"Modules LoRA is applied to (space-separated), or 'all'")
	f.Lookup("lora-target-modules").NoOptDefVal = " "
	f.Lookup("lora-target-modules").DefValue = ""

	// Accepted for sglang command-line compatibility and ignored: they tune how
	// adapter weights are computed and moved, which a simulated adapter has none
	// of. A value sglang would refuse is refused here too.
	var dummyBool bool
	var dummyFloat float64
	f.Var(newChoiceValue("csgmv", "triton", "csgmv", "ascend", "torch_native"), "lora-backend",
		"Kernel backend for multi-LoRA serving, ignored")
	f.Var(newChoiceValue("16", "16", "32", "64", "128"), "max-lora-chunk-size",
		"Maximum chunk size for the ChunkedSGMV LoRA backend, ignored")
	f.Var(newChoiceValue("lru", "lru", "fifo"), "lora-eviction-policy",
		"LoRA adapter eviction policy when memory pool is full, ignored")
	f.Float64Var(&dummyFloat, "lora-drain-wait-threshold", 0,
		"Wait time after which a running adapter is drained to make room, ignored")
	common.AddToggle(f, &dummyBool, "enable-lora-overlap-loading",
		"Overlap LoRA weight loading with compute, ignored", "Don't overlap LoRA weight loading with compute, ignored")
	common.AddToggle(f, &dummyBool, "lora-strict-loading",
		"Fail on mismatched LoRA adapter weight keys, ignored", "Don't fail on mismatched LoRA adapter weight keys, ignored")
	common.AddToggle(f, &dummyBool, "experts-shared-outer-loras",
		"Share outer LoRA weights across MoE experts, ignored", "Don't share outer LoRA weights across MoE experts, ignored")
	f.BoolVar(&dummyBool, "lora-use-virtual-experts", false, "Use virtual expert computation for MoE models, ignored")
}

// choiceValue is a flag value restricted to a fixed set of choices, the way
// argparse's choices restrict one.
type choiceValue struct {
	value   string
	choices []string
}

func newChoiceValue(value string, choices ...string) *choiceValue {
	return &choiceValue{value: value, choices: choices}
}

func (c *choiceValue) Set(s string) error {
	if !slices.Contains(c.choices, s) {
		return fmt.Errorf("invalid choice: '%s' (choose from %s)", s, strings.Join(c.choices, ", "))
	}
	c.value = s
	return nil
}
func (c *choiceValue) Type() string   { return "string" }
func (c *choiceValue) String() string { return c.value }

// optionalToggle is common.AddToggle's toggle over an optional boolean, which
// stays nil until one of its two spellings is given.
type optionalToggle struct {
	ptr **bool
	val bool
}

func (t optionalToggle) Set(s string) error {
	v, err := strconv.ParseBool(s)
	if err != nil {
		return fmt.Errorf("invalid boolean value %q", s)
	}
	enabled := t.val == v
	*t.ptr = &enabled
	return nil
}
func (optionalToggle) Type() string   { return "bool" }
func (optionalToggle) String() string { return "" }

// addOptionalToggle registers name and its "no-" negation the way
// common.AddToggle does, over an optional boolean.
func addOptionalToggle(f *pflag.FlagSet, ptr **bool, name, nameUsage, noNameUsage string) {
	for _, spelling := range []struct {
		name, usage string
		val         bool
	}{{name, nameUsage, true}, {"no-" + name, noNameUsage, false}} {
		f.Var(optionalToggle{ptr, spelling.val}, spelling.name, spelling.usage)
		f.Lookup(spelling.name).NoOptDefVal = "true"
		f.Lookup(spelling.name).DefValue = ""
	}
}

// claimLoraYAML builds cfg.Lora from a config file's "lora" block and claims
// it. The block is decoded strictly, so that a key of another engine's LoRA
// block, e.g. vLLM's max-loras, is reported rather than silently dropped.
// rawYAML is nil when no --config file was given.
func claimLoraYAML(cfg *common.Configuration, rawYAML map[string]any) error {
	v, ok := rawYAML["lora"]
	if !ok {
		return nil
	}
	delete(rawYAML, "lora")

	data, err := yaml.Marshal(v)
	if err != nil {
		return fmt.Errorf("failed to marshal \"lora\": %w", err)
	}
	// Seeded from the current values so keys the block omits keep their
	// defaults, while an explicit zero still comes through to be rejected by
	// ValidateConfig rather than silently ignored.
	ly := loraYAML{MaxLorasPerBatch: cfg.Lora.MaxLoras, MaxLoadedLoras: cfg.Lora.MaxCPULoras,
		MaxLoraRank: cfg.Lora.MaxLoraRank}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&ly); err != nil && !errors.Is(err, io.EOF) {
		return fmt.Errorf("failed to unmarshal \"lora\": %w", err)
	}

	cfg.Lora.EnableLora = ly.EnableLora
	cfg.Lora.MaxLoras = ly.MaxLorasPerBatch
	cfg.Lora.MaxCPULoras = ly.MaxLoadedLoras
	cfg.Lora.MaxLoraRank = ly.MaxLoraRank
	cfg.Lora.TargetModules = ly.TargetModules
	return parseLoraPaths(cfg, ly.LoraPaths)
}

// parseLoraPaths reconciles items (from a YAML config file's "lora" block or
// the --lora-paths flag) into cfg.Lora.LoraModules, the way sglang's
// LoRAPathAction and check_lora_server_args read them: an item wrapped in
// braces is JSON, one containing "=" is split on the first "=" into name and
// path, and anything else is a path that also serves as the name. Only a JSON
// item can pin its adapter.
func parseLoraPaths(cfg *common.Configuration, items []string) error {
	cfg.Lora.LoraModules = make([]common.LoraModule, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		var lora common.LoraModule
		switch {
		case strings.HasPrefix(item, "{") && strings.HasSuffix(item, "}"):
			var ref struct {
				Name   *string `json:"lora_name"`
				Path   *string `json:"lora_path"`
				Pinned bool    `json:"pinned"`
			}
			if err := json.Unmarshal([]byte(item), &ref); err != nil {
				return fmt.Errorf("invalid LoRA path %s: %w", item, err)
			}
			if ref.Name == nil || ref.Path == nil {
				return fmt.Errorf("LoRA path %s looks like a JSON str, but it does not contain 'lora_name' and 'lora_path' keys", item)
			}
			lora = common.LoraModule{Name: *ref.Name, Path: *ref.Path, Pinned: ref.Pinned}
		case strings.Contains(item, "="):
			name, path, _ := strings.Cut(item, "=")
			lora = common.LoraModule{Name: name, Path: path}
		default:
			lora = common.LoraModule{Name: item, Path: item}
		}
		cfg.Lora.LoraModules = append(cfg.Lora.LoraModules, lora)
	}
	return nil
}

// fakeMetrics stands in for this engine's fake-metrics configuration. It holds
// nothing: it exists so that a --fake-metrics request is visible to
// ValidateConfig, which rejects it.
type fakeMetrics struct{}

func (fakeMetrics) Validate() error         { return nil }
func (fakeMetrics) New() common.FakeMetrics { return fakeMetrics{} }
