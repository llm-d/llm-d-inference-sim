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
	"fmt"

	"github.com/spf13/pflag"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// notYet marks the usage string of a flag this engine accepts but does not act
// on yet.
const notYet = " (not simulated by the sglang engine yet)"

// BindFlags registers this engine's own CLI flags and builds the config groups
// whose wire format it owns. Must be called before f.Parse.
//
// What it registers is this engine's name for each of the three core fields the
// engines disagree on the name of, plus the toggles of two features this engine
// does not simulate yet, bound to the configuration fields they will eventually
// fill. Asking for one of those two is therefore parsed and then rejected by
// ValidateConfig, which names the feature, instead of failing as an unknown
// flag: these are features the engine is expected to grow, not mistakes. Both
// are spelled the same whatever the engine, which is why this engine can name
// them already. LoRA is not among them: its flags are spelled per engine, so
// this engine registers none until it has its own.
func (Engine) BindFlags(f *pflag.FlagSet, cfg *common.Configuration, rawYAML map[string]any) error {
	if err := claimYAMLKeys(cfg, rawYAML); err != nil {
		return err
	}

	if err := declareFields(f, cfg, rawYAML); err != nil {
		return err
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

// fakeMetrics stands in for this engine's fake-metrics configuration. It holds
// nothing: it exists so that a --fake-metrics request is visible to
// ValidateConfig, which rejects it.
type fakeMetrics struct{}

func (fakeMetrics) Validate() error         { return nil }
func (fakeMetrics) New() common.FakeMetrics { return fakeMetrics{} }
