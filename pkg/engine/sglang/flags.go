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
// engines disagree on the name of, the KV-cache toggle with sglang's
// --page-size, the metrics toggle, which sglang spells --enable-metrics and
// defaults to off, and the toggle of fake metrics, which this engine does not
// simulate yet. Asking for fake metrics is therefore parsed and then rejected
// by ValidateConfig, which names the feature, instead of failing as an unknown
// flag. LoRA is not among them: its flags are spelled per engine, so this
// engine registers none until it has its own.
func (Engine) BindFlags(f *pflag.FlagSet, cfg *common.Configuration, rawYAML map[string]any) error {
	if err := claimYAMLKeys(cfg, rawYAML); err != nil {
		return err
	}

	if err := declareFields(f, cfg, rawYAML); err != nil {
		return err
	}

	// Registered after the config file has been read, so each flag's default is
	// the file's value and an explicit flag still overrides it.
	common.AddToggle(f, &cfg.KVCache.EnableKVCache,
		"enable-kvcache", "Enable KV cache simulation", "Disable KV cache simulation")
	f.IntVar(&cfg.KVCache.TokenBlockSize, "page-size", cfg.KVCache.TokenBlockSize,
		"Number of tokens in a KV-cache page, the block size of the simulated KV cache")
	common.AddToggle(f, &cfg.EnableMetrics,
		"enable-metrics", "Enable Prometheus metrics", "Disable Prometheus metrics")

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

// claimBool reads key out of raw into out, leaving out at its current value for
// a valueless "key:" the way unmarshaling a null into a bool does, and claims
// the key either way.
func claimBool(raw map[string]any, key string, out *bool) error {
	v, ok := raw[key]
	if ok && v != nil {
		enabled, isBool := v.(bool)
		if !isBool {
			return fmt.Errorf("%s must be a boolean, got '%v'", key, v)
		}
		*out = enabled
	}
	delete(raw, key)
	return nil
}

// claimYAMLKeys reads this engine's own flat config-file keys out of a config
// file's raw tree: the two toggles whose flags BindFlags registers, page-size,
// and the fake-metrics block. A file asking for fake metrics is then refused by
// ValidateConfig naming the feature rather than reported as an unrecognized
// key. rawYAML is nil when no --config file was given; indexing and deleting a
// nil map are both no-ops.
//
// Only the flat enable-kvcache and page-size keys are claimed. A nested kvcache
// block carries the rest of the group, whose names this engine cannot commit to
// yet, so leaving the block unclaimed reports it, which is the honest outcome.
func claimYAMLKeys(cfg *common.Configuration, rawYAML map[string]any) error {
	if err := claimBool(rawYAML, "enable-kvcache", &cfg.KVCache.EnableKVCache); err != nil {
		return err
	}
	if err := claimBool(rawYAML, "enable-metrics", &cfg.EnableMetrics); err != nil {
		return err
	}

	if err := common.UnmarshalYAMLKey(rawYAML, "page-size", &cfg.KVCache.TokenBlockSize); err != nil {
		return err
	}
	delete(rawYAML, "page-size")

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
