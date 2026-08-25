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

// Package engine identifies which inference engine the simulator is simulating.
package engine

import (
	"fmt"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const (
	// VLLM identifies the vLLM engine, the only engine currently supported.
	VLLM = "vllm"

	// DefaultEngine is used when neither --engine nor the config file's "engine" key is set.
	DefaultEngine = VLLM
)

// Engine identifies the inference engine being simulated. It determines how the
// simulator's configuration is constructed and is available throughout the simulator's
// runtime for engine-specific behavior.
type Engine interface {
	// Name returns the engine's identifier, the value accepted by --engine.
	Name() string
	// NewConfiguration returns a Configuration populated with this engine's defaults,
	// before any YAML or command-line overrides are applied.
	NewConfiguration() *common.Configuration
}

// New creates the Engine identified by name, or returns an error listing the valid
// values if name is not recognized.
func New(name string) (Engine, error) {
	switch name {
	case VLLM:
		return vllmEngine{}, nil
	default:
		return nil, fmt.Errorf("unknown engine %q, valid values are: %s", name, VLLM)
	}
}

// Resolve determines which Engine to use, following the same precedence as every other
// setting: a --engine flag wins; otherwise, if --config points at a YAML file, its
// top-level "engine" key is peeked (without loading the full Configuration schema, since
// the engine isn't known yet); otherwise DefaultEngine applies. Both --engine and --config
// are hand-read from the command line here, before the full flag set exists, mirroring how
// common.ParseCommandParamsAndLoadConfig hand-reads --config.
func Resolve() (Engine, error) {
	name := DefaultEngine
	if values := common.GetParamValueFromArgs("engine"); len(values) == 1 {
		name = values[0]
	} else if configFiles := common.GetParamValueFromArgs("config"); len(configFiles) == 1 {
		fileEngine, err := common.PeekEngineFromFile(configFiles[0])
		if err != nil {
			return nil, err
		}
		if fileEngine != "" {
			name = fileEngine
		}
	}
	return New(name)
}
