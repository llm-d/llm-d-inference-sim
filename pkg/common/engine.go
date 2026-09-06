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

package common

import "github.com/spf13/pflag"

// Engine supplies the active engine's own CLI flags and configuration
// validation, for use by ParseCommandParamsAndLoadConfig.
type Engine interface {
	// Name identifies the engine backend, e.g. "vllm".
	Name() string
	// BindFlags registers the engine's own CLI flags on f and reconciles any
	// values that need parsing beyond what pflag can bind directly. Must be
	// called before f.Parse.
	BindFlags(f *pflag.FlagSet, cfg *Configuration) error
	// ValidateConfig checks the engine's own fields of cfg. Called after cfg's
	// common fields have already been validated.
	ValidateConfig(cfg *Configuration) error
}
