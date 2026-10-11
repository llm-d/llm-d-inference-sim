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

package engine

import (
	"fmt"
	"sort"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/engine/sglang"
	"github.com/llm-d/llm-d-inference-sim/pkg/engine/vllm"
)

// registry maps each engine's name to its constructor. Adding an engine means
// adding one entry here.
var registry = map[string]func() Engine{
	"vllm":   func() Engine { return vllm.New() },
	"sglang": func() Engine { return sglang.New() },
}

// Select returns the Engine implementation for the named engine.
func Select(name string) (Engine, error) {
	newEngine, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown engine '%s', supported engines are: %s",
			name, strings.Join(Names(), ", "))
	}
	return newEngine(), nil
}

// Names returns the registered engine names, sorted.
func Names() []string {
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
