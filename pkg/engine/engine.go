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

// Package engine defines the seam between the simulator's generic core and
// its concrete backends (currently only vLLM, see pkg/engine/vllm).
package engine

import (
	"context"
	"fmt"
	"sort"

	"github.com/buaazp/fasthttprouter"
	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/pflag"
	"google.golang.org/grpc"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	"github.com/llm-d/llm-d-inference-sim/pkg/engine/vllm"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
)

// Engine supplies one backend's own defaults, CLI flags, configuration
// validation, and HTTP/gRPC transport surface.
type Engine interface {
	// Name identifies the engine backend, e.g. "vllm".
	Name() string
	// NewRequestValidator compiles the backend's strict request validator.
	NewRequestValidator() (communication.RequestValidator, error)
	// ApplyDefaults fills in the default values of the configuration groups
	// the engine owns. Called on a freshly constructed Configuration, before
	// a config file is loaded and before BindFlags, so that a YAML value
	// overrides a default and a flag overrides both.
	ApplyDefaults(cfg *common.Configuration)
	// BindFlags registers the engine's own CLI flags on f and reconciles any
	// values that need parsing beyond what pflag can bind directly, including
	// its own engine-specific groups (e.g. lora) from rawYAML, the raw YAML
	// tree returned by Configuration.load (nil if no --config file was
	// given). Must be called before f.Parse.
	BindFlags(f *pflag.FlagSet, cfg *common.Configuration, rawYAML map[string]any) error
	// ApplyEnv applies the engine's own environment-variable settings to cfg.
	// Called after the flags have been parsed and before validation; changed
	// reports whether a given flag was set on the command line, so that an
	// env var can act as a fallback rather than an override.
	ApplyEnv(cfg *common.Configuration, changed func(flag string) bool)
	// ValidateConfig checks the engine's own fields of cfg. Called after cfg's
	// common fields have already been validated.
	ValidateConfig(cfg *common.Configuration) error
	// BindHTTP registers the engine's own HTTP routes on r, on top of the
	// common routes comm's own HTTP server already registers.
	BindHTTP(r *fasthttprouter.Router, comm *communication.Communication)
	// BindGRPC registers the engine's own gRPC service on server and reports
	// whether the engine has a gRPC surface at all. Communication does not
	// open a gRPC listener when it returns false.
	BindGRPC(server *grpc.Server, comm *communication.Communication) bool
	// NewMetricsAdapter builds the engine's own metrics adapter, registering
	// its collectors on registry. ctx must match the one passed to
	// metrics.NewMetricsBus.
	NewMetricsAdapter(ctx context.Context, registry *prometheus.Registry,
		logger logr.Logger, config common.Configuration) (metrics.MetricsAdapter, error)
	// NewKVEventEncoder builds the encoder that turns the block cache's
	// engine-independent events into this engine's KV-event wire format.
	NewKVEventEncoder(config common.Configuration) (kvcache.EventEncoder, error)
}

// registry maps each engine backend's name to its constructor. Adding a
// backend means adding one entry here.
var registry = map[string]func() Engine{
	"vllm": func() Engine { return vllm.New() },
}

// Select returns the Engine implementation for the named engine backend.
func Select(name string) (Engine, error) {
	newEngine, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown engine '%s'", name)
	}
	return newEngine(), nil
}

// Names returns the registered engine backend names, sorted.
func Names() []string {
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
