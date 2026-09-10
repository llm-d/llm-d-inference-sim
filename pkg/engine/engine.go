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
	"github.com/buaazp/fasthttprouter"
	"github.com/spf13/pflag"
	"google.golang.org/grpc"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
)

// Engine supplies one backend's own CLI flags, configuration validation, and
// HTTP/gRPC transport surface.
type Engine interface {
	// Name identifies the engine backend, e.g. "vllm".
	Name() string
	// BindFlags registers the engine's own CLI flags on f and reconciles any
	// values that need parsing beyond what pflag can bind directly. Must be
	// called before f.Parse.
	BindFlags(f *pflag.FlagSet, cfg *common.Configuration) error
	// ValidateConfig checks the engine's own fields of cfg. Called after cfg's
	// common fields have already been validated.
	ValidateConfig(cfg *common.Configuration) error
	// BindHTTP registers the engine's own HTTP routes on r, on top of the
	// common routes comm's own HTTP server already registers.
	BindHTTP(r *fasthttprouter.Router, comm *communication.Communication)
	// BindGRPC registers the engine's own gRPC service on server.
	BindGRPC(server *grpc.Server, comm *communication.Communication) bool
}
