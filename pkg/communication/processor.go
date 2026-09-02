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

package communication

import (
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/endpoint"
)

// Processor is the seam through which the transport layer submits requests
// to the simulator's queue, controls its lifecycle, and reaches its
// admin/metrics surface. It is satisfied implicitly by *simulator.Simulator,
// which keeps this package free of any dependency on the simulator package.
type Processor interface {
	// HandleRequest submits req to the simulator's worker queue.
	HandleRequest(req endpoint.Request) (numChoices int, isStream bool,
		channel *common.Channel[*endpoint.ResponseInfo], err *api.Error, errInjected bool)
	// OpenRequests reports the number of requests currently in flight.
	OpenRequests() int64
	// Stop cancels the simulator's processing loop.
	Stop()
	// MetricsRegistry returns the simulator's Prometheus registry.
	MetricsRegistry() *prometheus.Registry
	// LoadLoraAdaptor loads a LoRA adapter described by body.
	LoadLoraAdaptor(body []byte) error
	// UnloadLoraAdaptor unloads a LoRA adapter described by body.
	UnloadLoraAdaptor(body []byte) error
}
