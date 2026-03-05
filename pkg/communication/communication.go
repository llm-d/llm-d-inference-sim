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
	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
)

type Communication struct {
	logger    logr.Logger
	simulator *vllmsim.VllmSimulator
}

func New(logger logr.Logger, simulator *vllmsim.VllmSimulator) *Communication {
	return &Communication{logger: logger, simulator: simulator}
}

func (c *Communication) Start() {
	c.logger.V(logging.INFO).Info("Starting communication layer")
}
