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
	"context"

	"github.com/go-logr/logr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

var _ = Describe("Metrics adapter", func() {
	It("should expose no collectors", func() {
		registry := prometheus.NewPedanticRegistry()
		adapter, err := New().NewMetricsAdapter(context.Background(), registry, logr.Discard(), *common.NewConfig())
		Expect(err).NotTo(HaveOccurred())
		Expect(adapter.Start(context.Background())).To(Succeed())
		DeferCleanup(func() {
			Expect(adapter.Close()).To(Succeed())
		})

		Expect(registry.Gather()).To(BeEmpty())
	})
})
