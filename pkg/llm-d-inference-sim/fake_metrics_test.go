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

package llmdinferencesim

import (
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus"
)

var _ = Describe("total tokens", func() {
	It("should correctly calculate total tokens from bucket counts and boundaries", func() {
		tests := []struct {
			name        string
			counts      []int
			buckets     []float64
			expected    int64
			shouldBeNil bool
		}{
			{
				name:        "empty counts",
				counts:      []int{},
				buckets:     []float64{1, 2, 5},
				shouldBeNil: true,
				expected:    0,
			},
			{
				name:        "empty buckets",
				counts:      []int{10, 20},
				buckets:     []float64{},
				shouldBeNil: true,
				expected:    0,
			},
			{
				name:     "only first bucket has requests: [0,10]",
				counts:   []int{1},
				buckets:  []float64{10},
				expected: 10,
				// bucket0: [0,10] → 1*10 = 10
				// total = 10
			},
			{
				name:     "first two buckets: [0,10], (10,20]",
				counts:   []int{2, 3},
				buckets:  []float64{10, 20},
				expected: 80,
				// bucket0: [0,10] →  2*10 = 20
				// bucket1: (10,20] → 3*20 = 60
				// total = 80
			},
			{
				name:     "three finite buckets + last (+Inf) bucket",
				counts:   []int{1, 1, 1, 1},
				buckets:  []float64{10, 20, 50},
				expected: 131,
				// bucket0: [0,10] → 1*10 = 10
				// bucket1: (10,20] → 1*20 = 20
				// bucket2: (20,50] → 1*50 = 50
				// bucket3: (50,+Inf) → 1*(50+1)=51
				// total = 131
			},
			{
				name:     "zero counts in some buckets",
				counts:   []int{0, 5, 0, 2},
				buckets:  []float64{1, 10, 100},
				expected: 252,
				// bucket1: (1,10] →  5*10 = 50
				// bucket3: (100,+Inf) → 2*(100+1) = 202
				// total = 252
			},
			{
				name:     "only last bucket has requests",
				counts:   []int{0, 0, 0, 4},
				buckets:  []float64{10, 100, 1000},
				expected: 4004,
				// bucket3: (1000,+Inf) → 4*(1000+1) = 4004
			},
			{
				name:     "collaborator example: [10,20,30] with long buckets",
				counts:   []int{10, 20, 30},
				buckets:  []float64{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000},
				expected: 200,
				// bucket0: [0,1] → 10*1 = 10
				// bucket1: (1,2] → 20*2 = 40
				// bucket2: (2,5] → 30*5 = 150
				// total = 200
			},
			{
				name:     "counts shorter than buckets (trailing zeros omitted)",
				counts:   []int{1, 1},
				buckets:  []float64{10, 100, 1000, 10000},
				expected: 110,
				// bucket0: [0,10] → 1*10 = 10
				// bucket1: (10,100] → 1*100 = 100
				// total = 110
			},
			{
				name:     "all zero counts",
				counts:   []int{0, 0, 0},
				buckets:  []float64{1, 10, 100},
				expected: 0,
				// all buckets have zero requests
			},
		}

		s := SimContext{Config: &common.Configuration{Model: "test", ServedModelNames: []string{"test"}}}

		for _, test := range tests {
			hist := prometheus.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "dummy",
					Help:    "Test histogram",
					Buckets: test.buckets,
				}, []string{vllmapi.PromLabelModelName},
			)
			result := s.initFakeHistogram(hist, test.buckets, test.counts)
			if test.shouldBeNil {
				Expect(result).To(BeNil())
			} else {
				Expect(*result).To(Equal(test.expected), "test case: %s", test.name)
			}
		}
	})
})
