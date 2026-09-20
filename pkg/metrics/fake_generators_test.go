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

package metrics

import (
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

//nolint:unparam // period is intentionally parameterized for readability at call sites even though every current caller passes time.Second
func params(start, end float64, period time.Duration) *common.FunctionInfo {
	return &common.FunctionInfo{Start: start, End: end, Period: period}
}

var _ = Describe("Fake metric generators", func() {
	const tolerance = 1e-9

	Describe("Oscillate", func() {
		p := params(0, 10, time.Second)

		It("returns mid at t=0", func() {
			Expect(Oscillate(p, 0)).To(BeNumerically("~", 5, tolerance))
		})
		It("returns peak at t=period/4", func() {
			Expect(Oscillate(p, 250*time.Millisecond)).To(BeNumerically("~", 10, tolerance))
		})
		It("returns mid at t=period/2", func() {
			Expect(Oscillate(p, 500*time.Millisecond)).To(BeNumerically("~", 5, tolerance))
		})
	})

	Describe("Ramp", func() {
		p := params(0, 10, time.Second)

		It("returns Start at t=0", func() {
			Expect(Ramp(p, 0)).To(BeNumerically("~", 0, tolerance))
		})
		It("returns mid at t=period/2", func() {
			Expect(Ramp(p, 500*time.Millisecond)).To(BeNumerically("~", 5, tolerance))
		})
		It("returns End at t=period", func() {
			Expect(Ramp(p, time.Second)).To(BeNumerically("~", 10, tolerance))
		})
		It("stays at End past t=period", func() {
			Expect(Ramp(p, 3*time.Second)).To(BeNumerically("~", 10, tolerance))
		})
	})

	Describe("RampWithReset", func() {
		p := params(0, 10, time.Second)

		It("returns Start at t=0", func() {
			Expect(RampWithReset(p, 0)).To(BeNumerically("~", 0, tolerance))
		})
		It("returns mid at t=period/2", func() {
			Expect(RampWithReset(p, 500*time.Millisecond)).To(BeNumerically("~", 5, tolerance))
		})
		It("wraps to Start at t=period", func() {
			Expect(RampWithReset(p, time.Second)).To(BeNumerically("~", 0, tolerance))
		})
		It("wraps mid-way through the second period", func() {
			Expect(RampWithReset(p, 1500*time.Millisecond)).To(BeNumerically("~", 5, tolerance))
		})
	})

	Describe("Squarewave", func() {
		p := params(2, 8, time.Second)

		It("returns Start at t=0", func() {
			Expect(Squarewave(p, 0)).To(BeNumerically("~", 2, tolerance))
		})
		It("returns Start at t=period/4", func() {
			Expect(Squarewave(p, 250*time.Millisecond)).To(BeNumerically("~", 2, tolerance))
		})
		It("returns End at t=period/2", func() {
			Expect(Squarewave(p, 500*time.Millisecond)).To(BeNumerically("~", 8, tolerance))
		})
		It("returns End at t=3period/4", func() {
			Expect(Squarewave(p, 750*time.Millisecond)).To(BeNumerically("~", 8, tolerance))
		})
		It("wraps to Start at t=period", func() {
			Expect(Squarewave(p, time.Second)).To(BeNumerically("~", 2, tolerance))
		})
	})

	Describe("Dispatch", func() {
		p := params(0, 10, time.Second)
		at := 250 * time.Millisecond

		DescribeTable("resolves the correct generator by name",
			func(name string, want Generator) {
				got := Dispatch(name)
				Expect(got).NotTo(BeNil())
				// Compare by evaluated output at a sampling point rather than function
				// pointer identity, Go does not guarantee identity for named functions.
				Expect(got(p, at)).To(Equal(want(p, at)))
			},
			Entry("oscillate", common.OscillateFuncName, Generator(Oscillate)),
			Entry("ramp", common.RampFuncName, Generator(Ramp)),
			Entry("ramp with reset", common.RampWithResetFuncName, Generator(RampWithReset)),
			Entry("squarewave", common.SquarewaveFuncName, Generator(Squarewave)),
		)

		It("returns nil for an unknown name", func() {
			Expect(Dispatch("nope")).To(BeNil())
		})
	})
})
