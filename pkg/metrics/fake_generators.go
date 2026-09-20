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

// Generator functions for fake metrics. Each generator maps elapsed time to a
// value in [Start, End] according to a specific shape. Dispatch resolves the
// function name from common.FunctionInfo to its implementation.

package metrics

import (
	"math"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// Generator maps elapsed time to a fake-metric value using the parameters in
// FunctionInfo. Producer code evaluates one on every refresh tick.
type Generator func(params *common.FunctionInfo, t time.Duration) float64

// Dispatch returns the Generator registered for name, or nil for an unknown
// name. Names come from common.FunctionInfo.Name, populated by the parser from
// the "fun:start:end:period" wire form.
func Dispatch(name string) Generator {
	switch name {
	case common.OscillateFuncName:
		return Oscillate
	case common.RampFuncName:
		return Ramp
	case common.RampWithResetFuncName:
		return RampWithReset
	case common.SquarewaveFuncName:
		return Squarewave
	}
	return nil
}

// Oscillate generates a smooth sine wave between Start and End over each period.
func Oscillate(params *common.FunctionInfo, t time.Duration) float64 {
	phase := (2 * math.Pi * t.Seconds()) / params.Period.Seconds()
	amp := (params.End - params.Start) / 2
	mid := (params.Start + params.End) / 2
	return mid + amp*math.Sin(phase)
}

// Ramp interpolates linearly from Start to End over one period and then stays at End.
func Ramp(params *common.FunctionInfo, t time.Duration) float64 {
	frac := t.Seconds() / params.Period.Seconds()
	if frac >= 1 {
		return params.End
	}
	return params.Start + frac*(params.End-params.Start)
}

// RampWithReset interpolates linearly from Start to End over each period, then
// jumps back to Start and repeats.
func RampWithReset(params *common.FunctionInfo, t time.Duration) float64 {
	elapsedSec := (t % params.Period).Seconds()
	periodSec := params.Period.Seconds()
	frac := elapsedSec / periodSec
	if frac > 1 {
		frac = 1
	}
	return params.Start + frac*(params.End-params.Start)
}

// Squarewave alternates between Start and End, staying at each level for half of the period.
func Squarewave(params *common.FunctionInfo, t time.Duration) float64 {
	within := t % params.Period
	half := params.Period / 2
	if within < half {
		return params.Start
	}
	return params.End
}
