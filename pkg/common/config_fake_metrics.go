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

// Contains the engine-neutral value primitives a FakeMetrics implementation is
// built from. Which metrics an engine can fake is engine-specific and lives
// with that engine (see pkg/engine/vllm/fakemetrics); how a single faked value
// is expressed -- a fixed number or a generator function -- is not.

package common

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	OscillateFuncName     = "oscillate"
	RampFuncName          = "ramp"
	RampWithResetFuncName = "rampreset"
	SquarewaveFuncName    = "squarewave"
)

// FakeMetricWithFunction is a single faked metric value: either a fixed number
// or a generator function producing values over time from the parameters
// start, end, and period. Supported functions are:
//   - oscillate: Generates a smooth sine-wave between start and end over each period.
//   - ramp: Interpolates linearly from start to end over one period and then stays at end.
//   - rampreset: Interpolates linearly from start to end over each period, then jumps back to start and repeats.
//   - squarewave: Alternates between start and end, staying at each level for half of the period.
//
// The configuration format is: fun:start:end:period, for example: ramp:10:0:5s or oscillate:0:10:5s.
type FakeMetricWithFunction struct {
	FixedValue float64
	Function   *FunctionInfo
	IsFunction bool
}

type FunctionInfo struct {
	Name   string
	Start  float64
	End    float64
	Period time.Duration
}

func parseFunc(parts []string) (*FunctionInfo, error) {
	if len(parts) != 4 {
		return nil, errors.New("need func:start:end:period in fake metric generation function")
	}
	start, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		return nil, err
	}
	end, err := strconv.ParseFloat(parts[2], 64)
	if err != nil {
		return nil, err
	}
	period, err := time.ParseDuration(parts[3])
	if err != nil {
		return nil, err
	}
	return &FunctionInfo{Name: parts[0], Start: start, End: end, Period: period}, nil
}

func (f *FakeMetricWithFunction) parseFunction(s string) error {
	parts := strings.Split(s, ":")
	if config, err := parseFunc(parts); err != nil {
		return fmt.Errorf("unknown format in fake metric generation function: %s", err.Error())
	} else {
		f.Function = config
		f.IsFunction = true
		return nil
	}
}

func (f *FakeMetricWithFunction) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind == yaml.ScalarNode {
		// Try number first
		if n, err := strconv.ParseFloat(value.Value, 64); err == nil {
			f.FixedValue = n
			return nil
		}
	}

	return f.parseFunction(value.Value)
}

// MarshalJSON emits a JSON value that round-trips through UnmarshalJSON: a
// number for fixed values, or the canonical "func:start:end:period" string
// for generator functions. This keeps Configuration.Copy()'s marshal+unmarshal
// symmetric and gives /admin/config a clean wire form.
func (f *FakeMetricWithFunction) MarshalJSON() ([]byte, error) {
	if f.IsFunction && f.Function != nil {
		return json.Marshal(fmt.Sprintf("%s:%g:%g:%s",
			f.Function.Name, f.Function.Start, f.Function.End, f.Function.Period))
	}
	return json.Marshal(f.FixedValue)
}

func (f *FakeMetricWithFunction) UnmarshalJSON(data []byte) error {
	// Try number first
	var n float64
	if err := json.Unmarshal(data, &n); err == nil {
		f.FixedValue = n
		return nil
	}

	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}

	return f.parseFunction(s)
}

type LorasMetrics struct {
	// RunningLoras is a comma separated list of running LoRAs
	RunningLoras string `json:"running"`
	// WaitingLoras is a comma separated list of waiting LoRAs
	WaitingLoras string `json:"waiting"`
	// Timestamp is the timestamp of the metric
	Timestamp float64 `json:"timestamp"`
}

// Validate checks the generator function's name and parameters. Exported for
// use by an engine's own FakeMetrics.Validate, which owns the metrics these
// functions are attached to.
func (g *FunctionInfo) Validate() error {
	if g == nil {
		return nil
	}
	if g.Name != OscillateFuncName && g.Name != RampFuncName && g.Name != RampWithResetFuncName && g.Name != SquarewaveFuncName {
		return fmt.Errorf("invalid fake metrics generation function %s, must be one of the following: %s, %s, %s, %s",
			g.Name, OscillateFuncName, RampFuncName, RampWithResetFuncName, SquarewaveFuncName)
	}
	if g.End < 0 || g.Start < 0 || g.Period < 0 {
		return errors.New("invalid fake metrics generation parameter: start and end must not be negative")
	}
	if g.Period <= 0 {
		return errors.New("invalid fake metrics generation parameter: period must be positive")
	}
	return nil
}
