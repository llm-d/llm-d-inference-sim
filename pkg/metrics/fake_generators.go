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
	"context"
	"math"
	"sync"
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

// FakeGenerators owns an adapter's registered generators together with the
// ticker that re-evaluates them on every refresh. One generator is registered
// per scalar metric a fake-metrics configuration gives a function rather than a
// fixed value.
type FakeGenerators struct {
	ctx      context.Context
	interval time.Duration

	mu            sync.Mutex
	set           GeneratorSet
	started       bool
	tickerRunning bool
	tickerCancel  context.CancelFunc
	tickerStart   time.Time
}

// GeneratorSet is the mutable view of the generator map, handed to the callback
// of Apply. Its methods hold no lock of their own and must not be called from
// anywhere else.
type GeneratorSet struct {
	generators map[string]activeGenerator
}

// activeGenerator is one registered generator bound to the gauge channel it
// feeds. The ticker evaluates it on every refresh and pushes the result.
type activeGenerator struct {
	fn         Generator
	params     *common.FunctionInfo
	roundToInt bool
	update     func(upd GaugeUpdate)
}

// NewFakeGenerators returns an empty set whose ticker fires every interval once
// Start has been called. ctx must match the one the adapter's channels were
// created with, so that cancelling it stops the ticker along with them.
func NewFakeGenerators(ctx context.Context, interval time.Duration) *FakeGenerators {
	return &FakeGenerators{
		ctx:      ctx,
		interval: interval,
		set:      GeneratorSet{generators: make(map[string]activeGenerator)},
	}
}

// Apply runs mutate against the generator set with the lock held, then brings
// the ticker up or down to match the set mutate leaves behind. The ticker stays
// down until Start has been called, whatever the set holds.
func (g *FakeGenerators) Apply(mutate func(set *GeneratorSet)) {
	g.mu.Lock()
	defer g.mu.Unlock()

	wereEmpty := len(g.set.generators) == 0
	mutate(&g.set)
	areEmpty := len(g.set.generators) == 0

	if !g.started {
		return
	}
	switch {
	case wereEmpty && !areEmpty:
		g.startTickerLocked()
	case !wereEmpty && areEmpty:
		g.stopTickerLocked()
	}
}

// Start marks the adapter started and brings the ticker up if Apply has already
// registered a generator.
func (g *FakeGenerators) Start() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.started = true
	if len(g.set.generators) > 0 && !g.tickerRunning {
		g.startTickerLocked()
	}
}

// Stop stops the ticker and clears the started flag, so a later Apply does not
// bring it back up.
func (g *FakeGenerators) Stop() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.stopTickerLocked()
	g.started = false
}

// Running reports whether the ticker goroutine is live.
func (g *FakeGenerators) Running() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.tickerRunning
}

// Set registers key's generator and pushes its value at t=0 when fm is
// function-valued; a fixed value clears any generator registered for key and
// pushes the value itself. roundToInt truncates the result, for a metric
// counting whole requests.
func (s *GeneratorSet) Set(key string, fm *common.FakeMetricWithFunction,
	update func(upd GaugeUpdate), roundToInt bool) {
	if fm.IsFunction {
		gen := activeGenerator{
			fn:         Dispatch(fm.Function.Name),
			params:     fm.Function,
			roundToInt: roundToInt,
			update:     update,
		}
		s.generators[key] = gen
		update(GaugeReset(gen.value(0)))
		return
	}
	delete(s.generators, key)
	update(GaugeReset(fm.FixedValue))
}

func (gen activeGenerator) value(t time.Duration) float64 {
	v := gen.fn(gen.params, t)
	if gen.roundToInt {
		v = float64(int64(v))
	}
	return v
}

func (g *FakeGenerators) startTickerLocked() {
	tickerCtx, cancel := context.WithCancel(g.ctx)
	g.tickerCancel = cancel
	g.tickerRunning = true
	g.tickerStart = time.Now()
	go g.runTicker(tickerCtx, g.tickerStart)
}

func (g *FakeGenerators) stopTickerLocked() {
	if !g.tickerRunning {
		return
	}
	g.tickerCancel()
	g.tickerCancel = nil
	g.tickerRunning = false
}

func (g *FakeGenerators) runTicker(ctx context.Context, start time.Time) {
	ticker := time.NewTicker(g.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			g.tick(time.Since(start))
		}
	}
}

// tick evaluates and pushes every generator with the lock held, so a concurrent
// Apply cannot be overwritten by a value from a generator it has just removed.
func (g *FakeGenerators) tick(t time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()

	for _, gen := range g.set.generators {
		gen.update(GaugeReset(gen.value(t)))
	}
}
