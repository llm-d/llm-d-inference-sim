/*
Copyright 2025 The llm-d-inference-sim Authors.

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
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

// LoRAs usage info for requests execution
type lorasUsageInfo struct {
	mux sync.RWMutex
	// lora adapter name -> reference count (number of currently running requests)
	loadedLoras map[string]int
	// loraIDs indices of loaded loras, element i holds the name of the lora at index i+1, empty means a free slot
	loraIDs []string
	// channel for "there is a LoRA that can be removed" event
	loraRemovable common.Channel[int]
	// maximum number of LoRAs that can be used simultaneously
	maxLoras int
}

type SimContext struct {
	// logger is used for information and errors logging
	logger logr.Logger
	// metrics contains all Prometheus metrics related data
	metrics metricsData
	// config holds the simulator's configuration as an atomic pointer so that
	// admin updates can swap it under concurrent readers. Access via Config()/SetConfig().
	config atomic.Pointer[common.Configuration]
	// adminMu serializes admin-config updates so two concurrent updates can't
	// each load-then-store with a stale value.
	adminMu sync.Mutex
	// loraAdaptors contains list of LoRA available adaptors
	loraAdaptors sync.Map
	// loras contains information about which LoRAs are in use
	loras *lorasUsageInfo
	// rand with a configurable seed to generate reproducible Random responses
	Random *common.Random
	// kv cache functionality
	kvcacheHelper *kvcache.KVCacheHelper
	// dataset is used for token generation in responses
	dataset dataset.Dataset
	// latencyCalculator calculates the delays in simulator's responses.
	// Held in an atomic.Pointer (via a holder struct) so admin-config updates
	// can swap in a fresh calculator without racing against the workers that
	// read it on every request. A holder is needed because the three
	// calculator types (default/constant/per-token) are different concrete
	// types implementing LatencyCalculator, which atomic.Value would reject.
	latencyCalculator atomic.Pointer[latencyCalcHolder]
	// Tokenizer used for request tokenization and in /tokenize
	Tokenizer tokenizer.Tokenizer
}

type latencyCalcHolder struct {
	calc latencyCalculator
}

// latencyCalc returns the current latency calculator. Safe for concurrent
// reads while admin updates rebuild it.
func (s *SimContext) latencyCalc() latencyCalculator {
	return s.latencyCalculator.Load().calc
}

// rebuildLatencyCalculator constructs a calculator from the current config
// and atomically replaces the existing one. Called both at init and after
// each successful admin-config update.
func (s *SimContext) rebuildLatencyCalculator() {
	var calc latencyCalculator
	switch s.Config().LatencyCalculator {
	case common.DefaultLatencyCalculator:
		calc = newDefaultCalculator(s.Config(), s.Random)
	case common.ConstantLatencyCalculator:
		calc = newConstantCalculator(s.Config(), s.Random)
	case common.PerPromptTokenLatencyCalculator:
		calc = newPerTokenCalculator(s.Config(), s.Random)
	}
	s.latencyCalculator.Store(&latencyCalcHolder{calc: calc})
}

// Config returns the current configuration. Safe for concurrent reads while
// admin updates swap the pointer via SetConfig.
func (s *SimContext) Config() *common.Configuration {
	return s.config.Load()
}

// SetConfig atomically replaces the configuration pointer.
func (s *SimContext) SetConfig(c *common.Configuration) {
	s.config.Store(c)
}

// ApplyConfigUpdate validates the partial JSON body against the current
// configuration and atomically swaps in the resulting configuration. Updates
// are serialized so concurrent callers cannot lose each other's changes.
//
// A "fake-metrics" field in the body is applied to Prometheus collectors via
// updateFakeMetrics; this runs after Configuration.Update has validated the
// merged result but before the config swap, so a Prometheus side-effect
// failure aborts the whole update.
func (s *SimContext) ApplyConfigUpdate(body []byte) error {
	s.adminMu.Lock()
	defer s.adminMu.Unlock()

	next, update, latencyChanged, err := s.Config().Update(body)
	if err != nil {
		return err
	}
	if update.FakeMetrics != nil {
		if s.Config().FakeMetrics == nil {
			return errors.New("the simulator is reporting real metrics; fake metrics cannot be updated")
		}
		if err := s.updateFakeMetrics(update.FakeMetrics, s.Config().FakeMetrics); err != nil {
			return fmt.Errorf("failed to update fake metrics: %w", err)
		}
	}
	s.SetConfig(next)
	// The calculator caches latency-related fields at construction time, so
	// rebuild it whenever any of those fields was updated.
	if latencyChanged {
		s.rebuildLatencyCalculator()
	}
	return nil
}

func (s *SimContext) initialize(ctx context.Context) error {
	s.Random = common.NewRandom(s.Config().Seed, s.Config().Port)

	s.rebuildLatencyCalculator()

	for _, lora := range s.Config().LoraModules {
		s.loraAdaptors.Store(lora.Name, lora.Path)
	}
	s.loras.maxLoras = s.Config().MaxLoras
	s.loras.loraIDs = make([]string, s.Config().MaxLoras)
	s.loras.loraRemovable = common.Channel[int]{
		Channel: make(chan int, s.Config().MaxNumSeqs),
		Name:    "loraRemovable",
		Done:    ctx.Done(),
	}

	// initialize prometheus metrics
	err := s.createAndRegisterPrometheus(ctx)
	if err != nil {
		return err
	}

	// KVCache doesn't support images at the moment, so in mm-encoder only mode
	// we don't start it.
	if s.Config().EnableKVCache && !s.Config().MMEncoderOnly {
		s.kvcacheHelper, err = kvcache.NewKVCacheHelper(ctx, s.Config(), s.logger,
			s.metrics.kvCacheUsageChan, s.metrics.prefixCacheStatsChan, s.Tokenizer)
		if err != nil {
			return err
		}

		go s.kvcacheHelper.Run(ctx)
	}

	err = s.initDataset(ctx)
	if err != nil {
		return fmt.Errorf("dataset initialization error: %w", err)
	}

	return nil
}

func (s *SimContext) initDataset(ctx context.Context) error {
	if s.Config().MMEncoderOnly {
		var err error
		s.dataset, err = dataset.NewMMEncoderOnlyDataset(s.logger, s.Tokenizer)
		if err != nil {
			return fmt.Errorf("failed to initialize dataset for mm-encoder-only mode: %w", err)
		}
		return nil
	}

	if s.Config().Mode == common.ModeEcho {
		s.dataset = &dataset.EchoDataset{}
		return nil
	}

	if s.Config().DatasetPath == "" && s.Config().DatasetURL == "" {
		// use predefined sentences as responses
		randDataset := &dataset.DefaultDataset{}
		err := randDataset.Init(ctx, s.logger, s.Random, s.Config().MaxModelLen, s.Tokenizer)
		if err != nil {
			return fmt.Errorf("failed to initialize random dataset: %w", err)
		}
		s.logger.V(logging.INFO).Info("No dataset path or URL provided, using random text for responses")
		s.dataset = randDataset
		return nil
	}

	// use dataset containing responses
	custDataset := &dataset.CustomDataset{}
	err := custDataset.Init(ctx, s.logger, s.Random, s.Config().DatasetPath, s.Config().DatasetTableName,
		s.Config().DatasetInMemory, s.Config().MaxModelLen, s.Tokenizer)

	if err == nil {
		s.dataset = custDataset
		return nil
	}

	return err
}

// isLora returns true if the given model name is one of loaded LoRAs
func (s *SimContext) isLora(model string) bool {
	for _, lora := range s.getLoras() {
		if model == lora {
			return true
		}
	}

	return false
}

// getDisplayedModelName returns the model name that must appear in API
// responses.  LoRA adapters keep their explicit name, while all base-model
// requests are surfaced as the first alias from --served-model-name.
func (s *SimContext) getDisplayedModelName(reqModel string) string {
	if s.isLora(reqModel) {
		return reqModel
	}
	return s.Config().ServedModelNames[0]
}

func (s *SimContext) simulateTTFT(respCtx ResponseContext) {
	startPrefill := time.Now()
	// time to first token delay
	params := TTFTParams{
		PromptTokens:       respCtx.UsageData().PromptTokens,
		CachedPromptTokens: respCtx.NumberCachedPromptTokens(),
		DoRemotePrefill:    respCtx.doRemotePrefill(),
		RunningReqs:        s.metrics.nRunningReqs,
	}
	ttft := s.latencyCalc().GetTimeToFirstToken(&params)
	time.Sleep(ttft)
	// report ttft in seconds
	common.WriteToChannel(s.metrics.ttftChan, ttft.Seconds(), s.logger)
	common.WriteToChannel(s.metrics.reqPrefillTimeChan, time.Since(startPrefill).Seconds(), s.logger)
}

func (s *SimContext) simulateInterTokenLatency() {
	perTokenLatency := s.latencyCalc().GetInterTokenLatency(&InterTokenParams{
		RunningReqs: s.metrics.nRunningReqs})
	time.Sleep(perTokenLatency)

	// report tpot in seconds
	common.WriteToChannel(s.metrics.tpotChan, perTokenLatency.Seconds(), s.logger)
}

// CreateModelsResponse creates and returns ModelResponse for the current state, returned array of models contains the base model + LoRA adapters if exist
func (s *SimContext) CreateModelsResponse() *api.ModelsResponse {
	modelsResp := api.ModelsResponse{Object: "list", Data: []api.ModelsResponseModelInfo{}}

	// Advertise every public model alias
	for _, alias := range s.Config().ServedModelNames {
		modelsResp.Data = append(modelsResp.Data, api.ModelsResponseModelInfo{
			ID:          alias,
			Object:      api.ObjectModel,
			Created:     time.Now().Unix(),
			OwnedBy:     "vllm",
			Root:        s.Config().Model,
			Parent:      nil,
			MaxModelLen: s.Config().MaxModelLen,
		})
	}

	// add LoRA adapter's info
	parent := s.Config().ServedModelNames[0]
	for _, lora := range s.getLoras() {
		modelsResp.Data = append(modelsResp.Data, api.ModelsResponseModelInfo{
			ID:          lora,
			Object:      api.ObjectModel,
			Created:     time.Now().Unix(),
			OwnedBy:     "vllm",
			Root:        s.getLoraPath(lora),
			Parent:      &parent,
			MaxModelLen: s.Config().MaxModelLen,
		})
	}

	return &modelsResp
}
