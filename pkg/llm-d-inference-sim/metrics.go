/*
Copyright 2025 The llm-d-inference-simference-sim Authors.

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

// Contains functions related to prometheus metrics

package llmdinferencesim

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

// createAndRegisterPrometheus creates and registers prometheus metrics used by vLLM simulator
// Metrics reported:
// - lora_requests_info
func (s *VllmSimulator) createAndRegisterPrometheus() error {
	s.registry = prometheus.NewRegistry()

	s.loraInfo = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      "vllm:lora_requests_info",
			Help:      "Running stats on lora requests.",
		},
		[]string{vllmapi.PromLabelMaxLora, vllmapi.PromLabelRunningLoraAdapters, vllmapi.PromLabelWaitingLoraAdapters},
	)

	if err := s.registry.Register(s.loraInfo); err != nil {
		s.logger.Error(err, "Prometheus lora info gauge register failed")
		return err
	}

	s.runningRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      "vllm:num_requests_running",
			Help:      "Number of requests currently running on GPU.",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.registry.Register(s.runningRequests); err != nil {
		s.logger.Error(err, "Prometheus number of running requests gauge register failed")
		return err
	}

	// not supported for now, reports constant value
	s.waitingRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      "vllm:num_requests_waiting",
			Help:      "Prometheus metric for the number of queued requests.",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.registry.Register(s.waitingRequests); err != nil {
		s.logger.Error(err, "Prometheus number of requests in queue gauge register failed")
		return err
	}

	// not supported for now, reports constant value
	s.kvCacheUsagePercentage = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: "",
			Name:      "vllm:gpu_cache_usage_perc",
			Help:      "Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).",
		},
		[]string{vllmapi.PromLabelModelName},
	)

	if err := s.registry.Register(s.kvCacheUsagePercentage); err != nil {
		s.logger.Error(err, "Prometheus kv cache usage percentage gauge register failed")
		return err
	}

	s.setInitialPrometheusMetrics()

	return nil
}

// setInitialPrometheusMetrics sends the default values to prometheus or
// the fake metrics if set
func (s *VllmSimulator) setInitialPrometheusMetrics() {
	var nRunningReqs, nWaitingReqs, kvCacheUsage float64
	if s.config.FakeMetrics != nil {
		nRunningReqs = float64(s.config.FakeMetrics.RunningRequests)
		nWaitingReqs = float64(s.config.FakeMetrics.WaitingRequests)
		kvCacheUsage = float64(s.config.FakeMetrics.KVCacheUsagePercentage)
	}
	modelName := s.getDisplayedModelName(s.config.Model)
	s.runningRequests.WithLabelValues(modelName).Set(nRunningReqs)
	s.waitingRequests.WithLabelValues(modelName).Set(nWaitingReqs)
	s.kvCacheUsagePercentage.WithLabelValues(modelName).Set(kvCacheUsage)

	if s.config.FakeMetrics != nil && len(s.config.FakeMetrics.LoraMetrics) != 0 {
		for _, metrics := range s.config.FakeMetrics.LoraMetrics {
			s.loraInfo.WithLabelValues(
				strconv.Itoa(s.config.MaxLoras),
				metrics.RunningLoras,
				metrics.WaitingLoras).Set(metrics.Timestamp)
		}
	} else {
		s.loraInfo.WithLabelValues(
			strconv.Itoa(s.config.MaxLoras),
			"",
			"").Set(float64(time.Now().Unix()))
	}
}

// reportLoras sets information about loaded LoRA adapters
func (s *VllmSimulator) reportLoras() {
	if s.config.FakeMetrics != nil {
		return
	}
	if s.loraInfo == nil {
		// Happens in the tests
		return
	}

	var runningLoras []string
	s.runningLoras.Range(func(key any, _ any) bool {
		if lora, ok := key.(string); ok {
			runningLoras = append(runningLoras, lora)
		}
		return true
	})
	var waitingLoras []string
	s.waitingLoras.Range(func(key any, _ any) bool {
		if lora, ok := key.(string); ok {
			waitingLoras = append(waitingLoras, lora)
		}
		return true
	})

	s.loraInfo.WithLabelValues(
		strconv.Itoa(s.config.MaxLoras),
		strings.Join(runningLoras, ","),
		strings.Join(waitingLoras, ",")).Set(float64(time.Now().Unix()))
}

// reportRunningRequests sets information about running completion requests
func (s *VllmSimulator) reportRunningRequests() {
	if s.config.FakeMetrics != nil {
		return
	}
	if s.runningRequests != nil {
		s.runningRequests.WithLabelValues(
			s.getDisplayedModelName(s.config.Model)).Set(float64(s.nRunningReqs))
	}
}

// reportWaitingRequests sets information about waiting completion requests
func (s *VllmSimulator) reportWaitingRequests() {
	if s.config.FakeMetrics != nil {
		return
	}
	if s.waitingRequests != nil {
		s.waitingRequests.WithLabelValues(
			s.getDisplayedModelName(s.config.Model)).Set(float64(s.nWaitingReqs))
	}
}

// reportKVCacheUsage sets information about kv cache usage
func (s *VllmSimulator) reportKVCacheUsage(value float64) {
	if s.config.FakeMetrics != nil {
		return
	}
	if s.kvCacheUsagePercentage != nil {
		s.kvCacheUsagePercentage.WithLabelValues(
			s.getDisplayedModelName(s.config.Model)).Set(value)
	}
}

// startMetricsUpdaters starts the various metrics updaters
func (s *VllmSimulator) startMetricsUpdaters(ctx context.Context) {
	go s.waitingRequestsUpdater(ctx)
	go s.runningRequestsUpdater(ctx)
	go s.lorasUpdater(ctx)
	go s.kvCacheUsageUpdater(ctx)
}

// waitingRequestsUpdater updates the waiting requests metric by listening on the relevant channel
func (s *VllmSimulator) waitingRequestsUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case inc := <-s.waitingReqChan:
			s.nWaitingReqs += inc
			s.reportWaitingRequests()
		}
	}
}

// runningRequestsUpdater updates the running requests metric by listening on the relevant channel
func (s *VllmSimulator) runningRequestsUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case inc := <-s.runReqChan:
			s.nRunningReqs += inc
			s.reportRunningRequests()
		}
	}
}

// kvCacheUsageUpdater updates the kv cache usage  metric by listening on the relevant channel
func (s *VllmSimulator) kvCacheUsageUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-s.kvCacheUsageChan:
			s.reportKVCacheUsage(value)
		}
	}
}

// lorasUpdater updates the running loras metric by listening on the relevant channel
// one function updates both waiting and running loras since they a part of the same prometheus gauge
func (s *VllmSimulator) lorasUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case loraUpdate := <-s.lorasChan:
			switch loraUpdate.state {
			case waitingUsageState:
				s.incrementLoraRefCount(loraUpdate.name, &s.waitingLoras)
			case runningUsageState:
				s.decrementLoraRefCount(loraUpdate.name, &s.waitingLoras)
				s.incrementLoraRefCount(loraUpdate.name, &s.runningLoras)
			case doneUsageState:
				s.decrementLoraRefCount(loraUpdate.name, &s.runningLoras)
			}
			s.reportLoras()
		}
	}
}

func (s *VllmSimulator) incrementLoraRefCount(lora string, theMap *sync.Map) {
	count := 0
	if value, ok := theMap.Load(lora); ok {
		// if lora is already in the map - increment its counter
		count = value.(int)
	}
	theMap.Store(lora, count+1)
}

func (s *VllmSimulator) decrementLoraRefCount(lora string, theMap *sync.Map) {
	if value, ok := theMap.Load(lora); ok {
		count := value.(int)
		if count > 1 {
			theMap.Store(lora, count-1)
		} else {
			// last lora instance stopped its execution - remove from the map
			theMap.Delete(lora)
		}
	} else {
		s.logger.Error(nil, "Zero model reference", "model", lora)
	}
}
