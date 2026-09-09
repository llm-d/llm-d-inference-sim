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

package common

import (
	"os"
	"path/filepath"
	"reflect"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func createConfigWithModel(model string, servedModelNames []string) *Configuration {
	c := NewConfig()

	c.Model = model
	if len(servedModelNames) > 0 {
		c.ServedModelNames = servedModelNames
	} else {
		c.ServedModelNames = []string{c.Model}
	}

	c.DisplayModelName = c.ServedModelNames[0]

	return c
}

func createDefaultConfig(model string, servedModelNames []string) *Configuration {
	c := createConfigWithModel(model, servedModelNames)

	c.MaxNumSeqs = 5
	c.MaxLoras = 2
	c.MaxCPULoras = 5
	c.TimeToFirstToken = 2000 * time.Millisecond
	c.InterTokenLatency = 1000 * time.Millisecond
	c.KVCacheTransferLatency = 100 * time.Millisecond
	c.Seed = 100100100
	c.LoraModules = []LoraModule{}
	return c
}

var _ = Describe("ApplyAdminUpdate", func() {
	var base *Configuration

	BeforeEach(func() {
		base = createDefaultConfig("model", nil)
		base.FailureInjectionRate = 10
		base.FailureTypes = []string{FailureTypeRateLimit}
	})

	It("updates failure-injection-rate and returns a new Configuration", func() {
		next, update, latencyChanged, err := base.Update([]byte(`{"failure-injection-rate": 42}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(latencyChanged).To(BeFalse())
		Expect(update.FakeMetrics).To(BeNil())
		Expect(next).ToNot(BeIdenticalTo(base))
		Expect(next.FailureInjectionRate).To(Equal(42))
		Expect(next.FailureTypes).To(Equal([]string{FailureTypeRateLimit}))
		// original is unchanged
		Expect(base.FailureInjectionRate).To(Equal(10))
	})

	It("updates failure-types", func() {
		next, _, _, err := base.Update([]byte(`{"failure-types": ["server_error", "model_not_found"]}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(next.FailureTypes).To(Equal([]string{FailureTypeServerError, FailureTypeModelNotFound}))
		Expect(next.FailureInjectionRate).To(Equal(10))
		Expect(base.FailureTypes).To(Equal([]string{FailureTypeRateLimit}))
	})

	It("updates both fields at once", func() {
		next, _, _, err := base.Update([]byte(`{"failure-injection-rate": 5, "failure-types": ["invalid_request"]}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(next.FailureInjectionRate).To(Equal(5))
		Expect(next.FailureTypes).To(Equal([]string{FailureTypeInvalidRequest}))
	})

	It("returns the parsed fake-metrics partial via update.FakeMetrics", func() {
		next, update, _, err := base.Update([]byte(
			`{"failure-injection-rate": 50, "fake-metrics": {"running-requests": 7}}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(next.FailureInjectionRate).To(Equal(50))
		Expect(update.FakeMetrics).ToNot(BeNil())
		Expect(update.FakeMetrics.RunningRequests).ToNot(BeNil())
		Expect(update.FakeMetrics.RunningRequests.FixedValue).To(Equal(float64(7)))
		// Fields not in the body are nil on the fake-metrics partial.
		Expect(update.FakeMetrics.WaitingRequests).To(BeNil())
	})

	DescribeTable("flags latencyChanged according to the body keys",
		func(body string, expected bool) {
			_, _, latencyChanged, err := base.Update([]byte(body))
			Expect(err).ToNot(HaveOccurred())
			Expect(latencyChanged).To(Equal(expected))
		},
		Entry("only failure-injection-rate -> false",
			`{"failure-injection-rate": 0}`, false),
		Entry("only failure-types -> false",
			`{"failure-types": ["rate_limit"]}`, false),
		Entry("time-to-first-token -> true",
			`{"time-to-first-token": "250ms"}`, true),
		Entry("inter-token-latency -> true",
			`{"inter-token-latency": "1ms"}`, true),
		Entry("time-factor-under-load -> true",
			`{"time-factor-under-load": 1.5}`, true),
		Entry("latency-calculator -> true",
			`{"latency-calculator": "constant"}`, true),
		Entry("std-dev field -> true",
			`{"time-to-first-token": "1s", "time-to-first-token-std-dev": "100ms"}`, true),
		Entry("mixed latency + non-latency -> true",
			`{"failure-injection-rate": 0, "prefill-overhead": "1ms"}`, true),
		Entry("time-to-generate-image -> false",
			`{"time-to-generate-image": "500ms"}`, false),
		Entry("time-to-generate-image-std-dev -> false",
			`{"time-to-generate-image": "500ms", "time-to-generate-image-std-dev": "50ms"}`, false),
	)

	It("returns latencyChanged=false when validation fails on a latency body", func() {
		// The 30% std-dev rule trips, but we still expect a clean error path
		// that does not claim latencyChanged.
		_, _, latencyChanged, err := base.Update([]byte(
			`{"time-to-first-token": "1ms", "time-to-first-token-std-dev": "0.5ms"}`))
		Expect(err).To(HaveOccurred())
		Expect(latencyChanged).To(BeFalse())
	})

	It("rejects an invalid duration string", func() {
		_, _, _, err := base.Update([]byte(`{"time-to-first-token": "notaduration"}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("time-to-first-token"))
	})

	It("rejects fields that are not admin-configurable", func() {
		_, _, _, err := base.Update([]byte(`{"port": 9000}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("not admin-configurable"))
	})

	It("rejects an out-of-range failure-injection-rate", func() {
		_, _, _, err := base.Update([]byte(`{"failure-injection-rate": 150}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failure injection rate"))
	})

	It("rejects an unknown failure type", func() {
		_, _, _, err := base.Update([]byte(`{"failure-types": ["bogus"]}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("invalid failure type"))
	})

	It("rejects malformed JSON", func() {
		_, _, _, err := base.Update([]byte(`not json`))
		Expect(err).To(HaveOccurred())
	})
})

var _ = Describe("Configuration.Copy", func() {
	It("should round-trip a non-nil FakeMetrics with a fixed-value metric", func() {
		c := &Configuration{
			FakeMetrics: &FakeMetrics{
				RunningRequests: &FakeMetricWithFunction{FixedValue: 5},
			},
		}

		got, err := c.Copy()
		Expect(err).NotTo(HaveOccurred())
		Expect(got.FakeMetrics).NotTo(BeNil())
		Expect(got.FakeMetrics.RunningRequests).NotTo(BeNil())
		Expect(got.FakeMetrics.RunningRequests.IsFunction).To(BeFalse())
		Expect(got.FakeMetrics.RunningRequests.FixedValue).To(Equal(5.0))
	})

	It("should round-trip a non-nil FakeMetrics with a function-valued metric", func() {
		c := &Configuration{
			FakeMetrics: &FakeMetrics{
				WaitingRequests: &FakeMetricWithFunction{
					IsFunction: true,
					Function: &FunctionInfo{
						Name:   OscillateFuncName,
						Start:  0,
						End:    10,
						Period: 5 * time.Second,
					},
				},
			},
		}

		got, err := c.Copy()
		Expect(err).NotTo(HaveOccurred())
		Expect(got.FakeMetrics).NotTo(BeNil())
		Expect(got.FakeMetrics.WaitingRequests).NotTo(BeNil())
		Expect(got.FakeMetrics.WaitingRequests.IsFunction).To(BeTrue())
		Expect(got.FakeMetrics.WaitingRequests.Function).NotTo(BeNil())
		Expect(got.FakeMetrics.WaitingRequests.Function.Name).To(Equal(OscillateFuncName))
		Expect(got.FakeMetrics.WaitingRequests.Function.Start).To(Equal(0.0))
		Expect(got.FakeMetrics.WaitingRequests.Function.End).To(Equal(10.0))
		Expect(got.FakeMetrics.WaitingRequests.Function.Period).To(Equal(5 * time.Second))
	})

	It("should round-trip an explicit-zero metric (non-nil pointer to zero-value struct)", func() {
		c := &Configuration{
			FakeMetrics: &FakeMetrics{
				RunningRequests: &FakeMetricWithFunction{},
			},
		}

		got, err := c.Copy()
		Expect(err).NotTo(HaveOccurred())
		Expect(got.FakeMetrics).NotTo(BeNil())
		Expect(got.FakeMetrics.RunningRequests).NotTo(BeNil())
		Expect(got.FakeMetrics.RunningRequests.IsFunction).To(BeFalse())
		Expect(got.FakeMetrics.RunningRequests.FixedValue).To(Equal(0.0))
	})
})

var _ = Describe("admin struct tags", func() {
	It("has no unrecognized tag values", func() {
		t := reflect.TypeOf(Configuration{})
		for i := range t.NumField() {
			f := t.Field(i)
			Expect(f.Tag.Get("admin")).To(BeElementOf("", "configurable"),
				"field %s has unexpected admin tag %q", f.Name, f.Tag.Get("admin"))
			Expect(f.Tag.Get("rebuild")).To(BeElementOf("", "latency"),
				"field %s has unexpected rebuild tag %q", f.Name, f.Tag.Get("rebuild"))
			if f.Tag.Get("rebuild") == "latency" {
				Expect(f.Tag.Get("admin")).To(Equal("configurable"),
					"field %s has rebuild:\"latency\" but missing admin:\"configurable\"", f.Name)
			}
		}
	})

	It("configurableFields contains exactly the expected entries with their rebuild tags", func() {
		Expect(configurableFields).To(Equal(map[string]string{
			"time-to-first-token":               "latency",
			"time-to-first-token-std-dev":       "latency",
			"inter-token-latency":               "latency",
			"inter-token-latency-std-dev":       "latency",
			"kv-cache-transfer-latency":         "latency",
			"kv-cache-transfer-latency-std-dev": "latency",
			"prefill-overhead":                  "latency",
			"prefill-time-per-token":            "latency",
			"prefill-time-std-dev":              "latency",
			"kv-cache-transfer-time-per-token":  "latency",
			"kv-cache-transfer-time-std-dev":    "latency",
			"time-factor-under-load":            "latency",
			"latency-calculator":                "latency",
			"failure-injection-rate":            "",
			"failure-types":                     "",
			"fake-metrics":                      "",
			"image-emission-rate":               "",
			"time-to-generate-image":            "",
			"time-to-generate-image-std-dev":    "",
		}))
	})

})

var _ = Describe("Configuration.load kv-cache YAML folding", func() {
	writeConfig := func(contents string) string {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "config.yaml")
		Expect(os.WriteFile(path, []byte(contents), 0o644)).To(Succeed())
		return path
	}

	It("populates KVCache from the nested kvcache block", func() {
		c := NewConfig()
		Expect(c.load(writeConfig(`
model: test-model
kvcache:
  enable-kvcache: true
  kv-cache-size: 2048
  block-size: 32
`))).To(Succeed())

		Expect(c.KVCache.EnableKVCache).To(BeTrue())
		Expect(c.KVCache.KVCacheSize).To(Equal(2048))
		Expect(c.KVCache.TokenBlockSize).To(Equal(32))
	})

	It("populates KVCache from legacy flat top-level keys", func() {
		c := NewConfig()
		Expect(c.load(writeConfig(`
model: test-model
enable-kvcache: true
kv-cache-size: 2048
block-size: 32
`))).To(Succeed())

		Expect(c.KVCache.EnableKVCache).To(BeTrue())
		Expect(c.KVCache.KVCacheSize).To(Equal(2048))
		Expect(c.KVCache.TokenBlockSize).To(Equal(32))
	})

	It("errors when kv-cache settings mix the flat and nested layouts", func() {
		c := NewConfig()
		Expect(c.load(writeConfig(`
model: test-model
kv-cache-size: 111
kvcache:
  kv-cache-size: 222
`))).ToNot(Succeed())
	})

	It("errors when a flat kv-cache key is set alongside an unrelated nested key", func() {
		c := NewConfig()
		Expect(c.load(writeConfig(`
model: test-model
kv-cache-size: 111
kvcache:
  block-size: 32
`))).ToNot(Succeed())
	})
})
