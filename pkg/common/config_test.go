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
	"encoding/json"
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
	c.Lora.MaxLoras = 2
	c.Lora.MaxCPULoras = 5
	c.Latencies.TimeToFirstToken = 2000 * time.Millisecond
	c.Latencies.InterTokenLatency = 1000 * time.Millisecond
	c.Latencies.KVCacheTransferLatency = 100 * time.Millisecond
	c.Seed = 100100100
	c.Lora.LoraModules = []LoraModule{}
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
		// A fake-metrics partial only makes sense against an already-configured
		// FakeMetrics value (see Configuration.Update); the base fixture starts
		// with a nil one.
		base.FakeMetrics = &stubFakeMetrics{}

		next, update, _, err := base.Update([]byte(
			`{"failure-injection-rate": 50, "fake-metrics": {"running-requests": 7}}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(next.FailureInjectionRate).To(Equal(50))
		Expect(update.FakeMetrics).ToNot(BeNil())
		updateFakeMetrics, ok := update.FakeMetrics.(*stubFakeMetrics)
		Expect(ok).To(BeTrue())
		Expect(updateFakeMetrics.RunningRequests).ToNot(BeNil())
		Expect(updateFakeMetrics.RunningRequests.FixedValue).To(Equal(float64(7)))
		// Fields not in the body are nil on the fake-metrics partial.
		Expect(updateFakeMetrics.WaitingRequests).To(BeNil())
	})

	It("rejects a fake-metrics partial when no fake-metrics is configured", func() {
		// Without a concrete FakeMetrics to unmarshal into, the partial would be
		// silently dropped, so it is rejected instead.
		_, _, _, err := base.Update([]byte(`{"fake-metrics": {"running-requests": 7}}`))
		Expect(err).To(HaveOccurred())
	})

	It("treats an explicit null fake-metrics as a no-op when none is configured", func() {
		next, update, _, err := base.Update([]byte(`{"fake-metrics": null}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(update.FakeMetrics).To(BeNil())
		Expect(next.FakeMetrics).To(BeNil())
	})

	It("leaves update.FakeMetrics nil for a body that does not mention fake metrics", func() {
		// update reports only what the body asked to change, so the caller can
		// use it to decide whether to touch Prometheus at all.
		base.FakeMetrics = &stubFakeMetrics{RunningRequests: &FakeMetricWithFunction{FixedValue: 3}}

		next, update, _, err := base.Update([]byte(`{"failure-injection-rate": 50}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(update.FakeMetrics).To(BeNil())
		// The configured metrics survive untouched on the merged result.
		nextFakeMetrics, ok := next.FakeMetrics.(*stubFakeMetrics)
		Expect(ok).To(BeTrue())
		Expect(nextFakeMetrics.RunningRequests.FixedValue).To(Equal(3.0))
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
		Entry("time-to-generate-image -> true",
			`{"time-to-generate-image": "500ms"}`, true),
		Entry("time-to-generate-image-std-dev -> true",
			`{"time-to-generate-image": "500ms", "time-to-generate-image-std-dev": "50ms"}`, true),
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

	It("accepts latency fields nested under a top-level latencies object", func() {
		next, _, latencyChanged, err := base.Update([]byte(
			`{"latencies": {"time-to-first-token": "500ms", "inter-token-latency": "20ms"}}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(latencyChanged).To(BeTrue())
		Expect(next.Latencies.TimeToFirstToken).To(Equal(500 * time.Millisecond))
		Expect(next.Latencies.InterTokenLatency).To(Equal(20 * time.Millisecond))
	})

	It("accepts a nested latencies object alongside unrelated flat fields", func() {
		next, _, latencyChanged, err := base.Update([]byte(
			`{"failure-injection-rate": 7, "latencies": {"time-to-first-token": "500ms"}}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(latencyChanged).To(BeTrue())
		Expect(next.FailureInjectionRate).To(Equal(7))
		Expect(next.Latencies.TimeToFirstToken).To(Equal(500 * time.Millisecond))
	})

	It("rejects a field set both flat and inside the nested latencies object", func() {
		_, _, _, err := base.Update([]byte(
			`{"time-to-first-token": "100ms", "latencies": {"time-to-first-token": "200ms"}}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("time-to-first-token"))
	})

	It("rejects an unknown field inside the nested latencies object", func() {
		_, _, _, err := base.Update([]byte(`{"latencies": {"bogus-field": "1s"}}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("not a latencies field"))
	})

	It("rejects latency-calculator inside the nested latencies object", func() {
		_, _, _, err := base.Update([]byte(`{"latencies": {"latency-calculator": "constant"}}`))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("latency-calculator"))
	})

	It("rejects a non-object nested latencies value", func() {
		_, _, _, err := base.Update([]byte(`{"latencies": "not-an-object"}`))
		Expect(err).To(HaveOccurred())
	})
})

var _ = Describe("Configuration.MarshalCleaned", func() {
	It("nests latency fields under a top-level latencies object", func() {
		c := createDefaultConfig("model", nil)
		data, err := c.MarshalCleaned()
		Expect(err).ToNot(HaveOccurred())

		var m map[string]any
		Expect(json.Unmarshal(data, &m)).To(Succeed())

		Expect(m).To(HaveKey("latencies"))
		latencies, ok := m["latencies"].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(latencies).To(HaveKeyWithValue("time-to-first-token", "2s"))
		Expect(latencies).To(HaveKeyWithValue("inter-token-latency", "1s"))

		for _, key := range latenciesYAMLKeys {
			Expect(m).ToNot(HaveKey(key), "latency field %q must not remain at the top level", key)
		}

		Expect(m).To(HaveKey("latency-calculator"), "latency-calculator is a top-level field, not part of latencies")
		Expect(latencies).ToNot(HaveKey("latency-calculator"))
	})

	DescribeTable("nests fields under their own group and none remain at the top level",
		func(groupKey string, groupYAMLKeys []string) {
			c := createDefaultConfig("model", nil)
			data, err := c.MarshalCleaned()
			Expect(err).ToNot(HaveOccurred())

			var m map[string]any
			Expect(json.Unmarshal(data, &m)).To(Succeed())

			Expect(m).To(HaveKey(groupKey))
			_, ok := m[groupKey].(map[string]any)
			Expect(ok).To(BeTrue())

			for _, key := range groupYAMLKeys {
				Expect(m).ToNot(HaveKey(key), "field %q must not remain at the top level", key)
			}
		},
		Entry("tool-calls", "tool-calls", toolCallYAMLKeys),
		Entry("dataset", "dataset", datasetYAMLKeys),
		Entry("ssl", "ssl", sslYAMLKeys),
		// LoraConfig's JSON keys, listed literally: its wire format is engine-owned
		// (see pkg/engine/vllm), so there's no common reflection-derived list for it.
		Entry("lora", "lora", []string{"max-loras", "max-cpu-loras", "lora-modules"}),
	)
})

// stubFakeMetrics is a minimal FakeMetrics implementation, standing in for a
// real engine's own. The concrete implementations live with their engine (see
// pkg/engine/vllm/fakemetrics), which this package cannot import, and what
// Copy/Update need to get right here is generic anyway: pre-seeding the
// interface field with the source's concrete type so encoding/json has
// something to unmarshal the partial into.
type stubFakeMetrics struct {
	RunningRequests *FakeMetricWithFunction `json:"running-requests,omitempty"`
	WaitingRequests *FakeMetricWithFunction `json:"waiting-requests,omitempty"`
}

func (s *stubFakeMetrics) Validate() error  { return nil }
func (s *stubFakeMetrics) New() FakeMetrics { return &stubFakeMetrics{} }

var _ = Describe("Configuration.Copy", func() {
	It("should round-trip a non-nil FakeMetrics with a fixed-value metric", func() {
		c := &Configuration{
			FakeMetrics: &stubFakeMetrics{
				RunningRequests: &FakeMetricWithFunction{FixedValue: 5},
			},
		}

		got, err := c.Copy()
		Expect(err).NotTo(HaveOccurred())
		Expect(got.FakeMetrics).NotTo(BeNil())
		gotFakeMetrics, ok := got.FakeMetrics.(*stubFakeMetrics)
		Expect(ok).To(BeTrue())
		Expect(gotFakeMetrics.RunningRequests).NotTo(BeNil())
		Expect(gotFakeMetrics.RunningRequests.IsFunction).To(BeFalse())
		Expect(gotFakeMetrics.RunningRequests.FixedValue).To(Equal(5.0))
	})

	It("should round-trip a non-nil FakeMetrics with a function-valued metric", func() {
		c := &Configuration{
			FakeMetrics: &stubFakeMetrics{
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
		gotFakeMetrics, ok := got.FakeMetrics.(*stubFakeMetrics)
		Expect(ok).To(BeTrue())
		Expect(gotFakeMetrics.WaitingRequests).NotTo(BeNil())
		Expect(gotFakeMetrics.WaitingRequests.IsFunction).To(BeTrue())
		Expect(gotFakeMetrics.WaitingRequests.Function).NotTo(BeNil())
		Expect(gotFakeMetrics.WaitingRequests.Function.Name).To(Equal(OscillateFuncName))
		Expect(gotFakeMetrics.WaitingRequests.Function.Start).To(Equal(0.0))
		Expect(gotFakeMetrics.WaitingRequests.Function.End).To(Equal(10.0))
		Expect(gotFakeMetrics.WaitingRequests.Function.Period).To(Equal(5 * time.Second))
	})

	It("should round-trip an explicit-zero metric (non-nil pointer to zero-value struct)", func() {
		c := &Configuration{
			FakeMetrics: &stubFakeMetrics{
				RunningRequests: &FakeMetricWithFunction{},
			},
		}

		got, err := c.Copy()
		Expect(err).NotTo(HaveOccurred())
		Expect(got.FakeMetrics).NotTo(BeNil())
		gotFakeMetrics, ok := got.FakeMetrics.(*stubFakeMetrics)
		Expect(ok).To(BeTrue())
		Expect(gotFakeMetrics.RunningRequests).NotTo(BeNil())
		Expect(gotFakeMetrics.RunningRequests.IsFunction).To(BeFalse())
		Expect(gotFakeMetrics.RunningRequests.FixedValue).To(Equal(0.0))
	})
})

var _ = Describe("admin struct tags", func() {
	It("has no unrecognized tag values", func() {
		// Latencies is a named, non-anonymous field, so a field walk over
		// Configuration does not descend into it; check it separately.
		checkTags := func(t reflect.Type) {
			for _, f := range reflect.VisibleFields(t) {
				Expect(f.Tag.Get("admin")).To(BeElementOf("", "configurable"),
					"field %s has unexpected admin tag %q", f.Name, f.Tag.Get("admin"))
				Expect(f.Tag.Get("rebuild")).To(BeElementOf("", "latency"),
					"field %s has unexpected rebuild tag %q", f.Name, f.Tag.Get("rebuild"))
				if f.Tag.Get("rebuild") == "latency" {
					Expect(f.Tag.Get("admin")).To(Equal("configurable"),
						"field %s has rebuild:\"latency\" but missing admin:\"configurable\"", f.Name)
				}
			}
		}
		checkTags(reflect.TypeOf(Configuration{}))
		checkTags(reflect.TypeOf(LatenciesConfig{}))
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
			"time-to-generate-image":            "latency",
			"time-to-generate-image-std-dev":    "latency",
			"failure-injection-rate":            "",
			"failure-types":                     "",
			"fake-metrics":                      "",
			"image-emission-rate":               "",
		}))
	})

})

var _ = Describe("Configuration.load latencies YAML folding", func() {
	writeConfig := func(contents string) string {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "config.yaml")
		Expect(os.WriteFile(path, []byte(contents), 0o644)).To(Succeed())
		return path
	}

	It("populates Latencies from the nested latencies block", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
latencies:
  time-to-first-token: 250ms
  inter-token-latency: 10ms
`))
		Expect(err).To(Succeed())

		Expect(c.Latencies.TimeToFirstToken).To(Equal(250 * time.Millisecond))
		Expect(c.Latencies.InterTokenLatency).To(Equal(10 * time.Millisecond))
	})

	It("populates Latencies from legacy flat top-level keys", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
time-to-first-token: 250ms
inter-token-latency: 10ms
`))
		Expect(err).To(Succeed())

		Expect(c.Latencies.TimeToFirstToken).To(Equal(250 * time.Millisecond))
		Expect(c.Latencies.InterTokenLatency).To(Equal(10 * time.Millisecond))
	})

	It("errors when latencies settings mix the flat and nested layouts", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
time-to-first-token: 100ms
latencies:
  time-to-first-token: 200ms
`))
		Expect(err).ToNot(Succeed())
	})

	It("errors when a flat latency key is set alongside an unrelated nested key", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
time-to-first-token: 100ms
latencies:
  inter-token-latency: 10ms
`))
		Expect(err).ToNot(Succeed())
	})
})

var _ = Describe("Configuration.load tool-calls YAML folding", func() {
	writeConfig := func(contents string) string {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "config.yaml")
		Expect(os.WriteFile(path, []byte(contents), 0o644)).To(Succeed())
		return path
	}

	It("populates ToolCalls from the nested tool-calls block", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
tool-calls:
  max-tool-call-integer-param: 50
  skip-tool-validation: true
`))
		Expect(err).To(Succeed())

		Expect(c.ToolCalls.MaxToolCallIntegerParam).To(Equal(50))
		Expect(c.ToolCalls.SkipToolValidation).To(BeTrue())
	})

	It("populates ToolCalls from legacy flat top-level keys", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
max-tool-call-integer-param: 50
skip-tool-validation: true
`))
		Expect(err).To(Succeed())

		Expect(c.ToolCalls.MaxToolCallIntegerParam).To(Equal(50))
		Expect(c.ToolCalls.SkipToolValidation).To(BeTrue())
	})

	It("errors when tool-call settings mix the flat and nested layouts", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
max-tool-call-integer-param: 50
tool-calls:
  max-tool-call-integer-param: 60
`))
		Expect(err).ToNot(Succeed())
	})

	It("errors when a flat tool-call key is set alongside an unrelated nested key", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
max-tool-call-integer-param: 50
tool-calls:
  skip-tool-validation: true
`))
		Expect(err).ToNot(Succeed())
	})
})

var _ = Describe("Configuration.load dataset YAML folding", func() {
	writeConfig := func(contents string) string {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "config.yaml")
		Expect(os.WriteFile(path, []byte(contents), 0o644)).To(Succeed())
		return path
	}

	It("populates Dataset from the nested dataset block", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
dataset:
  dataset-path: /tmp/data.db
  dataset-in-memory: true
`))
		Expect(err).To(Succeed())

		Expect(c.Dataset.DatasetPath).To(Equal("/tmp/data.db"))
		Expect(c.Dataset.DatasetInMemory).To(BeTrue())
	})

	It("populates Dataset from legacy flat top-level keys", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
dataset-path: /tmp/data.db
dataset-in-memory: true
`))
		Expect(err).To(Succeed())

		Expect(c.Dataset.DatasetPath).To(Equal("/tmp/data.db"))
		Expect(c.Dataset.DatasetInMemory).To(BeTrue())
	})

	It("errors when dataset settings mix the flat and nested layouts", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
dataset-path: /tmp/data.db
dataset:
  dataset-path: /tmp/other.db
`))
		Expect(err).ToNot(Succeed())
	})

	It("errors when a flat dataset key is set alongside an unrelated nested key", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
dataset-path: /tmp/data.db
dataset:
  dataset-in-memory: true
`))
		Expect(err).ToNot(Succeed())
	})
})

var _ = Describe("Configuration.load ssl YAML folding", func() {
	writeConfig := func(contents string) string {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "config.yaml")
		Expect(os.WriteFile(path, []byte(contents), 0o644)).To(Succeed())
		return path
	}

	It("populates SSL from the nested ssl block", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
ssl:
  ssl-certfile: /tmp/cert.pem
  ssl-keyfile: /tmp/key.pem
`))
		Expect(err).To(Succeed())

		Expect(c.SSL.SSLCertFile).To(Equal("/tmp/cert.pem"))
		Expect(c.SSL.SSLKeyFile).To(Equal("/tmp/key.pem"))
	})

	It("populates SSL from legacy flat top-level keys", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
ssl-certfile: /tmp/cert.pem
ssl-keyfile: /tmp/key.pem
`))
		Expect(err).To(Succeed())

		Expect(c.SSL.SSLCertFile).To(Equal("/tmp/cert.pem"))
		Expect(c.SSL.SSLKeyFile).To(Equal("/tmp/key.pem"))
	})

	It("errors when ssl settings mix the flat and nested layouts", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
ssl-certfile: /tmp/cert.pem
ssl:
  ssl-certfile: /tmp/other.pem
`))
		Expect(err).ToNot(Succeed())
	})

	It("errors when a flat ssl key is set alongside an unrelated nested key", func() {
		c := NewConfig()
		_, err := c.load(writeConfig(`
model: test-model
ssl-certfile: /tmp/cert.pem
ssl:
  self-signed-certs: true
`))
		Expect(err).ToNot(Succeed())
	})
})
