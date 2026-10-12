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
	"encoding/json"
	"os"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// createSimConfig runs the full parse-and-load path with this engine selected,
// the same way main does.
func createSimConfig(args []string) (*common.Configuration, error) {
	oldArgs := os.Args
	defer func() {
		os.Args = oldArgs
	}()
	os.Args = args

	return common.ParseCommandParamsAndLoadConfig(New())
}

// writeConfig writes body to a temporary YAML config file and returns its path.
// writeConfig writes a config file for one spec and returns its path.
func writeConfig(contents string) string {
	return common.WriteConfigFile(GinkgoT().TempDir(), contents)
}

var _ = Describe("Configuration", func() {
	It("should load this engine's own shipped config file", func() {
		config, err := createSimConfig([]string{"cmd", "--config", "../../../manifests/sglang-config.yaml"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.Model).To(Equal("Qwen/Qwen2-VL-2B-Instruct"))
		Expect(config.MaxModelLen).To(Equal(2048))
		Expect(config.MaxNumSeqs).To(Equal(5))
		Expect(config.MaxWaitingQueueLength).To(Equal(1000))
		// A single LoRA slot is all ApplyDefaults claims; no adapter can occupy it.
		Expect(config.Lora.MaxLoras).To(Equal(1))
		Expect(config.Lora.LoraModules).To(BeEmpty())
		Expect(config.KVCache).To(Equal(common.KVCacheConfig{}))
		Expect(config.FakeMetrics).To(BeNil())
	})

	// The shared example names none of the three settings the engines name
	// themselves, so it must load under this engine as it does under any other.
	It("should load the engine-independent shipped config file", func() {
		config, err := createSimConfig([]string{"cmd", "--config", "../../../manifests/basic-config.yaml"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.Port).To(Equal(8001))
		Expect(config.Model).To(Equal("Qwen/Qwen2-VL-2B-Instruct"))
		Expect(config.Seed).To(Equal(int64(100100100)))
	})

	// A key for a feature this engine does not implement and cannot yet spell is
	// left unclaimed, so the parser reports it by name.
	DescribeTable("should reject an unsupported configuration key",
		func(body string, expectedKeys ...string) {
			_, err := createSimConfig([]string{"cmd", "--model", common.TestModelName,
				"--config", writeConfig(body)})
			Expect(err).To(MatchError(ContainSubstring("the 'sglang' engine does not recognize")))
			for _, key := range expectedKeys {
				Expect(err.Error()).To(ContainSubstring(key))
			}
		},
		Entry("nested lora block", "lora:\n  max-loras: 2\n", "lora"),
		Entry("nested kvcache block", "kvcache:\n  enable-kvcache: true\n", "kvcache"),
		Entry("deeper flat kv-cache key", "block-size: 16\n", "block-size"),
		Entry("flat lora keys", "max-loras: 2\nlora-modules:\n- '{\"name\":\"lora1\",\"path\":\"/p\"}'\n",
			"lora-modules", "max-loras"),
		// Both are vLLM's spelling and neither has a flag under this engine, so a
		// config file must get the same answer the command line does. mm-encoder-only
		// in particular is not inert: it drops /v1/embeddings and changes which
		// dataset serves responses.
		Entry("enable-sleep-mode", "enable-sleep-mode: true\n", "enable-sleep-mode"),
		Entry("mm-encoder-only", "mm-encoder-only: true\n", "mm-encoder-only"),
	)

	// The same settings on the command line, for the symmetry the report above is
	// there to keep: neither channel may quietly apply them.
	DescribeTable("should reject an unsupported flag",
		func(flag string) {
			_, err := createSimConfig([]string{"cmd", "--model", common.TestModelName, flag})
			Expect(err).To(MatchError(ContainSubstring("unknown flag")))
		},
		Entry("enable-sleep-mode", "--enable-sleep-mode"),
		Entry("mm-encoder-only", "--mm-encoder-only"),
	)

	// The two settings this engine does claim from a config file, so that a file
	// gets the same answer the matching flag does rather than being told the key
	// is unrecognized.
	DescribeTable("should name the feature behind a configuration key it cannot honor",
		func(body string, expected string) {
			_, err := createSimConfig([]string{"cmd", "--model", common.TestModelName,
				"--config", writeConfig(body)})
			Expect(err).To(MatchError(ContainSubstring(expected)))
		},
		Entry("enable-kvcache", "enable-kvcache: true\n",
			"KV-cache simulation is not implemented by the sglang engine yet"),
		Entry("fake-metrics", "fake-metrics:\n  running-requests: 16\n",
			"fake metrics are not implemented by the sglang engine yet"),
	)

	DescribeTable("should accept a claimed configuration key that asks for nothing",
		func(body string) {
			config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName,
				"--config", writeConfig(body)})
			Expect(err).NotTo(HaveOccurred())
			Expect(config.KVCache.EnableKVCache).To(BeFalse())
			Expect(config.FakeMetrics).To(BeNil())
		},
		Entry("enable-kvcache disabled", "enable-kvcache: false\n"),
		// A valueless key must leave the default rather than fail, matching how
		// unmarshaling a null into a bool behaves for the engines that read it.
		Entry("enable-kvcache with no value", "enable-kvcache:\n"),
		// Every setting commented out is a common real-world state, and fake
		// metrics suppress every real metric, so it must not turn them on.
		Entry("empty fake-metrics block", "fake-metrics:\n  # running-requests: 5\n"),
	)

	// Both the groups this engine does not own and the keys it names its own way
	// are left unclaimed, so the report covers them together.
	It("should reject a config file written for vLLM", func() {
		_, err := createSimConfig([]string{"cmd", "--config", "../../../manifests/vllm-config.yaml"})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("lora-modules, max-cpu-loras, max-loras"))
		Expect(err.Error()).To(ContainSubstring("max-num-seqs"))
	})

	// A flag for a feature this engine is expected to grow is registered and
	// then refused by name, so the reason reaches the user instead of pflag's
	// "unknown flag".
	DescribeTable("should name the feature behind a flag it cannot honor",
		func(expected string, args ...string) {
			_, err := createSimConfig(append([]string{"cmd", "--model", common.TestModelName}, args...))
			Expect(err).To(MatchError(ContainSubstring(expected)))
		},
		Entry("enable-kvcache", "KV-cache simulation is not implemented by the sglang engine yet",
			"--enable-kvcache"),
		// An attached value reaches the same rejection: the feature is refused
		// after parsing, not by the flag failing to parse.
		Entry("enable-kvcache with an explicit value",
			"KV-cache simulation is not implemented by the sglang engine yet", "--enable-kvcache=true"),
		Entry("fake-metrics", "fake metrics are not implemented by the sglang engine yet",
			"--fake-metrics", `{"running-requests":16}`),
	)

	DescribeTable("should accept a flag it can honor for a feature it lacks",
		func(args ...string) {
			config, err := createSimConfig(append([]string{"cmd", "--model", common.TestModelName}, args...))
			Expect(err).NotTo(HaveOccurred())
			Expect(config.KVCache.EnableKVCache).To(BeFalse())
			Expect(config.FakeMetrics).To(BeNil())
		},
		// Written with "=" or as the negative spelling, since pflag gives a bool
		// flag a NoOptDefVal: in the separate-argument form the value is a
		// positional and the flag is true.
		Entry("enable-kvcache set to false", "--enable-kvcache=false"),
		Entry("enable-kvcache turned off by its no- form", "--no-enable-kvcache"),
		Entry("no flag at all"),
	)

	// LoRA is spelled differently by every engine, so this one registers no LoRA
	// flag rather than borrowing vLLM's names to reject them.
	It("should not register a LoRA flag", func() {
		_, err := createSimConfig([]string{"cmd", "--model", common.TestModelName, "--max-loras", "2"})
		Expect(err).To(MatchError(ContainSubstring("unknown flag")))
	})

	// The KV-cache transfer durations of a disaggregated P/D deployment are core:
	// the latency model applies them whichever engine runs, so the flags exist
	// here even though this engine simulates no KV cache of its own.
	It("should accept the KV-cache transfer latencies", func() {
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName,
			"--kv-cache-transfer-latency", "70ms", "--kv-cache-transfer-latency-std-dev", "20ms",
			"--kv-cache-transfer-time-per-token", "5ms", "--kv-cache-transfer-time-std-dev", "1ms"})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.Latencies.KVCacheTransferLatency).To(Equal(70 * time.Millisecond))
		Expect(config.Latencies.KVCacheTransferLatencyStdDev).To(Equal(20 * time.Millisecond))
		Expect(config.Latencies.KVCacheTransferTimePerToken).To(Equal(5 * time.Millisecond))
		Expect(config.Latencies.KVCacheTransferTimeStdDev).To(Equal(time.Millisecond))
	})

	It("should validate the KV-cache transfer latencies", func() {
		_, err := createSimConfig([]string{"cmd", "--model", common.TestModelName,
			"--kv-cache-transfer-latency", "70ms", "--kv-cache-transfer-latency-std-dev", "35ms"})
		Expect(err).To(MatchError(ContainSubstring(
			"kv-cache transfer standard deviation cannot be more than 30% of kv-cache transfer")))
	})
})

var _ = Describe("Setting names", func() {
	// sglang's own names for the three core fields each engine names for itself.
	// This engine registers these flags, so a command line written for a
	// different engine must fail rather than be honored here.
	DescribeTable("should accept a setting under sglang's own flag",
		func(flag string, read func(*common.Configuration) int) {
			config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName, flag, "4"})
			Expect(err).NotTo(HaveOccurred())
			Expect(read(config)).To(Equal(4))
		},
		Entry("context-length", "--context-length",
			func(c *common.Configuration) int { return c.MaxModelLen }),
		Entry("max-running-requests", "--max-running-requests",
			func(c *common.Configuration) int { return c.MaxNumSeqs }),
		Entry("max-queued-requests", "--max-queued-requests",
			func(c *common.Configuration) int { return c.MaxWaitingQueueLength }),
	)

	DescribeTable("should not register another engine's name for the setting",
		func(flag string) {
			_, err := createSimConfig([]string{"cmd", "--model", common.TestModelName, flag, "4"})
			Expect(err).To(MatchError(ContainSubstring("unknown flag")))
		},
		Entry("max-model-len", "--max-model-len"),
		Entry("max-num-seqs", "--max-num-seqs"),
		Entry("max-waiting-queue-length", "--max-waiting-queue-length"),
	)

	It("should read those settings from a config file under sglang's keys", func() {
		config, err := createSimConfig([]string{"cmd", "--config", writeConfig(
			"model: " + common.TestModelName + "\ncontext-length: 512\n" +
				"max-running-requests: 4\nmax-queued-requests: 2\n")})
		Expect(err).NotTo(HaveOccurred())
		Expect(config.MaxModelLen).To(Equal(512))
		Expect(config.MaxNumSeqs).To(Equal(4))
		Expect(config.MaxWaitingQueueLength).To(Equal(2))
	})

	// /admin/config and the startup log report these under the same names, so
	// what a reader sees is what they would pass back on the command line.
	It("should report the settings under sglang's names for external display", func() {
		config, err := createSimConfig([]string{"cmd", "--model", common.TestModelName,
			"--context-length", "512"})
		Expect(err).NotTo(HaveOccurred())

		body, err := config.MarshalCleaned()
		Expect(err).NotTo(HaveOccurred())
		var shown map[string]any
		Expect(json.Unmarshal(body, &shown)).To(Succeed())

		Expect(shown).To(HaveKeyWithValue("context-length", BeEquivalentTo(512)))
		Expect(shown).NotTo(HaveKey("max-model-len"))
	})
})
