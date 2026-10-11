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

package tests

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/engine"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
)

// loadManifestConfig parses path the way main does, under engineName. The engine
// is passed on the command line rather than left to resolveEngine's precedence,
// since SIM_ENGINE outranks the key a manifest declares and would otherwise read
// one engine's examples as another's.
func loadManifestConfig(path string, engineName string) (*common.Configuration, error) {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = []string{"cmd", "--model", common.TestModelName,
		"--engine", engineName, "--config", path}

	eng, err := resolveEngine()
	if err != nil {
		return nil, err
	}
	return common.ParseCommandParamsAndLoadConfig(eng)
}

// manifestEngines reports the engines a manifest is expected to load under: the
// one it declares, or every registered engine when it names none.
func manifestEngines(contents []byte) []string {
	var declared struct {
		Engine string `yaml:"engine"`
	}
	Expect(yaml.Unmarshal(contents, &declared)).To(Succeed())
	if declared.Engine != "" {
		return []string{declared.Engine}
	}
	return engine.Names()
}

var _ = Describe("configuration manifests", func() {
	It("loads every configuration manifest the simulator ships", func() {
		paths, err := filepath.Glob("../../manifests/*.yaml")
		Expect(err).NotTo(HaveOccurred())
		profiles, err := filepath.Glob("../../manifests/latency-profiles/*.yaml")
		Expect(err).NotTo(HaveOccurred())
		Expect(profiles).NotTo(BeEmpty())

		for _, path := range append(paths, profiles...) {
			contents, err := os.ReadFile(path)
			Expect(err).NotTo(HaveOccurred())
			// Skip the Kubernetes manifests that share the directory.
			if strings.Contains(string(contents), "apiVersion:") {
				continue
			}
			for _, engineName := range manifestEngines(contents) {
				_, err = loadManifestConfig(path, engineName)
				Expect(err).NotTo(HaveOccurred(), "failed to load %s under the %s engine", path, engineName)
			}
		}
	})
})
