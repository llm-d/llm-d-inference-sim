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

package engine

import (
	"os"
	"path/filepath"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func withArgs(args []string, f func()) {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = args
	f()
}

var _ = Describe("Get", func() {
	It("returns the vllm engine", func() {
		e, err := Get(VLLM)
		Expect(err).NotTo(HaveOccurred())
		Expect(e.Name()).To(Equal(VLLM))
	})

	It("errors on an unknown engine", func() {
		_, err := Get("bogus")
		Expect(err).To(HaveOccurred())
	})
})

var _ = Describe("Resolve", func() {
	It("defaults to vllm when --engine is absent and there is no config file", func() {
		withArgs([]string{"cmd"}, func() {
			e, err := Resolve()
			Expect(err).NotTo(HaveOccurred())
			Expect(e.Name()).To(Equal(VLLM))
		})
	})

	It("uses --engine when present", func() {
		withArgs([]string{"cmd", "--engine", "vllm"}, func() {
			e, err := Resolve()
			Expect(err).NotTo(HaveOccurred())
			Expect(e.Name()).To(Equal(VLLM))
		})
	})

	It("errors on an unknown --engine value", func() {
		withArgs([]string{"cmd", "--engine", "bogus"}, func() {
			_, err := Resolve()
			Expect(err).To(HaveOccurred())
		})
	})

	It("falls back to the config file's engine key when --engine is absent", func() {
		path := writeConfigFile("engine: vllm\n")
		withArgs([]string{"cmd", "--config", path}, func() {
			e, err := Resolve()
			Expect(err).NotTo(HaveOccurred())
			Expect(e.Name()).To(Equal(VLLM))
		})
	})

	It("defaults to vllm when the config file has no engine key", func() {
		path := writeConfigFile("model: some-model\n")
		withArgs([]string{"cmd", "--config", path}, func() {
			e, err := Resolve()
			Expect(err).NotTo(HaveOccurred())
			Expect(e.Name()).To(Equal(VLLM))
		})
	})

	It("errors on an unknown engine value in the config file", func() {
		path := writeConfigFile("engine: bogus\n")
		withArgs([]string{"cmd", "--config", path}, func() {
			_, err := Resolve()
			Expect(err).To(HaveOccurred())
		})
	})

	It("prefers --engine over the config file's engine key", func() {
		path := writeConfigFile("engine: bogus\n")
		withArgs([]string{"cmd", "--engine", "vllm", "--config", path}, func() {
			e, err := Resolve()
			Expect(err).NotTo(HaveOccurred())
			Expect(e.Name()).To(Equal(VLLM))
		})
	})
})

func writeConfigFile(content string) string {
	path := filepath.Join(GinkgoT().TempDir(), "config.yaml")
	Expect(os.WriteFile(path, []byte(content), 0o600)).To(Succeed())
	return path
}
