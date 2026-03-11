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

package tests

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

var tokenizerMngr *tokenizer.TokenizerManager = tokenizer.NewTokenizerManager()

func TestVllmSimulator(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Tests Suite")
}

var _ = BeforeSuite(func() {
	err := tokenizerMngr.Init(context.Background(), klog.Background())
	Expect(err).ShouldNot(HaveOccurred())
})

var _ = AfterSuite(func() {
	tokenizerMngr.Clean()
})
