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

package simulator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
)

var _ = Describe("SimContext", func() {
	DescribeTable("RandomResponses",
		func(ds dataset.Dataset, want bool) {
			Expect((&SimContext{dataset: ds}).RandomResponses()).To(Equal(want))
		},
		Entry("random text", &dataset.DefaultDataset{}, true),
		Entry("echo", &dataset.EchoDataset{}, false),
		Entry("dataset file", &dataset.CustomDataset{}, false),
		Entry("mm-encoder-only", &dataset.MMEncoderOnlyDataset{}, false),
	)
})
