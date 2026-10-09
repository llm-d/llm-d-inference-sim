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

package common

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
)

var _ = DescribeTable("FakeMetricWithFunction numeric decoding",
	func(decode func([]byte, any) error, value string) {
		var metric FakeMetricWithFunction
		Expect(json.Unmarshal([]byte(`"ramp:0:10:5s"`), &metric)).To(Succeed())
		Expect(decode([]byte(value), &metric)).To(Succeed())
		data, err := json.Marshal(&metric)
		Expect(err).ToNot(HaveOccurred())
		Expect(data).To(MatchJSON(value))
	},
	Entry("JSON nonzero", json.Unmarshal, "7"),
	Entry("JSON zero", json.Unmarshal, "0"),
	Entry("YAML nonzero", yaml.Unmarshal, "7"),
	Entry("YAML zero", yaml.Unmarshal, "0"),
)
