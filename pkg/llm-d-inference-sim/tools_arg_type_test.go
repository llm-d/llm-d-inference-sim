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
	"encoding/json"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("createArgument type normalization", func() {
	var (
		config *common.Configuration
		random *common.Random
	)

	BeforeEach(func() {
		config = &common.Configuration{
			Model:                                     "test",
			ServedModelNames:                          []string{"test"},
			MinToolCallArrayParamLength:               1,
			MaxToolCallArrayParamLength:               3,
			MinToolCallIntegerParam:                   0,
			MaxToolCallIntegerParam:                   10,
			MinToolCallNumberParam:                    0,
			MaxToolCallNumberParam:                    10,
			ObjectToolCallNotRequiredParamProbability: 100,
		}
		random = common.NewRandom(0, 0)
	})

	// property unmarshals a JSON schema fragment the way production code receives it.
	property := func(schemaJSON string) any {
		var value any
		Expect(json.Unmarshal([]byte(schemaJSON), &value)).To(Succeed())
		return value
	}

	It("treats a property with no type as a string", func() {
		arg, err := createArgument(property(`{"description": "a field"}`), config, random)

		Expect(err).NotTo(HaveOccurred())
		Expect(arg).To(BeAssignableToTypeOf(""))
	})

	It("resolves a nullable object union to its object branch", func() {
		arg, err := createArgument(
			property(`{"type": ["null", "object"], "properties": {"city": {"type": "string"}}, "required": ["city"]}`),
			config, random)

		Expect(err).NotTo(HaveOccurred())
		Expect(arg).To(HaveKey("city"))
	})

	It("resolves a nullable array union to its array branch", func() {
		arg, err := createArgument(
			property(`{"type": ["null", "array"], "items": {"type": "string"}}`),
			config, random)

		Expect(err).NotTo(HaveOccurred())
		Expect(arg).To(BeAssignableToTypeOf([]any{}))
		Expect(arg).NotTo(BeEmpty())
	})

	It("resolves a union to its first non-null branch regardless of position", func() {
		arg, err := createArgument(property(`{"type": ["string", "null"]}`), config, random)

		Expect(err).NotTo(HaveOccurred())
		Expect(arg).To(BeAssignableToTypeOf(""))
	})

	It("generates null for a property typed only null", func() {
		arg, err := createArgument(property(`{"type": "null"}`), config, random)

		Expect(err).NotTo(HaveOccurred())
		Expect(arg).To(BeNil())
	})

	It("still rejects a genuinely unsupported type", func() {
		_, err := createArgument(property(`{"type": "wombat"}`), config, random)

		Expect(err).To(HaveOccurred())
	})
})
