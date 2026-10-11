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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/spf13/pflag"
)

var _ = Describe("AddToggle", func() {
	// The positive and negative spellings share one variable, so an explicit
	// value has to be honored for them to stay each other's negation. Discarding
	// it silently turned --omni=false into --omni.
	DescribeTable("should apply the value a flag carries",
		func(arg string, expected bool) {
			value := false
			f := pflag.NewFlagSet("test", pflag.ContinueOnError)
			AddToggle(f, &value, "omni", "Enable omni mode", "Disable omni mode")

			Expect(f.Parse([]string{arg})).To(Succeed())
			Expect(value).To(Equal(expected))
		},
		Entry("bare positive", "--omni", true),
		Entry("positive set true", "--omni=true", true),
		Entry("positive set false", "--omni=false", false),
		Entry("bare negative", "--no-omni", false),
		Entry("negative set true", "--no-omni=true", false),
		Entry("negative set false", "--no-omni=false", true),
	)

	It("should start from the value the variable already holds", func() {
		value := true
		f := pflag.NewFlagSet("test", pflag.ContinueOnError)
		AddToggle(f, &value, "omni", "Enable omni mode", "Disable omni mode")

		Expect(f.Parse(nil)).To(Succeed())
		Expect(value).To(BeTrue())
	})

	It("should reject a value that is not a boolean", func() {
		value := false
		f := pflag.NewFlagSet("test", pflag.ContinueOnError)
		f.SetOutput(&discard{})
		AddToggle(f, &value, "omni", "Enable omni mode", "Disable omni mode")

		Expect(f.Parse([]string{"--omni=maybe"})).To(MatchError(ContainSubstring("invalid boolean value")))
	})
})

// discard swallows a FlagSet's usage output, which it prints on a parse error.
type discard struct{}

func (discard) Write(p []byte) (int, error) { return len(p), nil }

var _ = Describe("rejectSeparateBoolValue", func() {
	newFlagSet := func() *pflag.FlagSet {
		f := pflag.NewFlagSet("test", pflag.ContinueOnError)
		f.SetOutput(&discard{})
		var value bool
		AddToggle(f, &value, "omni", "Enable omni mode", "Disable omni mode")
		return f
	}

	// pflag takes no value for a boolean flag, so the separate-argument form
	// leaves the value as a positional argument and sets the flag regardless.
	// Refusing it is what stops "--flag false" from silently meaning "--flag",
	// and real vLLM refuses the form whatever the value.
	DescribeTable("should refuse a value written as a separate argument",
		func(flag string, args ...string) {
			f := newFlagSet()
			Expect(f.Parse(args)).To(Succeed())
			Expect(rejectSeparateBoolValue(f, args)).To(MatchError(ContainSubstring(flag + " does not take a value")))
		},
		Entry("positive spelling", "--omni", "--omni", "false"),
		// Every spelling an explicit "--flag=<value>" would have accepted, since
		// each one means the opposite of what was written in this form.
		Entry("zero", "--omni", "--omni", "0"),
		Entry("capitalized", "--omni", "--omni", "False"),
		Entry("abbreviated", "--omni", "--omni", "f"),
		// A true value is refused as well: the flag would be set either way, but
		// the argument does nothing and real vLLM does not accept it.
		Entry("true", "--omni", "--omni", "true"),
		Entry("one", "--omni", "--omni", "1"),
		// Nor is a value that is not a boolean at all silently dropped.
		Entry("not a boolean", "--omni", "--omni", "maybe"),
		// The negative spelling is refused the same way. The suggestion cannot be
		// "--no-no-omni": the negation of a negative flag is the flag itself.
		Entry("negative spelling", "--no-omni", "--no-omni", "false"),
	)
})
