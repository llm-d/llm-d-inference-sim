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

package openaiserverapi

import (
	"encoding/base64"
	"encoding/binary"
	"math"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("EmbeddingInput", func() {
	Context("UnmarshalJSON", func() {
		It("unmarshals empty input", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(""))
			Expect(err).NotTo(HaveOccurred())
			Expect(e.TextInputs()).To(BeNil())
			Expect(e.TokenInputs()).To(BeNil())
			Expect(e.Len()).To(Equal(0))
		})

		It("unmarshals single string", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`"hello"`))
			Expect(err).NotTo(HaveOccurred())
			Expect(e.TextInputs()).To(Equal([]string{"hello"}))
			Expect(e.TokenInputs()).To(BeNil())
			Expect(e.IsTokenInput()).To(BeFalse())
			Expect(e.Len()).To(Equal(1))
		})

		It("unmarshals array of strings", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`["a","b","c"]`))
			Expect(err).NotTo(HaveOccurred())
			Expect(e.TextInputs()).To(Equal([]string{"a", "b", "c"}))
			Expect(e.TokenInputs()).To(BeNil())
			Expect(e.IsTokenInput()).To(BeFalse())
			Expect(e.Len()).To(Equal(3))
		})

		It("unmarshals single token sequence (array of numbers)", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`[1,2,3]`))
			Expect(err).NotTo(HaveOccurred())
			Expect(e.TextInputs()).To(BeNil())
			Expect(e.TokenInputs()).To(Equal([][]int64{{1, 2, 3}}))
			Expect(e.IsTokenInput()).To(BeTrue())
			Expect(e.Len()).To(Equal(1))
		})

		It("unmarshals multiple token sequences (array of arrays of numbers)", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`[[1,2],[3,4]]`))
			Expect(err).NotTo(HaveOccurred())
			Expect(e.TextInputs()).To(BeNil())
			Expect(e.TokenInputs()).To(Equal([][]int64{{1, 2}, {3, 4}}))
			Expect(e.IsTokenInput()).To(BeTrue())
			Expect(e.Len()).To(Equal(2))
		})

		It("unmarshals empty array", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`[]`))
			Expect(err).NotTo(HaveOccurred())
			Expect(e.TextInputs()).To(BeNil())
			Expect(e.TokenInputs()).To(BeNil())
			Expect(e.Len()).To(Equal(0))
		})

		It("unmarshals token sequence with float64 numbers from JSON", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`[10.0, 20.0, 30.0]`))
			Expect(err).NotTo(HaveOccurred())
			Expect(e.TokenInputs()).To(Equal([][]int64{{10, 20, 30}}))
			Expect(e.Len()).To(Equal(1))
		})

		It("clears inputs when array has mixed types (string and number)", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`["a", 1]`))
			Expect(err).NotTo(HaveOccurred())
			Expect(e.TextInputs()).To(BeNil())
			Expect(e.TokenInputs()).To(BeNil())
			Expect(e.Len()).To(Equal(0))
		})

		It("sets nil entry when array of arrays has non-number element in one inner array", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`[[1, 2], ["x"]]`))
			Expect(err).NotTo(HaveOccurred())
			Expect(e.TextInputs()).To(BeNil())
			// Implementation stores nil for the inner array that failed to parse as numbers
			Expect(e.TokenInputs()).To(HaveLen(2))
			Expect(e.TokenInputs()[0]).To(Equal([]int64{1, 2}))
			Expect(e.TokenInputs()[1]).To(BeNil())
		})

		It("returns error for invalid JSON on string input", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`"unclosed`))
			Expect(err).To(HaveOccurred())
		})

		It("returns error for invalid JSON on array input", func() {
			var e EmbeddingInput
			err := e.UnmarshalJSON([]byte(`[1,2,`))
			Expect(err).To(HaveOccurred())
		})
	})
})

var _ = Describe("EncodeEmbeddingBase64", func() {
	It("encodes empty slice as empty string", func() {
		out := EncodeEmbeddingBase64(nil)
		Expect(out).To(Equal(""))
	})

	It("encodes single float32 correctly", func() {
		emb := []float32{1.0}
		out := EncodeEmbeddingBase64(emb)
		dec, err := base64.StdEncoding.DecodeString(out)
		Expect(err).NotTo(HaveOccurred())
		Expect(dec).To(HaveLen(4))
		bits := binary.LittleEndian.Uint32(dec)
		Expect(math.Float32frombits(bits)).To(Equal(float32(1.0)))
	})

	It("encodes multiple float32 values as little-endian", func() {
		emb := []float32{0.0, 1.0, -1.0, 3.14}
		out := EncodeEmbeddingBase64(emb)
		dec, err := base64.StdEncoding.DecodeString(out)
		Expect(err).NotTo(HaveOccurred())
		Expect(dec).To(HaveLen(4 * len(emb)))
		for i, expected := range emb {
			bits := binary.LittleEndian.Uint32(dec[4*i : 4*(i+1)])
			Expect(math.Float32frombits(bits)).To(Equal(expected))
		}
	})

	It("round-trips arbitrary float32 slice", func() {
		emb := []float32{0.5, -0.25, 1e-6, 1e6, float32(math.Pi)}
		out := EncodeEmbeddingBase64(emb)
		dec, err := base64.StdEncoding.DecodeString(out)
		Expect(err).NotTo(HaveOccurred())
		Expect(dec).To(HaveLen(4 * len(emb)))
		for i, expected := range emb {
			bits := binary.LittleEndian.Uint32(dec[4*i : 4*(i+1)])
			Expect(math.Float32frombits(bits)).To(Equal(expected))
		}
	})
})
