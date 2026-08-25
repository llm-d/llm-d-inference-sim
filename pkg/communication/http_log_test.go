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

package communication

import (
	"bytes"
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
)

var _ = Describe("bodyForLog", func() {
	const payload = `{"model":"test"}`

	It("returns an unencoded body as is", func() {
		Expect(bodyForLog([]byte(payload), nil)).To(Equal(payload))
	})

	It("decodes a gzip body", func() {
		body := fasthttp.AppendGzipBytes(nil, []byte(payload))

		Expect(bodyForLog(body, []byte("gzip"))).To(Equal(payload))
	})

	It("returns the raw body when decoding fails", func() {
		Expect(bodyForLog([]byte(payload), []byte("gzip"))).To(Equal(payload))
	})

	It("logs an empty string for a nil or empty body", func() {
		Expect(bodyForLog(nil, []byte("gzip"))).To(Equal(""))
		Expect(bodyForLog([]byte{}, nil)).To(Equal(""))
	})

	It("truncates the decoded body at the limit", func() {
		body := fasthttp.AppendGzipBytes(nil, bytes.Repeat([]byte("a"), maxHTTPLogBodyBytes+1))

		got := bodyForLog(body, []byte("gzip"))
		Expect(got).To(HavePrefix("aaa"))
		Expect(got).To(HaveSuffix(fmt.Sprintf("[truncated, total %d bytes]", maxHTTPLogBodyBytes+1)))
	})
})
