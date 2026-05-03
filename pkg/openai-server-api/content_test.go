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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Content ReadableText", func() {
	It("returns raw string when content is plain text", func() {
		c := ChatComplContent{Raw: "hello world"}
		Expect(c.ReadableText()).To(Equal("hello world"))
	})

	It("returns text block content for structured text", func() {
		c := ChatComplContent{
			Structured: []ChatComplContentBlock{
				{Type: "text", Text: "Describe this"},
			},
		}
		Expect(c.ReadableText()).To(Equal("Describe this"))
	})

	It("returns image_url block as image: <url>", func() {
		c := ChatComplContent{
			Structured: []ChatComplContentBlock{
				{Type: "image_url", ImageURL: ChatComplImageBlock{Url: "https://example.com/img.png"}},
			},
		}
		Expect(c.ReadableText()).To(Equal("image: https://example.com/img.png"))
	})

	It("joins multiple blocks with newlines", func() {
		c := ChatComplContent{
			Structured: []ChatComplContentBlock{
				{Type: "text", Text: "Describe this"},
				{Type: "image_url", ImageURL: ChatComplImageBlock{Url: "https://example.com/img.png"}},
			},
		}
		Expect(c.ReadableText()).To(Equal("Describe this\nimage: https://example.com/img.png"))
	})

	It("returns empty string for empty structured content", func() {
		c := ChatComplContent{Structured: []ChatComplContentBlock{}}
		Expect(c.ReadableText()).To(Equal(""))
	})

	It("returns empty string for zero-value content", func() {
		c := ChatComplContent{}
		Expect(c.ReadableText()).To(Equal(""))
	})

	It("skips unknown block types", func() {
		c := ChatComplContent{
			Structured: []ChatComplContentBlock{
				{Type: "text", Text: "hello"},
				{Type: "unknown", Text: "ignored"},
				{Type: "image_url", ImageURL: ChatComplImageBlock{Url: "https://example.com/img.png"}},
			},
		}
		Expect(c.ReadableText()).To(Equal("hello\nimage: https://example.com/img.png"))
	})

	It("does not modify PlainText behavior", func() {
		c := ChatComplContent{
			Structured: []ChatComplContentBlock{
				{Type: "text", Text: "hello"},
				{Type: "image_url", ImageURL: ChatComplImageBlock{Url: "https://example.com/img.png"}},
			},
		}
		Expect(c.PlainText()).To(Equal("hello "))
	})
})
