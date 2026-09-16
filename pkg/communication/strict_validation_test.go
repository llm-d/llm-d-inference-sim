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
	"testing"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/endpoint"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
)

const (
	strictChatPath       = "/v1/chat/completions"
	strictCompletionPath = "/v1/completions"
)

var _ = Describe("strict vLLM request validation", func() {
	Describe("content type", func() {
		It("accepts JSON with parameters", func() {
			Expect(validateStrictContentType("application/json; charset=utf-8")).To(BeNil())
		})

		It("matches vLLM's unsupported media type response", func() {
			Expect(validateStrictContentType("text/plain")).To(Equal(&api.Error{
				Message: "1 validation error:\n  Unsupported Media Type: Only 'application/json' is allowed " +
					"[\"Unsupported Media Type: Only 'application/json' is allowed\"]",
				Type: "Bad Request",
				Code: fasthttp.StatusBadRequest,
			}))
		})
	})
})

type strictTestRuntime struct {
	endpoint.Runtime
	config *common.Configuration
}

func (r strictTestRuntime) Config() *common.Configuration { return r.config }

type strictTestValidator struct{ calls int }

func (v *strictTestValidator) Validate(body []byte, path string) *api.Error {
	v.calls++
	if path != strictChatPath && path != strictCompletionPath {
		panic("unexpected endpoint")
	}
	return badRequest("engine rejected request", nil)
}

func TestStrictValidationAtReceipt(t *testing.T) {
	for _, path := range []string{strictChatPath, strictCompletionPath} {
		for _, strict := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/strict=%v", path, strict), func(t *testing.T) {
				cfg := common.NewConfig()
				cfg.StrictRequestValidation = strict
				cfg.EnableRequestIDHeaders = true
				validator := &strictTestValidator{}
				c := New(logr.Discard(), nil, strictTestRuntime{config: cfg})
				c.strictValidator = validator
				ctx := &fasthttp.RequestCtx{}
				ctx.Request.SetRequestURI(path)
				ctx.Request.Header.SetContentType("application/json")
				// A deterministic response lets the lenient path stop before generation.
				ctx.Request.Header.Set(XReturnErrorHeader, "418")
				ctx.Request.Header.Set(RequestIDHeader, "strict-test")
				ctx.Request.SetBodyString(`{"model":"m","messages":[{"role":"user","content":"hi"}],"prompt":"hi"}`)
				if path == strictChatPath {
					c.HandleChatCompletions(ctx)
				} else {
					c.HandleTextCompletions(ctx)
				}
				if strict {
					if validator.calls != 1 || ctx.Response.StatusCode() != 400 || !bytes.Contains(ctx.Response.Body(), []byte("engine rejected request")) {
						t.Fatalf("calls=%d response=%s", validator.calls, &ctx.Response)
					}
				} else if validator.calls != 0 || ctx.Response.StatusCode() != 418 {
					t.Fatalf("calls=%d response=%s", validator.calls, &ctx.Response)
				}
			})
		}
	}
}

func TestStrictMediaTypeBeforeEngine(t *testing.T) {
	cfg := common.NewConfig()
	cfg.StrictRequestValidation = true
	validator := &strictTestValidator{}
	c := New(logr.Discard(), nil, strictTestRuntime{config: cfg})
	c.strictValidator = validator
	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetRequestURI(strictChatPath)
	ctx.Request.Header.SetContentType("text/plain")
	ctx.Request.SetBodyString(`{}`)
	c.HandleChatCompletions(ctx)
	if validator.calls != 0 || ctx.Response.StatusCode() != 400 {
		t.Fatalf("calls=%d response=%s", validator.calls, &ctx.Response)
	}
}
