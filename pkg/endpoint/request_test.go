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

package endpoint

import (
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp"
)

// ptrInt64 and ptrInt return pointers to their argument; small helpers for
// building request fixtures without leaking temp vars into each test.
func ptrInt64(v int64) *int64 { return &v }
func ptrInt(v int) *int       { return &v }

// newTextCompletionsFixture builds a TextCompletionsParsedRequest populated
// with a cross-section of fields that Split must preserve on each sub-request.
// Keep this exhaustive — if you add a field to the parsed type that should
// survive the copy, add it here too.
func newTextCompletionsFixture() *TextCompletionsParsedRequest {
	req := &TextCompletionsParsedRequest{}
	req.RequestID = "req-abc"
	req.Model = "test-model"
	req.DisplayedModel = "test-model-alias"
	req.Stream = true
	req.IgnoreEOS = true
	req.StreamOptions = &api.StreamOptions{IncludeUsage: true}
	threshold := 0.25
	req.CacheHitThreshold = &threshold
	req.Prompt = []api.PromptInput{{Text: "one"}, {Text: "two"}}
	req.MaxTokens = ptrInt64(42)
	req.Logprobs = ptrInt(3)
	return req
}

var _ = Describe("convertInputToMessages", func() {
	It("converts text-only input to raw message content", func() {
		input := []api.InputItem{
			&api.InputMessage{
				Type: "message",
				Role: api.RoleUser,
				Content: []api.InputContent{
					{Type: api.ResponsesInputText, Text: "Hello"},
				},
			},
		}
		messages := convertInputToMessages(input)
		Expect(messages).To(HaveLen(1))
		Expect(messages[0].Role).To(Equal(api.RoleUser))
		Expect(messages[0].Content.Raw).To(Equal("Hello"))
	})

	It("converts image input to structured content with image_url block", func() {
		input := []api.InputItem{
			&api.InputMessage{
				Type: "message",
				Role: api.RoleUser,
				Content: []api.InputContent{
					{Type: api.ResponsesInputText, Text: "Describe this"},
					{Type: api.ResponsesInputImage, ImageURL: "https://example.com/img.png"},
				},
			},
		}
		messages := convertInputToMessages(input)
		Expect(messages).To(HaveLen(1))
		Expect(messages[0].Content.Structured).To(HaveLen(2))
		Expect(messages[0].Content.Structured[0].Type).To(Equal("text"))
		Expect(messages[0].Content.Structured[0].Text).To(Equal("Describe this"))
		Expect(messages[0].Content.Structured[1].Type).To(Equal("image_url"))
		Expect(messages[0].Content.Structured[1].ImageURL.Url).To(Equal("https://example.com/img.png"))
	})

	It("converts audio input to structured content with input_audio block", func() {
		input := []api.InputItem{
			&api.InputMessage{
				Type: "message",
				Role: api.RoleUser,
				Content: []api.InputContent{
					{Type: api.ResponsesInputText, Text: "What does this say?"},
					{Type: api.ResponsesInputAudio, AudioData: "base64data", AudioFormat: "wav"},
				},
			},
		}
		messages := convertInputToMessages(input)
		Expect(messages).To(HaveLen(1))
		Expect(messages[0].Content.Structured).To(HaveLen(2))
		Expect(messages[0].Content.Structured[0].Type).To(Equal("text"))
		Expect(messages[0].Content.Structured[0].Text).To(Equal("What does this say?"))
		Expect(messages[0].Content.Structured[1].Type).To(Equal("input_audio"))
		Expect(messages[0].Content.Structured[1].InputAudio.Data).To(Equal("base64data"))
		Expect(messages[0].Content.Structured[1].InputAudio.Format).To(Equal("wav"))
	})

	It("converts mixed text, image, and audio content to structured blocks", func() {
		input := []api.InputItem{
			&api.InputMessage{
				Type: "message",
				Role: api.RoleUser,
				Content: []api.InputContent{
					{Type: api.ResponsesInputText, Text: "Look at this"},
					{Type: api.ResponsesInputImage, ImageURL: "https://example.com/a.png"},
					{Type: api.ResponsesInputAudio, AudioData: "audio==", AudioFormat: "mp3"},
				},
			},
		}
		messages := convertInputToMessages(input)
		Expect(messages).To(HaveLen(1))
		blocks := messages[0].Content.Structured
		Expect(blocks).To(HaveLen(3))
		Expect(blocks[0].Type).To(Equal("text"))
		Expect(blocks[1].Type).To(Equal("image_url"))
		Expect(blocks[2].Type).To(Equal("input_audio"))
	})

	It("uses raw content for single text-only message", func() {
		input := []api.InputItem{
			&api.InputMessage{
				Type: "message",
				Role: api.RoleUser,
				Content: []api.InputContent{
					{Type: api.ResponsesInputText, Text: "Simple message"},
				},
			},
		}
		messages := convertInputToMessages(input)
		Expect(messages).To(HaveLen(1))
		// Single text-only input -> raw content (not structured)
		Expect(messages[0].Content.Raw).To(Equal("Simple message"))
		Expect(messages[0].Content.Structured).To(BeNil())
	})

	It("uses structured content for single image-only message", func() {
		input := []api.InputItem{
			&api.InputMessage{
				Type: "message",
				Role: api.RoleUser,
				Content: []api.InputContent{
					{Type: api.ResponsesInputImage, ImageURL: "https://example.com/img.png"},
				},
			},
		}
		messages := convertInputToMessages(input)
		Expect(messages).To(HaveLen(1))
		// Even single image -> structured content (multimodal)
		Expect(messages[0].Content.Structured).To(HaveLen(1))
		Expect(messages[0].Content.Structured[0].Type).To(Equal("image_url"))
	})
})

var _ = Describe("ChatCompletionsRequest.Validate", func() {
	It("rejects an empty messages array", func() {
		req := &ChatCompletionsRequest{}
		req.Messages = []api.Message{}

		err := req.Validate(nil)
		Expect(err).NotTo(BeNil())
		Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
		Expect(err.Message).To(Equal("messages must not be empty"))
	})

	It("accepts a request carrying at least one message", func() {
		req := &ChatCompletionsRequest{}
		req.Messages = []api.Message{{Role: api.RoleUser}}

		Expect(req.Validate(nil)).To(BeNil())
	})
})

var _ = Describe("TextCompletionsParsedRequest.Split", func() {
	It("returns one sub-request per array element with suffixed RequestIDs", func() {
		orig := newTextCompletionsFixture()

		subs := orig.Split()

		Expect(subs).To(HaveLen(2))
		first := subs[0].(*TextCompletionsRequest)
		second := subs[1].(*TextCompletionsRequest)
		Expect(first.GetRequestID()).To(Equal("req-abc-0"))
		Expect(first.Prompt).To(Equal(api.PromptInput{Text: "one"}))
		Expect(second.GetRequestID()).To(Equal("req-abc-1"))
		Expect(second.Prompt).To(Equal(api.PromptInput{Text: "two"}))
	})

	It("preserves non-prompt fields on each sub-request", func() {
		orig := newTextCompletionsFixture()

		subs := orig.Split()
		sub := subs[0].(*TextCompletionsRequest)

		Expect(sub.GetModel()).To(Equal(orig.GetModel()))
		Expect(sub.GetDisplayedModel()).To(Equal(orig.GetDisplayedModel()))
		Expect(sub.IsStream()).To(Equal(orig.IsStream()))
		Expect(sub.IncludeUsage()).To(Equal(orig.IncludeUsage()))
		Expect(sub.GetIgnoreEOS()).To(Equal(orig.GetIgnoreEOS()))
		Expect(sub.ExtractMaxTokens()).To(Equal(orig.ExtractMaxTokens()))
		Expect(sub.GetLogprobs()).To(Equal(orig.GetLogprobs()))
		Expect(sub.GetCacheHitThreshold()).To(Equal(orig.GetCacheHitThreshold()))
	})

	It("carries token-id prompts through Split and pre-populates TokenizedPrompt", func() {
		orig := &TextCompletionsParsedRequest{}
		orig.RequestID = "req-xyz"
		orig.Model = "test-model"
		orig.Prompt = []api.PromptInput{{Tokens: []uint32{10, 20, 30}}}

		subs := orig.Split()

		Expect(subs).To(HaveLen(1))
		sub := subs[0].(*TextCompletionsRequest)
		Expect(sub.GetRequestID()).To(Equal("req-xyz-0"))
		Expect(sub.Prompt).To(Equal(api.PromptInput{Tokens: []uint32{10, 20, 30}}))
		// AsSingle pre-populates TokenizedPrompt so the worker skips encode().
		Expect(sub.TokenizedPrompt()).To(Equal(&api.Tokenized{Tokens: []uint32{10, 20, 30}}))
	})

	It("does not mutate the parsed request", func() {
		orig := newTextCompletionsFixture()
		origPrompt := append([]api.PromptInput(nil), orig.Prompt...)
		origID := orig.GetRequestID()

		_ = orig.Split()

		Expect(orig.GetRequestID()).To(Equal(origID))
		Expect(orig.Prompt).To(Equal(origPrompt))
	})
})

var _ = Describe("HandleRequest", func() {
	It("returns a server error when response tokens cannot be generated", func() {
		runtime := &fakeRuntime{
			config:            &common.Configuration{MaxModelLen: 1024, Mode: common.ModeRandom},
			responseTokensErr: errors.New("invalid gen_tokens"),
		}
		req := &TextCompletionsRequest{}
		req.RequestID = "req-err"
		req.Prompt = api.PromptInput{Tokens: []uint32{1, 2, 3}}
		reqCtx := req.BuildRequestContext(runtime, common.Channel[*ResponseInfo]{}, 0, func() {})

		respCtx, err := reqCtx.HandleRequest()

		Expect(respCtx).To(BeNil())
		Expect(err).NotTo(BeNil())
		Expect(err.Code).To(Equal(fasthttp.StatusInternalServerError))
		Expect(err.Message).To(ContainSubstring("invalid gen_tokens"))
	})
})

var _ = Describe("baseRequestContext.validateTokenizedRequest", func() {
	It("rejects min_tokens above the context-clamped response maximum in random mode", func() {
		minTokens := int64(6)
		maxTokens := int64(20)
		req := &ChatCompletionsRequest{
			ChatCompletionsRequest: api.ChatCompletionsRequest{
				MinTokens:           &minTokens,
				MaxCompletionTokens: &maxTokens,
			},
		}
		req.SetTokenizedPrompt(&api.Tokenized{Tokens: make([]uint32, 95)})

		runtime := &fakeRuntime{config: &common.Configuration{
			Mode:        common.ModeRandom,
			MaxModelLen: 100,
		}}
		reqCtx := &chatCompletionReqCtx{
			baseRequestContext: baseRequestContext{runtime: runtime},
			req:                req,
		}
		reqCtx.RequestContext = reqCtx

		message, statusCode := reqCtx.validateTokenizedRequest()

		Expect(statusCode).To(Equal(400))
		Expect(message).To(ContainSubstring("min_tokens"))
	})

	It("rejects min_tokens above the default max when max_tokens is omitted", func() {
		minTokens := int64(6)
		req := &ChatCompletionsRequest{
			ChatCompletionsRequest: api.ChatCompletionsRequest{
				MinTokens: &minTokens,
			},
		}
		req.SetTokenizedPrompt(&api.Tokenized{Tokens: make([]uint32, 95)})

		runtime := &fakeRuntime{config: &common.Configuration{
			Mode:        common.ModeRandom,
			MaxModelLen: 100,
		}}
		reqCtx := &chatCompletionReqCtx{
			baseRequestContext: baseRequestContext{runtime: runtime},
			req:                req,
		}
		reqCtx.RequestContext = reqCtx

		message, statusCode := reqCtx.validateTokenizedRequest()

		Expect(statusCode).To(Equal(400))
		Expect(message).To(ContainSubstring("min_tokens must be less than or equal to max_tokens=5"))
	})

	DescribeTable("in echo mode",
		func(promptLen int, echoLen int, minTokens *int64, maxTokens *int64, ignoreEOS bool, maxModelLen int,
			expectedMessage string) {
			req := &ChatCompletionsRequest{
				ChatCompletionsRequest: api.ChatCompletionsRequest{MinTokens: minTokens, MaxTokens: maxTokens},
			}
			req.IgnoreEOS = ignoreEOS
			req.SetTokenizedPrompt(&api.Tokenized{Tokens: make([]uint32, promptLen)})
			req.SetTokenizedPromptForEcho(&api.Tokenized{Tokens: make([]uint32, echoLen)})

			runtime := &fakeRuntime{config: &common.Configuration{
				Mode:        common.ModeEcho,
				MaxModelLen: maxModelLen,
			}}
			reqCtx := &chatCompletionReqCtx{
				baseRequestContext: baseRequestContext{runtime: runtime},
				req:                req,
			}
			reqCtx.RequestContext = reqCtx

			message, statusCode := reqCtx.validateTokenizedRequest()

			if expectedMessage == "" {
				Expect(statusCode).To(Equal(200))
				Expect(message).To(BeEmpty())
			} else {
				Expect(statusCode).To(Equal(400))
				Expect(message).To(ContainSubstring(expectedMessage))
			}
		},
		Entry("accepts min_tokens within the echo length", 50, 10, new(int64(10)), nil, false, 100, ""),
		Entry("accepts min_tokens above the echo length", 5, 5, new(int64(20)), nil, false, 100, ""),
		Entry("rejects min_tokens above max_tokens", 50, 10, new(int64(5)), new(int64(4)), false, 100,
			"min_tokens must be less than or equal to max_tokens=4, got 5."),
		Entry("rejects min_tokens above the room left in the context window", 50, 10, new(int64(60)), nil, false, 100,
			"min_tokens must be less than or equal to max_tokens=50, got 60."),
		Entry("accepts min_tokens above the echo length when the echo is padded to max_tokens", 50, 3, new(int64(10)), new(int64(20)), true, 100, ""),
		Entry("accepts an echo response that does not fit in the context window", 50, 50, nil, nil, false, 51, ""),
		Entry("accepts an echo padded to max_tokens that does not fit in the context window", 50, 10, nil, new(int64(20)), true, 60, ""),
		Entry("rejects a prompt that fills the context window", 50, 10, nil, nil, false, 50,
			"This model's maximum context length is 50 tokens. However, you requested 50 tokens in the messages."),
	)
})

var _ = Describe("validateRequest", func() {
	It("accepts min_tokens set to zero", func() {
		minTokens := int64(0)
		maxTokens := int64(20)
		req := &api.ChatCompletionsRequest{
			MinTokens:           &minTokens,
			MaxCompletionTokens: &maxTokens,
		}

		Expect(validateRequest(req)).To(BeNil())
	})
})

var _ = Describe("Decision API", func() {
	DescribeTable("rejects malformed JSON and field types during decoding",
		func(protocol DecisionProtocol, body string, expectedMessage ...string) {
			req := &DecisionRequest{Protocol: protocol}
			err := req.Unmarshal([]byte(body))
			Expect(err).NotTo(Succeed())
			if len(expectedMessage) > 0 {
				Expect(err).To(MatchError(expectedMessage[0]))
			}
		},
		Entry("malformed JSON", SystemOneProtocol, `{"state":`),
		Entry("question object required", SystemOneProtocol, `{"state":"x","questions":[]}`),
		Entry("question array required", DecisionsProtocol, `{"model":"test","input":"x","questions":{}}`),
		Entry("unsupported choice value type", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"choice","instructions":"q","choices":[{"value":1},{"value":2}]}]}`),
		Entry("unknown question field", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"noul","depends_on":["x"]}}}`),
		Entry("unknown question type", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"other"}}}`),
		Entry("bad noul criteria", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"noul","criteria":["yes","no"]}}}`, "noul criteria must be an object"),
		Entry("unknown noul criteria key", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"noul","criteria":{"yes":"desc"}}}}`, "noul criteria keys must be 'true' or 'false'"),
		Entry("duplicate criteria", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"choice","criteria":{"a":null,"a":null}}}}`),
		Entry("null name", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"predicate","instructions":"q","name":null}]}`),
		Entry("null description", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"choice","instructions":"q","choices":[{"value":"a","description":null},{"value":"b"}]}]}`),
		Entry("image input", DecisionsProtocol, `{"model":"test","input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}],"questions":[{"type":"predicate","instructions":"q"}]}`),
		Entry("predicate with choices", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"predicate","instructions":"q","choices":null}]}`),
		Entry("choice with levels", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"choice","instructions":"q","levels":null}]}`),
		Entry("score with choices", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"score","instructions":"q","choices":null}]}`),
		Entry("trailing JSON", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"noul"}}} {}`),
		Entry("question type casing", SystemOneProtocol, `{"state":"x","questions":{"q":{"Type":"noul"}}}`),
		Entry("score label casing", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"score","instructions":"q","levels":[{"Label":"low"},{"label":"high"}]}]}`),
	)

	DescribeTable("separates semantic validation from decoding",
		func(protocol DecisionProtocol, body, message string) {
			req := &DecisionRequest{Protocol: protocol, DefaultModel: common.TestModelName}
			Expect(req.Unmarshal([]byte(body))).To(Succeed())
			err := req.Validate(nil)
			Expect(err).NotTo(BeNil())
			Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
			Expect(err.Message).To(ContainSubstring(message))
		},
		Entry("required state", SystemOneProtocol, `{"questions":{"q":{"type":"noul"}}}`, "state is required"),
		Entry("null state", SystemOneProtocol, `{"state":null,"questions":{"q":{"type":"noul"}}}`, "state is required"),
		Entry("required model", DecisionsProtocol, `{"input":"x","questions":[{"type":"predicate","instructions":"q"}]}`, "model is required"),
		Entry("required input", DecisionsProtocol, `{"model":"test","questions":[{"type":"predicate","instructions":"q"}]}`, "input is required"),
		Entry("null input", DecisionsProtocol, `{"model":"test","input":null,"questions":[{"type":"predicate","instructions":"q"}]}`, "input is required"),
		Entry("empty questions", SystemOneProtocol, `{"state":"x","questions":{}}`, "questions must contain"),
		Entry("empty question id", SystemOneProtocol, `{"state":"x","questions":{"":{"type":"noul"}}}`, "question ids"),
		Entry("empty System One choices", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"choice","criteria":{}}}}`, "choice criteria must not be empty"),
		Entry("empty Decisions choices", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"choice","instructions":"q","choices":[]}]}`, "choices must not be empty"),
		Entry("empty System One score levels", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"score","criteria":[]}}}`, "score criteria must not be empty"),
		Entry("empty Decisions score levels", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"score","instructions":"q","levels":[]}]}`, "score levels must not be empty"),
		Entry("duplicate choices", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"choice","instructions":"q","choices":[{"value":true},{"value":true}]}]}`, "distinct"),
		Entry("converted score labels", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"score","criteria":[true,"True"]}}}`, "distinct"),
		Entry("required instructions", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"predicate"}]}`, "instructions is required"),
		Entry("required score label", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"score","instructions":"q","levels":[{},{"label":"high"}]}]}`, "label is required"),
		Entry("null score label", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"score","instructions":"q","levels":[{"label":null},{"label":"high"}]}]}`, "label is required"),
		Entry("thinking", SystemOneProtocol, `{"state":"x","chat_template_kwargs":{"enable_thinking":true},"questions":{"q":{"type":"noul"}}}`, "thinking must be off"),
		Entry("streaming", SystemOneProtocol, `{"state":"x","stream":true,"questions":{"q":{"type":"noul"}}}`, "does not support streaming"),
		Entry("media", SystemOneProtocol, `{"state":"x","audio":["url"],"questions":{"q":{"type":"noul"}}}`, "does not support"),
	)

	It("accepts long instructions", func() {
		text := strings.Repeat("x", 2*1024*1024)
		raw, err := json.Marshal(map[string]any{
			"model": "test", "input": "x",
			"questions": []map[string]any{
				{"type": "predicate", "instructions": text},
			},
		})
		Expect(err).NotTo(HaveOccurred())
		req := &DecisionRequest{Protocol: DecisionsProtocol}
		Expect(req.Unmarshal(raw)).To(Succeed())
		Expect(req.Validate(nil)).To(BeNil())
		Expect(req.questions[0].instructions).To(Equal(&text))
	})

	DescribeTable("uses shared cache salt semantics",
		func(salt string) {
			for _, protocol := range []DecisionProtocol{SystemOneProtocol, DecisionsProtocol} {
				body := map[string]any{
					"model": "test", "cache_salt": salt, "input": "x",
					"questions": json.RawMessage(`[{"type":"predicate","instructions":"q"}]`),
				}
				if protocol == SystemOneProtocol {
					delete(body, "input")
					body["state"] = "x"
					body["questions"] = json.RawMessage(`{"q":{"type":"noul"}}`)
				}
				raw, err := json.Marshal(body)
				Expect(err).NotTo(HaveOccurred())
				req := &DecisionRequest{Protocol: protocol}
				Expect(req.Unmarshal(raw)).To(Succeed())
				Expect(req.Validate(nil)).To(BeNil())
				if salt == "" {
					Expect(req.GetCacheSalt()).To(BeNil())
				} else {
					Expect(req.GetCacheSalt()).To(Equal(&salt))
				}
				Expect(req.Split()[0].GetCacheSalt()).To(Equal(req.GetCacheSalt()))
			}
		},
		Entry("empty", ""),
		Entry("long with special characters", strings.Repeat("tenant/a@b\\c\x00", 32)),
	)

	DescribeTable("validates normalized question count bounds",
		func(protocol DecisionProtocol, maximum int) {
			body := `{"model":"test","input":"x","questions":[{"type":"predicate","instructions":"q"}]}`
			if protocol == SystemOneProtocol {
				body = `{"state":"x","questions":{"q":{"type":"noul"}}}`
			}
			for _, size := range []int{0, 1, maximum, maximum + 1} {
				req := &DecisionRequest{Protocol: protocol}
				Expect(req.Unmarshal([]byte(body))).To(Succeed())
				q := req.questions[0]
				req.questions = make([]decisionQuestion, size)
				for i := range size {
					q.id = strconv.Itoa(i)
					req.questions[i] = q
				}
				apiErr := req.Validate(nil)
				if size > 0 && size <= maximum {
					Expect(apiErr).To(BeNil())
				} else {
					Expect(apiErr).NotTo(BeNil())
					Expect(apiErr.Code).To(Equal(fasthttp.StatusBadRequest))
				}
			}
		},
		Entry("System One questions", SystemOneProtocol, 64),
		Entry("Decisions questions", DecisionsProtocol, 200),
	)

	DescribeTable("accepts large option lists",
		func(protocol DecisionProtocol, kind string) {
			const size = 300
			body := `{"model":"test","input":"x","questions":[]}`
			if protocol == SystemOneProtocol {
				body = `{"state":"x","questions":{}}`
			}
			instructions := "q"
			req := &DecisionRequest{Protocol: protocol}
			Expect(req.Unmarshal([]byte(body))).To(Succeed())
			q := decisionQuestion{id: "q", kind: kind, instructions: &instructions, values: make([]any, size)}
			for i := range size {
				q.values[i] = strconv.Itoa(i)
				if kind == "score" {
					q.legend = append(q.legend, json.RawMessage(strconv.Quote(strconv.Itoa(i))))
					if protocol == DecisionsProtocol {
						q.values[i] = i
					}
				}
			}
			req.questions = []decisionQuestion{q}
			Expect(req.Validate(nil)).To(BeNil())
			if protocol == SystemOneProtocol {
				answers, _ := req.SystemOneAnswers()
				Expect(answers[0].Probabilities).To(HaveLen(size))
			} else {
				Expect(req.DecisionAnswers()[0].Probabilities).To(HaveLen(size))
			}
		},
		Entry("System One choices", SystemOneProtocol, "choice"),
		Entry("System One levels", SystemOneProtocol, "score"),
		Entry("Decisions choices", DecisionsProtocol, "choice"),
		Entry("Decisions levels", DecisionsProtocol, "score"),
	)

	DescribeTable("converts nested score levels without per-level copies",
		func(prefix, suffix, expectedPrefix string) {
			const depth = 1024
			text := strings.Repeat("x", 64*1024)
			level := strings.Repeat(prefix, depth) + strconv.Quote(text) + strings.Repeat(suffix, depth)
			body := []byte(`{"state":"x","questions":{"q":{"type":"score","criteria":[` + level + `,"high"]}}}`)
			var req *DecisionRequest
			result := testing.Benchmark(func(b *testing.B) {
				for range b.N {
					req = &DecisionRequest{Protocol: SystemOneProtocol}
					if err := req.Unmarshal(body); err != nil {
						b.Fatal(err)
					}
				}
			})
			// Allow decoder overhead, but not a full string copy at each nesting level.
			Expect(result.AllocedBytesPerOp()).To(BeNumerically("<", 128*len(body)))
			Expect(req.Validate(nil)).To(BeNil())
			var label string
			Expect(json.Unmarshal(req.questions[0].legend[0], &label)).To(Succeed())
			Expect(label).To(Equal(strings.Repeat(expectedPrefix, depth) + "'" + text + "'" + strings.Repeat(suffix, depth)))
		},
		Entry("arrays", "[", "]", "["),
		Entry("objects", `{"x":`, "}", "{'x': "),
	)

	DescribeTable("shares input without expanding prompts or client request IDs",
		func(protocol DecisionProtocol, body string) {
			req := &DecisionRequest{Protocol: protocol}
			Expect(req.Unmarshal([]byte(body))).To(Succeed())
			req.input = strings.Repeat("x", 3*1024*1024)
			req.SetRequestID(req.input)
			Expect(req.Validate(nil)).To(BeNil())
			parts := req.Split()
			Expect(parts).To(HaveLen(2))
			for _, part := range parts {
				sub := part.(*DecisionRequest)
				Expect(sub.Prompt.Text).To(BeEmpty())
				Expect(sub.input).To(Equal(req.input))
				Expect(sub.questions).To(HaveLen(1))
				Expect(sub.GetRequestID()).To(HaveLen(38))
			}
			Expect(parts[0].GetRequestID()).NotTo(Equal(parts[1].GetRequestID()))
			Expect(req.GetRequestID()).To(Equal(req.input))
		},
		Entry("System One", SystemOneProtocol, `{"state":"x","questions":{"q":{"type":"noul"},"r":{"type":"noul"}}}`),
		Entry("Decisions", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"predicate","instructions":"q"},{"type":"predicate","instructions":"r"}]}`),
	)

	// Answer vectors from vLLM 429a3da6654facfa792995bbbc75ee64cf83be56:
	// tests/entrypoints/generate/structured_decisions/test_{question_types,decisions}.py.
	// These are supplied probabilities, not model outputs or simulator defaults.
	DescribeTable("builds protocol answers from supplied probabilities",
		func(protocol DecisionProtocol, question string, probabilities []float64, labelMass float64, expected string) {
			body := `{"model":"test","input":"x","questions":[` + question + `]}`
			if protocol == SystemOneProtocol {
				body = `{"state":"x","questions":{"q":` + question + `}}`
			}
			req := &DecisionRequest{Protocol: protocol}
			Expect(req.Unmarshal([]byte(body))).To(Succeed())
			Expect(req.Validate(nil)).To(BeNil())
			Expect(req.questions).To(HaveLen(1))
			var answer any
			if protocol == SystemOneProtocol {
				answer = systemOneAnswer(req.questions[0], probabilities, labelMass)
			} else {
				answer = decisionAnswer(req.questions[0], probabilities, labelMass)
			}
			raw, err := json.Marshal(answer)
			Expect(err).NotTo(HaveOccurred())
			var got, want map[string]any
			Expect(json.Unmarshal(raw, &got)).To(Succeed())
			Expect(json.Unmarshal([]byte(expected), &want)).To(Succeed())
			for _, field := range []string{"confidence", "score", "noul", "probability"} {
				if value, ok := want[field]; ok {
					Expect(got).To(HaveKey(field))
					Expect(got[field]).To(BeNumerically("~", value, 1e-12), field)
					delete(got, field)
					delete(want, field)
				}
			}
			// Probability entries are copied unchanged; compare their types and order exactly.
			Expect(got).To(Equal(want))
		},
		Entry("System One choice selects a non-first maximum", SystemOneProtocol,
			`{"type":"choice","instructions":"Which team?","criteria":{"billing":"money","outage":null,"other":null}}`,
			[]float64{0.2, 0.7, 0.1}, 0.5,
			`{"type":"choice","choice":"outage","probabilities":{"billing":0.2,"outage":0.7,"other":0.1},"confidence":0.35}`),
		Entry("System One singleton confidence uses label mass", SystemOneProtocol,
			`{"type":"choice","criteria":{"only":null}}`, []float64{1}, 0.4,
			`{"type":"choice","choice":"only","probabilities":{"only":1},"confidence":0.4}`),
		Entry("System One noul returns the yes probability", SystemOneProtocol,
			`{"type":"noul","instructions":"Reply within the hour?"}`, []float64{0.8, 0.2}, 0.5,
			`{"type":"noul","noul":0.8,"probabilities":{"yes":0.8,"no":0.2},"confidence":0.4}`),
		Entry("System One score uses zero-based indices and a legend", SystemOneProtocol,
			`{"type":"score","instructions":"How angry?","criteria":["calm","annoyed","furious"]}`,
			[]float64{0.5, 0.25, 0.25}, 0.4,
			`{"type":"score","score":0.75,"legend":{"0":"calm","1":"annoyed","2":"furious"},"probabilities":{"0":0.5,"1":0.25,"2":0.25},"confidence":0.2}`),
		Entry("Decisions predicate has null name and no confidence", DecisionsProtocol,
			`{"type":"predicate","instructions":"Is it damaged?"}`, []float64{0.1, 0.9}, 0.5,
			`{"type":"predicate","name":null,"probability":0.9}`),
		Entry("Decisions choice preserves boolean and string values", DecisionsProtocol,
			`{"type":"choice","name":"decision","instructions":"Select a value.","choices":[{"value":true},{"value":"true"},{"value":false}]}`,
			[]float64{0.7, 0.2, 0.1}, 0.5,
			`{"type":"choice","name":"decision","choice":true,"probabilities":[{"value":true,"probability":0.7},{"value":"true","probability":0.2},{"value":false,"probability":0.1}],"confidence":0.35}`),
		Entry("Decisions score uses weighted indices and maximum confidence", DecisionsProtocol,
			`{"type":"score","instructions":"Rate severity.","levels":[{"label":"low"},{"label":"medium"},{"label":"high"}]}`,
			[]float64{0.1, 0.7, 0.2}, 0.8,
			`{"type":"score","name":null,"score":1.1,"probabilities":[{"value":0,"label":"low","probability":0.1},{"value":1,"label":"medium","probability":0.7},{"value":2,"label":"high","probability":0.2}],"confidence":0.56}`),
		Entry("Decisions repeated score labels retain distinct positions", DecisionsProtocol,
			`{"type":"score","instructions":"Rate severity.","levels":[{"label":"damage","description":"cosmetic"},{"label":"damage","description":"unusable"}]}`,
			[]float64{0.1, 0.9}, 0.8,
			`{"type":"score","name":null,"score":0.9,"probabilities":[{"value":0,"label":"damage","probability":0.1},{"value":1,"label":"damage","probability":0.9}],"confidence":0.72}`),
		// Additional regressions for criteria, zero values, and argmax ties.
		Entry("System One noul accepts empty criteria", SystemOneProtocol,
			`{"type":"noul","criteria":{}}`, []float64{0.8, 0.2}, 0.5,
			`{"type":"noul","noul":0.8,"probabilities":{"yes":0.8,"no":0.2},"confidence":0.4}`),
		Entry("System One noul accepts true criteria alone", SystemOneProtocol,
			`{"type":"noul","criteria":{"true":"desc"}}`, []float64{0.8, 0.2}, 0.5,
			`{"type":"noul","noul":0.8,"probabilities":{"yes":0.8,"no":0.2},"confidence":0.4}`),
		Entry("System One noul accepts false criteria alone", SystemOneProtocol,
			`{"type":"noul","criteria":{"false":"desc"}}`, []float64{0.8, 0.2}, 0.5,
			`{"type":"noul","noul":0.8,"probabilities":{"yes":0.8,"no":0.2},"confidence":0.4}`),
		Entry("System One singleton score is zero", SystemOneProtocol,
			`{"type":"score","criteria":["only"]}`, []float64{1}, 1.0,
			`{"type":"score","score":0,"legend":{"0":"only"},"probabilities":{"0":1},"confidence":1}`),
		Entry("Decisions singleton choice has probability one", DecisionsProtocol,
			`{"type":"choice","instructions":"q","choices":[{"value":"only"}]}`, []float64{1}, 1.0,
			`{"type":"choice","name":null,"choice":"only","probabilities":[{"value":"only","probability":1}],"confidence":1}`),
		Entry("Decisions singleton score is zero", DecisionsProtocol,
			`{"type":"score","instructions":"q","levels":[{"label":"only"}]}`, []float64{1}, 1.0,
			`{"type":"score","name":null,"score":0,"probabilities":[{"value":0,"label":"only","probability":1}],"confidence":1}`),
		Entry("System One noul confidence can exceed the yes probability", SystemOneProtocol,
			`{"type":"noul"}`, []float64{0, 1}, 0.5,
			`{"type":"noul","noul":0,"probabilities":{"yes":0,"no":1},"confidence":0.5}`),
		Entry("System One tied choices retain caller order", SystemOneProtocol,
			`{"type":"choice","criteria":{"zebra":null,"apple":null}}`, []float64{0.5, 0.5}, 0.4,
			`{"type":"choice","choice":"zebra","probabilities":{"zebra":0.5,"apple":0.5},"confidence":0.2}`),
		Entry("System One score retains a zero result", SystemOneProtocol,
			`{"type":"score","criteria":["low","high"]}`, []float64{1, 0}, 0.5,
			`{"type":"score","score":0,"legend":{"0":"low","1":"high"},"probabilities":{"0":1,"1":0},"confidence":0.5}`),
		Entry("Decisions selects a non-first boolean false with zero confidence", DecisionsProtocol,
			`{"type":"choice","instructions":"q","choices":[{"value":"false"},{"value":false}]}`, []float64{0.1, 0.9}, 0.0,
			`{"type":"choice","name":null,"choice":false,"probabilities":[{"value":"false","probability":0.1},{"value":false,"probability":0.9}],"confidence":0}`),
		Entry("Decisions tied choices preserve an empty string and name", DecisionsProtocol,
			`{"type":"choice","name":"","instructions":"q","choices":[{"value":""},{"value":false}]}`, []float64{0.5, 0.5}, 0.4,
			`{"type":"choice","name":"","choice":"","probabilities":[{"value":"","probability":0.5},{"value":false,"probability":0.5}],"confidence":0.2}`),
		Entry("Decisions predicate retains zero probability", DecisionsProtocol,
			`{"type":"predicate","instructions":"q"}`, []float64{1, 0}, 0.5,
			`{"type":"predicate","name":null,"probability":0}`),
		Entry("Decisions score retains a zero result", DecisionsProtocol,
			`{"type":"score","instructions":"q","levels":[{"label":"low"},{"label":"high"}]}`, []float64{1, 0}, 0.5,
			`{"type":"score","name":null,"score":0,"probabilities":[{"value":0,"label":"low","probability":1},{"value":1,"label":"high","probability":0}],"confidence":0.5}`),
		Entry("System One score legends use Python strings", SystemOneProtocol,
			`{"type":"score","criteria":[null,true,1.0,{"z":"discarded","a":[true,false,"low"],"z":null}]}`, []float64{0.5, 0.25, 0.125, 0.125}, 1.0,
			`{"type":"score","score":0.875,"legend":{"0":"None","1":"True","2":"1.0","3":"{'z': None, 'a': [True, False, 'low']}"},"probabilities":{"0":0.5,"1":0.25,"2":0.125,"3":0.125},"confidence":0.5}`),
	)

	DescribeTable("uses decision token counts instead of the response dataset",
		func(protocol DecisionProtocol, body string) {
			req := &DecisionRequest{Protocol: protocol, DefaultModel: common.TestModelName}
			Expect(req.Unmarshal([]byte(body))).To(Succeed())
			Expect(req.Validate(nil)).To(BeNil())
			req.SetRequestID("decision-counts")
			req.SetDisplayedModel("alias")
			runtime := &fakeRuntime{
				config:            &common.Configuration{Mode: common.ModeEcho, MaxModelLen: 1024},
				tokenizer:         tokenizer.NewSimpleTokenizer(),
				responseTokensErr: errors.New("the response dataset must not be used"),
			}
			parts := req.Split()
			Expect(parts).To(HaveLen(len(req.questions)))
			for _, part := range parts {
				Expect(part.GetRequestID()).NotTo(Equal(req.GetRequestID()))
				Expect(part.GetDisplayedModel()).To(Equal("alias"))
			}
			part := parts[0]
			resp, err := part.BuildRequestContext(runtime, common.Channel[*ResponseInfo]{}, 0, nil).HandleRequest()
			Expect(err).To(BeNil())
			Expect(resp.UsageData().PromptTokens).To(BeNumerically(">", 0))
			Expect(resp.UsageData().CompletionTokens).To(Equal(1))
			Expect(resp.UsageData().TotalTokens).To(Equal(resp.UsageData().PromptTokens + 1))
			Expect(resp.ResponseTokens().Length()).To(Equal(1))
			Expect(resp.DisplayModel()).To(Equal("alias"))
			Expect(part.TokenizedPrompt().Strings).To(BeEmpty())
			runtime.config.MaxModelLen = 1
			_, err = part.BuildRequestContext(runtime, common.Channel[*ResponseInfo]{}, 0, nil).HandleRequest()
			Expect(err).NotTo(BeNil())
			Expect(err.Code).To(Equal(fasthttp.StatusBadRequest))
		},
		Entry("vLLM System One", SystemOneProtocol, `{"state":"x","State":0,"Priority":null,"questions":{"z":{"type":"noul"},"a":{"type":"noul"}}}`),
		Entry("vLLM Decisions", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"predicate","instructions":"q"}]}`),
		Entry("text messages", DecisionsProtocol, `{"model":"test","input":[{"role":"user","content":"hello"},{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_text","text":"there"}]}],"questions":[{"type":"predicate","instructions":"q"}]}`),
		Entry("empty score label", DecisionsProtocol, `{"model":"test","input":"x","questions":[{"type":"score","instructions":"q","levels":[{"label":""},{"label":"high"}]}]}`),
	)
})
