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

package tests

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// messagesSSEEvent holds a single parsed Anthropic SSE event.
type messagesSSEEvent struct {
	EventType string
	Data      json.RawMessage
}

// buildMessagesBody serialises a /v1/messages request body.
// Pass a non-empty system string for a system prompt, and tools/toolChoice when
// you need tool-call behaviour.
func buildMessagesBody(model string, stream bool, messages []map[string]any,
	system string, tools []map[string]any, toolChoice map[string]any) string {

	body := map[string]any{
		"model":      model,
		"messages":   messages,
		"max_tokens": 100,
		"stream":     stream,
	}
	if system != "" {
		body["system"] = system
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	if toolChoice != nil {
		body["tool_choice"] = toolChoice
	}
	b, err := json.Marshal(body)
	Expect(err).NotTo(HaveOccurred())
	return string(b)
}

// sendMessagesRequest sends a non-streaming POST to /v1/messages and returns the parsed response.
func sendMessagesRequest(client *http.Client, body string) *api.MessagesResponse {
	resp, err := client.Post("http://localhost/v1/messages", "application/json", strings.NewReader(body))
	Expect(err).NotTo(HaveOccurred())
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	Expect(resp.StatusCode).To(Equal(http.StatusOK), "response body: %s", string(data))

	var result api.MessagesResponse
	Expect(json.Unmarshal(data, &result)).To(Succeed())
	return &result
}

// readMessagesSSEStream sends a streaming POST to /v1/messages and returns all parsed SSE events.
func readMessagesSSEStream(client *http.Client, body string) []messagesSSEEvent {
	resp, err := client.Post("http://localhost/v1/messages", "application/json", strings.NewReader(body))
	Expect(err).NotTo(HaveOccurred())
	Expect(resp.StatusCode).To(Equal(http.StatusOK))
	defer resp.Body.Close() //nolint:errcheck

	var events []messagesSSEEvent
	scanner := bufio.NewScanner(resp.Body)
	var currentType string
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			currentType = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			raw := []byte(strings.TrimPrefix(line, "data: "))
			events = append(events, messagesSSEEvent{EventType: currentType, Data: raw})
			currentType = ""
		}
	}
	return events
}

// anthropicGetWeatherTool returns a minimal Anthropic-format tool definition.
var anthropicGetWeatherTool = map[string]any{
	"name":        functionNameGetWeather,
	"description": "Get weather at the given location",
	"input_schema": map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]string{"type": "string"},
		},
		"required": []string{"location"},
	},
}

var anthropicGetTemperatureTool = map[string]any{
	"name":        functionNameGetTemperature,
	"description": "Get temperature at the given location",
	"input_schema": map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]string{"type": "string"},
			"unit": map[string]any{
				"type": "string",
				"enum": []string{"C", "F"},
			},
		},
		"required": []string{"city", "unit"},
	},
}

// messagesBadRequestType is the error type each engine reports for a 400 on
// /v1/messages. Each engine's full status-to-type mapping is covered in its own
// package; this is the single value the specs below need.
var messagesBadRequestType = map[string]string{
	"vllm":   "BadRequestError",
	"sglang": "invalid_request_error",
}

var _ = Describe("Simulator for /v1/messages (Anthropic Messages API)", forEachEngine(func() {
	var (
		ctx   context.Context
		model string
	)

	BeforeEach(func() {
		ctx = context.TODO()
		model = common.TestModelName
	})

	Describe("non-streaming", func() {
		It("returns a basic text response", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := buildMessagesBody(model, false, []map[string]any{
				{"role": "user", "content": testUserMessage},
			}, "", nil, nil)

			resp := sendMessagesRequest(client, body)

			Expect(resp.ID).To(HavePrefix(api.MessagesIDPrefix))
			Expect(resp.Type).To(Equal(api.MessagesType))
			Expect(resp.Role).To(Equal(api.RoleAssistant))
			Expect(resp.Model).To(Equal(model))

			Expect(resp.Content).To(HaveLen(1))
			Expect(resp.Content[0].Type).To(Equal("text"))
			Expect(resp.Content[0].Text).NotTo(BeEmpty())

			Expect(resp.StopReason).NotTo(BeNil())
			Expect(*resp.StopReason).To(BeElementOf(
				api.MessagesStopReasonEndTurn,
				api.MessagesStopReasonMaxTokens,
			))
			Expect(resp.StopSequence).To(BeNil())

			Expect(resp.Usage.InputTokens).To(BeNumerically(">", 0))
			Expect(resp.Usage.OutputTokens).To(BeNumerically(">", 0))
		})

		It("returns a text response with a system prompt", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := buildMessagesBody(model, false, []map[string]any{
				{"role": "user", "content": testUserMessage},
			}, "You are a helpful assistant.", nil, nil)

			resp := sendMessagesRequest(client, body)

			Expect(resp.Type).To(Equal(api.MessagesType))
			Expect(resp.Content).To(HaveLen(1))
			Expect(resp.Content[0].Type).To(Equal("text"))
			Expect(*resp.StopReason).To(BeElementOf(
				api.MessagesStopReasonEndTurn,
				api.MessagesStopReasonMaxTokens,
			))
			Expect(resp.Usage.InputTokens).To(BeNumerically(">", 0))
		})

		It("returns a text response with multi-turn messages", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := buildMessagesBody(model, false, []map[string]any{
				{"role": "user", "content": "Hello!"},
				{"role": "assistant", "content": "Hi there! How can I help?"},
				{"role": "user", "content": testUserMessage},
			}, "", nil, nil)

			resp := sendMessagesRequest(client, body)

			Expect(resp.Type).To(Equal(api.MessagesType))
			Expect(resp.Content).To(HaveLen(1))
			Expect(resp.Content[0].Type).To(Equal("text"))
			Expect(*resp.StopReason).To(BeElementOf(
				api.MessagesStopReasonEndTurn,
				api.MessagesStopReasonMaxTokens,
			))
		})

		It("returns tool_use content blocks when tool_choice is any", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := buildMessagesBody(model, false, []map[string]any{
				{"role": "user", "content": testUserMessage},
			}, "", []map[string]any{anthropicGetWeatherTool}, map[string]any{"type": "any"})

			resp := sendMessagesRequest(client, body)

			Expect(resp.Type).To(Equal(api.MessagesType))
			Expect(*resp.StopReason).To(Equal(api.MessagesStopReasonToolUse))
			Expect(resp.Content).NotTo(BeEmpty())
			for _, block := range resp.Content {
				Expect(block.Type).To(Equal("tool_use"))
				Expect(block.ID).To(HavePrefix(common.MessagesToolIDPrefix))
				Expect(block.Name).To(Equal(functionNameGetWeather))
				Expect(block.Input).NotTo(BeNil())
				Expect(block.Input).To(HaveKey("location"))
			}
			Expect(resp.Usage.OutputTokens).To(BeNumerically(">", 0))
		})

		It("routes to the specific tool when tool_choice is tool", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := buildMessagesBody(model, false, []map[string]any{
				{"role": "user", "content": testUserMessage},
			}, "",
				[]map[string]any{anthropicGetWeatherTool, anthropicGetTemperatureTool},
				map[string]any{"type": "tool", "name": functionNameGetTemperature},
			)

			resp := sendMessagesRequest(client, body)

			Expect(*resp.StopReason).To(Equal(api.MessagesStopReasonToolUse))
			for _, block := range resp.Content {
				Expect(block.Name).To(Equal(functionNameGetTemperature))
				args, ok := block.Input["city"]
				Expect(ok).To(BeTrue(), "expected 'city' in input, got %v", block.Input)
				_ = args
				Expect(block.Input).To(HaveKey("unit"))
			}
		})

		// The error carries the Messages API's own envelope, not the OpenAI one
		// the other endpoints use: every engine answers this route the way the
		// Anthropic SDK expects, so that its typed error classes still parse.
		// What differs is the error type inside that envelope, which the engine
		// owns: vLLM keeps the spelling it uses on every other route, sglang maps
		// it to Anthropic's own vocabulary.
		It("returns 400 for a request with no messages", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := `{"model":"` + model + `","max_tokens":100,"messages":[]}`
			resp, err := client.Post("http://localhost/v1/messages", "application/json", strings.NewReader(body))
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close() //nolint:errcheck
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

			respBody, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			var errResp api.MessagesErrorResponse
			Expect(json.Unmarshal(respBody, &errResp)).To(Succeed())
			Expect(errResp.Type).To(Equal(api.MessagesTypeError))
			expectedType, known := messagesBadRequestType[currentEngine]
			Expect(known).To(BeTrue(), "no expected error type recorded for engine %s", currentEngine)
			Expect(errResp.Error.Type).To(Equal(expectedType))
			Expect(errResp.Error.Message).To(ContainSubstring("messages must not be empty"))
		})
	})

	Describe("streaming", func() {
		It("returns correct event sequence for a text response", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := buildMessagesBody(model, true, []map[string]any{
				{"role": "user", "content": testUserMessage},
			}, "", nil, nil)

			events := readMessagesSSEStream(client, body)
			Expect(events).NotTo(BeEmpty())

			// Verify the event sequence
			types := make([]string, len(events))
			for i, e := range events {
				types[i] = e.EventType
			}

			Expect(types[0]).To(Equal(api.MessagesEventMessageStart))
			Expect(types).To(ContainElement(api.MessagesEventContentBlockStart))
			Expect(types).To(ContainElement(api.MessagesEventPing))
			Expect(types).To(ContainElement(api.MessagesEventContentBlockDelta))
			Expect(types).To(ContainElement(api.MessagesEventContentBlockStop))
			Expect(types).To(ContainElement(api.MessagesEventMessageDelta))
			Expect(types[len(types)-1]).To(Equal(api.MessagesEventMessageStop))

			// message_start carries the initial message object
			var startEvent api.MessagesMessageStartEvent
			Expect(json.Unmarshal(events[0].Data, &startEvent)).To(Succeed())
			Expect(startEvent.Message).NotTo(BeNil())
			Expect(startEvent.Message.ID).To(HavePrefix(api.MessagesIDPrefix))
			Expect(startEvent.Message.Role).To(Equal(api.RoleAssistant))
			Expect(startEvent.Message.Usage.InputTokens).To(BeNumerically(">", 0))

			// content_block_start opens a text block
			cbsIdx := findEventIndex(types, api.MessagesEventContentBlockStart)
			var blockStart api.MessagesContentBlockStartEvent
			Expect(json.Unmarshal(events[cbsIdx].Data, &blockStart)).To(Succeed())
			Expect(blockStart.ContentBlock.Type).To(Equal("text"))
			Expect(blockStart.Index).To(Equal(0))

			// content_block_delta carries text
			deltasText := collectTextDeltas(events)
			Expect(strings.Join(deltasText, "")).NotTo(BeEmpty())

			// message_delta carries stop_reason and output token count
			mdIdx := findEventIndex(types, api.MessagesEventMessageDelta)
			var msgDelta api.MessagesMessageDeltaEvent
			Expect(json.Unmarshal(events[mdIdx].Data, &msgDelta)).To(Succeed())
			Expect(msgDelta.Delta.StopReason).NotTo(BeNil())
			Expect(*msgDelta.Delta.StopReason).To(BeElementOf(
				api.MessagesStopReasonEndTurn,
				api.MessagesStopReasonMaxTokens,
			))
			Expect(msgDelta.Usage.OutputTokens).To(BeNumerically(">", 0))
		})

		It("returns correct event sequence for tool_use", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := buildMessagesBody(model, true, []map[string]any{
				{"role": "user", "content": testUserMessage},
			}, "", []map[string]any{anthropicGetWeatherTool}, map[string]any{"type": "any"})

			events := readMessagesSSEStream(client, body)
			Expect(events).NotTo(BeEmpty())

			types := make([]string, len(events))
			for i, e := range events {
				types[i] = e.EventType
			}

			Expect(types[0]).To(Equal(api.MessagesEventMessageStart))
			Expect(types).To(ContainElement(api.MessagesEventContentBlockStart))
			Expect(types).To(ContainElement(api.MessagesEventContentBlockDelta))
			Expect(types).To(ContainElement(api.MessagesEventContentBlockStop))
			Expect(types).To(ContainElement(api.MessagesEventMessageDelta))
			Expect(types[len(types)-1]).To(Equal(api.MessagesEventMessageStop))

			// content_block_start must describe a tool_use block
			cbsIdx := findEventIndex(types, api.MessagesEventContentBlockStart)
			var blockStart api.MessagesContentBlockStartEvent
			Expect(json.Unmarshal(events[cbsIdx].Data, &blockStart)).To(Succeed())
			Expect(blockStart.ContentBlock.Type).To(Equal("tool_use"))
			Expect(blockStart.ContentBlock.ID).To(HavePrefix(common.MessagesToolIDPrefix))
			Expect(blockStart.ContentBlock.Name).To(Equal(functionNameGetWeather))

			// content_block_delta events carry input_json_delta
			for _, e := range events {
				if e.EventType != api.MessagesEventContentBlockDelta {
					continue
				}
				var delta api.MessagesContentBlockDeltaEvent
				Expect(json.Unmarshal(e.Data, &delta)).To(Succeed())
				Expect(delta.Delta.Type).To(Equal("input_json_delta"))
				Expect(delta.Delta.PartialJSON).NotTo(BeEmpty())
			}

			// message_delta carries tool_use stop_reason
			mdIdx := findEventIndex(types, api.MessagesEventMessageDelta)
			var msgDelta api.MessagesMessageDeltaEvent
			Expect(json.Unmarshal(events[mdIdx].Data, &msgDelta)).To(Succeed())
			Expect(*msgDelta.Delta.StopReason).To(Equal(api.MessagesStopReasonToolUse))
		})
	})

	Describe("disaggregation params", func() {
		It("returns kv_transfer_params when do_remote_decode is true", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := `{"model":"` + model + `","max_tokens":100,` +
				`"messages":[{"role":"user","content":"hello"}],` +
				`"kv_transfer_params":{"do_remote_decode":true}}`
			status, respBody := postMessagesRoute(client, "/v1/messages", body)
			Expect(status).To(Equal(http.StatusOK), "response body: %s", string(respBody))

			var respObj map[string]any
			Expect(json.Unmarshal(respBody, &respObj)).To(Succeed())
			kvParams, ok := respObj["kv_transfer_params"].(map[string]any)
			Expect(ok).To(BeTrue(), "kv_transfer_params must be present")
			Expect(kvParams["do_remote_prefill"]).To(BeTrue())
			Expect(kvParams["do_remote_decode"]).To(BeFalse())
			Expect(kvParams["remote_host"]).NotTo(BeEmpty())
			Expect(kvParams["remote_port"]).To(BeNumerically(">", 0))
			Expect(kvParams["remote_block_ids"]).NotTo(BeEmpty())
			Expect(kvParams["remote_engine_id"]).NotTo(BeEmpty())
		})

		It("does not return kv_transfer_params when do_remote_decode is absent", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := buildMessagesBody(model, false, []map[string]any{
				{"role": "user", "content": testUserMessage},
			}, "", nil, nil)
			status, respBody := postMessagesRoute(client, "/v1/messages", body)
			Expect(status).To(Equal(http.StatusOK), "response body: %s", string(respBody))

			var respObj map[string]any
			Expect(json.Unmarshal(respBody, &respObj)).To(Succeed())
			Expect(respObj).NotTo(HaveKey("kv_transfer_params"))
		})
	})

	Describe("count_tokens", func() {
		It("counts the tokens of the prompt, without requiring max_tokens", func() {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			body := `{"model":"` + model + `","system":"You are a helpful assistant.",` +
				`"messages":[{"role":"user","content":"` + testUserMessage + `"}]}`
			status, respBody := postMessagesRoute(client, "/v1/messages/count_tokens", body)
			Expect(status).To(Equal(http.StatusOK), "response body: %s", string(respBody))
			var resp api.MessagesCountTokensResponse
			Expect(json.Unmarshal(respBody, &resp)).To(Succeed())
			Expect(resp.InputTokens).To(BeNumerically(">", 0))
			// context_management is vLLM's extension; sglang answers input_tokens alone
			if currentEngine == "sglang" {
				Expect(resp.ContextManagement).To(BeNil())
			} else {
				Expect(resp.ContextManagement).NotTo(BeNil())
				Expect(resp.ContextManagement.OriginalInputTokens).To(Equal(resp.InputTokens))
			}
		})
	})

	// count_tokens answers in the Messages API's error envelope. render answers
	// as the other render endpoints do: vLLM serves it from the render router,
	// which frames errors in the OpenAI shape.
	DescribeTable("count_tokens request validation",
		func(reqBody string, expectedMsg string) {
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			status, respBody := postMessagesRoute(client, "/v1/messages/count_tokens", reqBody)
			Expect(status).To(Equal(http.StatusBadRequest))

			var errResp api.MessagesErrorResponse
			Expect(json.Unmarshal(respBody, &errResp)).To(Succeed())
			Expect(errResp.Type).To(Equal(api.MessagesTypeError))
			Expect(errResp.Error.Message).To(ContainSubstring(expectedMsg))
		},
		Entry("rejects empty messages",
			`{"model":"`+common.TestModelName+`","messages":[]}`, "messages must not be empty"),
		Entry("rejects an image block without a source",
			`{"model":"`+common.TestModelName+`","messages":[{"role":"user","content":[{"type":"image"}]}]}`,
			"missing required 'source' field"),
	)
}))

// The render endpoints are vLLM's: sglang serves none of them, so these specs
// name no other engine. Errors on /v1/messages/render are framed like the other
// render endpoints, which vLLM serves from its render router.
var _ = Describe("Messages API render", func() {
	var (
		ctx    = context.TODO()
		model  = common.TestModelName
		client *http.Client
	)

	BeforeEach(func() {
		var err error
		client, err = startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())
	})

	It("returns the token ids of the prompt", func() {
		body := buildMessagesBody(model, false, []map[string]any{
			{"role": "user", "content": testUserMessage},
		}, "You are a helpful assistant.", nil, nil)
		status, respBody := postMessagesRoute(client, "/v1/messages/render", body)
		Expect(status).To(Equal(http.StatusOK), "response body: %s", string(respBody))

		var resp api.RenderResponse
		Expect(json.Unmarshal(respBody, &resp)).To(Succeed())
		Expect(resp.TokenIDs).NotTo(BeEmpty())
		Expect(resp.Features).To(BeNil())
	})

	It("synthesizes mm features for image blocks", func() {
		body := `{"model":"` + model + `","max_tokens":100,"messages":[{"role":"user","content":[` +
			`{"type":"text","text":"describe"},` +
			`{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}},` +
			`{"type":"image","source":{"type":"url","url":"https://example.com/b.png"}}` +
			`]}]}`
		status, respBody := postMessagesRoute(client, "/v1/messages/render", body)
		Expect(status).To(Equal(http.StatusOK), "response body: %s", string(respBody))

		var resp api.RenderResponse
		Expect(json.Unmarshal(respBody, &resp)).To(Succeed())
		Expect(resp.TokenIDs).NotTo(BeEmpty())
		Expect(resp.Features).NotTo(BeNil())
		Expect(resp.Features.MMHashes).To(HaveKey("image"))
		Expect(resp.Features.MMHashes["image"]).To(HaveLen(2))
	})

	It("matches the count_tokens count", func() {
		prompt := `"model":"` + model + `","system":"You are a helpful assistant.",` +
			`"messages":[{"role":"user","content":"` + testUserMessage + `"}]`

		status, renderBody := postMessagesRoute(client, "/v1/messages/render", `{`+prompt+`,"max_tokens":100}`)
		Expect(status).To(Equal(http.StatusOK), "response body: %s", string(renderBody))
		var rendered api.RenderResponse
		Expect(json.Unmarshal(renderBody, &rendered)).To(Succeed())

		status, countBody := postMessagesRoute(client, "/v1/messages/count_tokens", `{`+prompt+`}`)
		Expect(status).To(Equal(http.StatusOK), "response body: %s", string(countBody))
		var counted api.MessagesCountTokensResponse
		Expect(json.Unmarshal(countBody, &counted)).To(Succeed())
		Expect(counted.InputTokens).To(Equal(len(rendered.TokenIDs)))
	})

	DescribeTable("render request validation",
		func(reqBody string, expectedMsg string) {
			status, respBody := postMessagesRoute(client, "/v1/messages/render", reqBody)
			Expect(status).To(Equal(http.StatusBadRequest))
			Expect(string(respBody)).To(ContainSubstring(expectedMsg))

			// not the Anthropic envelope {"type":"error","error":{...}}
			var errResp api.MessagesErrorResponse
			Expect(json.Unmarshal(respBody, &errResp)).To(Succeed())
			Expect(errResp.Type).NotTo(Equal(api.MessagesTypeError))
			var openAIErr api.ErrorResponse
			Expect(json.Unmarshal(respBody, &openAIErr)).To(Succeed())
			Expect(openAIErr.Error.Message).To(ContainSubstring(expectedMsg))
		},
		Entry("rejects empty messages",
			`{"model":"`+common.TestModelName+`","max_tokens":100,"messages":[]}`, "messages must not be empty"),
		Entry("rejects an image block without a source",
			`{"model":"`+common.TestModelName+`","max_tokens":100,"messages":[{"role":"user","content":[{"type":"image"}]}]}`,
			"missing required 'source' field"),
		Entry("rejects a missing max_tokens",
			`{"model":"`+common.TestModelName+`","messages":[{"role":"user","content":"hi"}]}`,
			"max_tokens is required"),
		Entry("rejects a non-positive max_tokens",
			`{"model":"`+common.TestModelName+`","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}`,
			common.InvalidMaxTokensErrMsg),
	)
})

// Encoder-only mode is vLLM's own, so the Messages API specs that turn it on
// name no other engine.
var _ = Describe("Messages API in encoder-only mode", func() {
	var client *http.Client

	BeforeEach(func() {
		var err error
		client, err = startServerWithArgs(context.TODO(), []string{"cmd", "--model", common.TestModelName,
			"--mm-encoder-only", "--mm-processor-kwargs", "args", "--ec-transfer-config", "cfg",
			"--enforce-eager", "--no-enable-prefix-caching"})
		Expect(err).NotTo(HaveOccurred())
	})

	It("returns ec_transfer_params when messages contain images", func() {
		body := `{"model":"` + common.TestModelName + `","max_tokens":100,"messages":[{"role":"user","content":[` +
			`{"type":"text","text":"describe"},` +
			`{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}},` +
			`{"type":"image","source":{"type":"url","url":"https://example.com/b.png"}}` +
			`]}]}`
		status, respBody := postMessagesRoute(client, "/v1/messages", body)
		Expect(status).To(Equal(http.StatusOK), "response body: %s", string(respBody))

		var respObj map[string]any
		Expect(json.Unmarshal(respBody, &respObj)).To(Succeed())
		ecParams, ok := respObj["ec_transfer_params"].(map[string]any)
		Expect(ok).To(BeTrue(), "ec_transfer_params must be present")
		Expect(ecParams).To(HaveLen(2))
		for _, v := range ecParams {
			params, ok := v.(map[string]any)
			Expect(ok).To(BeTrue())
			Expect(params["peer_host"]).NotTo(BeEmpty())
			Expect(params["peer_port"]).To(BeNumerically(">", 0))
			Expect(params["size_bytes"]).To(BeNumerically(">", 0))
			Expect(params["nixl_agent_metadata_b64"]).NotTo(BeEmpty())
		}
	})

	It("does not return ec_transfer_params when messages contain no images", func() {
		body := buildMessagesBody(common.TestModelName, false, []map[string]any{
			{"role": "user", "content": testUserMessage},
		}, "", nil, nil)
		status, respBody := postMessagesRoute(client, "/v1/messages", body)
		Expect(status).To(Equal(http.StatusOK), "response body: %s", string(respBody))

		var respObj map[string]any
		Expect(json.Unmarshal(respBody, &respObj)).To(Succeed())
		Expect(respObj).NotTo(HaveKey("ec_transfer_params"))
	})
})

// LoRA adapters are vLLM's own, so the Messages API specs that load one name no
// other engine. vLLM validates the model of /v1/messages/render and
// /v1/messages/count_tokens against the base model and the loaded adapters.
var _ = Describe("Messages API with LoRA adapters", func() {
	const loraName = "lora3"

	var client *http.Client

	BeforeEach(func() {
		var err error
		client, err = startServerWithArgs(context.TODO(), []string{"cmd", "--model", common.TestModelName,
			"--mode", common.ModeEcho, "--lora-modules", `{"name":"` + loraName + `","path":"/path/to/lora3"}`})
		Expect(err).NotTo(HaveOccurred())
	})

	body := func(model string) string {
		return buildMessagesBody(model, false, []map[string]any{
			{"role": "user", "content": testUserMessage},
		}, "", nil, nil)
	}

	DescribeTable("accepts a loaded adapter as the model",
		func(path string) {
			status, respBody := postMessagesRoute(client, path, body(loraName))
			Expect(status).To(Equal(http.StatusOK), "response body: %s", string(respBody))
		},
		Entry("render", "/v1/messages/render"),
		Entry("count_tokens", "/v1/messages/count_tokens"),
	)

	DescribeTable("rejects an unknown model",
		func(path string) {
			status, respBody := postMessagesRoute(client, path, body("lora1"))
			Expect(status).To(Equal(http.StatusNotFound))
			Expect(string(respBody)).To(ContainSubstring("does not exist"))
		},
		Entry("render", "/v1/messages/render"),
		Entry("count_tokens", "/v1/messages/count_tokens"),
	)

	It("leaves the other render endpoints rejecting an adapter", func() {
		status, respBody := postMessagesRoute(client, "/v1/chat/completions/render",
			`{"model":"`+loraName+`","messages":[{"role":"user","content":"hi"}]}`)
		Expect(status).To(Equal(http.StatusBadRequest))
		Expect(string(respBody)).To(ContainSubstring("is a LoRA adapter"))
	})
})

// postMessagesRoute sends body to the given /v1/messages route and returns the
// status code and response body.
func postMessagesRoute(client *http.Client, path string, body string) (int, []byte) {
	resp, err := client.Post("http://localhost"+path, "application/json", strings.NewReader(body))
	Expect(err).NotTo(HaveOccurred())
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	return resp.StatusCode, data
}

// findEventIndex returns the index of the first event with the given type.
func findEventIndex(types []string, eventType string) int {
	for i, t := range types {
		if t == eventType {
			return i
		}
	}
	return -1
}

// collectTextDeltas returns the Text field from every content_block_delta event that carries a text_delta.
func collectTextDeltas(events []messagesSSEEvent) []string {
	var out []string
	for _, e := range events {
		if e.EventType != api.MessagesEventContentBlockDelta {
			continue
		}
		var delta api.MessagesContentBlockDeltaEvent
		if err := json.Unmarshal(e.Data, &delta); err != nil {
			continue
		}
		if delta.Delta.Type == "text_delta" {
			out = append(out, delta.Delta.Text)
		}
	}
	return out
}
