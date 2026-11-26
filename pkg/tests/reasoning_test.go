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
	"strconv"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("chat completion reasoning simulation", func() {
	DescribeTable("partitions echo tokens in non-streamed responses",
		func(pct int) {
			client, err := startServerWithArgs(context.TODO(), []string{
				"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
				"--reasoning-percentage", strconv.Itoa(pct),
			})
			Expect(err).NotTo(HaveOccurred())
			prompt := "alpha beta gamma delta"
			body := `{"model":"` + common.TestModelName + `","messages":[{"role":"user","content":"` + prompt + `"}],"max_tokens":20}`
			resp, err := client.Post(baseURL+"/chat/completions", "application/json", strings.NewReader(body))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			data, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			var out struct {
				Choices []struct {
					Message struct {
						Reasoning string `json:"reasoning"`
						Content   string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
				Usage struct {
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
			}
			Expect(json.Unmarshal(data, &out)).To(Succeed())
			Expect(out.Choices).To(HaveLen(1))
			Expect(out.Choices[0].Message.Reasoning + out.Choices[0].Message.Content).To(Equal(prompt))
			Expect(out.Usage.CompletionTokens).To(BeNumerically(">", 0))
			switch pct {
			case 0:
				Expect(out.Choices[0].Message.Reasoning).To(BeEmpty())
			case 100:
				Expect(out.Choices[0].Message.Content).To(BeEmpty())
			default:
				Expect(out.Choices[0].Message.Reasoning).NotTo(BeEmpty())
				Expect(out.Choices[0].Message.Content).NotTo(BeEmpty())
			}
		},
		Entry("no reasoning", 0),
		Entry("half the tokens", 50),
		Entry("every token", 100),
	)

	It("streams reasoning before content with usage unchanged", func() {
		client, err := startServerWithArgs(context.TODO(), []string{
			"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
			"--reasoning-percentage", "50", "--max-num-seqs", "4",
		})
		Expect(err).NotTo(HaveOccurred())
		prompt := "alpha beta gamma delta"
		body := `{"model":"` + common.TestModelName + `","messages":[{"role":"user","content":"` + prompt + `"}],"max_tokens":20,"stream":true,"stream_options":{"include_usage":true},"n":2}`
		resp, err := client.Post(baseURL+"/chat/completions", "application/json", strings.NewReader(body))
		Expect(err).NotTo(HaveOccurred())
		defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		scanner := bufio.NewScanner(resp.Body)
		reason := map[int]string{}
		content := map[int]string{}
		seenContent := map[int]bool{}
		var usage int
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
				continue
			}
			var frame struct {
				Choices []struct {
					Index int `json:"index"`
					Delta struct {
						Reasoning string `json:"reasoning"`
						Content   string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
				Usage *struct {
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
			}
			Expect(json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame)).To(Succeed())
			if frame.Usage != nil {
				usage = frame.Usage.CompletionTokens
			}
			for _, choice := range frame.Choices {
				if choice.Delta.Reasoning != "" {
					Expect(seenContent[choice.Index]).To(BeFalse())
					reason[choice.Index] += choice.Delta.Reasoning
				}
				if choice.Delta.Content != "" {
					seenContent[choice.Index] = true
					content[choice.Index] += choice.Delta.Content
				}
			}
		}
		Expect(scanner.Err()).NotTo(HaveOccurred())
		Expect(reason).To(HaveLen(2))
		Expect(content).To(HaveLen(2))
		for i := 0; i < 2; i++ {
			Expect(reason[i] + content[i]).To(Equal(prompt))
		}
		Expect(usage).To(BeNumerically(">", 0))
	})
})

// reasoningMessage is one assistant message of a non-streamed chat completion,
// with both field names an engine may use. Pointers tell an absent field from
// an empty one, which is what the field-name and structured-format specs need.
type reasoningMessage struct {
	Reasoning        *string           `json:"reasoning"`
	ReasoningContent *string           `json:"reasoning_content"`
	Content          string            `json:"content"`
	ToolCalls        []json.RawMessage `json:"tool_calls"`
}

// postChatCompletions posts body to /v1/chat/completions with the given extra
// headers and returns the response, leaving the body for the caller to read.
func postChatCompletions(client *http.Client, body string, headers map[string]string) *http.Response {
	req, err := http.NewRequest("POST", baseURL+"/chat/completions", strings.NewReader(body))
	Expect(err).NotTo(HaveOccurred())
	req.Header.Set("Content-Type", "application/json")
	for name, value := range headers {
		req.Header.Set(name, value)
	}
	resp, err := client.Do(req)
	Expect(err).NotTo(HaveOccurred())
	return resp
}

// readReasoningMessage decodes the first choice's message of a non-streamed
// chat completion.
func readReasoningMessage(resp *http.Response) reasoningMessage {
	defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
	data, err := io.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	var out struct {
		Choices []struct {
			Message reasoningMessage `json:"message"`
		} `json:"choices"`
	}
	Expect(json.Unmarshal(data, &out)).To(Succeed())
	Expect(out.Choices).To(HaveLen(1))
	return out.Choices[0].Message
}

// startEchoWithReasoning starts an echo-mode server that reserves pct of each
// chat completion's tokens for reasoning.
func startEchoWithReasoning(pct int) *http.Client {
	client, err := startServerWithArgs(context.TODO(), []string{
		"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
		"--reasoning-percentage", strconv.Itoa(pct),
	})
	Expect(err).NotTo(HaveOccurred())
	return client
}

// echoReasoningBody is a chat completion request over a four-word prompt, so a
// share of it lands in reasoning and the rest in content.
func echoReasoningBody() string {
	return `{"model":"` + common.TestModelName +
		`","messages":[{"role":"user","content":"alpha beta gamma delta"}],"max_tokens":20}`
}

var _ = Describe("chat completion reasoning controls", func() {
	It("honors X-Reasoning-Percentage for one request only", func() {
		client := startEchoWithReasoning(0)
		body := echoReasoningBody()

		withHeader := readReasoningMessage(postChatCompletions(client, body,
			map[string]string{communication.XReasoningPercentageHeader: "100"}))
		Expect(withHeader.Reasoning).NotTo(BeNil())
		Expect(*withHeader.Reasoning).To(Equal("alpha beta gamma delta"))
		Expect(withHeader.Content).To(BeEmpty())

		// The header covered that request alone: the configured 0 still applies.
		withoutHeader := readReasoningMessage(postChatCompletions(client, body, nil))
		Expect(withoutHeader.Reasoning).To(BeNil())
		Expect(withoutHeader.Content).To(Equal("alpha beta gamma delta"))
	})

	It("rejects an X-Reasoning-Percentage outside 0-100", func() {
		client := startEchoWithReasoning(0)
		for _, value := range []string{"-1", "101", "abc", "50.5", ""} {
			resp := postChatCompletions(client, echoReasoningBody(),
				map[string]string{communication.XReasoningPercentageHeader: value})
			Expect(resp.Body.Close()).To(Succeed())
			if value == "" {
				// An empty header is indistinguishable from an absent one.
				Expect(resp.StatusCode).To(Equal(http.StatusOK))
				continue
			}
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
		}
	})

	It("applies a reasoning-percentage update from POST /admin/config", func() {
		client := startEchoWithReasoning(0)
		body := echoReasoningBody()

		Expect(readReasoningMessage(postChatCompletions(client, body, nil)).Reasoning).To(BeNil())

		resp := postAdminConfig(client, `{"reasoning-percentage":100}`)
		Expect(resp.Body.Close()).To(Succeed())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		updated := readReasoningMessage(postChatCompletions(client, body, nil))
		Expect(updated.Reasoning).NotTo(BeNil())
		Expect(*updated.Reasoning).To(Equal("alpha beta gamma delta"))
	})

	// The field name follows the engine, so a client that reads the wrong one
	// fails here the way it would against the real backend.
	DescribeTable("emits reasoning under the field the engine names",
		func(engineName string, expected, unexpected string) {
			client, err := startServerWithArgs(context.TODO(), []string{
				"cmd", "--engine", engineName, "--model", common.TestModelName,
				"--mode", common.ModeEcho, "--reasoning-percentage", "100",
			})
			Expect(err).NotTo(HaveOccurred())

			message := readReasoningMessage(postChatCompletions(client, echoReasoningBody(), nil))
			switch expected {
			case api.ReasoningField:
				Expect(message.Reasoning).NotTo(BeNil())
				Expect(*message.Reasoning).To(Equal("alpha beta gamma delta"))
			default:
				Expect(message.ReasoningContent).NotTo(BeNil())
				Expect(*message.ReasoningContent).To(Equal("alpha beta gamma delta"))
			}

			if unexpected == api.ReasoningField {
				Expect(message.Reasoning).To(BeNil())
			} else {
				Expect(message.ReasoningContent).To(BeNil())
			}
		},
		Entry("vllm", "vllm", api.ReasoningField, api.ReasoningContentField),
		Entry("sglang", "sglang", api.ReasoningContentField, api.ReasoningField),
	)

	// A reasoning prefix would land in front of a JSON value and break it, but a
	// plain-text answer can carry one -- including response_format: text, which
	// used to be treated as structured.
	DescribeTable("suppresses reasoning only for structured response formats",
		func(responseFormat string, wantReasoning bool) {
			client := startEchoWithReasoning(50)
			body := `{"model":"` + common.TestModelName +
				`","messages":[{"role":"user","content":"alpha beta gamma delta"}],"max_tokens":20,` +
				`"response_format":` + responseFormat + `}`

			message := readReasoningMessage(postChatCompletions(client, body, nil))
			if wantReasoning {
				Expect(message.Reasoning).NotTo(BeNil())
				Expect(*message.Reasoning).NotTo(BeEmpty())
				return
			}
			Expect(message.Reasoning).To(BeNil())
		},
		Entry("text", `{"type":"text"}`, true),
		Entry("json_object", `{"type":"json_object"}`, false),
		Entry("json_schema", `{"type":"json_schema","json_schema":{"name":"r","schema":`+
			`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}}`, false),
	)

	// A tool call choice carries no reasoning: the message is the call itself, and
	// a reasoning prefix would sit in front of the arguments the client awaits.
	It("adds no reasoning to a tool call", func() {
		client := startEchoWithReasoning(50)
		body := `{"model":"` + common.TestModelName +
			`","messages":[{"role":"user","content":"weather in Paris"}],"max_tokens":20,` +
			`"tool_choice":"required","tools":[{"type":"function","function":{"name":"get_weather",` +
			`"parameters":{"type":"object","properties":{"location":{"type":"string"}},` +
			`"required":["location"]}}}]}`

		message := readReasoningMessage(postChatCompletions(client, body, nil))
		Expect(message.ToolCalls).NotTo(BeEmpty())
		Expect(message.Reasoning).To(BeNil())
		Expect(message.ReasoningContent).To(BeNil())
	})

	// Random responses give every choice its own length, so a budget shared across
	// the choices of one request would starve the later ones. Echo mode cannot show
	// that, because every choice there has the same number of tokens.
	It("reserves reasoning for every choice of a multi-choice response", func() {
		client, err := startServerWithArgs(context.TODO(), []string{
			"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom,
			"--reasoning-percentage", "50", "--max-num-seqs", "4",
		})
		Expect(err).NotTo(HaveOccurred())

		body := `{"model":"` + common.TestModelName +
			`","messages":[{"role":"user","content":"alpha beta gamma delta"}],` +
			`"min_tokens":8,"max_tokens":40,"n":3}`
		resp := postChatCompletions(client, body, nil)
		defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
		data, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		var out struct {
			Choices []struct {
				Message reasoningMessage `json:"message"`
			} `json:"choices"`
		}
		Expect(json.Unmarshal(data, &out)).To(Succeed())
		Expect(out.Choices).To(HaveLen(3))
		for i, choice := range out.Choices {
			Expect(choice.Message.Reasoning).NotTo(BeNil(), "choice %d", i)
			Expect(*choice.Message.Reasoning).NotTo(BeEmpty(), "choice %d", i)
			Expect(choice.Message.Content).NotTo(BeEmpty(), "choice %d", i)
		}
	})

	// With a stream interval above one, a single batch can straddle the boundary
	// and still has to be split. The reasoning frame is then the only place where
	// several reasoning tokens share one frame.
	It("splits a multi-token chunk at the reasoning boundary", func() {
		client, err := startServerWithArgs(context.TODO(), []string{
			"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
			"--reasoning-percentage", "50", "--stream-interval", "3", "--max-num-seqs", "2",
		})
		Expect(err).NotTo(HaveOccurred())

		// Six echo tokens put three in reasoning. The first token streams alone, then
		// batches of three, so the boundary lands inside the second batch and that
		// batch has to yield a two-token reasoning frame plus a one-token content one.
		frames := collectChatStream(client, `{"model":"`+common.TestModelName+
			`","messages":[{"role":"user","content":"alpha beta gamma delta epsilon zeta"}],"max_tokens":20,`+
			`"stream":true,"logprobs":true,"top_logprobs":1}`)

		maxReasoning := 0
		for _, frame := range frames {
			if len(frame.Choices) == 0 {
				continue
			}
			choice := frame.Choices[0]
			if choice.Delta.Reasoning == "" || choice.Logprobs == nil {
				continue
			}
			if n := len(choice.Logprobs.Content); n > maxReasoning {
				maxReasoning = n
			}
		}
		Expect(maxReasoning).To(BeNumerically(">", 1),
			"a stream interval of 3 must put several reasoning tokens in one frame")
	})

	// Numbering continues across the split, so the logprobs streamed frame by frame
	// match the ones a non-streamed response reports for the whole output.
	It("numbers logprobs continuously across the split", func() {
		client, err := startServerWithArgs(context.TODO(), []string{
			"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
			"--reasoning-percentage", "50", "--stream-interval", "3", "--max-num-seqs", "2",
		})
		Expect(err).NotTo(HaveOccurred())

		base := `{"model":"` + common.TestModelName +
			`","messages":[{"role":"user","content":"alpha beta gamma delta"}],"max_tokens":20,` +
			`"logprobs":true,"top_logprobs":2`

		var streamed []logprobEntry
		for _, frame := range collectChatStream(client, base+`,"stream":true}`) {
			for _, choice := range frame.Choices {
				if choice.Logprobs != nil {
					streamed = append(streamed, choice.Logprobs.Content...)
				}
			}
		}

		whole, message := readWholeChatLogprobs(client, base+`}`)
		Expect(message.Reasoning).NotTo(BeNil())
		Expect(message.Content).NotTo(BeEmpty())
		Expect(streamed).To(HaveLen(len(whole)))
		for i := range whole {
			Expect(streamed[i].Token).To(Equal(whole[i].Token), "token %d", i)
			Expect(streamed[i].Logprob).To(BeNumerically("~", whole[i].Logprob, 1e-9), "logprob %d", i)
		}
	})
})

// logprobEntry is one token's logprob, as both the streamed and the non-streamed
// chat completion report it.
type logprobEntry struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

// chatStreamFrame is one SSE frame of a streamed chat completion, reduced to the
// per-choice fields the reasoning specs look at.
type chatStreamFrame struct {
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Reasoning string `json:"reasoning"`
			Content   string `json:"content"`
		} `json:"delta"`
		Logprobs *struct {
			Content []logprobEntry `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
}

// collectChatStream posts body as a streaming chat completion and returns its
// JSON frames, skipping the [DONE] sentinel.
func collectChatStream(client *http.Client, body string) []chatStreamFrame {
	resp := postChatCompletions(client, body, nil)
	defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
	Expect(resp.StatusCode).To(Equal(http.StatusOK))

	var frames []chatStreamFrame
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var frame chatStreamFrame
		Expect(json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame)).To(Succeed())
		frames = append(frames, frame)
	}
	Expect(scanner.Err()).NotTo(HaveOccurred())
	return frames
}

// readWholeChatLogprobs decodes the message and the logprobs of one non-streamed
// chat completion.
func readWholeChatLogprobs(client *http.Client, body string) ([]logprobEntry, reasoningMessage) {
	resp := postChatCompletions(client, body, nil)
	defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
	data, err := io.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	var out struct {
		Choices []struct {
			Message  reasoningMessage `json:"message"`
			Logprobs *struct {
				Content []logprobEntry `json:"content"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	Expect(json.Unmarshal(data, &out)).To(Succeed())
	Expect(out.Choices).To(HaveLen(1))
	if out.Choices[0].Logprobs == nil {
		return nil, out.Choices[0].Message
	}
	return out.Choices[0].Logprobs.Content, out.Choices[0].Message
}
