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
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/valyala/fasthttp"
)

var _ = Describe("Simulator", func() {

	Context("responses API", func() {
		responseParts := []string{
			api.ResponsesEventCreated,
			api.ResponsesEventInProgress,
			api.ResponsesEventOutputItemAdded,
			api.ResponsesEventContentPartAdded,
			api.ResponsesEventTextDelta,
			api.ResponsesEventTextDone,
			api.ResponsesEventContentPartDone,
			api.ResponsesEventOutputItemDone,
			api.ResponsesEventCompleted}

		DescribeTable("responses with string and array input",
			func(model string, mode string, useStringInput bool) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", model, "--mode", mode}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClientAndResponsesParams(client, model, testUserMessage)
				if !useStringInput {
					params.Input = responses.ResponseNewParamsInputUnion{
						OfInputItemList: responses.ResponseInputParam{
							responses.ResponseInputItemUnionParam{
								OfMessage: &responses.EasyInputMessageParam{
									Role:    responses.EasyInputMessageRoleUser,
									Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(testUserMessage)},
								},
							},
						},
					}
				}

				resp, err := openaiclient.Responses.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).NotTo(BeNil())

				outputText := resp.OutputText()
				Expect(outputText).NotTo(BeEmpty())
				if mode == common.ModeEcho {
					Expect(outputText).To(Equal(testUserMessage))
				} else {
					Expect(dataset.IsValidText(outputText)).To(BeTrue())
				}

				Expect(resp.Usage.InputTokens).To(BeNumerically(">", 0))
				Expect(resp.Usage.OutputTokens).To(BeNumerically(">", 0))
				Expect(resp.Usage.TotalTokens).To(Equal(resp.Usage.InputTokens + resp.Usage.OutputTokens))

				Expect(resp.ID).To(HavePrefix(api.ResponsesIDPrefix))
				Expect(resp.Status).To(Equal(responses.ResponseStatusCompleted))
				Expect(resp.Instructions.AsString()).To(BeEmpty())

				Expect(resp.Output).NotTo(BeEmpty())
				firstItem := resp.Output[0]
				Expect(string(firstItem.Role)).To(Equal("assistant"))
				Expect(firstItem.Content).NotTo(BeEmpty())
				Expect(firstItem.Content[0].Type).To(Equal(api.ResponsesOutputText))
			},
			func(model string, mode string, useStringInput bool) string {
				inputType := "array"
				if useStringInput {
					inputType = "string"
				}
				return fmt.Sprintf("model: %s mode: %s input: %s", model, mode, inputType)
			},
			Entry(nil, common.TestModelName, common.ModeRandom, true),
			Entry(nil, common.TestModelName, common.ModeEcho, true),
			Entry(nil, common.TestModelName, common.ModeRandom, false),
			Entry(nil, common.TestModelName, common.ModeEcho, false),
			Entry(nil, common.QwenModelName, common.ModeRandom, true),
			Entry(nil, common.QwenModelName, common.ModeEcho, true),
			Entry(nil, common.QwenModelName, common.ModeRandom, false),
			Entry(nil, common.QwenModelName, common.ModeEcho, false),
		)

		It("Should echo instructions in the response", func() {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			const testInstructions = "Reply in French"
			openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, testUserMessage, testInstructions)

			resp, err := openaiclient.Responses.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Instructions.AsString()).To(Equal(testInstructions))
		})

		It("Should respect max_output_tokens", func() {
			ctx := context.TODO()
			server, _, client, err := startServerHandle(ctx, common.ModeRandom, nil, nil)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, testUserMessage)
			params.MaxOutputTokens = param.NewOpt(int64(2))

			resp, err := openaiclient.Responses.New(ctx, params)
			Expect(err).NotTo(HaveOccurred())

			outputText := resp.OutputText()
			_, tokens, err := server.Context.Tokenizer.RenderText(outputText)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically("<=", 2))
		})

		It("Should return error for invalid model", func() {
			ctx := context.TODO()
			client, err := startServer(ctx, common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			openaiclient, params := getOpenAIClientAndResponsesParams(client, "nonexistent-model", testUserMessage)
			_, err = openaiclient.Responses.New(ctx, params)
			Expect(err).To(HaveOccurred())
			var openaiError *openai.Error
			Expect(errors.As(err, &openaiError)).To(BeTrue())
			Expect(openaiError.StatusCode).To(Equal(fasthttp.StatusNotFound))
		})

		DescribeTable("responses streaming",
			func(model string, mode string) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", model, "--mode", mode}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClientAndResponsesParams(client, model, testUserMessage)

				stream := openaiclient.Responses.NewStreaming(ctx, params)
				defer func() {
					Expect(stream.Close()).NotTo(HaveOccurred())
				}()

				var eventTypes []string
				var deltas []string

				for stream.Next() {
					event := stream.Current()
					eventTypes = append(eventTypes, event.Type)
					switch event.Type {
					case api.ResponsesEventCreated:
						created := event.AsResponseCreated()
						Expect(string(created.Response.Status)).To(Equal(api.ResponsesStatusInProgress))
					case api.ResponsesEventOutputItemAdded:
						added := event.AsResponseOutputItemAdded()
						Expect(added.OutputIndex).To(Equal(int64(0)))
					case api.ResponsesEventTextDelta:
						delta := event.AsResponseOutputTextDelta()
						deltas = append(deltas, delta.Delta)
					case api.ResponsesEventTextDone:
						done := event.AsResponseOutputTextDone()
						Expect(done.Text).NotTo(BeEmpty())
						Expect(done.Text).To(Equal(strings.Join(deltas, "")))
					case api.ResponsesEventCompleted:
						completed := event.AsResponseCompleted()
						Expect(completed.Response.Usage.InputTokens).To(BeNumerically(">", 0))
						Expect(completed.Response.Usage.OutputTokens).To(BeNumerically(">", 0))
						Expect(completed.Response.Usage.TotalTokens).To(Equal(
							completed.Response.Usage.InputTokens + completed.Response.Usage.OutputTokens))
						Expect(string(completed.Response.Status)).To(Equal(api.ResponsesStatusCompleted))
					}
				}
				Expect(stream.Err()).NotTo(HaveOccurred())

				// Verify the mandatory fixed positions in the event sequence:
				// [0] created, [1] in_progress, [2] output_item.added, [3] content_part.added,
				// [4..n-5] deltas, [n-4] text.done, [n-3] content_part.done,
				// [n-2] output_item.done, [n-1] completed
				Expect(len(eventTypes)).To(BeNumerically(">=", 9), "expected at least 9 events")
				Expect(eventTypes[0]).To(Equal(api.ResponsesEventCreated))
				Expect(eventTypes[1]).To(Equal(api.ResponsesEventInProgress))
				Expect(eventTypes[2]).To(Equal(api.ResponsesEventOutputItemAdded))
				Expect(eventTypes[3]).To(Equal(api.ResponsesEventContentPartAdded))
				// deltas occupy positions [4 .. len-5]
				nDeltas := len(eventTypes) - 8
				Expect(nDeltas).To(BeNumerically(">=", 1), "expected at least one delta event")
				for i := 4; i < 4+nDeltas; i++ {
					Expect(eventTypes[i]).To(Equal(api.ResponsesEventTextDelta))
				}
				Expect(eventTypes[len(eventTypes)-4]).To(Equal(api.ResponsesEventTextDone))
				Expect(eventTypes[len(eventTypes)-3]).To(Equal(api.ResponsesEventContentPartDone))
				Expect(eventTypes[len(eventTypes)-2]).To(Equal(api.ResponsesEventOutputItemDone))
				Expect(eventTypes[len(eventTypes)-1]).To(Equal(api.ResponsesEventCompleted))

				fullText := strings.Join(deltas, "")
				if mode == common.ModeEcho {
					Expect(fullText).To(Equal(testUserMessage))
				} else {
					Expect(dataset.IsValidText(fullText)).To(BeTrue())
				}
			},
			func(model string, mode string) string {
				return fmt.Sprintf("model: %s mode: %s", model, mode)
			},
			Entry(nil, common.TestModelName, common.ModeRandom),
			Entry(nil, common.TestModelName, common.ModeEcho),
			Entry(nil, common.QwenModelName, common.ModeRandom),
			Entry(nil, common.QwenModelName, common.ModeEcho),
		)

		DescribeTable("responses with logprobs",
			func(includeLogprobs bool, topLogprobs int) {
				ctx := context.TODO()
				client, err := startServer(ctx, common.ModeEcho)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, testUserMessage)
				if includeLogprobs {
					params.Include = []responses.ResponseIncludable{responses.ResponseIncludableMessageOutputTextLogprobs}
					if topLogprobs > 0 {
						params.TopLogprobs = param.NewOpt(int64(topLogprobs))
					}
				}

				resp, err := openaiclient.Responses.New(ctx, params)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).NotTo(BeNil())
				Expect(resp.Output).NotTo(BeEmpty())

				contentItem := resp.Output[0].Content[0]
				if includeLogprobs {
					Expect(contentItem.Logprobs).NotTo(BeEmpty())
					Expect(contentItem.Logprobs[0].Token).NotTo(BeEmpty())
					Expect(contentItem.Logprobs[0].Logprob).To(BeNumerically("<=", 0))
					if topLogprobs > 0 {
						Expect(contentItem.Logprobs[0].TopLogprobs).To(HaveLen(topLogprobs))
						Expect(contentItem.Logprobs[0].TopLogprobs[0].Token).To(Equal(contentItem.Logprobs[0].Token))
					} else {
						Expect(contentItem.Logprobs[0].TopLogprobs).To(BeEmpty())
					}
				} else {
					Expect(contentItem.Logprobs).To(BeEmpty())
				}
			},
			func(includeLogprobs bool, topLogprobs int) string {
				return fmt.Sprintf("includeLogprobs: %t top_logprobs: %d", includeLogprobs, topLogprobs)
			},
			Entry(nil, true, 0),  // logprobs requested, no top alternatives
			Entry(nil, true, 2),  // logprobs requested, 2 top alternatives
			Entry(nil, false, 0), // logprobs not requested
		)

		DescribeTable("responses streaming with logprobs",
			func(includeLogprobs bool, topLogprobs int) {
				ctx := context.TODO()
				client, err := startServer(ctx, common.ModeEcho)
				Expect(err).NotTo(HaveOccurred())

				openaiclient, params := getOpenAIClientAndResponsesParams(client, common.TestModelName, testUserMessage)
				if includeLogprobs {
					params.Include = []responses.ResponseIncludable{responses.ResponseIncludableMessageOutputTextLogprobs}
					if topLogprobs > 0 {
						params.TopLogprobs = param.NewOpt(int64(topLogprobs))
					}
				}

				stream := openaiclient.Responses.NewStreaming(ctx, params)
				defer func() {
					Expect(stream.Close()).NotTo(HaveOccurred())
				}()

				deltaCount := 0
				deltaLogprobsCount := 0

				for stream.Next() {
					event := stream.Current()
					switch event.Type {
					case api.ResponsesEventTextDelta:
						delta := event.AsResponseOutputTextDelta()
						Expect(delta.Delta).NotTo(BeEmpty())
						deltaCount++
						if includeLogprobs {
							Expect(delta.JSON.Logprobs.Valid()).To(BeTrue(),
								"delta event should have logprobs field present")
							Expect(delta.Logprobs).NotTo(BeEmpty(),
								"delta event should have non-empty logprobs when requested")
							for _, lp := range delta.Logprobs {
								Expect(lp.Token).NotTo(BeEmpty())
								Expect(lp.Logprob).To(BeNumerically("<=", 0))
								if topLogprobs > 0 {
									Expect(lp.TopLogprobs).To(HaveLen(topLogprobs))
								}
							}
							deltaLogprobsCount++
						} else {
							Expect(delta.Logprobs).To(BeEmpty(),
								"delta event should have no logprobs when not requested")
						}
					case api.ResponsesEventTextDone:
						done := event.AsResponseOutputTextDone()
						Expect(done.Text).NotTo(BeEmpty())
						if includeLogprobs {
							Expect(done.JSON.Logprobs.Valid()).To(BeTrue(),
								"done event should have logprobs field present (as [])")
							Expect(done.Logprobs).To(BeEmpty(),
								"done event logprobs should be empty array, not populated")
						} else {
							Expect(done.JSON.Logprobs.Valid()).To(BeFalse(),
								"done event should not have logprobs field when not requested")
						}
					}
				}
				Expect(stream.Err()).NotTo(HaveOccurred())

				Expect(deltaCount).To(BeNumerically(">", 0), "should have received delta events")
				if includeLogprobs {
					Expect(deltaLogprobsCount).To(Equal(deltaCount),
						"all delta events should have logprobs when requested")
				}
			},
			func(includeLogprobs bool, topLogprobs int) string {
				return fmt.Sprintf("includeLogprobs: %t top_logprobs: %d", includeLogprobs, topLogprobs)
			},
			Entry(nil, true, 0),  // logprobs requested, no top alternatives
			Entry(nil, true, 2),  // logprobs requested, 2 top alternatives
			Entry(nil, false, 0), // logprobs not requested
		)

		DescribeTable("responses streaming logprobs per chunk type",
			func(includeLogprobs bool, topLogprobs int) {
				ctx := context.TODO()
				client, err := startServer(ctx, common.ModeEcho)
				Expect(err).NotTo(HaveOccurred())

				reqBody := fmt.Sprintf(`{"model":%q,"input":%q,"stream":true`, common.TestModelName, testUserMessage)
				if includeLogprobs {
					reqBody += `,"include":["message.output_text.logprobs"]`
					if topLogprobs > 0 {
						reqBody += fmt.Sprintf(`,"top_logprobs":%d`, topLogprobs)
					}
				}
				reqBody += "}"

				req, err := http.NewRequest("POST", "http://localhost/v1/responses", strings.NewReader(reqBody))
				Expect(err).NotTo(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				httpResp, err := client.Do(req)
				Expect(err).NotTo(HaveOccurred())
				defer func() { Expect(httpResp.Body.Close()).To(Succeed()) }()
				Expect(httpResp.StatusCode).To(Equal(http.StatusOK))

				checkLogprobsMissing := func(part map[string]any, partType string) {
					_, ok := part["logprobs"]
					Expect(ok).To(BeFalse(), partType+": logprobs must be absent when not requested")
				}
				checkLogprobEmpty := func(partObj map[string]any, partType string, isNullExpected bool) {
					if includeLogprobs {
						logprobs, ok := partObj["logprobs"]
						Expect(ok).To(BeTrue(), partType+": part.logprobs must be present when requested")
						if isNullExpected {
							Expect(logprobs).To(BeNil(), partType+": part.logprobs must be null")
						} else {
							Expect(logprobs.([]any)).To(BeEmpty(), partType+": part.logprobs must be empty []")
						}
					} else {
						checkLogprobsMissing(partObj, partType)
					}
				}

				seenTypes := map[string]bool{}
				var deltaLogprobs []any
				reader := bufio.NewReader(httpResp.Body)
				for {
					line, err := reader.ReadString('\n')
					if err == io.EOF {
						break
					}
					Expect(err).NotTo(HaveOccurred())
					if !strings.HasPrefix(line, api.SSEDataPrefix) {
						continue
					}
					data := strings.TrimSpace(strings.TrimPrefix(line, api.SSEDataPrefix))
					if data == api.SSEDoneMarker {
						break
					}
					var event map[string]any
					Expect(json.Unmarshal([]byte(data), &event)).To(Succeed())
					eventType, _ := event["type"].(string)
					seenTypes[eventType] = true

					switch eventType {
					case api.ResponsesEventContentPartAdded:
						// part.logprobs: [] when requested, absent when not
						partObj, _ := event["part"].(map[string]any)
						checkLogprobEmpty(partObj, eventType, false)

					case api.ResponsesEventTextDelta:
						// logprobs: populated when requested, absent when not
						if includeLogprobs {
							logprobsArr, _ := event["logprobs"].([]any)
							Expect(logprobsArr).NotTo(BeEmpty(), "text.delta: logprobs must be non-empty")
							for _, lp := range logprobsArr {
								lpMap, _ := lp.(map[string]any)
								Expect(lpMap["token"]).NotTo(BeEmpty())
								Expect(lpMap["logprob"].(float64)).To(BeNumerically("<=", 0))
								if topLogprobs > 0 {
									Expect(lpMap["top_logprobs"].([]any)).To(HaveLen(topLogprobs))
								}
							}
							deltaLogprobs = append(deltaLogprobs, logprobsArr...)
						} else {
							checkLogprobsMissing(event, eventType)
						}

					case api.ResponsesEventTextDone:
						checkLogprobEmpty(event, eventType, false)

					case api.ResponsesEventContentPartDone:
						// part.logprobs: null when requested (signals per-token entries already streamed), absent when not
						partObj, _ := event["part"].(map[string]any)
						checkLogprobEmpty(partObj, eventType, true)

					case api.ResponsesEventOutputItemDone:
						// item.content[0].logprobs: null when requested, absent when not
						item, ok := event["item"].(map[string]any)
						Expect(ok).To(BeTrue(), "output_item.done: event.item must be a map")
						contentArr, ok := item["content"].([]any)
						Expect(ok).To(BeTrue(), "output_item.done: item.content must be an array")
						Expect(contentArr).NotTo(BeEmpty(), "output_item.done: item.content must not be empty")
						firstContent, ok := contentArr[0].(map[string]any)
						Expect(ok).To(BeTrue(), "output_item.done: item.content[0] must be a map")
						checkLogprobEmpty(firstContent, eventType, true)

					case api.ResponsesEventCompleted:
						// response.output[0].content[0].logprobs: accumulated entries when requested, absent when not
						response, ok := event["response"].(map[string]any)
						Expect(ok).To(BeTrue(), "completed: event.response must be a map")
						outputArr, ok := response["output"].([]any)
						Expect(ok).To(BeTrue(), "completed: response.output must be an array")
						Expect(outputArr).NotTo(BeEmpty(), "completed: response.output must not be empty")
						firstOutput, ok := outputArr[0].(map[string]any)
						Expect(ok).To(BeTrue(), "completed: response.output[0] must be a map")
						contentArr, ok := firstOutput["content"].([]any)
						Expect(ok).To(BeTrue(), "completed: response.output[0].content must be an array")
						Expect(contentArr).NotTo(BeEmpty(), "completed: response.output[0].content must not be empty")
						firstContent, ok := contentArr[0].(map[string]any)
						Expect(ok).To(BeTrue(), "completed: response.output[0].content[0] must be a map")
						if includeLogprobs {
							logprobsArr, _ := firstContent["logprobs"].([]any)
							Expect(logprobsArr).NotTo(BeEmpty(), "completed: content[0].logprobs must have accumulated entries")
							Expect(logprobsArr).To(HaveLen(len(deltaLogprobs)),
								"completed: accumulated logprobs count must equal sum of all delta logprobs")
							for i, lp := range logprobsArr {
								Expect(lp).To(Equal(deltaLogprobs[i]),
									"completed: logprobs[%d] must match the corresponding delta logprob entry", i)
							}
						} else {
							checkLogprobsMissing(firstContent, eventType)
						}
					}
				}

				// check that all chunk types were received
				for _, et := range responseParts {
					Expect(seenTypes[et]).To(BeTrue(), "event type %q was not received in the stream", et)
				}
			},
			func(includeLogprobs bool, topLogprobs int) string {
				return fmt.Sprintf("includeLogprobs: %t top_logprobs: %d", includeLogprobs, topLogprobs)
			},
			Entry(nil, true, 2),  // logprobs with 2 top alternatives
			Entry(nil, true, 0),  // logprobs with no top alternatives
			Entry(nil, false, 0), // no logprobs: logprobs fields must be absent in all chunks
		)
	})

	Context("responses API with multimodal content", func() {
		It("Should accept input_image content in responses request", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"input": [{
					"role": "user",
					"content": [
						{"type": "input_text", "text": "Describe this image"},
						{"type": "input_image", "image_url": "https://example.com/cat.jpg"}
					]
				}]
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/v1/responses", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var respObj map[string]any
			Expect(json.Unmarshal(body, &respObj)).To(Succeed())
			Expect(respObj["status"]).To(Equal(api.ResponsesStatusCompleted))
			Expect(respObj["id"]).To(HavePrefix(api.ResponsesIDPrefix))

			// Verify output contains text content
			output := respObj["output"].([]any)
			Expect(output).NotTo(BeEmpty())
			firstOutput := output[0].(map[string]any)
			Expect(firstOutput["role"]).To(Equal("assistant"))
			content := firstOutput["content"].([]any)
			Expect(content).NotTo(BeEmpty())
			firstContent := content[0].(map[string]any)
			Expect(firstContent["type"]).To(Equal(api.ResponsesOutputText))
			Expect(firstContent["text"]).NotTo(BeEmpty())

			// Verify usage
			usage := respObj["usage"].(map[string]any)
			Expect(usage["input_tokens"]).To(BeNumerically(">", 0))
			Expect(usage["output_tokens"]).To(BeNumerically(">", 0))
		})

		It("Should accept input_audio content in responses request", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"input": [{
					"role": "user",
					"content": [
						{"type": "input_text", "text": "Transcribe this audio"},
						{"type": "input_audio", "data": "base64encodedaudiodata", "format": "wav"}
					]
				}]
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/v1/responses", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var respObj map[string]any
			Expect(json.Unmarshal(body, &respObj)).To(Succeed())
			Expect(respObj["status"]).To(Equal(api.ResponsesStatusCompleted))

			// Verify output
			output := respObj["output"].([]any)
			Expect(output).NotTo(BeEmpty())
			firstOutput := output[0].(map[string]any)
			content := firstOutput["content"].([]any)
			Expect(content).NotTo(BeEmpty())
			firstContent := content[0].(map[string]any)
			Expect(firstContent["text"]).NotTo(BeEmpty())

			usage := respObj["usage"].(map[string]any)
			Expect(usage["input_tokens"]).To(BeNumerically(">", 0))
			Expect(usage["output_tokens"]).To(BeNumerically(">", 0))
		})

		It("Should accept mixed image, audio, and text content in responses request", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"input": [{
					"role": "user",
					"content": [
						{"type": "input_text", "text": "What do you see and hear?"},
						{"type": "input_image", "image_url": "https://example.com/photo.png"},
						{"type": "input_audio", "data": "audiobase64data", "format": "mp3"}
					]
				}]
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/v1/responses", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var respObj map[string]any
			Expect(json.Unmarshal(body, &respObj)).To(Succeed())
			Expect(respObj["status"]).To(Equal(api.ResponsesStatusCompleted))

			output := respObj["output"].([]any)
			Expect(output).NotTo(BeEmpty())

			usage := respObj["usage"].(map[string]any)
			Expect(usage["input_tokens"]).To(BeNumerically(">", 0))
			Expect(usage["output_tokens"]).To(BeNumerically(">", 0))
			Expect(usage["total_tokens"]).To(Equal(usage["input_tokens"].(float64) + usage["output_tokens"].(float64)))
		})

		It("Should echo image reference in echo mode", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeEcho}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"input": [{
					"role": "user",
					"content": [
						{"type": "input_text", "text": "Describe this"},
						{"type": "input_image", "image_url": "https://example.com/img.png"}
					]
				}]
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/v1/responses", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var respObj map[string]any
			Expect(json.Unmarshal(body, &respObj)).To(Succeed())

			// In echo mode the last message's plain text is echoed back.
			// PlainText for image content includes "image: <url>"
			output := respObj["output"].([]any)
			Expect(output).NotTo(BeEmpty())
			firstOutput := output[0].(map[string]any)
			content := firstOutput["content"].([]any)
			firstContent := content[0].(map[string]any)
			outputText := firstContent["text"].(string)
			Expect(outputText).To(ContainSubstring("Describe this"))
			Expect(outputText).To(ContainSubstring("image: https://example.com/img.png"))
		})

		It("Should stream responses with image content", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"stream": true,
				"input": [{
					"role": "user",
					"content": [
						{"type": "input_text", "text": "Describe this image"},
						{"type": "input_image", "image_url": "https://example.com/test.jpg"}
					]
				}]
			}`, common.TestModelName)

			req, err := http.NewRequest("POST", "http://localhost/v1/responses", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")

			httpResp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(httpResp.Body.Close()).To(Succeed()) }()
			Expect(httpResp.StatusCode).To(Equal(http.StatusOK))

			seenTypes := map[string]bool{}
			var deltas []string
			reader := bufio.NewReader(httpResp.Body)
			for {
				line, err := reader.ReadString('\n')
				if err == io.EOF {
					break
				}
				Expect(err).NotTo(HaveOccurred())
				if !strings.HasPrefix(line, api.SSEDataPrefix) {
					continue
				}
				data := strings.TrimSpace(strings.TrimPrefix(line, api.SSEDataPrefix))
				if data == api.SSEDoneMarker {
					break
				}
				var event map[string]any
				Expect(json.Unmarshal([]byte(data), &event)).To(Succeed())
				eventType, _ := event["type"].(string)
				seenTypes[eventType] = true

				if eventType == api.ResponsesEventTextDelta {
					delta, _ := event["delta"].(string)
					deltas = append(deltas, delta)
				}
			}

			Expect(seenTypes[api.ResponsesEventCreated]).To(BeTrue())
			Expect(seenTypes[api.ResponsesEventCompleted]).To(BeTrue())
			Expect(seenTypes[api.ResponsesEventTextDelta]).To(BeTrue())
			Expect(deltas).NotTo(BeEmpty())
			fullText := strings.Join(deltas, "")
			Expect(fullText).NotTo(BeEmpty())
		})

		It("Should reject unsupported content type in responses request", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"input": [{
					"role": "user",
					"content": [
						{"type": "input_video", "url": "https://example.com/video.mp4"}
					]
				}]
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/v1/responses", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(body)).To(ContainSubstring("unsupported input content type"))
		})
	})
})
