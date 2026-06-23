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
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Simulator", func() {

	Context("generate API", func() {
		DescribeTable("Should return correct response to /inference/v1/generate",
			func(model string) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", model, "--mode", common.ModeRandom}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				reqBody := fmt.Sprintf(`{
					"model": "%s",
					"token_ids": [1, 2, 3, 4],
					"sampling_params": {"max_tokens": 5}
				}`, model)
				resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					err := resp.Body.Close()
					Expect(err).NotTo(HaveOccurred())
				}()

				Expect(resp.StatusCode).To(Equal(http.StatusOK))

				body, err := io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())

				var generateResp api.GenerateResponse
				Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
				Expect(generateResp.GenRequestID).NotTo(BeEmpty())
				Expect(generateResp.Choices).To(HaveLen(1))
				Expect(generateResp.Choices[0].FinishReason).NotTo(BeNil())
				Expect(generateResp.Choices[0].TokenIDs).NotTo(BeEmpty())
				Expect(int64(len(generateResp.Choices[0].TokenIDs))).To(BeNumerically("<=", 5))
			},
			func(model string) string {
				return "model: " + model
			},
			Entry(nil, common.TestModelName),
			Entry(nil, common.QwenModelName),
		)

		DescribeTable("Should return 400 when required fields are missing",
			func(reqBody string, expectedErrMsg string) {
				ctx := context.TODO()
				args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
				client, err := startServerWithArgs(ctx, args)
				Expect(err).NotTo(HaveOccurred())

				resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					err := resp.Body.Close()
					Expect(err).NotTo(HaveOccurred())
				}()

				Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

				body, err := io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())
				Expect(string(body)).To(ContainSubstring(expectedErrMsg))
			},
			Entry("missing token_ids",
				fmt.Sprintf(`{"model": "%s", "sampling_params": {"max_tokens": 5}}`, common.TestModelName),
				"Missing input token_ids",
			),
			Entry("missing sampling_params",
				fmt.Sprintf(`{"model": "%s", "token_ids": [1, 2, 3]}`, common.TestModelName),
				"Missing sampling_params field",
			),
		)

		It("Should return ec_transfer_params in MMEncoderOnly mode when features are present", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mm-encoder-only"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 32000, 32000, 32000],
				"features": {
					"mm_hashes": {"image": ["abc123hash", "def456hash"]},
					"mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
					"kwargs_data": {"image": ["<base64-encoded-pixel-tensor-1>"]}
				},
				"sampling_params": {"max_tokens": 1}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.GenRequestID).NotTo(BeEmpty())
			Expect(generateResp.Choices).To(HaveLen(1))
			Expect(generateResp.ECTransferParams).To(HaveLen(2))
			Expect(generateResp.ECTransferParams).To(HaveKey("abc123hash"))
			Expect(generateResp.ECTransferParams).To(HaveKey("def456hash"))
			Expect(generateResp.ECTransferParams["abc123hash"].PeerPort).To(BeNumerically(">", 0))
		})

		It("Should not return ec_transfer_params when features are nil", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mm-encoder-only"}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 1}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.ECTransferParams).To(BeNil())
		})

		It("Should return kv_transfer_params when do_remote_decode is true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5},
				"kv_transfer_params": {"do_remote_decode": true}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.KVParams).NotTo(BeNil())
			Expect(generateResp.KVParams.DoRemotePrefill).To(BeTrue())
			Expect(generateResp.KVParams.DoRemoteDecode).To(BeFalse())
			Expect(generateResp.KVParams.RemoteHost).NotTo(BeEmpty())
			Expect(generateResp.KVParams.RemotePort).To(BeNumerically(">", 0))
			Expect(generateResp.KVParams.RemoteBlockIds).NotTo(BeEmpty())
			Expect(generateResp.KVParams.RemoteEngineId).NotTo(BeEmpty())
		})

		It("Should not return kv_transfer_params when do_remote_decode is absent", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.KVParams).To(BeNil())
		})

		It("Should return kv_transfer_params when do_remote_decode is true inside sampling_params.extra_args", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5, "extra_args": {"kv_transfer_params": {"do_remote_decode": true}}}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())

			var generateResp api.GenerateResponse
			Expect(json.Unmarshal(body, &generateResp)).To(Succeed())
			Expect(generateResp.KVParams).NotTo(BeNil())
			Expect(generateResp.KVParams.DoRemotePrefill).To(BeTrue())
			Expect(generateResp.KVParams.DoRemoteDecode).To(BeFalse())
			Expect(generateResp.KVParams.RemoteHost).NotTo(BeEmpty())
			Expect(generateResp.KVParams.RemotePort).To(BeNumerically(">", 0))
			Expect(generateResp.KVParams.RemoteBlockIds).NotTo(BeEmpty())
			Expect(generateResp.KVParams.RemoteEngineId).NotTo(BeEmpty())
		})

		It("Should stream SSE chunks for /inference/v1/generate with stream=true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5},
				"stream": true
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			Expect(resp.Header.Get("Content-Type")).To(Equal("text/event-stream"))

			reader := bufio.NewReader(resp.Body)
			var tokenChunks []api.GenerateStreamResponse
			var finishChunk *api.GenerateStreamResponse
			var usageChunk *api.GenerateStreamResponse
			gotDone := false

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
					gotDone = true
					break
				}

				var streamResp api.GenerateStreamResponse
				Expect(json.Unmarshal([]byte(data), &streamResp)).To(Succeed(), "failed to parse SSE chunk: %s", data)
				if len(streamResp.Choices) == 0 {
					Expect(streamResp.Usage).NotTo(BeNil(), "empty choices chunk must carry usage")
					usageChunk = &streamResp
					continue
				}
				choice := streamResp.Choices[0]
				if choice.TokenIDs != nil {
					tokenChunks = append(tokenChunks, streamResp)
				}
				if choice.FinishReason != nil {
					finishChunk = &streamResp
				}
			}

			Expect(tokenChunks).NotTo(BeEmpty(), "should have received at least one streaming chunk with token_ids")
			for _, tc := range tokenChunks {
				Expect(tc.RequestID).NotTo(BeEmpty())
				Expect(tc.Choices[0].TokenIDs).NotTo(BeEmpty())
			}

			Expect(finishChunk).NotTo(BeNil(), "should have received a chunk with finish_reason")
			Expect(finishChunk.RequestID).NotTo(BeEmpty())
			Expect(*finishChunk.Choices[0].FinishReason).NotTo(BeEmpty())
			Expect(finishChunk.Choices[0].TokenIDs).NotTo(BeNil(), "finish_reason must be in the last token chunk, not a separate empty chunk")
			Expect(finishChunk.Usage).To(BeNil(), "finish chunk should not carry usage")

			Expect(usageChunk).To(BeNil(), "should not receive usage chunk without stream_options.include_usage")

			Expect(gotDone).To(BeTrue(), "stream should end with [DONE]")
		})

		It("Should send length finish_reason in last token chunk for generate streaming", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			// max_tokens: 1 forces length finish reason
			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 1},
				"stream": true
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			reader := bufio.NewReader(resp.Body)
			var lastTokenChunk *api.GenerateStreamResponse
			gotDone := false

			for {
				line, err := reader.ReadString('\n')
				if err == io.EOF {
					break
				}
				Expect(err).NotTo(HaveOccurred())

				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				data := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
				if data == api.SSEDoneMarker {
					gotDone = true
					break
				}

				var streamResp api.GenerateStreamResponse
				Expect(json.Unmarshal([]byte(data), &streamResp)).To(Succeed(), "failed to parse SSE chunk: %s", data)
				if len(streamResp.Choices) == 0 {
					continue
				}
				choice := streamResp.Choices[0]
				if choice.TokenIDs != nil {
					lastTokenChunk = &streamResp
				}
				// finish_reason must never appear in a separate empty chunk
				if choice.FinishReason != nil {
					Expect(choice.TokenIDs).NotTo(BeNil(), "finish_reason must be carried by a token chunk, not a separate empty chunk")
				}
			}

			Expect(lastTokenChunk).NotTo(BeNil(), "should have received at least one token chunk")
			Expect(lastTokenChunk.Choices[0].FinishReason).NotTo(BeNil(), "last token chunk should carry finish_reason")
			Expect(*lastTokenChunk.Choices[0].FinishReason).To(Equal(common.LengthFinishReason))
			Expect(gotDone).To(BeTrue(), "stream should end with [DONE]")
		})

		It("Should include usage chunk when stream_options.include_usage is true", func() {
			ctx := context.TODO()
			args := []string{"cmd", "--model", common.TestModelName, "--mode", common.ModeRandom}
			client, err := startServerWithArgs(ctx, args)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{
				"model": "%s",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 5},
				"stream": true,
				"stream_options": {"include_usage": true}
			}`, common.TestModelName)

			resp, err := client.Post("http://localhost/inference/v1/generate", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				err := resp.Body.Close()
				Expect(err).NotTo(HaveOccurred())
			}()

			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			reader := bufio.NewReader(resp.Body)
			var usageChunk *api.GenerateStreamResponse
			gotDone := false

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
					gotDone = true
					break
				}

				var streamResp api.GenerateStreamResponse
				Expect(json.Unmarshal([]byte(data), &streamResp)).To(Succeed(), "failed to parse SSE chunk: %s", data)
				if len(streamResp.Choices) == 0 {
					Expect(streamResp.Usage).NotTo(BeNil(), "empty choices chunk must carry usage")
					usageChunk = &streamResp
					continue
				}
			}

			Expect(usageChunk).NotTo(BeNil(), "should have received a usage chunk with choices:[]")
			Expect(usageChunk.Choices).To(BeEmpty(), "usage chunk should have empty choices")
			Expect(usageChunk.Usage).NotTo(BeNil())
			Expect(usageChunk.Usage.PromptTokens).To(BeNumerically(">", 0))
			Expect(usageChunk.Usage.CompletionTokens).To(BeNumerically(">", 0))

			Expect(gotDone).To(BeTrue(), "stream should end with [DONE]")
		})
	})

})
