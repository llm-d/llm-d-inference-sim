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
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

var _ = Describe("Data Parallel", func() {
	It("Start with data-parallel-size=3 serves requests on all ranks", func() {
		ctx := context.TODO()

		clients, err := startDataParallelServers(ctx, []string{
			"cmd",
			"--model", common.TestModelName,
			"--mode", common.ModeRandom,
			"--data-parallel-size", "3",
			"--force-dummy-tokenizer",
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(clients).To(HaveLen(3))

		for rank, httpClient := range clients {
			c := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(httpClient),
				option.WithMaxRetries(0))
			resp, err := c.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("hello from rank"),
				},
				Model: common.TestModelName,
			})
			Expect(err).NotTo(HaveOccurred(), "rank %d request failed", rank)
			Expect(resp.Choices).ShouldNot(BeEmpty(), "rank %d got empty choices", rank)
		}
	})

	It("ZMQ kv-events are published for every rank with data-parallel-size=3", func() {
		ctx := context.TODO()
		model := common.QwenModelName
		longPrompt := "This is a test message for kv cache events, has to be long enough to be tokenized into multiple blocks."

		// Each data-parallel rank serves on its own port (baseServingPort + rank)
		// and therefore publishes KV events on a per-rank topic
		// kv@<ip>:<port>@<model>, so the EPP block index can distinguish ranks.
		// Rank 0's ZMQ PUB endpoint gets a random port; ranks 1 and 2 get port+1
		// and port+2 respectively (the simulator calls
		// common.OffsetEndpointPort(base, rank) for ranks > 0).
		const baseServingPort = 8000
		topic0 := kvcache.CreateKVEventsTopic("localhost", baseServingPort, model)
		sub0, zmqEndpoint := common.CreateSub(ctx, topic0)
		defer sub0.Close() //nolint:errcheck

		// Derive fixed ZMQ endpoints for rank 1 and rank 2 from rank 0's port.
		lastColon := strings.LastIndex(zmqEndpoint, ":")
		Expect(lastColon).To(BeNumerically(">", 0))
		baseZMQPort, err := strconv.Atoi(zmqEndpoint[lastColon+1:])
		Expect(err).NotTo(HaveOccurred())

		topic1 := kvcache.CreateKVEventsTopic("localhost", baseServingPort+1, model)
		sub1 := common.NewSub(ctx)
		common.StartSub(sub1, fmt.Sprintf("tcp://*:%d", baseZMQPort+1), topic1)
		defer sub1.Close() //nolint:errcheck

		topic2 := kvcache.CreateKVEventsTopic("localhost", baseServingPort+2, model)
		sub2 := common.NewSub(ctx)
		common.StartSub(sub2, fmt.Sprintf("tcp://*:%d", baseZMQPort+2), topic2)
		defer sub2.Close() //nolint:errcheck

		clients, err := startDataParallelServers(ctx, []string{
			"cmd",
			"--model", model,
			"--mode", common.ModeRandom,
			"--data-parallel-size", "3",
			"--port", strconv.Itoa(baseServingPort),
			"--force-dummy-tokenizer",
			"--enable-kvcache", "true", "--kv-cache-size", "16", "--block-size", "8",
			"--event-batch-size", "1", "--zmq-endpoint", zmqEndpoint,
		}, map[string]string{"POD_IP": "localhost"})
		Expect(err).NotTo(HaveOccurred())
		Expect(clients).To(HaveLen(3))

		// Send one request to each rank concurrently.
		for rank, httpClient := range clients {
			c := openai.NewClient(
				option.WithBaseURL(baseURL),
				option.WithHTTPClient(httpClient),
				option.WithMaxRetries(0))

			go func() {
				time.Sleep(200 * time.Millisecond)
				resp, err := c.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.UserMessage(longPrompt),
					},
					Model: model,
				})
				Expect(err).NotTo(HaveOccurred(), "rank %d request failed", rank)
				Expect(resp.Choices).ShouldNot(BeEmpty(), "rank %d got empty choices", rank)
			}()
		}

		// Assert that each rank published at least one store event on its own
		// per-rank topic, and that the DataParallelRank embedded in the batch
		// matches the rank index.
		rankTopics := []string{topic0, topic1, topic2}
		for expectedRank, sub := range []zmq4.Socket{sub0, sub1, sub2} {
			msg, err := sub.Recv()
			Expect(err).NotTo(HaveOccurred())
			storedCount, removedCount, _ := kvcache.CountKVEventBlocks(msg.Frames, rankTopics[expectedRank], 1)
			Expect(storedCount).To(BeNumerically(">", 0), "expected store events from rank %d", expectedRank)
			Expect(removedCount).To(Equal(0))
			Expect(kvcache.ParseKVBatchRank(msg.Frames)).To(Equal(expectedRank),
				"DataParallelRank mismatch for rank %d", expectedRank)
		}
	})
})
