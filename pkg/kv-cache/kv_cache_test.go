/*
Copyright 2025 The llm-d-inference-sim Authors.

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

package kvcache

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	zmq "github.com/pebbe/zmq4"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvevents"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const (
	req1ID           = "req1"
	req2ID           = "req2"
	req3ID           = "req3"
	wildcardEndpoint = "tcp://*:*"
)

type ActionType int

const (
	actionStartRequest ActionType = iota
	actionFinishRequest
)

type testRequest struct {
	id     string
	blocks []uint64
}

type expectedBlockInfo struct {
	exists   bool
	refCount int
}

type testAction struct {
	action                 ActionType
	request                testRequest
	isError                bool
	errMsg                 string
	expectedActiveRequests int
	expectedTotalBlocks    int
	expectedUnusedBlocks   int
	expectedBlocksInfo     map[uint64]expectedBlockInfo
}

func newStartAction(request testRequest) testAction {
	return testAction{
		action:                 actionStartRequest,
		request:                request,
		isError:                false,
		expectedActiveRequests: -1,
		expectedTotalBlocks:    -1,
		expectedUnusedBlocks:   -1,
	}
}
func newInvalidTestAction(action ActionType, request testRequest, errMsg string) testAction {
	return testAction{
		action:                 action,
		request:                request,
		isError:                true,
		errMsg:                 errMsg,
		expectedActiveRequests: -1,
		expectedTotalBlocks:    -1,
		expectedUnusedBlocks:   -1,
	}
}
func newTestActionWithExpectedValues(action ActionType, request testRequest, expectedActiveRequests int,
	expectedTotalBlocks int, expectedUnusedBlocks int, expectedBlocksInfo map[uint64]expectedBlockInfo) testAction {
	return testAction{
		action:                 action,
		request:                request,
		isError:                false,
		expectedActiveRequests: expectedActiveRequests,
		expectedTotalBlocks:    expectedTotalBlocks,
		expectedUnusedBlocks:   expectedUnusedBlocks,
		expectedBlocksInfo:     expectedBlocksInfo,
	}
}

type testCase struct {
	name                  string
	cacheSize             int
	actions               []testAction
	expectedRemovedBlocks int
	expectedStoredBlocks  int
}

type threadTestCase struct {
	name              string
	cacheSize         int
	numGoroutines     int
	numOperations     int
	minBlockLen       int
	maxBlockLen       int
	maxHashValue      uint64
	shouldUseAllCache bool
}

var _ = Describe("KV cache", Ordered, func() {
	common.InitRandom(time.Now().UnixNano())

	Context("general tests", func() {
		// check single request processing, ensure cache is valid after request processing started
		// and after the processing was finished
		req1 := testRequest{req1ID, []uint64{1, 2}}
		req2 := testRequest{req2ID, []uint64{3, 4}}
		req2_1 := testRequest{req2ID, []uint64{1, 3}}
		req3 := testRequest{req3ID, []uint64{5, 6}}

		testCases := []testCase{
			{
				name:      "single request",
				cacheSize: 3,
				actions: []testAction{
					newTestActionWithExpectedValues(actionStartRequest, req1, 1, 2, 0, nil),
					newTestActionWithExpectedValues(actionFinishRequest, req1, 0, 2, 2, nil),
				},
				expectedRemovedBlocks: 0,
				expectedStoredBlocks:  2,
			},
			{
				name:      "two requests",
				cacheSize: 5,
				actions: []testAction{
					newStartAction(req1),
					newTestActionWithExpectedValues(actionStartRequest, req2, 2, 4, 0, nil),
					newTestActionWithExpectedValues(actionFinishRequest, req1, 1, 4, 2, nil),
					newTestActionWithExpectedValues(actionFinishRequest, req2, 0, 4, 4, nil),
				},
				expectedRemovedBlocks: 0,
				expectedStoredBlocks:  4,
			},
			{
				name:      "reusing blocks",
				cacheSize: 5,
				actions: []testAction{
					newStartAction(req1),
					// Check block '1' reference count (should be 2)
					newTestActionWithExpectedValues(actionStartRequest, req2_1, 2, 3, 0, map[uint64]expectedBlockInfo{1: {true, 2}}),
					// Check block '1' reference count (should be 1)
					newTestActionWithExpectedValues(actionFinishRequest, req1, 1, 3, 1, map[uint64]expectedBlockInfo{1: {true, 1}}),
				},
				expectedRemovedBlocks: 0,
				expectedStoredBlocks:  3,
			},
			{
				name:      "block eviction",
				cacheSize: 4,
				actions: []testAction{
					newStartAction(req1),
					newStartAction(req2),
					newTestActionWithExpectedValues(actionFinishRequest, req2, -1, -1, -1, map[uint64]expectedBlockInfo{3: {true, 0}}),
					newTestActionWithExpectedValues(actionStartRequest, req3, -1, -1, -1, map[uint64]expectedBlockInfo{
						5: {true, 1},
						3: {false, 0},
					}),
				},
				expectedRemovedBlocks: 2,
				expectedStoredBlocks:  6,
			},
			{
				name:      "cache full, no eviction",
				cacheSize: 4,
				actions: []testAction{
					newStartAction(req1),
					newStartAction(req2),
					newInvalidTestAction(actionStartRequest, req3, capacityError),
				},
				expectedRemovedBlocks: 0,
				expectedStoredBlocks:  4,
			},
		}

		for _, test := range testCases {
			It(test.name, func() {
				time.Sleep(300 * time.Millisecond)

				config := &common.Configuration{
					Port:                  1234,
					Model:                 "model",
					KVCacheSize:           test.cacheSize,
					ZMQMaxConnectAttempts: 3,
					EventBatchSize:        1,
				}

				sub, topic := createSub(config)
				//nolint
				defer sub.Close()

				ctx, cancel := context.WithCancel(context.Background())

				wg := sync.WaitGroup{}
				wg.Add(1)

				blockCache, err := newBlockCache(config, GinkgoLogr, nil)
				Expect(err).NotTo(HaveOccurred())

				go func() {
					blockCache.start(ctx)
					wg.Done()
				}()

				defer func() {
					cancel()
					wg.Wait() // wait for goroutine to exit
				}()

				go func() {
					// Make sure that the subscriber listens before the events are published
					time.Sleep(time.Second)

					for _, action := range test.actions {
						var err error
						switch action.action {
						case actionStartRequest:
							_, err = blockCache.startRequest(action.request.id, action.request.blocks)
						case actionFinishRequest:
							err = blockCache.finishRequest(action.request.id)
						}

						if action.isError {
							Expect(err).To(HaveOccurred())
							if len(action.errMsg) > 0 {
								Expect(err.Error()).To(Equal(action.errMsg))
							}
							continue
						}

						// ensure that error has not occurred
						Expect(err).NotTo(HaveOccurred())

						// check cache info if required
						if action.expectedActiveRequests >= 0 || action.expectedTotalBlocks >= 0 || action.expectedUnusedBlocks >= 0 {
							activeRequests, totalBlocks, unusedBlocks := blockCache.getStats()
							if action.expectedActiveRequests >= 0 {
								Expect(activeRequests).To(Equal(action.expectedActiveRequests))
							}
							if action.expectedTotalBlocks >= 0 {
								Expect(totalBlocks).To(Equal(action.expectedTotalBlocks))
							}
							if action.expectedUnusedBlocks >= 0 {
								Expect(unusedBlocks).To(Equal(action.expectedUnusedBlocks))
							}
						}

						// check specific blocks info if required
						if len(action.expectedBlocksInfo) > 0 {
							for block, expectedInfo := range action.expectedBlocksInfo {
								refCount, exists := blockCache.getBlockInfo(block)
								if expectedInfo.exists {
									Expect(exists).To(BeTrue())
								} else {
									Expect(exists).To(BeFalse())
								}
								if expectedInfo.refCount >= 0 {
									Expect(refCount).To(Equal(expectedInfo.refCount))
								}
							}
						}
					}
				}()

				storedCount := 0
				removedCount := 0
				for i := range test.expectedRemovedBlocks + test.expectedStoredBlocks {
					parts, err := sub.RecvMessageBytes(0)
					Expect(err).NotTo(HaveOccurred())
					stored, removed := parseEvent(parts, topic, uint64(i+1))
					storedCount += len(stored)
					removedCount += len(removed)
				}
				Expect(removedCount).To(Equal(test.expectedRemovedBlocks))
				Expect(storedCount).To(Equal(test.expectedStoredBlocks))
			})
		}
	})

	Context("events", func() {

		It("should send events correctly", func() {
			config := &common.Configuration{
				Port:                  1234,
				Model:                 "model",
				KVCacheSize:           4,
				ZMQMaxConnectAttempts: 3,
			}

			sub, topic := createSub(config)
			//nolint
			defer sub.Close()

			ctx, cancel := context.WithCancel(context.Background())

			wg := sync.WaitGroup{}
			wg.Add(1)

			blockCache, err := newBlockCache(config, GinkgoLogr, nil)
			Expect(err).NotTo(HaveOccurred())

			go func() {
				blockCache.start(ctx)
				wg.Done()
			}()

			defer func() {
				cancel()
				wg.Wait() // wait for goroutine to exit
			}()

			expectedRemovedBlocks := []uint64{2, 4}
			expectedStoredBlocks := []uint64{1, 2, 3, 4, 5, 6}

			go func() {
				// Make sure that the subscriber listens before the events are published
				time.Sleep(time.Second)

				req1 := testRequest{"req1", []uint64{1, 2}}
				req2 := testRequest{"req2", []uint64{3, 4}}
				req3 := testRequest{"req3", []uint64{1, 3}}
				req4 := testRequest{"req4", []uint64{5, 6}}

				// blocks 1 and 2 stored
				alreadyInCache, err := blockCache.startRequest(req1.id, req1.blocks)
				Expect(err).NotTo(HaveOccurred())
				Expect(alreadyInCache).To(Equal(0))
				// blocks 3 and 4 stored
				alreadyInCache, err = blockCache.startRequest(req2.id, req2.blocks)
				Expect(err).NotTo(HaveOccurred())
				Expect(alreadyInCache).To(Equal(0))
				// no new blocks stored, reuse of 1 and 3
				alreadyInCache, err = blockCache.startRequest(req3.id, req3.blocks)
				Expect(err).NotTo(HaveOccurred())
				Expect(alreadyInCache).To(Equal(2))
				// no space left - should fail
				alreadyInCache, err = blockCache.startRequest(req4.id, req4.blocks)
				Expect(err).To(HaveOccurred())
				Expect(alreadyInCache).To(Equal(0))

				err = blockCache.finishRequest(req1.id)
				Expect(err).NotTo(HaveOccurred())
				err = blockCache.finishRequest(req2.id)
				Expect(err).NotTo(HaveOccurred())
				// now 2 and 4 are not in use

				// blocks 2 and 4 should be removed, and 5 and 6 stored
				alreadyInCache, err = blockCache.startRequest(req4.id, req4.blocks)
				Expect(err).NotTo(HaveOccurred())
				Expect(alreadyInCache).To(Equal(0))
			}()

			removedBlocks := make([]uint64, 0)
			storedBlocks := make([]uint64, 0)
			count := uint64(1)
			for {
				parts, err := sub.RecvMessageBytes(0)
				Expect(err).NotTo(HaveOccurred())
				stored, removed := parseEvent(parts, topic, count)
				storedBlocks = append(storedBlocks, stored...)
				removedBlocks = append(removedBlocks, removed...)
				count++

				if len(removedBlocks) == len(expectedRemovedBlocks) && len(storedBlocks) == len(expectedStoredBlocks) {
					break
				}
			}
			Expect(removedBlocks).To(Equal(expectedRemovedBlocks))
			Expect(storedBlocks).To(Equal(expectedStoredBlocks))
		})

	})

	Context("thread safety", func() {
		testCases := []threadTestCase{{
			name:              "run add/remove requests in parallel, use partial cache",
			cacheSize:         1000,
			numGoroutines:     50,
			numOperations:     100,
			minBlockLen:       2,
			maxBlockLen:       10,
			maxHashValue:      100,
			shouldUseAllCache: false,
		}, {
			name:              "run add/remove requests in parallel, use all cache",
			cacheSize:         100,
			numGoroutines:     50,
			numOperations:     10,
			minBlockLen:       2,
			maxBlockLen:       10,
			maxHashValue:      100,
			shouldUseAllCache: true,
		}}

		for _, testCase := range testCases {
			It(testCase.name, func() {
				config := common.Configuration{
					Port:                  1234,
					Model:                 "model",
					KVCacheSize:           testCase.cacheSize,
					ZMQMaxConnectAttempts: 3,
				}
				blockCache, err := newBlockCache(&config, GinkgoLogr, nil)
				Expect(err).NotTo(HaveOccurred())
				var wg sync.WaitGroup

				// Start multiple goroutines performing concurrent operations
				for i := range testCase.numGoroutines {
					wg.Add(1)
					go func(id int) {
						defer wg.Done()

						for j := range testCase.numOperations {
							reqID := fmt.Sprintf("req_%d_%d", id, j)
							blocks := createRandomArray(testCase.minBlockLen, testCase.maxBlockLen, testCase.maxHashValue)

							_, err := blockCache.startRequest(reqID, blocks)
							if err != nil {
								// some operations may fail due to cache being full, which is expected
								Expect(err.Error()).To(Equal(capacityError))
								continue
							}

							time.Sleep(time.Duration(common.RandomInt(1, 100)) * time.Microsecond)

							err = blockCache.finishRequest(reqID)
							Expect(err).NotTo(HaveOccurred())
						}
					}(i)
				}

				wg.Wait()

				activeReqs, totalBlocks, unusedBlocks := blockCache.getStats()
				fmt.Printf("Thread safety test completed. Final stats: Active requests: %d, Total blocks: %d, Unused blocks: %d\n",
					activeReqs, totalBlocks, unusedBlocks)
				if testCase.shouldUseAllCache {
					Expect(totalBlocks).To(Equal(testCase.cacheSize))
				}
				Expect(totalBlocks).To(Equal(unusedBlocks))
			})
		}
	})
})

func createRandomArray(minArrLen, maxArrLen int, maxValue uint64) []uint64 {
	// Random length between a and b (inclusive)
	length := common.RandomInt(minArrLen, maxArrLen)

	// Create array with random values
	arr := make([]uint64, 0)
	seen := make(map[uint64]struct{})

	for len(arr) < length {
		val := uint64(common.RandomInt(0, int(maxValue)))
		if _, exists := seen[val]; !exists {
			seen[val] = struct{}{}
			arr = append(arr, val)
		}
	}

	return arr
}

func parseEvent(parts [][]byte, expectedTopic string, expectedSeq uint64) ([]uint64, []uint64) {
	// The message should be [topic, seq, payload]
	Expect(parts).To(HaveLen(3))

	Expect(string(parts[0])).To(Equal(expectedTopic))

	seq := binary.BigEndian.Uint64(parts[1])
	Expect(seq).To(Equal(expectedSeq))

	removed := make([]uint64, 0)
	stored := make([]uint64, 0)

	var eventBatch kvevents.EventBatch
	err := msgpack.Unmarshal(parts[2], &eventBatch)
	Expect(err).NotTo(HaveOccurred())
	for _, rawEvent := range eventBatch.Events {
		var taggedUnion []msgpack.RawMessage
		err := msgpack.Unmarshal(rawEvent, &taggedUnion)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(taggedUnion)).To(BeNumerically(">", 1))

		payloadBytes, err := msgpack.Marshal(taggedUnion[1:])
		Expect(err).NotTo(HaveOccurred())

		var tag string
		err = msgpack.Unmarshal(taggedUnion[0], &tag)
		Expect(err).NotTo(HaveOccurred())

		switch tag {
		case kvevents.BlockStoredEventTag:
			var bs kvevents.BlockStored
			err = msgpack.Unmarshal(payloadBytes, &bs)
			stored = append(stored, bs.BlockHashes...)
		case kvevents.BlockRemovedEventTag:
			var br kvevents.BlockRemoved
			err = msgpack.Unmarshal(payloadBytes, &br)
			removed = append(removed, br.BlockHashes...)

		default:
			Fail("unexpected tag " + tag)
			continue
		}
		Expect(err).NotTo(HaveOccurred())
	}
	return stored, removed
}

func createSub(config *common.Configuration) (*zmq.Socket, string) {
	zctx, err := zmq.NewContext()
	Expect(err).NotTo(HaveOccurred())
	sub, err := zctx.NewSocket(zmq.SUB)
	Expect(err).NotTo(HaveOccurred())
	err = sub.Bind(wildcardEndpoint)
	Expect(err).NotTo(HaveOccurred())
	// get the actual port
	endpoint, err := sub.GetLastEndpoint()
	Expect(err).NotTo(HaveOccurred())
	config.ZMQEndpoint = endpoint
	topic := createTopic(config)
	err = sub.SetSubscribe(topic)
	Expect(err).NotTo(HaveOccurred())
	return sub, topic
}
