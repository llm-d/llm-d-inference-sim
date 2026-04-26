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
	"fmt"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const (
	req1ID    = "req1"
	req2ID    = "req2"
	req3ID    = "req3"
	localhost = "127.0.0.1"
)

type ActionType int

const (
	actionStartRequest ActionType = iota
	actionFinishRequest
)

type testRequest struct {
	id          string
	model       string
	loraName    *string
	loraID      *int
	blockHashes []uint64
	tokens      [][]uint32
}

// ensure testRequest implements the Request interface
var _ Request = (*testRequest)(nil)

func (t *testRequest) GetRequestID() string {
	return t.id
}

func (t *testRequest) GetDisplayedModel() string {
	if t.model != "" {
		return t.model
	}
	return common.TestModelName
}

func (t *testRequest) GetLoraName() *string {
	return t.loraName
}

func (t *testRequest) GetLoraID() *int {
	return t.loraID
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
	random := common.NewRandom(time.Now().UnixNano(), 8080)

	Context("general tests", func() {
		// check single request processing, ensure cache is valid after request processing started
		// and after the processing was finished
		req1 := testRequest{id: req1ID, blockHashes: []uint64{1, 2}, tokens: [][]uint32{{1}, {2}}}
		req2 := testRequest{id: req2ID, blockHashes: []uint64{3, 4}, tokens: [][]uint32{{3}, {4}}}
		req2_1 := testRequest{id: req2ID, blockHashes: []uint64{1, 3}, tokens: [][]uint32{{1}, {3}}}
		req3 := testRequest{id: req3ID, blockHashes: []uint64{5, 6}, tokens: [][]uint32{{5}, {6}}}

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

				ctx, cancel := context.WithCancel(context.Background())

				config := &common.Configuration{
					IP:             localhost,
					Port:           1234,
					Model:          "model",
					KVCacheSize:    test.cacheSize,
					EventBatchSize: 1,
				}

				topic := CreateKVEventsTopic(localhost, config.Model)
				sub, endpoint := common.CreateSub(ctx, topic)
				config.ZMQEndpoint = endpoint
				//nolint
				defer sub.Close()

				wg := sync.WaitGroup{}
				wg.Add(1)

				blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
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
							_, err = blockCache.startRequest(&action.request, action.request.blockHashes, action.request.tokens)
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
								refCount, exists := blockCache.getBlockInfo(blockKey{hash: block, modelName: action.request.GetDisplayedModel()})
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
				expectedTotal := test.expectedRemovedBlocks + test.expectedStoredBlocks

				for i, seq := 0, uint64(1); i < expectedTotal; i, seq = storedCount+removedCount, seq+1 {
					msg, err := sub.Recv()
					Expect(err).NotTo(HaveOccurred())
					stored, removed, _ := CountKVEventBlocks(msg.Frames, topic, seq)
					storedCount += stored
					removedCount += removed
				}
				Expect(removedCount).To(Equal(test.expectedRemovedBlocks))
				Expect(storedCount).To(Equal(test.expectedStoredBlocks))
			})
		}
	})

	Context("events", func() {

		It("should send events correctly", func() {
			ctx, cancel := context.WithCancel(context.Background())

			config := &common.Configuration{
				IP:          localhost,
				Port:        1234,
				Model:       "model",
				KVCacheSize: 4,
			}

			topic := CreateKVEventsTopic(localhost, config.Model)
			sub, endpoint := common.CreateSub(ctx, topic)
			config.ZMQEndpoint = endpoint
			//nolint
			defer sub.Close()

			wg := sync.WaitGroup{}
			wg.Add(1)

			blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
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

				req1 := testRequest{id: "req1", blockHashes: []uint64{1, 2}, tokens: [][]uint32{{1}, {2}}}
				req2 := testRequest{id: "req2", blockHashes: []uint64{3, 4}, tokens: [][]uint32{{1}, {2}}}
				req3 := testRequest{id: "req3", blockHashes: []uint64{1, 3}, tokens: [][]uint32{{1}, {2}}}
				req4 := testRequest{id: "req4", blockHashes: []uint64{5, 6}, tokens: [][]uint32{{1}, {2}}}

				// blocks 1 and 2 stored
				alreadyInCache, err := blockCache.startRequest(&req1, req1.blockHashes, req1.tokens)
				Expect(err).NotTo(HaveOccurred())
				Expect(alreadyInCache).To(Equal(0))
				// blocks 3 and 4 stored
				alreadyInCache, err = blockCache.startRequest(&req2, req2.blockHashes, req2.tokens)
				Expect(err).NotTo(HaveOccurred())
				Expect(alreadyInCache).To(Equal(0))
				// no new blocks stored, reuse of 1 and 3
				alreadyInCache, err = blockCache.startRequest(&req3, req3.blockHashes, req3.tokens)
				Expect(err).NotTo(HaveOccurred())
				Expect(alreadyInCache).To(Equal(2))
				// no space left - should fail
				alreadyInCache, err = blockCache.startRequest(&req4, req4.blockHashes, req4.tokens)
				Expect(err).To(HaveOccurred())
				Expect(alreadyInCache).To(Equal(0))

				err = blockCache.finishRequest(req1.id)
				Expect(err).NotTo(HaveOccurred())
				err = blockCache.finishRequest(req2.id)
				Expect(err).NotTo(HaveOccurred())
				// now 2 and 4 are not in use

				// blocks 2 and 4 should be removed, and 5 and 6 stored
				alreadyInCache, err = blockCache.startRequest(&req4, req4.blockHashes, req4.tokens)
				Expect(err).NotTo(HaveOccurred())
				Expect(alreadyInCache).To(Equal(0))
			}()

			removedBlocks := make([]uint64, 0)
			storedBlocks := make([]uint64, 0)
			count := uint64(1)
			for {
				msg, err := sub.Recv()
				Expect(err).NotTo(HaveOccurred())
				storedEvents, removed, _ := ParseKVEvent(msg.Frames, topic, count)
				for _, e := range storedEvents {
					storedBlocks = append(storedBlocks, e.BlockHashes...)
				}
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
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()

				config := common.Configuration{
					IP:          localhost,
					Port:        1234,
					Model:       "model",
					KVCacheSize: testCase.cacheSize,
				}
				blockCache, err := newBlockCache(ctx, &config, GinkgoLogr, nil)
				Expect(err).NotTo(HaveOccurred())
				var wg sync.WaitGroup

				// Start multiple goroutines performing concurrent operations
				for i := range testCase.numGoroutines {
					wg.Add(1)
					go func(id int) {
						defer wg.Done()

						for j := range testCase.numOperations {
							reqID := fmt.Sprintf("req_%d_%d", id, j)
							hashes, tokens := createRandomArray(testCase.minBlockLen, testCase.maxBlockLen,
								testCase.maxHashValue, random)

							req := testRequest{
								id:          reqID,
								blockHashes: hashes,
								tokens:      tokens,
							}
							_, err := blockCache.startRequest(&req, hashes, tokens)
							if err != nil {
								// some operations may fail due to cache being full, which is expected
								Expect(err.Error()).To(Equal(capacityError))
								continue
							}

							time.Sleep(time.Duration(random.RandomInt(1, 100)) * time.Microsecond)

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

	Context("model-aware blocks", func() {
		const loraModel = "loraX"

		It("same hash different model should be stored as separate blocks", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			config := &common.Configuration{
				IP:          localhost,
				Port:        1234,
				Model:       common.TestModelName,
				KVCacheSize: 10,
			}

			blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
			Expect(err).NotTo(HaveOccurred())

			reqA := testRequest{id: "reqA", model: common.TestModelName, blockHashes: []uint64{1, 2}, tokens: [][]uint32{{1}, {2}}}
			reqB := testRequest{id: "reqB", model: loraModel, blockHashes: []uint64{1, 2}, tokens: [][]uint32{{1}, {2}}}

			_, err = blockCache.startRequest(&reqA, reqA.blockHashes, reqA.tokens)
			Expect(err).NotTo(HaveOccurred())
			_, err = blockCache.startRequest(&reqB, reqB.blockHashes, reqB.tokens)
			Expect(err).NotTo(HaveOccurred())

			// 4 distinct blocks (2 per model), 2 active requests
			activeReqs, totalBlocks, unusedBlocks := blockCache.getStats()
			Expect(activeReqs).To(Equal(2))
			Expect(totalBlocks).To(Equal(4))
			Expect(unusedBlocks).To(Equal(0))

			// blocks are independent per model
			refCount, exists := blockCache.getBlockInfo(blockKey{hash: 1, modelName: common.TestModelName})
			Expect(exists).To(BeTrue())
			Expect(refCount).To(Equal(1))

			refCount, exists = blockCache.getBlockInfo(blockKey{hash: 1, modelName: loraModel})
			Expect(exists).To(BeTrue())
			Expect(refCount).To(Equal(1))

			// finishing reqA only affects base model blocks
			err = blockCache.finishRequest(reqA.id)
			Expect(err).NotTo(HaveOccurred())

			refCount, exists = blockCache.getBlockInfo(blockKey{hash: 1, modelName: common.TestModelName})
			Expect(exists).To(BeTrue())
			Expect(refCount).To(Equal(0)) // unused

			refCount, exists = blockCache.getBlockInfo(blockKey{hash: 1, modelName: loraModel})
			Expect(exists).To(BeTrue())
			Expect(refCount).To(Equal(1)) // still in use
		})

		It("should not reuse blocks across models", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			config := &common.Configuration{
				IP:          localhost,
				Port:        1234,
				Model:       common.TestModelName,
				KVCacheSize: 10,
			}

			blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
			Expect(err).NotTo(HaveOccurred())

			reqA := testRequest{id: "reqA", model: common.TestModelName, blockHashes: []uint64{1, 2}, tokens: [][]uint32{{1}, {2}}}
			_, err = blockCache.startRequest(&reqA, reqA.blockHashes, reqA.tokens)
			Expect(err).NotTo(HaveOccurred())
			err = blockCache.finishRequest(reqA.id)
			Expect(err).NotTo(HaveOccurred())

			// same hashes but different model - should NOT find them in cache
			reqB := testRequest{id: "reqB", model: loraModel, blockHashes: []uint64{1, 2}, tokens: [][]uint32{{1}, {2}}}
			alreadyInCache, err := blockCache.startRequest(&reqB, reqB.blockHashes, reqB.tokens)
			Expect(err).NotTo(HaveOccurred())
			Expect(alreadyInCache).To(Equal(0))

			_, totalBlocks, _ := blockCache.getStats()
			Expect(totalBlocks).To(Equal(4))
		})

		It("countCachedBlockPrefix should be model-scoped", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			config := &common.Configuration{
				IP:          localhost,
				Port:        1234,
				Model:       common.TestModelName,
				KVCacheSize: 10,
			}

			blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
			Expect(err).NotTo(HaveOccurred())

			req := testRequest{id: "req1", model: common.TestModelName, blockHashes: []uint64{1, 2, 3}, tokens: [][]uint32{{1}, {2}, {3}}}
			_, err = blockCache.startRequest(&req, req.blockHashes, req.tokens)
			Expect(err).NotTo(HaveOccurred())

			Expect(blockCache.countCachedBlockPrefix([]uint64{1, 2, 3}, common.TestModelName)).To(Equal(3))
			Expect(blockCache.countCachedBlockPrefix([]uint64{1, 2, 3}, loraModel)).To(Equal(0))
		})
	})

	Context("model-aware eviction", func() {
		const lora1 = "lora1"
		const lora2 = "lora2"

		It("should evict unloaded lora blocks before loaded lora blocks", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			config := &common.Configuration{
				IP:          localhost,
				Port:        1234,
				Model:       common.TestModelName,
				KVCacheSize: 4,
			}

			blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
			Expect(err).NotTo(HaveOccurred())

			// lora1 is loaded, lora2 is not
			blockCache.setModelLoaded(lora1)

			reqL1 := testRequest{id: "reqL1", model: lora1, blockHashes: []uint64{10, 20}, tokens: [][]uint32{{10}, {20}}}
			reqL2 := testRequest{id: "reqL2", model: lora2, blockHashes: []uint64{30, 40}, tokens: [][]uint32{{30}, {40}}}

			_, err = blockCache.startRequest(&reqL1, reqL1.blockHashes, reqL1.tokens)
			Expect(err).NotTo(HaveOccurred())
			_, err = blockCache.startRequest(&reqL2, reqL2.blockHashes, reqL2.tokens)
			Expect(err).NotTo(HaveOccurred())

			// finish both - all 4 blocks become unused, cache is full
			err = blockCache.finishRequest(reqL1.id)
			Expect(err).NotTo(HaveOccurred())
			err = blockCache.finishRequest(reqL2.id)
			Expect(err).NotTo(HaveOccurred())

			// add a new block - should evict from lora2 (unloaded) first
			reqNew := testRequest{id: "reqNew", model: lora1, blockHashes: []uint64{50}, tokens: [][]uint32{{50}}}
			_, err = blockCache.startRequest(&reqNew, reqNew.blockHashes, reqNew.tokens)
			Expect(err).NotTo(HaveOccurred())

			// lora2 block 30 (oldest unloaded) should be evicted
			_, exists := blockCache.getBlockInfo(blockKey{hash: 30, modelName: lora2})
			Expect(exists).To(BeFalse())

			// lora1 blocks should still exist
			_, exists = blockCache.getBlockInfo(blockKey{hash: 10, modelName: lora1})
			Expect(exists).To(BeTrue())
			_, exists = blockCache.getBlockInfo(blockKey{hash: 20, modelName: lora1})
			Expect(exists).To(BeTrue())
		})

		It("should fall back to oldest loaded-model block when no unloaded blocks exist", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			config := &common.Configuration{
				IP:          localhost,
				Port:        1234,
				Model:       common.TestModelName,
				KVCacheSize: 3,
			}

			blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
			Expect(err).NotTo(HaveOccurred())

			blockCache.setModelLoaded(lora1)

			// fill cache with lora1 blocks only (all loaded)
			req1 := testRequest{id: "req1", model: lora1, blockHashes: []uint64{1, 2, 3}, tokens: [][]uint32{{1}, {2}, {3}}}
			_, err = blockCache.startRequest(&req1, req1.blockHashes, req1.tokens)
			Expect(err).NotTo(HaveOccurred())
			err = blockCache.finishRequest(req1.id)
			Expect(err).NotTo(HaveOccurred())

			// add a new block - must evict from loaded model (only option)
			reqNew := testRequest{id: "reqNew", model: lora1, blockHashes: []uint64{99}, tokens: [][]uint32{{99}}}
			_, err = blockCache.startRequest(&reqNew, reqNew.blockHashes, reqNew.tokens)
			Expect(err).NotTo(HaveOccurred())

			// exactly one of the three original blocks should be evicted
			surviving := 0
			for i := 1; i <= 3; i++ {
				_, exists := blockCache.getBlockInfo(blockKey{hash: uint64(i), modelName: lora1})
				if exists {
					surviving++
				}
			}
			Expect(surviving).To(Equal(2), "expected exactly 1 eviction out of 3 blocks")
		})

		It("should prefer newly unloaded model blocks after setModelUnloaded", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			config := &common.Configuration{
				IP:          localhost,
				Port:        1234,
				Model:       common.TestModelName,
				KVCacheSize: 4,
			}

			blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
			Expect(err).NotTo(HaveOccurred())

			// both loras loaded
			blockCache.setModelLoaded(lora1)
			blockCache.setModelLoaded(lora2)

			reqL1 := testRequest{id: "reqL1", model: lora1, blockHashes: []uint64{10, 20}, tokens: [][]uint32{{10}, {20}}}
			reqL2 := testRequest{id: "reqL2", model: lora2, blockHashes: []uint64{30, 40}, tokens: [][]uint32{{30}, {40}}}

			_, err = blockCache.startRequest(&reqL1, reqL1.blockHashes, reqL1.tokens)
			Expect(err).NotTo(HaveOccurred())
			_, err = blockCache.startRequest(&reqL2, reqL2.blockHashes, reqL2.tokens)
			Expect(err).NotTo(HaveOccurred())

			err = blockCache.finishRequest(reqL1.id)
			Expect(err).NotTo(HaveOccurred())
			err = blockCache.finishRequest(reqL2.id)
			Expect(err).NotTo(HaveOccurred())

			// unload lora2 - its blocks become low-priority eviction candidates
			blockCache.setModelUnloaded(lora2)

			// force eviction
			reqNew := testRequest{id: "reqNew", model: lora1, blockHashes: []uint64{50}, tokens: [][]uint32{{50}}}
			_, err = blockCache.startRequest(&reqNew, reqNew.blockHashes, reqNew.tokens)
			Expect(err).NotTo(HaveOccurred())

			// one of lora2 blocks (unloaded) should be evicted first
			_, exists30 := blockCache.getBlockInfo(blockKey{hash: 30, modelName: lora2})
			_, exists40 := blockCache.getBlockInfo(blockKey{hash: 40, modelName: lora2})
			Expect(exists30 && exists40).To(BeFalse())

			// lora1 blocks should remain
			_, exists := blockCache.getBlockInfo(blockKey{hash: 10, modelName: lora1})
			Expect(exists).To(BeTrue())
			_, exists = blockCache.getBlockInfo(blockKey{hash: 20, modelName: lora1})
			Expect(exists).To(BeTrue())
		})
	})

	Context("lora fields in events", func() {

		It("store events should carry lora metadata", func() {
			ctx, cancel := context.WithCancel(context.Background())

			config := &common.Configuration{
				IP:             localhost,
				Port:           1234,
				Model:          common.TestModelName,
				KVCacheSize:    10,
				EventBatchSize: 1,
			}

			topic := CreateKVEventsTopic(localhost, config.Model)
			sub, endpoint := common.CreateSub(ctx, topic)
			config.ZMQEndpoint = endpoint
			//nolint
			defer sub.Close()

			wg := sync.WaitGroup{}
			wg.Add(1)

			blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
			Expect(err).NotTo(HaveOccurred())

			go func() {
				blockCache.start(ctx)
				wg.Done()
			}()

			defer func() {
				cancel()
				wg.Wait()
			}()

			loraName := "lora1"
			loraID := 1

			go func() {
				time.Sleep(time.Second)

				// base model request - no lora info
				reqBase := testRequest{
					id:          "reqBase",
					model:       common.TestModelName,
					blockHashes: []uint64{1, 2},
					tokens:      [][]uint32{{1}, {2}},
				}
				_, err := blockCache.startRequest(&reqBase, reqBase.blockHashes, reqBase.tokens)
				Expect(err).NotTo(HaveOccurred())

				// lora request - with lora info
				reqLora := testRequest{
					id:          "reqLora",
					model:       loraName,
					loraName:    &loraName,
					loraID:      &loraID,
					blockHashes: []uint64{3, 4},
					tokens:      [][]uint32{{3}, {4}},
				}
				_, err = blockCache.startRequest(&reqLora, reqLora.blockHashes, reqLora.tokens)
				Expect(err).NotTo(HaveOccurred())
			}()

			// collect 2 store events
			storedEvents := make([]StoredEventInfo, 0)
			seq := uint64(1)
			for len(storedEvents) < 2 {
				msg, err := sub.Recv()
				Expect(err).NotTo(HaveOccurred())
				events, _, _ := ParseKVEvent(msg.Frames, topic, seq)
				storedEvents = append(storedEvents, events...)
				seq++
			}

			// first event (base model) should have nil lora fields
			Expect(storedEvents[0].LoraName).To(BeNil())
			Expect(storedEvents[0].LoraID).To(BeNil())
			Expect(storedEvents[0].BlockHashes).To(Equal([]uint64{1, 2}))

			// second event (lora) should carry lora metadata
			Expect(storedEvents[1].LoraName).NotTo(BeNil())
			Expect(*storedEvents[1].LoraName).To(Equal(loraName))
			Expect(storedEvents[1].LoraID).NotTo(BeNil())
			Expect(*storedEvents[1].LoraID).To(Equal(loraID))
			Expect(storedEvents[1].BlockHashes).To(Equal([]uint64{3, 4}))
		})

		It("same prompt with base model and lora should produce separate events", func() {
			ctx, cancel := context.WithCancel(context.Background())

			config := &common.Configuration{
				IP:             localhost,
				Port:           1234,
				Model:          common.TestModelName,
				KVCacheSize:    10,
				EventBatchSize: 1,
			}

			topic := CreateKVEventsTopic(localhost, config.Model)
			sub, endpoint := common.CreateSub(ctx, topic)
			config.ZMQEndpoint = endpoint
			//nolint
			defer sub.Close()

			wg := sync.WaitGroup{}
			wg.Add(1)

			blockCache, err := newBlockCache(ctx, config, GinkgoLogr, nil)
			Expect(err).NotTo(HaveOccurred())

			go func() {
				blockCache.start(ctx)
				wg.Done()
			}()

			defer func() {
				cancel()
				wg.Wait()
			}()

			loraName := "lora1"
			loraID := 1

			go func() {
				time.Sleep(time.Second)

				// same hashes, different models
				req1 := testRequest{
					id:          "req1",
					model:       common.TestModelName,
					blockHashes: []uint64{10, 20},
					tokens:      [][]uint32{{10}, {20}},
				}
				_, err := blockCache.startRequest(&req1, req1.blockHashes, req1.tokens)
				Expect(err).NotTo(HaveOccurred())

				req2 := testRequest{
					id:          "req2",
					model:       loraName,
					loraName:    &loraName,
					loraID:      &loraID,
					blockHashes: []uint64{10, 20},
					tokens:      [][]uint32{{10}, {20}},
				}
				_, err = blockCache.startRequest(&req2, req2.blockHashes, req2.tokens)
				Expect(err).NotTo(HaveOccurred())
			}()

			// both requests store new blocks (4 total) since models differ
			storedEvents := make([]StoredEventInfo, 0)
			seq := uint64(1)
			totalStoredHashes := 0
			for totalStoredHashes < 4 {
				msg, err := sub.Recv()
				Expect(err).NotTo(HaveOccurred())
				events, _, _ := ParseKVEvent(msg.Frames, topic, seq)
				for _, e := range events {
					totalStoredHashes += len(e.BlockHashes)
				}
				storedEvents = append(storedEvents, events...)
				seq++
			}

			Expect(totalStoredHashes).To(Equal(4))
			Expect(storedEvents).To(HaveLen(2))

			// first event: base model, no lora
			Expect(storedEvents[0].LoraName).To(BeNil())
			Expect(storedEvents[0].BlockHashes).To(Equal([]uint64{10, 20}))

			// second event: lora
			Expect(storedEvents[1].LoraName).NotTo(BeNil())
			Expect(*storedEvents[1].LoraName).To(Equal(loraName))
			Expect(*storedEvents[1].LoraID).To(Equal(loraID))
			Expect(storedEvents[1].BlockHashes).To(Equal([]uint64{10, 20}))
		})
	})
})

// returns kv event content - array of blocks hash values and array of tokens for each block
// both arrays are of the same length, tokens array is the same for all blocks
func createRandomArray(minArrLen, maxArrLen int, maxValue uint64, random *common.Random) ([]uint64, [][]uint32) {
	// Random length between a and b (inclusive)
	length := random.RandomInt(minArrLen, maxArrLen)

	// Create array with random values
	hashes := make([]uint64, 0)
	tokens := make([][]uint32, 0)
	seen := make(map[uint64]struct{})

	for len(hashes) < length {
		val := uint64(random.RandomInt(0, int(maxValue)))
		if _, exists := seen[val]; !exists {
			seen[val] = struct{}{}
			hashes = append(hashes, val)
			tokens = append(tokens, []uint32{1, 2})
		}
	}

	return hashes, tokens
}
