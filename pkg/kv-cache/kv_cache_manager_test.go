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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const (
	req1ID = "req1"
	req2ID = "req2"
	req3ID = "req3"
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

func newTestAction(action ActionType, request testRequest) testAction {
	return testAction{
		action:                 action,
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
	name      string
	cacheSize int
	actions   []testAction
}

var _ = Describe("KV cache", func() {
	Context("blocks cache tests", func() {
		// check single request processing, ensure cache is valid after request processing started
		// and after the processing was finished
		req1 := testRequest{req1ID, []uint64{1, 2}}
		req2 := testRequest{req2ID, []uint64{3, 4}}
		req2_1 := testRequest{req2ID, []uint64{1, 3}}
		req3 := testRequest{req3ID, []uint64{5, 6}}

		testCases := []testCase{{
			name:      "single request",
			cacheSize: 3,
			actions: []testAction{
				newTestActionWithExpectedValues(actionStartRequest, req1, 1, 2, 0, nil),
				newTestActionWithExpectedValues(actionFinishRequest, req1, 0, 2, 2, nil),
			},
		}, {
			name:      "two requests",
			cacheSize: 5,
			actions: []testAction{
				newTestAction(actionStartRequest, req1),
				newTestActionWithExpectedValues(actionStartRequest, req2, 2, 4, 0, nil),
				newTestActionWithExpectedValues(actionFinishRequest, req1, 1, 4, 2, nil),
				newTestActionWithExpectedValues(actionFinishRequest, req2, 0, 4, 4, nil),
			},
		}, {
			name:      "reusing blocks",
			cacheSize: 5,
			actions: []testAction{
				newTestAction(actionStartRequest, req1),
				// Check block '1' reference count (should be 2)
				newTestActionWithExpectedValues(actionStartRequest, req2_1, 2, 3, 0, map[uint64]expectedBlockInfo{1: {true, 2}}),
				// Check block '1' reference count (should be 1)
				newTestActionWithExpectedValues(actionFinishRequest, req1, 1, 3, 1, map[uint64]expectedBlockInfo{1: {true, 1}}),
			},
		}, {
			name:      "block eviction",
			cacheSize: 4,
			actions: []testAction{
				newTestAction(actionStartRequest, req1),
				newTestAction(actionStartRequest, req2),
				newTestActionWithExpectedValues(actionFinishRequest, req2, -1, -1, -1, map[uint64]expectedBlockInfo{3: {true, 0}}),
				newTestActionWithExpectedValues(actionStartRequest, req3, -1, -1, -1, map[uint64]expectedBlockInfo{
					3: {false, -1},
					5: {true, 1},
				}),
			},
		}, {
			name:      "cache full, no eviction",
			cacheSize: 4,
			actions: []testAction{
				newTestAction(actionStartRequest, req1),
				newTestAction(actionStartRequest, req2),
				newInvalidTestAction(actionStartRequest, req3, "cache is full and no blocks available for eviction"),
			},
		}}

		for _, test := range testCases {
			It(test.name, func() {
				blockCache := newBlockCache(test.cacheSize)

				for _, action := range test.actions {
					var err error

					switch action.action {
					case actionStartRequest:
						err = blockCache.startRequest(action.request.id, action.request.blocks)
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

					// ensure that error does not accured
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
							info, exists := blockCache.getBlockInfo(block)
							if expectedInfo.exists {
								Expect(exists).To(BeTrue())
							} else {
								Expect(exists).To(BeFalse())
							}
							if expectedInfo.refCount >= 0 {
								Expect(info.refCount).To(Equal(expectedInfo.refCount))
							}
						}
					}
				}
			})
		}
	})
})

// 	// Test 4: Cache full with no evictable blocks
// 	fmt.Println("\n=== Test 4: Cache Full Scenario ===")

// 	// Try to add more blocks than capacity allows
// 	err = cache.StartRequest("req4", []string{"block6", "block7"})
// 	if err != nil {
// 		fmt.Printf("Expected error when cache is full: %v\n", err)
// 	}

// 	// Test 5: Thread safety test
// 	fmt.Println("\n=== Test 5: Thread Safety ===")
// 	testThreadSafety()

// 	fmt.Println("\nAll tests completed!")
// }

// func testThreadSafety() {
// 	cache := NewBlockCache(10)

// 	var wg sync.WaitGroup
// 	numGoroutines := 50
// 	numOperations := 100

// 	// Start multiple goroutines performing concurrent operations
// 	for i := 0; i < numGoroutines; i++ {
// 		wg.Add(1)
// 		go func(id int) {
// 			defer wg.Done()

// 			for j := 0; j < numOperations; j++ {
// 				reqID := fmt.Sprintf("req_%d_%d", id, j)
// 				blockHashes := []string{
// 					fmt.Sprintf("block_%d_%d_1", id, j),
// 					fmt.Sprintf("block_%d_%d_2", id, j),
// 				}

// 				// Start request
// 				err := cache.StartRequest(reqID, blockHashes)
// 				if err != nil {
// 					// Some operations may fail due to cache being full, which is expected
// 					continue
// 				}

// 				// Small delay to simulate work
// 				time.Sleep(time.Microsecond)

// 				// Finish request
// 				cache.FinishRequest(reqID)
// 			}
// 		}(i)
// 	}

// 	wg.Wait()

// 	activeReqs, totalBlocks, unusedBlocks := cache.GetStats()
// 	fmt.Printf("Thread safety test completed. Final stats: Active requests: %d, Total blocks: %d, Unused blocks: %d\n",
// 		activeReqs, totalBlocks, unusedBlocks)
// }
