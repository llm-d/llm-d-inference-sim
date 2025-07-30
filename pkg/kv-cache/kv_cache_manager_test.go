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

var _ = Describe("KV cache", func() {
	Context("blocks cache tests", func() {
		It("single request", func() {
			// check single request processing, ensure cache is valid after request processing started
			// and after the processing was finished
			blockCache := newBlockCache(3)
			err := blockCache.startRequest(req1ID, []uint64{1, 2})
			Expect(err).NotTo(HaveOccurred())
			activeRequests, totalBlocks, unusedBlocks := blockCache.getStats()
			Expect(activeRequests).To(Equal(1))
			Expect(totalBlocks).To(Equal(2))
			Expect(unusedBlocks).To(Equal(0))

			err = blockCache.finishRequest(req1ID)
			Expect(err).NotTo(HaveOccurred())
			activeRequests, totalBlocks, unusedBlocks = blockCache.getStats()
			Expect(activeRequests).To(Equal(0))
			Expect(totalBlocks).To(Equal(2))
			Expect(unusedBlocks).To(Equal(2))
		})
		It("two requests", func() {
			blockCache := newBlockCache(5)

			err := blockCache.startRequest(req1ID, []uint64{1, 2})
			Expect(err).NotTo(HaveOccurred())
			err = blockCache.startRequest(req2ID, []uint64{3, 4})
			Expect(err).NotTo(HaveOccurred())
			activeRequests, totalBlocks, unusedBlocks := blockCache.getStats()
			Expect(activeRequests).To(Equal(2))
			Expect(totalBlocks).To(Equal(4))
			Expect(unusedBlocks).To(Equal(0))

			err = blockCache.finishRequest(req1ID)
			Expect(err).NotTo(HaveOccurred())
			activeRequests, totalBlocks, unusedBlocks = blockCache.getStats()
			Expect(activeRequests).To(Equal(1))
			Expect(totalBlocks).To(Equal(4))
			Expect(unusedBlocks).To(Equal(2))

			err = blockCache.finishRequest(req2ID)
			Expect(err).NotTo(HaveOccurred())
			activeRequests, totalBlocks, unusedBlocks = blockCache.getStats()
			Expect(activeRequests).To(Equal(0))
			Expect(totalBlocks).To(Equal(4))
			Expect(unusedBlocks).To(Equal(4))
		})
		It("reusing blocks", func() {
			blockCache := newBlockCache(5)

			err := blockCache.startRequest(req1ID, []uint64{1, 2})
			Expect(err).NotTo(HaveOccurred())

			err = blockCache.startRequest(req2ID, []uint64{1, 3})
			Expect(err).NotTo(HaveOccurred())
			activeRequests, totalBlocks, unusedBlocks := blockCache.getStats()
			Expect(activeRequests).To(Equal(2))
			Expect(totalBlocks).To(Equal(3))
			Expect(unusedBlocks).To(Equal(0))
			// Check block '1' reference count (should be 2)
			info, exists := blockCache.getBlockInfo(1)
			Expect(exists).To(BeTrue())
			Expect(info.refCount).To(Equal(2))

			err = blockCache.finishRequest(req1ID)
			Expect(err).NotTo(HaveOccurred())
			activeRequests, totalBlocks, unusedBlocks = blockCache.getStats()
			Expect(activeRequests).To(Equal(1))
			Expect(totalBlocks).To(Equal(3))
			Expect(unusedBlocks).To(Equal(1))
			// Check block '1' reference count (should be 1)
			info, exists = blockCache.getBlockInfo(1)
			Expect(exists).To(BeTrue())
			Expect(info.refCount).To(Equal(1))

		})
		It("block eviction", func() {
			// text eviction
			blockCache := newBlockCache(4)

			err := blockCache.startRequest(req1ID, []uint64{1, 2})
			Expect(err).NotTo(HaveOccurred())
			err = blockCache.startRequest(req2ID, []uint64{3, 4})
			Expect(err).NotTo(HaveOccurred())
			err = blockCache.finishRequest(req2ID)
			Expect(err).NotTo(HaveOccurred())
			info, exists := blockCache.getBlockInfo(3)
			Expect(exists).To(BeTrue())
			Expect(info.refCount).To(Equal(0))
			err = blockCache.startRequest(req3ID, []uint64{5, 6})
			Expect(err).NotTo(HaveOccurred())
			_, exists = blockCache.getBlockInfo(3)
			Expect(exists).To(BeFalse())
			info, exists = blockCache.getBlockInfo(5)
			Expect(exists).To(BeTrue())
			Expect(info.refCount).To(Equal(1))
		})
		It("cache full, no eviction", func() {
			// text eviction
			blockCache := newBlockCache(4)

			err := blockCache.startRequest(req1ID, []uint64{1, 2})
			Expect(err).NotTo(HaveOccurred())
			err = blockCache.startRequest(req2ID, []uint64{3, 4})
			Expect(err).NotTo(HaveOccurred())
			err = blockCache.startRequest(req3ID, []uint64{5, 6})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("cache is full and no blocks available for eviction"))
		})
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
