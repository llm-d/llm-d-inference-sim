package kvcache

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// blockInfo holds metadata about a cached block
type blockInfo struct {
	refCount int
	lastUsed time.Time
}

// blockCache represents a thread-safe cache for blocks with eviction policy
type blockCache struct {
	mu              sync.RWMutex
	requestToBlocks map[string][]uint64   // request id -> array of it blocks (block hashes)
	blockToInfo     map[uint64]*blockInfo // block hash -> block info
	maxBlocks       int                   // maximum number of blocks in the cache
}

// newBlockCache creates a new blockCache with the specified maximum number of blocks
func newBlockCache(maxBlocks int) *blockCache {
	return &blockCache{
		requestToBlocks: make(map[string][]uint64),
		blockToInfo:     make(map[uint64]*blockInfo),
		maxBlocks:       maxBlocks,
	}
}

// startRequest adds a request with its associated block hashes to the cache
func (bc *blockCache) startRequest(requestID string, blocks []uint64) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	if _, exists := bc.requestToBlocks[requestID]; exists {
		// request with the same id already exists
		return fmt.Errorf("request already exists for id %s", requestID)
	}

	// array of blocks that need to be added
	blocksToAdd := make([]uint64, 0)
	now := time.Now()

	for _, blockHash := range blocks {
		if info, exists := bc.blockToInfo[blockHash]; exists {
			// block already exists - increase reference count and update time
			info.refCount++
			info.lastUsed = now
		} else {
			// block doesn't exist - needs to be added
			blocksToAdd = append(blocksToAdd, blockHash)
		}
	}

	// try to add new blocks
	// TODO calculate how many blocks should be evicted and evict them at once instead of doing this one by one
	for _, blockHash := range blocksToAdd {
		if len(bc.blockToInfo) >= bc.maxBlocks {
			// cache is full, try to evict
			if !bc.evictOldestUnusedBlock() {
				// no blocks available for eviction
				return errors.New("cache is full and no blocks available for eviction")
			}
		}

		// Add the new block
		bc.blockToInfo[blockHash] = &blockInfo{
			refCount: 1,
			lastUsed: now,
		}
	}

	// store the request mapping
	bc.requestToBlocks[requestID] = make([]uint64, len(blocks))
	copy(bc.requestToBlocks[requestID], blocks)

	return nil
}

// evictOldestUnusedBlock finds and removes the oldest block with reference count 0
// Returns true if a block was evicted, false if no evictable blocks found
func (bc *blockCache) evictOldestUnusedBlock() bool {
	var oldestUnusedHash uint64
	var oldestUnusedTime time.Time
	found := false

	// find the oldest block with reference count == 0
	for hash, info := range bc.blockToInfo {
		if info.refCount == 0 {
			if !found || info.lastUsed.Before(oldestUnusedTime) {
				oldestUnusedHash = hash
				oldestUnusedTime = info.lastUsed
				found = true
			}
		}
	}

	if found {
		fmt.Printf("Block %d evicted\n", oldestUnusedHash)
		delete(bc.blockToInfo, oldestUnusedHash)
		return true
	}

	return false
}

// finishRequest processes the completion of a request, decreasing reference counts
func (bc *blockCache) finishRequest(requestID string) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	// Get blocks associated with this request
	blockHashes, exists := bc.requestToBlocks[requestID]
	if !exists {
		return errors.New("request not found")
	}

	now := time.Now()

	// Decrease reference count for each block
	for _, blockHash := range blockHashes {
		if info, exists := bc.blockToInfo[blockHash]; exists {
			info.refCount--
			info.lastUsed = now
		}
	}

	// Remove the request mapping
	delete(bc.requestToBlocks, requestID)

	return nil
}

// GetStats returns current cache statistics (for testing/debugging)
func (bc *blockCache) getStats() (int, int, int) {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	activeRequests := len(bc.requestToBlocks)
	totalBlocks := len(bc.blockToInfo)
	unusedBlocks := 0

	for _, info := range bc.blockToInfo {
		if info.refCount == 0 {
			unusedBlocks++
		}
	}

	return activeRequests, totalBlocks, unusedBlocks
}

// getBlockInfo returns information about a specific block (for testing)
func (bc *blockCache) getBlockInfo(blockHash uint64) (*blockInfo, bool) {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	info, exists := bc.blockToInfo[blockHash]
	if !exists {
		return nil, false
	}

	// Return a copy to avoid race conditions
	return &blockInfo{
		refCount: info.refCount,
		lastUsed: info.lastUsed,
	}, true
}

// // Test functions
// func main() {
// 	fmt.Println("Testing BlockCache...")

// 	// Test 1: Basic functionality
// 	fmt.Println("\n=== Test 1: Basic Functionality ===")
// 	cache := NewBlockCache(3)

// 	// Start a request with 2 blocks
// 	err := cache.StartRequest("req1", []string{"block1", "block2"})
// 	if err != nil {
// 		fmt.Printf("Error: %v\n", err)
// 		return
// 	}

// 	activeReqs, totalBlocks, unusedBlocks := cache.GetStats()
// 	fmt.Printf("After req1: Active requests: %d, Total blocks: %d, Unused blocks: %d\n",
// 		activeReqs, totalBlocks, unusedBlocks)

// 	// Test 2: Reusing blocks
// 	fmt.Println("\n=== Test 2: Block Reuse ===")
// 	err = cache.StartRequest("req2", []string{"block1", "block3"})
// 	if err != nil {
// 		fmt.Printf("Error: %v\n", err)
// 		return
// 	}

// 	// Check block1 reference count (should be 2)
// 	if info, exists := cache.GetBlockInfo("block1"); exists {
// 		fmt.Printf("block1 reference count: %d\n", info.refCount)
// 	}

// 	activeReqs, totalBlocks, unusedBlocks = cache.GetStats()
// 	fmt.Printf("After req2: Active requests: %d, Total blocks: %d, Unused blocks: %d\n",
// 		activeReqs, totalBlocks, unusedBlocks)

// 	// Test 3: Cache eviction
// 	fmt.Println("\n=== Test 3: Cache Eviction ===")

// 	// Finish req1 to free up some blocks
// 	err = cache.FinishRequest("req1")
// 	if err != nil {
// 		fmt.Printf("Error: %v\n", err)
// 		return
// 	}

// 	activeReqs, totalBlocks, unusedBlocks = cache.GetStats()
// 	fmt.Printf("After finishing req1: Active requests: %d, Total blocks: %d, Unused blocks: %d\n",
// 		activeReqs, totalBlocks, unusedBlocks)

// 	// Try to add a request that would exceed capacity
// 	time.Sleep(10 * time.Millisecond) // Small delay to ensure different timestamps
// 	err = cache.StartRequest("req3", []string{"block4", "block5"})
// 	if err != nil {
// 		fmt.Printf("Error: %v\n", err)
// 		return
// 	}

// 	activeReqs, totalBlocks, unusedBlocks = cache.GetStats()
// 	fmt.Printf("After req3 (with eviction): Active requests: %d, Total blocks: %d, Unused blocks: %d\n",
// 		activeReqs, totalBlocks, unusedBlocks)

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
