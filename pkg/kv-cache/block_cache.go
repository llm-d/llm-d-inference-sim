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
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const (
	capacityError = "the kv cache does not have sufficient capacity to store this request"
	delay         = time.Second
)

// blockCache represents a thread-safe cache for blocks with eviction policy
type blockCache struct {
	mu              sync.RWMutex
	requestToBlocks map[string][]uint64  // request id -> array of it blocks (block hashes)
	usedBlocks      map[uint64]int       // block hash -> reference count
	unusedBlocks    map[uint64]time.Time // block hash -> last usage timestamp
	maxBlocks       int                  // maximum number of blocks in the cache
	eventSender     *KVEventSender       // emmits kv events
	eventChan       chan EventData       // channel for asynchronous event processing
	logger          logr.Logger
}

// newBlockCache creates a new blockCache with the specified maximum number of blocks
func newBlockCache(config *common.Configuration, logger logr.Logger) (*blockCache, error) {
	// TODO read size of channel from config
	eChan := make(chan EventData, 10000)

	publisher, err := common.NewPublisher(config.ZMQEndpoint, config.ZMQMaxConnectAttempts)
	if err != nil {
		return nil, err
	}

	return &blockCache{
		requestToBlocks: make(map[string][]uint64),
		usedBlocks:      make(map[uint64]int),
		unusedBlocks:    make(map[uint64]time.Time),
		maxBlocks:       config.KVCacheSize,
		eventChan:       eChan,
		eventSender:     NewKVEventSender(publisher, createTopic(config), eChan, config.EventBatchSize, delay, logger),
		logger:          logger,
	}, nil
}

func (b *blockCache) start(ctx context.Context) {
	err := b.eventSender.Run(ctx)
	if err != nil {
		b.logger.Info("sender stopped with error", "error", err)
	}
}

// startRequest adds a request with its associated block hashes to the cache
func (b *blockCache) startRequest(requestID string, blocks []uint64) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.requestToBlocks[requestID]; exists {
		// request with the same id already exists
		return fmt.Errorf("request already exists for id %s", requestID)
	}

	// divide list of blocks to three lists:
	// blockAreadyInUse - blocks, which are already used by currently running request
	// blockToMoveToUsed - blocks, which were used in past
	// blocksToAdd - new blocks
	blocksToAdd := make([]uint64, 0)
	blockToMoveToUsed := make([]uint64, 0)
	blockAreadyInUse := make([]uint64, 0)

	// first step - ensure that there is enough space for all blocks
	// count number of new blocks + number of blocks that are in the unused blocks
	// don't update the data until we are sure that it's ok
	for _, blockHash := range blocks {
		if _, exists := b.unusedBlocks[blockHash]; exists {
			blockToMoveToUsed = append(blockToMoveToUsed, blockHash)
		} else if _, exists := b.usedBlocks[blockHash]; !exists {
			blocksToAdd = append(blocksToAdd, blockHash)
		} else {
			blockAreadyInUse = append(blockAreadyInUse, blockHash)
		}
	}

	if len(b.usedBlocks)+len(blocksToAdd)+len(blockToMoveToUsed) > b.maxBlocks {
		return errors.New(capacityError)
	}

	// for blocks that are already in use - update the reference
	for _, block := range blockAreadyInUse {
		b.usedBlocks[block]++
	}

	// for block used in the past - move them to the used blocks collection
	for _, block := range blockToMoveToUsed {
		b.usedBlocks[block] = 1
		delete(b.unusedBlocks, block)
	}

	// for new block - add them, if there is no empty slots - evict the oldest block
	for _, block := range blocksToAdd {
		if len(b.usedBlocks)+len(b.unusedBlocks) == b.maxBlocks {
			// cache is full but contains unused blocks - evict the oldest
			var oldestUnusedHash uint64
			oldestUnusedTime := time.Now()

			for hash, t := range b.unusedBlocks {
				if t.Before(oldestUnusedTime) {
					oldestUnusedHash = hash
					oldestUnusedTime = t
				}
			}

			delete(b.unusedBlocks, oldestUnusedHash)
			b.eventChan <- EventData{action: eventActionRemove, hashValues: []uint64{oldestUnusedHash}}
		}

		// Add the new block
		b.usedBlocks[block] = 1
		b.eventChan <- EventData{action: eventActionStore, hashValues: []uint64{block}}
	}

	// store the request mapping
	b.requestToBlocks[requestID] = make([]uint64, len(blocks))
	copy(b.requestToBlocks[requestID], blocks)

	return nil
}

// finishRequest processes the completion of a request, decreasing reference counts
func (b *blockCache) finishRequest(requestID string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Get blocks associated with this request
	blockHashes, exists := b.requestToBlocks[requestID]
	if !exists {
		return errors.New("request not found")
	}

	now := time.Now()

	// Decrease reference count for each block
	errBlocks := make([]uint64, 0)
	for _, blockHash := range blockHashes {
		if refCount, exists := b.usedBlocks[blockHash]; exists {
			if refCount > 1 {
				// this block is in use by another request, just update reference count
				b.usedBlocks[blockHash] = refCount - 1
			} else {
				// this was the last block usage - move this block to unused
				b.unusedBlocks[blockHash] = now
				delete(b.usedBlocks, blockHash)
			}
		} else {
			errBlocks = append(errBlocks, blockHash)
		}
	}

	// Remove the request mapping
	delete(b.requestToBlocks, requestID)

	if len(errBlocks) > 0 {
		errMsg := "Not existing blocks "
		for _, bl := range errBlocks {
			errMsg += fmt.Sprintf("%d, ", bl)
		}
		return fmt.Errorf("%s for request %s", errMsg[:len(errMsg)-2], requestID)
	}

	return nil
}

// GetStats returns current cache statistics (for testing/debugging)
func (b *blockCache) getStats() (int, int, int) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	return len(b.requestToBlocks), len(b.usedBlocks) + len(b.unusedBlocks), len(b.unusedBlocks)
}

// getBlockInfo returns reference count and if it's in the cache for a specific block (for testing)
// if block is in use by currently running requests the count will be positive, boolean is true
// if block is in the unused list - count is 0, boolean is true
// if block is not in both collections - count is 0, boolean is false
func (b *blockCache) getBlockInfo(blockHash uint64) (int, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	refCount, exists := b.usedBlocks[blockHash]
	if exists {
		return refCount, true
	}
	_, exists = b.unusedBlocks[blockHash]
	if exists {
		return 0, true
	}

	return 0, false
}

func createTopic(config *common.Configuration) string {
	return fmt.Sprintf("kv@$localhost:%d@%s", config.Port, config.Model)
}
