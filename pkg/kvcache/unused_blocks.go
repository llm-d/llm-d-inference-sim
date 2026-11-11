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

package kvcache

import (
	"container/list"
	"time"
)

// unusedBlock is a cached block that no running request uses.
type unusedBlock struct {
	key   blockKey
	since time.Time // when the last request using it finished
	order uint64    // among blocks freed at the same time, the lower order is evicted first
}

// unusedBlocks holds the cached blocks that no running request uses, in the order they are
// evicted. It owns all of their state: one map finds a block, and each model's blocks are in a
// list with the next to evict at the front, so picking a block to evict only compares the fronts
// of the lists. It is not safe for concurrent use; blockCache guards it with its mutex.
type unusedBlocks struct {
	blocks  map[blockKey]*list.Element // block -> its element in its model's list (Value is *unusedBlock)
	byModel map[string]*list.List      // model name -> its blocks, the next to evict at the front
	order   uint64                     // the order given to the last block added
}

func newUnusedBlocks() *unusedBlocks {
	return &unusedBlocks{
		blocks:  make(map[blockKey]*list.Element),
		byModel: make(map[string]*list.List),
	}
}

// len returns the number of unused blocks.
func (u *unusedBlocks) len() int {
	return len(u.blocks)
}

// contains reports whether a block is unused.
func (u *unusedBlocks) contains(key blockKey) bool {
	_, exists := u.blocks[key]
	return exists
}

// add adds the blocks that a request stopped using when it finished at now, given in the order of
// the request, its prefix first. They are newer than every block already here (requests finish one
// at a time, so now is never earlier), and among them the deepest block is evicted first, so the
// shared prefix stays cached the longest.
func (u *unusedBlocks) add(keys []blockKey, now time.Time) {
	for i := len(keys) - 1; i >= 0; i-- {
		key := keys[i]
		u.remove(key) // a block that was already unused becomes the newest
		u.order++
		l, exists := u.byModel[key.modelName]
		if !exists {
			l = list.New()
			u.byModel[key.modelName] = l
		}
		u.blocks[key] = l.PushBack(&unusedBlock{key: key, since: now, order: u.order})
	}
}

// remove removes a block that is used again or evicted, and reports whether it was unused.
func (u *unusedBlocks) remove(key blockKey) bool {
	elem, exists := u.blocks[key]
	if !exists {
		return false
	}
	l := u.byModel[key.modelName]
	l.Remove(elem)
	if l.Len() == 0 {
		delete(u.byModel, key.modelName)
	}
	delete(u.blocks, key)
	return true
}

// evict removes and returns the block to evict, using priority:
// 1. oldest unused block of an unloaded model
// 2. oldest unused block of any model
// It returns false if there are no unused blocks.
func (u *unusedBlocks) evict(loadedModels map[string]struct{}) (blockKey, bool) {
	var best *unusedBlock
	bestUnloaded := false
	for model, l := range u.byModel {
		block := l.Front().Value.(*unusedBlock)
		_, loaded := loadedModels[model]
		unloaded := !loaded
		switch {
		case best == nil, unloaded && !bestUnloaded:
			// the first candidate, or the first of an unloaded model after loaded ones
			best, bestUnloaded = block, unloaded
		case unloaded == bestUnloaded && block.evictedBefore(best):
			best = block
		}
	}
	if best == nil {
		return blockKey{}, false
	}
	u.remove(best.key)
	return best.key, true
}

// evictedBefore reports whether b goes before other in eviction order.
func (b *unusedBlock) evictedBefore(other *unusedBlock) bool {
	return b.since.Before(other.since) || (b.since.Equal(other.since) && b.order < other.order)
}

// orderOf returns the order of an unused block, for tests.
func (u *unusedBlocks) orderOf(key blockKey) (uint64, bool) {
	elem, exists := u.blocks[key]
	if !exists {
		return 0, false
	}
	return elem.Value.(*unusedBlock).order, true
}
