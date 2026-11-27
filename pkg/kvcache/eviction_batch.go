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

import "container/heap"

// evictionBatch merges the ordered per-model lists for a sequence of evictions.
// Its lifetime is limited to one startRequest under the blockCache mutex. No
// other changes to unused blocks or loadedModels may occur during that lifetime.
type evictionBatch struct {
	unused     *unusedBlocks
	candidates evictionHeap
}

// newEvictionBatch builds a heap only when both dimensions justify its setup.
func (u *unusedBlocks) newEvictionBatch(loadedModels map[string]struct{}, expectedEvictions int) *evictionBatch {
	if len(u.byModel) < 8 || expectedEvictions < 8 {
		return nil
	}
	b := &evictionBatch{unused: u}
	b.candidates = make(evictionHeap, 0, len(u.byModel))
	for model, l := range u.byModel {
		_, loaded := loadedModels[model]
		b.candidates = append(b.candidates, evictionCandidate{
			block: l.Front().Value.(*unusedBlock), loaded: loaded,
		})
	}
	heap.Init(&b.candidates)
	return b
}

func (b *evictionBatch) evict() (blockKey, bool) {
	if len(b.candidates) == 0 {
		return blockKey{}, false
	}
	key := b.candidates[0].block.key
	l := b.unused.byModel[key.modelName]
	b.unused.remove(key)
	if l.Len() == 0 {
		last := len(b.candidates) - 1
		b.candidates[0] = b.candidates[last]
		b.candidates[last] = evictionCandidate{}
		b.candidates = b.candidates[:last]
	} else {
		b.candidates[0].block = l.Front().Value.(*unusedBlock)
	}
	if len(b.candidates) > 1 {
		heap.Fix(&b.candidates, 0)
	}
	return key, true
}

type evictionCandidate struct {
	block  *unusedBlock
	loaded bool
}

type evictionHeap []evictionCandidate

func (h evictionHeap) Len() int { return len(h) }
func (h evictionHeap) Less(i, j int) bool {
	if h[i].loaded != h[j].loaded {
		return !h[i].loaded
	}
	return h[i].block.evictedBefore(h[j].block)
}
func (h evictionHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *evictionHeap) Push(x any)   { *h = append(*h, x.(evictionCandidate)) }
func (h *evictionHeap) Pop() any {
	old := *h
	n := len(old) - 1
	x := old[n]
	old[n] = evictionCandidate{}
	*h = old[:n]
	return x
}
