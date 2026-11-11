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
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// expectConsistent checks that the map and the per-model lists describe the same blocks, and that
// every list is in eviction order.
func expectConsistent(u *unusedBlocks) {
	inLists := 0
	for model, l := range u.byModel {
		Expect(l.Len()).To(BeNumerically(">", 0), "model %s has an empty list", model)
		var prev *unusedBlock
		for elem := l.Front(); elem != nil; elem = elem.Next() {
			block := elem.Value.(*unusedBlock)
			Expect(block.key.modelName).To(Equal(model))
			Expect(u.blocks[block.key]).To(BeIdenticalTo(elem))
			if prev != nil {
				Expect(prev.evictedBefore(block)).To(BeTrue(), "model %s is out of eviction order", model)
			}
			prev = block
			inLists++
		}
	}
	Expect(inLists).To(Equal(u.len()))
}

func keys(model string, hashes ...uint64) []blockKey {
	out := make([]blockKey, len(hashes))
	for i, h := range hashes {
		out[i] = blockKey{hash: h, modelName: model}
	}
	return out
}

var _ = Describe("unused blocks", func() {
	now := time.Now()

	It("evicts a request's deepest block first, and earlier requests before later ones", func() {
		u := newUnusedBlocks()
		u.add(keys("m", 1, 2, 3), now)
		u.add(keys("m", 4, 5), now.Add(time.Second))
		expectConsistent(u)

		var evicted []uint64
		for u.len() > 0 {
			key, ok := u.evict(map[string]struct{}{"m": {}})
			Expect(ok).To(BeTrue())
			evicted = append(evicted, key.hash)
			expectConsistent(u)
		}
		Expect(evicted).To(Equal([]uint64{3, 2, 1, 5, 4}))
	})

	It("orders blocks freed at the same time by when they were added", func() {
		u := newUnusedBlocks()
		u.add(keys("m", 1, 2), now)
		u.add(keys("m", 3), now)
		order1, _ := u.orderOf(blockKey{hash: 1, modelName: "m"})
		order3, _ := u.orderOf(blockKey{hash: 3, modelName: "m"})
		Expect(order1).To(BeNumerically("<", order3))

		key, _ := u.evict(nil)
		Expect(key.hash).To(Equal(uint64(2)))
		key, _ = u.evict(nil)
		Expect(key.hash).To(Equal(uint64(1)))
	})

	It("evicts blocks of models that are not loaded before older blocks of loaded ones", func() {
		u := newUnusedBlocks()
		u.add(keys("base", 1), now)
		u.add(keys("lora1", 2), now.Add(time.Second))
		u.add(keys("lora2", 3), now.Add(2*time.Second))
		loaded := map[string]struct{}{"base": {}, "lora2": {}}

		var evicted []uint64
		for u.len() > 0 {
			key, _ := u.evict(loaded)
			evicted = append(evicted, key.hash)
			expectConsistent(u)
		}
		// lora1 is not loaded; then the base model's block is the oldest
		Expect(evicted).To(Equal([]uint64{2, 1, 3}))
	})

	It("removes blocks that are used again", func() {
		u := newUnusedBlocks()
		u.add(keys("a", 1, 2), now)
		u.add(keys("b", 3), now)
		Expect(u.remove(blockKey{hash: 3, modelName: "b"})).To(BeTrue())
		Expect(u.remove(blockKey{hash: 3, modelName: "b"})).To(BeFalse())
		Expect(u.remove(blockKey{hash: 1, modelName: "b"})).To(BeFalse(), "the same hash of another model is another block")
		Expect(u.contains(blockKey{hash: 1, modelName: "a"})).To(BeTrue())
		Expect(u.len()).To(Equal(2))
		Expect(u.byModel).NotTo(HaveKey("b"))
		expectConsistent(u)
	})

	It("moves a block that is added again to the back", func() {
		u := newUnusedBlocks()
		u.add(keys("m", 1), now)
		u.add(keys("m", 2), now)
		u.add(keys("m", 1), now.Add(time.Second))
		Expect(u.len()).To(Equal(2))
		expectConsistent(u)

		key, _ := u.evict(nil)
		Expect(key.hash).To(Equal(uint64(2)))
	})

	It("has nothing to evict when no block is unused", func() {
		u := newUnusedBlocks()
		_, ok := u.evict(nil)
		Expect(ok).To(BeFalse())
	})
})
