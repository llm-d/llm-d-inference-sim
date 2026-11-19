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
	"fmt"
	"testing"
	"time"
)

// BenchmarkUnusedBlockEvictModelCardinality isolates how eviction time changes
// as the number of distinct model/LoRA lists grows while block count stays fixed.
func BenchmarkUnusedBlockEvictModelCardinality(b *testing.B) {
	const totalBlocks = 8192
	const evictCount = 512
	for _, modelCount := range []int{1, 8, 64, 512} {
		b.Run(fmt.Sprintf("models_%d", modelCount), func(b *testing.B) {
			b.ReportAllocs()
			for iteration := 0; iteration < b.N; iteration++ {
				b.StopTimer()
				u := newUnusedBlocks()
				loaded := make(map[string]struct{}, modelCount)
				now := time.Unix(0, 0)
				blocksPerModel := totalBlocks / modelCount
				for modelIndex := 0; modelIndex < modelCount; modelIndex++ {
					model := fmt.Sprintf("model-%d", modelIndex)
					blocks := make([]blockKey, blocksPerModel)
					for blockIndex := range blocks {
						blocks[blockIndex] = blockKey{
							hash:      uint64(modelIndex*blocksPerModel + blockIndex + 1),
							modelName: model,
						}
					}
					u.add(blocks, now.Add(time.Duration(modelIndex)*time.Nanosecond))
					if modelIndex%2 == 0 {
						loaded[model] = struct{}{}
					}
				}
				b.StartTimer()
				for evicted := 0; evicted < evictCount; evicted++ {
					if _, ok := u.evict(loaded); !ok {
						b.Fatal("ran out of blocks during benchmark")
					}
				}
			}
		})
	}
}
