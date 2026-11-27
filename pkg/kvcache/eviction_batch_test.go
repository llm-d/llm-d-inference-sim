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
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// Compare against the existing scan implementation across interleaved release
// times, reused blocks, model-load changes between batches, and exhaustion.
func TestEvictionBatchMatchesScan(t *testing.T) {
	for _, modelCount := range []int{1, 7, 8, 64, 512} {
		for _, seed := range []int64{1, 2, 3, 4, 5} {
			t.Run(fmt.Sprintf("models_%d_seed_%d", modelCount, seed), func(t *testing.T) {
				rng := rand.New(rand.NewSource(seed))
				base, candidate := newUnusedBlocks(), newUnusedBlocks()
				now := time.Unix(0, 0)
				for round := 0; round < 6; round++ {
					for i := 0; i < modelCount*3; i++ {
						model := fmt.Sprintf("model-%d", rng.Intn(modelCount))
						keys := []blockKey{{hash: uint64(rng.Intn(32)), modelName: model}, {hash: uint64(32 + rng.Intn(32)), modelName: model}}
						now = now.Add(time.Duration(rng.Intn(2)) * time.Nanosecond)
						base.add(keys, now)
						candidate.add(keys, now)
						if rng.Intn(5) == 0 {
							base.remove(keys[0])
							candidate.remove(keys[0])
						}
					}
					loaded := make(map[string]struct{})
					for i := 0; i < modelCount; i++ {
						if round == 0 || (round > 1 && rng.Intn(2) == 0) {
							loaded[fmt.Sprintf("model-%d", i)] = struct{}{}
						}
					}
					count := modelCount*2 + 1
					if round == 5 {
						count = base.len() + 2
					}
					if round == 2 {
						count = 1
					}
					b := candidate.newEvictionBatch(loaded, count)
					for j := 0; j < count; j++ {
						want, wantOK := base.evict(loaded)
						var got blockKey
						var gotOK bool
						if b != nil {
							got, gotOK = b.evict()
						} else {
							got, gotOK = candidate.evict(loaded)
						}
						if got != want || gotOK != wantOK || candidate.len() != base.len() {
							t.Fatalf("round %d eviction %d: got %v %v len %d, want %v %v len %d", round, j, got, gotOK, candidate.len(), want, wantOK, base.len())
						}
					}
					for key := range base.blocks {
						if !candidate.contains(key) {
							t.Fatalf("missing remaining key %v", key)
						}
					}
				}
			})
		}
	}
}

func TestStartRequestBatchEviction(t *testing.T) {
	const capacity = 30
	config := &common.Configuration{
		IP:      "127.0.0.1",
		KVCache: common.KVCacheConfig{KVCacheSize: capacity},
	}
	cache, err := newBlockCache(t.Context(), config, logr.Discard(), nil, stubEncoder{})
	if err != nil {
		t.Fatal(err)
	}
	tokensFor := func(hashes []uint64) [][]uint32 {
		tokens := make([][]uint32, len(hashes))
		for i, hash := range hashes {
			tokens[i] = []uint32{uint32(hash)}
		}
		return tokens
	}
	for i := range 10 {
		model := fmt.Sprintf("adapter-%d", i)
		cache.setModelLoaded(model)
		first := uint64(3*i + 1)
		hashes := []uint64{first, first + 1, first + 2}
		if _, err := cache.startRequest(&testRequest{id: model, model: model}, hashes, tokensFor(hashes)); err != nil {
			t.Fatal(err)
		}
		if err := cache.finishRequest(model); err != nil {
			t.Fatal(err)
		}
		<-cache.eventChan.Channel
	}
	for _, model := range []string{"adapter-0", "adapter-7", "adapter-8", "adapter-9"} {
		cache.setModelUnloaded(model)
	}

	model, loraID := "adapter-0", 10
	request := &testRequest{id: "replacement", model: model, loraName: &model, loraID: &loraID}
	hashes := []uint64{1, 2, 3, 31, 32, 33, 34, 35, 36, 37, 38}
	if hits, err := cache.startRequest(request, hashes, tokensFor(hashes)); err != nil || hits != 3 {
		t.Fatalf("startRequest: got %d hits, error %v; want 3 hits", hits, err)
	}
	parent := uint64(3)
	want := make([]Event, 0, 9)
	for _, hash := range []uint64{24, 23, 22, 27, 26, 25, 30, 29} {
		want = append(want, Event{Action: ActionRemove, Hashes: []uint64{hash}, Tokens: []uint32{uint32(hash)}})
	}
	want = append(want, Event{
		Action: ActionStore, ParentHash: &parent, LoraName: &model, LoraID: &loraID,
		Hashes: hashes[3:], Tokens: []uint32{31, 32, 33, 34, 35, 36, 37, 38},
	})
	if got := len(cache.eventChan.Channel); got != len(want) {
		t.Fatalf("got %d KV events, want %d", got, len(want))
	}
	for i, expected := range want {
		if got := <-cache.eventChan.Channel; !reflect.DeepEqual(got, expected) {
			t.Fatalf("KV event %d: got %+v, want %+v", i, got, expected)
		}
	}
	if got := cache.cachedBlocks(model, hashes); got != len(hashes) {
		t.Fatalf("got %d cached request blocks, want %d", got, len(hashes))
	}
	if got := cache.cachedBlocks("adapter-9", []uint64{28, 29, 30}); got != 1 {
		t.Fatalf("got %d surviving blocks for adapter-9, want 1", got)
	}

	overflow := make([]uint64, 20)
	for i := range overflow {
		overflow[i] = uint64(100 + i)
	}
	rejected := &testRequest{id: "overflow", model: model}
	if _, err := cache.startRequest(rejected, overflow, tokensFor(overflow)); err == nil || err.Error() != capacityError {
		t.Fatalf("got error %v, want capacity error", err)
	}
	if requests, total, unused := cache.getStats(); requests != 1 || total != capacity || unused != capacity-len(hashes) {
		t.Fatalf("unexpected cache state after rejection: %d requests, %d blocks, %d unused", requests, total, unused)
	}
	if len(cache.eventChan.Channel) != 0 || len(cache.blockToTokens) != capacity {
		t.Fatal("rejection changed stored tokens or generated KV events")
	}
}

func BenchmarkEvictionBatch(b *testing.B) {
	for _, models := range []int{1, 8, 64, 512} {
		for _, count := range []int{1, 4, 8, 16, 512} {
			for _, optimized := range []bool{false, true} {
				b.Run(fmt.Sprintf("models_%d/evict_%d/batch_%t", models, count, optimized), func(b *testing.B) {
					b.ReportAllocs()
					for i := 0; i < b.N; i++ {
						b.StopTimer()
						u := newUnusedBlocks()
						loaded := make(map[string]struct{}, models)
						for m := 0; m < models; m++ {
							model := fmt.Sprintf("model-%d", m)
							keys := make([]blockKey, 8192/models)
							for k := range keys {
								keys[k] = blockKey{hash: uint64(k + 1), modelName: model}
							}
							u.add(keys, time.Unix(0, int64(m)))
							if m%2 == 0 {
								loaded[model] = struct{}{}
							}
						}
						b.StartTimer()
						if optimized {
							batch := u.newEvictionBatch(loaded, count)
							for j := 0; j < count; j++ {
								var ok bool
								if batch != nil {
									_, ok = batch.evict()
								} else {
									_, ok = u.evict(loaded)
								}
								if !ok {
									b.Fatal("empty")
								}
							}
						} else {
							for j := 0; j < count; j++ {
								if _, ok := u.evict(loaded); !ok {
									b.Fatal("empty")
								}
							}
						}
					}
				})
			}
		}
	}
}
