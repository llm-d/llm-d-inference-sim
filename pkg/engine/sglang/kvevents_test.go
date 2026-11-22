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

package sglang

import (
	"bytes"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/vmihailenco/msgpack/v5"
	"github.com/vmihailenco/msgpack/v5/msgpcode"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
)

// encodeEvent encodes ev with a page size of 4 and returns its keys with their
// raw values.
func encodeEvent(ev kvcache.Event) map[string]msgpack.RawMessage {
	var fields map[string]msgpack.RawMessage
	Expect(msgpack.Unmarshal(encodeRaw(ev), &fields)).To(Succeed())
	return fields
}

// encodeLoose encodes ev and decodes it into plain values, nil included.
func encodeLoose(ev kvcache.Event) map[string]any {
	var fields map[string]any
	Expect(msgpack.Unmarshal(encodeRaw(ev), &fields)).To(Succeed())
	return fields
}

func encodeRaw(ev kvcache.Event) []byte {
	cfg := common.Configuration{}
	cfg.KVCache.TokenBlockSize = 4
	enc, err := New().NewKVEventEncoder(cfg)
	Expect(err).NotTo(HaveOccurred())
	encoded, err := enc.EncodeEvent(ev)
	Expect(err).NotTo(HaveOccurred())
	return encoded
}

func keysOf(fields map[string]msgpack.RawMessage) []string {
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	return keys
}

func decodeField(fields map[string]msgpack.RawMessage, key string, v any) {
	Expect(fields).To(HaveKey(key))
	Expect(msgpack.Unmarshal(fields[key], v)).To(Succeed())
}

var _ = Describe("sglang KV-event encoder", func() {
	store := kvcache.Event{
		Action: kvcache.ActionStore,
		Hashes: []uint64{1, 2},
		Tokens: []uint32{10, 11, 12, 13, 20, 21, 22, 23},
	}

	It("encodes a store as a BlockStored map with sglang's required keys", func() {
		fields := encodeEvent(store)
		Expect(keysOf(fields)).To(ConsistOf(
			"type", "block_hashes", "parent_block_hash", "token_ids", "block_size", "lora_id", "medium"))

		var tag, medium string
		var hashes []int64
		var tokens []uint32
		var blockSize int
		decodeField(fields, "type", &tag)
		decodeField(fields, "block_hashes", &hashes)
		decodeField(fields, "token_ids", &tokens)
		decodeField(fields, "block_size", &blockSize)
		decodeField(fields, "medium", &medium)
		Expect(tag).To(Equal("BlockStored"))
		Expect(hashes).To(Equal([]int64{1, 2}))
		Expect(tokens).To(Equal(store.Tokens))
		Expect(blockSize).To(Equal(4))
		Expect(medium).To(Equal(gpu))

		loose := encodeLoose(store)
		Expect(loose["parent_block_hash"]).To(BeNil())
		Expect(loose["lora_id"]).To(BeNil())
	})

	It("sends block hashes as signed 64-bit integers", func() {
		high := uint64(1)<<63 | 5
		ev := store
		ev.Hashes = []uint64{high}
		ev.ParentHash = &high

		fields := encodeEvent(ev)
		var hashes []int64
		var parent int64
		decodeField(fields, "block_hashes", &hashes)
		decodeField(fields, "parent_block_hash", &parent)
		Expect(hashes).To(Equal([]int64{int64(high)}))
		Expect(parent).To(Equal(int64(high)))

		dec := msgpack.NewDecoder(bytes.NewReader(fields["block_hashes"]))
		_, err := dec.DecodeArrayLen()
		Expect(err).NotTo(HaveOccurred())
		code, err := dec.PeekCode()
		Expect(err).NotTo(HaveOccurred())
		Expect(code).To(Equal(msgpcode.Int64))
	})

	It("sends the parent hash, cache salt and adapter name when set", func() {
		parent := uint64(7)
		salt := "tenant-a"
		lora := "lora1"
		ev := store
		ev.ParentHash = &parent
		ev.CacheSalt = &salt
		ev.LoraName = &lora

		fields := encodeEvent(ev)
		var gotParent int64
		var gotSalt, gotLora string
		decodeField(fields, "parent_block_hash", &gotParent)
		decodeField(fields, "cache_salt", &gotSalt)
		decodeField(fields, "lora_name", &gotLora)
		Expect(gotParent).To(Equal(int64(7)))
		Expect(gotSalt).To(Equal(salt))
		Expect(gotLora).To(Equal(lora))
		Expect(encodeLoose(ev)["lora_id"]).To(BeNil())
	})

	It("encodes removals and clears", func() {
		removed := encodeEvent(kvcache.Event{Action: kvcache.ActionRemove, Hashes: []uint64{3, 4}})
		Expect(keysOf(removed)).To(ConsistOf("type", "block_hashes", "medium"))
		var tag string
		var hashes []int64
		decodeField(removed, "type", &tag)
		decodeField(removed, "block_hashes", &hashes)
		Expect(tag).To(Equal("BlockRemoved"))
		Expect(hashes).To(Equal([]int64{3, 4}))

		cleared := encodeEvent(kvcache.Event{Action: kvcache.ActionAllBlocksCleared})
		Expect(keysOf(cleared)).To(ConsistOf("type"))
		decodeField(cleared, "type", &tag)
		Expect(tag).To(Equal("AllBlocksCleared"))
	})

	It("rejects an unknown action", func() {
		enc, err := New().NewKVEventEncoder(common.Configuration{})
		Expect(err).NotTo(HaveOccurred())
		_, err = enc.EncodeEvent(kvcache.Event{Action: kvcache.EventAction(42)})
		Expect(err).To(HaveOccurred())
	})
})
