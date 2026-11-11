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
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
)

// gpu is the storage medium sglang reports for blocks in device memory.
const gpu = "GPU"

// The event types mirror sglang's KVCacheEvent classes
// (python/sglang/srt/disaggregation/kv_events.py): msgpack maps tagged with
// the class name under "type", with optional fields omitted while unset.
// parent_block_hash and lora_id have no default there, so they are always
// sent, as nil when unset. Block hashes are signed 64-bit integers.

type blockStoredEvent struct {
	Tag             string   `msgpack:"type"`
	BlockHashes     []int64  `msgpack:"block_hashes"`
	ParentBlockHash *int64   `msgpack:"parent_block_hash"`
	TokenIDs        []uint32 `msgpack:"token_ids"`
	BlockSize       int      `msgpack:"block_size"`
	// sglang names adapters with lora_name and leaves lora_id unset.
	LoraID    *int    `msgpack:"lora_id"`
	Medium    string  `msgpack:"medium"`
	CacheSalt *string `msgpack:"cache_salt,omitempty"`
	LoraName  *string `msgpack:"lora_name,omitempty"`
}

type blockRemovedEvent struct {
	Tag         string  `msgpack:"type"`
	BlockHashes []int64 `msgpack:"block_hashes"`
	Medium      string  `msgpack:"medium"`
}

type allBlocksClearedEvent struct {
	Tag string `msgpack:"type"`
}

// eventEncoder encodes KV-cache events in sglang's wire format.
type eventEncoder struct {
	blockSize int
}

// NewKVEventEncoder returns the encoder for sglang's KV-cache event format.
func (Engine) NewKVEventEncoder(cfg common.Configuration) (kvcache.EventEncoder, error) {
	return eventEncoder{blockSize: cfg.KVCache.TokenBlockSize}, nil
}

// EncodeEvent marshals ev in sglang's wire format. sglang sends one
// BlockStored per page; a store of several blocks is sent as one event with
// all of their hashes, which consumers read the same way.
func (e eventEncoder) EncodeEvent(ev kvcache.Event) ([]byte, error) {
	var event any
	switch ev.Action {
	case kvcache.ActionStore:
		var parent *int64
		if ev.ParentHash != nil {
			p := int64(*ev.ParentHash)
			parent = &p
		}
		event = &blockStoredEvent{
			Tag:             string(kvevents.EventTypeBlockStored),
			BlockHashes:     signedHashes(ev.Hashes),
			ParentBlockHash: parent,
			TokenIDs:        ev.Tokens,
			BlockSize:       e.blockSize,
			Medium:          gpu,
			CacheSalt:       ev.CacheSalt,
			LoraName:        ev.LoraName,
		}
	case kvcache.ActionRemove:
		event = &blockRemovedEvent{
			Tag:         string(kvevents.EventTypeBlockRemoved),
			BlockHashes: signedHashes(ev.Hashes),
			Medium:      gpu,
		}
	case kvcache.ActionAllBlocksCleared:
		event = &allBlocksClearedEvent{Tag: string(kvevents.EventTypeAllBlocksCleared)}
	default:
		return nil, fmt.Errorf("invalid event action %d", ev.Action)
	}

	encoded, err := msgpack.Marshal(event)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal event: %w", err)
	}
	return encoded, nil
}

func signedHashes(hashes []uint64) []int64 {
	signed := make([]int64, len(hashes))
	for i, h := range hashes {
		signed[i] = int64(h)
	}
	return signed
}
