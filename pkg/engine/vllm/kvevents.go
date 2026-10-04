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

package vllm

import (
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
)

// gpu is the device tier reported as the events' medium.
const gpu = "GPU"

type msgpackBlockStoredEvent struct {
	//nolint:unused
	_msgpack        struct{} `msgpack:",as_array"`
	Tag             string
	BlockHashes     []any
	ParentBlockHash any
	TokenIds        []uint32
	BlockSize       int
	LoraID          *int    `msgpack:",omitempty"`
	Medium          *string `msgpack:",omitempty"`
	LoraName        *string `msgpack:",omitempty"`
	ExtraKeys       []any   `msgpack:",omitempty"`
}

// The Tag field encodes the struct type name under the "type" key, matching
// the map-encoded format parsed by VLLMAdapter.
type kvCacheEvent struct {
	Tag string `msgpack:"type"`
}

type blockStoredEvent struct {
	kvCacheEvent
	BlockHashes     []any    `msgpack:"block_hashes"`
	ParentBlockHash any      `msgpack:"parent_block_hash"`
	TokenIds        []uint32 `msgpack:"token_ids"`
	BlockSize       int      `msgpack:"block_size"`
	LoraID          *int     `msgpack:"lora_id,omitempty"`
	Medium          *string  `msgpack:"medium,omitempty"`
	LoraName        *string  `msgpack:"lora_name,omitempty"`
	// The following fields are part of the vLLM BlockStoredEvent schema (vllm-project/vllm#42892)
	// and are reserved for forward-compatibility. They are never populated by this simulator.
	ExtraKeys                []any   `msgpack:"extra_keys,omitempty"`
	GroupIdx                 *int    `msgpack:"group_idx,omitempty"`
	KVCacheSpecKind          *string `msgpack:"kv_cache_spec_kind,omitempty"`
	KVCacheSpecSlidingWindow *int    `msgpack:"kv_cache_spec_sliding_window,omitempty"`
}

type msgpackBlockRemovedEvent struct {
	//nolint:unused
	_msgpack    struct{} `msgpack:",as_array"`
	Tag         string
	BlockHashes []any
	Medium      *string `msgpack:",omitempty"`
}

type blockRemovedEvent struct {
	kvCacheEvent
	BlockHashes []any   `msgpack:"block_hashes"`
	Medium      *string `msgpack:"medium,omitempty"`
}

type msgpackAllBlocksClearedEvent struct {
	//nolint:unused
	_msgpack struct{} `msgpack:",as_array"`
	Tag      string
}

type allBlocksClearedEvent struct {
	kvCacheEvent
}

// eventEncoder encodes KV-cache events in vLLM's wire format.
type eventEncoder struct {
	// mapFormat selects named-field maps (vLLM PR #42892) over positional arrays.
	mapFormat bool
	blockSize int
}

// NewKVEventEncoder returns the encoder for vLLM's KV-cache event format.
func (Engine) NewKVEventEncoder(cfg common.Configuration) (kvcache.EventEncoder, error) {
	return eventEncoder{
		mapFormat: cfg.KVCache.UseVllmMapEventFormat,
		blockSize: cfg.KVCache.TokenBlockSize,
	}, nil
}

// EncodeEvent marshals ev in vLLM's wire format.
//
// In the positional format a nil ParentHash is encoded as 0; in the map format
// it is encoded as msgpack nil, matching vLLM's ExternalBlockHash | None
// contract. VLLMAdapter maps nil back to 0 when parsing, so consumers see
// ParentHash == 0 for both formats when there is no cached parent.
func (e eventEncoder) EncodeEvent(ev kvcache.Event) ([]byte, error) {
	var event any

	switch ev.Action {
	case kvcache.ActionStore:
		medium := gpu
		if e.mapFormat {
			var parentBlockHash any
			if ev.ParentHash != nil {
				parentBlockHash = *ev.ParentHash
			}
			event = &blockStoredEvent{
				kvCacheEvent:    kvCacheEvent{Tag: string(kvevents.EventTypeBlockStored)},
				BlockHashes:     convertUint64ToAnySlice(ev.Hashes),
				ParentBlockHash: parentBlockHash,
				TokenIds:        ev.Tokens,
				BlockSize:       e.blockSize,
				LoraID:          ev.LoraID,
				Medium:          &medium,
				LoraName:        ev.LoraName,
			}
			break
		}
		var parentBlockHash uint64
		if ev.ParentHash != nil {
			parentBlockHash = *ev.ParentHash
		}
		event = &msgpackBlockStoredEvent{
			Tag:             string(kvevents.EventTypeBlockStored),
			BlockHashes:     convertUint64ToAnySlice(ev.Hashes),
			ParentBlockHash: parentBlockHash,
			TokenIds:        ev.Tokens,
			BlockSize:       e.blockSize,
			LoraID:          ev.LoraID,
			Medium:          &medium,
			LoraName:        ev.LoraName,
		}

	case kvcache.ActionRemove:
		medium := gpu
		if e.mapFormat {
			event = &blockRemovedEvent{
				kvCacheEvent: kvCacheEvent{Tag: string(kvevents.EventTypeBlockRemoved)},
				BlockHashes:  convertUint64ToAnySlice(ev.Hashes),
				Medium:       &medium,
			}
			break
		}
		event = &msgpackBlockRemovedEvent{
			Tag:         string(kvevents.EventTypeBlockRemoved),
			BlockHashes: convertUint64ToAnySlice(ev.Hashes),
			Medium:      &medium,
		}

	case kvcache.ActionAllBlocksCleared:
		if e.mapFormat {
			event = &allBlocksClearedEvent{
				kvCacheEvent: kvCacheEvent{Tag: string(kvevents.EventTypeAllBlocksCleared)},
			}
			break
		}
		event = &msgpackAllBlocksClearedEvent{
			Tag: string(kvevents.EventTypeAllBlocksCleared),
		}

	default:
		return nil, fmt.Errorf("invalid event action %d", ev.Action)
	}

	encoded, err := msgpack.Marshal(event)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal event: %w", err)
	}
	return encoded, nil
}

func convertUint64ToAnySlice(input []uint64) []any {
	result := make([]any, len(input))
	for i, v := range input {
		result[i] = v
	}
	return result
}
