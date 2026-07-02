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
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/vmihailenco/msgpack/v5"
)

type EventAction int

const (
	eventActionStore EventAction = iota
	eventActionRemove
	eventActionAllBlocksCleared
)

var GPU string = "GPU"

type msgpackEventBatch struct {
	//nolint:unused
	_msgpack         struct{} `msgpack:",as_array"`
	TS               float64
	Events           []msgpack.RawMessage
	DataParallelRank *int `msgpack:",omitempty"`
}

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

type EventData struct {
	action     EventAction
	tokens     []uint32
	hashes     []uint64
	parentHash *uint64 // nil means no parent (first block of sequence); non-nil is the last already-cached block hash
	loraName   *string
	loraID     *int
}

// batchEntry pairs a generic event with the original parentHash pointer so that
// the map-format encoder can distinguish "no parent" (nil) from a real zero hash,
// without relying on EmptyBlockHash=0 as a sentinel in the uint64 field.
type batchEntry struct {
	event      kvevents.GenericEvent
	parentHash *uint64 // only meaningful for BlockStoredEvent; nil = no parent
}

type KVEventSender struct {
	publisher             *common.Publisher
	topic                 string
	eventChan             common.Channel[EventData]
	maxBatchSize          int
	blockSize             int
	delay                 time.Duration
	batch                 []batchEntry
	logger                logr.Logger
	useVllmMapEventFormat bool
	dpRank                int
}

func NewKVEventSender(publisher *common.Publisher, topic string, ch common.Channel[EventData], maxBatchSize int,
	blockSize int, delay time.Duration, useVllmMapEventFormat bool, dpRank int, logger logr.Logger) *KVEventSender {
	return &KVEventSender{
		publisher:             publisher,
		topic:                 topic,
		eventChan:             ch,
		maxBatchSize:          maxBatchSize,
		blockSize:             blockSize,
		delay:                 delay,
		batch:                 make([]batchEntry, 0, maxBatchSize),
		logger:                logger,
		useVllmMapEventFormat: useVllmMapEventFormat,
		dpRank:                dpRank,
	}
}

func (s *KVEventSender) Run(ctx context.Context) error {
	timer := time.NewTimer(s.delay)
	defer timer.Stop()

	for {
		select {
		case <-ctx.Done():
			// Exiting, discard remaining events if any
			if len(s.batch) > 0 {
				s.logger.V(logging.INFO).Info("Exiting, discard remaining events", "num of events", len(s.batch))
			}
			return ctx.Err()

		case eventData, ok := <-s.eventChan.Channel:
			if !ok {
				// Channel closed, discard remaining events and exit
				if len(s.batch) > 0 {
					s.logger.V(logging.INFO).Info("Channel closed, discard remaining events", "num of events", len(s.batch))
				}
				return nil
			}

			if s.publisher == nil {
				continue
			}

			// Encode eventData's hash value to msgpack.RawMessage
			var entry batchEntry

			switch eventData.action {
			case eventActionStore:
				// Preserve parentHash as *uint64 so publishHelper can distinguish
				// nil (no parent) from a real hash that happens to be zero.
				ph := uint64(kvblock.EmptyBlockHash)
				if eventData.parentHash != nil {
					ph = *eventData.parentHash
				}
				entry = batchEntry{
					event: &kvevents.BlockStoredEvent{
						BlockHashes: eventData.hashes,
						Tokens:      eventData.tokens,
						DeviceTier:  GPU,
						ParentHash:  ph,
						LoraID:      eventData.loraID,
						LoraName:    eventData.loraName,
					},
					parentHash: eventData.parentHash,
				}
			case eventActionRemove:
				entry = batchEntry{event: &kvevents.BlockRemovedEvent{BlockHashes: eventData.hashes, DeviceTier: GPU}}
			case eventActionAllBlocksCleared:
				entry = batchEntry{event: &kvevents.AllBlocksClearedEvent{DeviceTier: GPU}}
			default:
				return fmt.Errorf("invalid event action %d", eventData.action)
			}

			s.batch = append(s.batch, entry)

			// check if batch is big enough to be sent
			if len(s.batch) >= s.maxBatchSize {
				if err := s.publishHelper(ctx); err != nil {
					return err
				}

				// reset timer
				if !timer.Stop() {
					<-timer.C
				}
				timer.Reset(s.delay)
			}

		case <-timer.C:
			if s.publisher == nil {
				continue
			}
			if err := s.publishHelper(ctx); err != nil {
				return err
			}
			timer.Reset(s.delay)
		}
	}
}

// encodeEvent converts a batchEntry to the msgpack-ready struct for the chosen
// wire format. mapFormat selects named-field maps (vLLM PR #42892); otherwise
// positional arrays (legacy) are used. blockSize is embedded in BlockStoredEvent.
//
// For BlockStoredEvent in map format, entry.parentHash (*uint64) is used rather
// than e.ParentHash (uint64) so that nil (no cached parent) is encoded as msgpack
// nil rather than 0, matching vLLM's ExternalBlockHash | None contract.
func encodeEvent(entry batchEntry, mapFormat bool, blockSize int) (interface{}, error) {
	switch e := entry.event.(type) {
	case *kvevents.BlockStoredEvent:
		if mapFormat {
			// In the map format, parent_block_hash is nil when there is no parent
			// (first block of the sequence), matching vLLM's ExternalBlockHash | None.
			// VLLMAdapter maps nil back to EmptyBlockHash (0) when parsing, so consumers
			// see ParentHash == 0 for both formats when there is no cached parent.
			var parentBlockHash any
			if entry.parentHash != nil {
				parentBlockHash = *entry.parentHash
			}
			return &blockStoredEvent{
				kvCacheEvent:    kvCacheEvent{Tag: string(kvevents.EventTypeBlockStored)},
				BlockHashes:     convertUint64ToAnySlice(e.BlockHashes),
				TokenIds:        e.Tokens,
				Medium:          &e.DeviceTier,
				LoraID:          e.LoraID,
				LoraName:        e.LoraName,
				ParentBlockHash: parentBlockHash,
				BlockSize:       blockSize,
			}, nil
		}
		return &msgpackBlockStoredEvent{
			Tag:             string(kvevents.EventTypeBlockStored),
			BlockHashes:     convertUint64ToAnySlice(e.BlockHashes),
			TokenIds:        e.Tokens,
			Medium:          &e.DeviceTier,
			LoraID:          e.LoraID,
			LoraName:        e.LoraName,
			ParentBlockHash: e.ParentHash,
			BlockSize:       blockSize,
		}, nil
	case *kvevents.BlockRemovedEvent:
		if mapFormat {
			return &blockRemovedEvent{
				kvCacheEvent: kvCacheEvent{Tag: string(kvevents.EventTypeBlockRemoved)},
				BlockHashes:  convertUint64ToAnySlice(e.BlockHashes),
				Medium:       &GPU,
			}, nil
		}
		return &msgpackBlockRemovedEvent{
			Tag:         string(kvevents.EventTypeBlockRemoved),
			BlockHashes: convertUint64ToAnySlice(e.BlockHashes),
			Medium:      &GPU,
		}, nil
	case *kvevents.AllBlocksClearedEvent:
		if mapFormat {
			return &allBlocksClearedEvent{
				kvCacheEvent: kvCacheEvent{Tag: string(kvevents.EventTypeAllBlocksCleared)},
			}, nil
		}
		return &msgpackAllBlocksClearedEvent{
			Tag: string(kvevents.EventTypeAllBlocksCleared),
		}, nil
	default:
		return nil, fmt.Errorf("unknown generic event type: %T", entry.event)
	}
}

// helper to publish collected batch if not empty
func (s *KVEventSender) publishHelper(ctx context.Context) error {
	if len(s.batch) == 0 {
		return nil
	}

	events := []msgpack.RawMessage{}

	for _, entry := range s.batch {
		raw, err := encodeEvent(entry, s.useVllmMapEventFormat, s.blockSize)
		if err != nil {
			return err
		}

		eventBytes, err := msgpack.Marshal(raw)
		if err != nil {
			return fmt.Errorf("failed to marshal event: %w", err)
		}
		events = append(events, msgpack.RawMessage(eventBytes))
	}

	dpRank := s.dpRank

	batch := msgpackEventBatch{
		TS:               float64(time.Now().UnixNano()) / 1e9,
		Events:           events,
		DataParallelRank: &dpRank,
	}

	err := s.publisher.PublishEvent(ctx, s.topic, batch)

	// reset batch
	s.batch = make([]batchEntry, 0, s.maxBatchSize)

	return err
}

func convertUint64ToAnySlice(input []uint64) []any {
	result := make([]any, len(input))
	for i, v := range input {
		result[i] = v
	}
	return result
}
