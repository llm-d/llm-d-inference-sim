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
	"encoding/binary"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var vllmAdapter *engineadapter.VLLMAdapter = engineadapter.NewVLLMAdapter()

// StoredEventInfo holds parsed metadata from a single BlockStoredEvent
type StoredEventInfo struct {
	BlockHashes []uint64
	LoraName    *string
	LoraID      *int
}

func parseBatch(parts [][]byte, expectedTopic string, expectedSeq uint64) kvevents.EventBatch {
	// The message should be [topic, seq, payload]
	gomega.Expect(parts).To(gomega.HaveLen(3))
	gomega.Expect(string(parts[0])).To(gomega.Equal(expectedTopic))

	seq := binary.BigEndian.Uint64(parts[1])
	gomega.Expect(seq).To(gomega.Equal(expectedSeq))

	rawMsg := kvevents.RawMessage{
		Topic:    string(parts[0]),
		Sequence: seq,
		Payload:  parts[2],
	}

	_, _, batch, err := vllmAdapter.ParseMessage(&rawMsg)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	return batch
}

// ParseKVEvent parses a ZMQ message and returns rich stored event info (with LoRA metadata),
// removed block hashes, and whether an all-blocks-cleared event was received.
func ParseKVEvent(parts [][]byte, expectedTopic string, expectedSeq uint64) ([]StoredEventInfo, []uint64, bool) {
	batch := parseBatch(parts, expectedTopic, expectedSeq)

	removed := make([]uint64, 0)
	storedEvents := make([]StoredEventInfo, 0)
	allCleared := false

	for _, genEvent := range batch.Events {
		switch genEvent.Type() {
		case kvevents.EventTypeBlockStored:
			storeEvent, ok := genEvent.(*kvevents.BlockStoredEvent)
			gomega.Expect(ok).To(gomega.BeTrue())
			storedEvents = append(storedEvents, StoredEventInfo{
				BlockHashes: storeEvent.BlockHashes,
				LoraName:    storeEvent.LoraName,
				LoraID:      storeEvent.LoraID,
			})
		case kvevents.EventTypeBlockRemoved:
			removeEvent, ok := genEvent.(*kvevents.BlockRemovedEvent)
			gomega.Expect(ok).To(gomega.BeTrue())
			removed = append(removed, removeEvent.BlockHashes...)
		case kvevents.EventTypeAllBlocksCleared:
			_, ok := genEvent.(*kvevents.AllBlocksClearedEvent)
			gomega.Expect(ok).To(gomega.BeTrue())
			allCleared = true
		default:
			ginkgo.Fail("unexpected tag " + string(genEvent.Type()))
		}
	}

	return storedEvents, removed, allCleared
}

// CountKVEventBlocks parses a ZMQ message and returns the total number of stored blocks,
// total number of removed blocks, and whether an all-blocks-cleared event was received.
func CountKVEventBlocks(parts [][]byte, expectedTopic string, expectedSeq uint64) (int, int, bool) {
	batch := parseBatch(parts, expectedTopic, expectedSeq)

	storedCount := 0
	removedCount := 0
	allCleared := false

	for _, genEvent := range batch.Events {
		switch genEvent.Type() {
		case kvevents.EventTypeBlockStored:
			storeEvent, ok := genEvent.(*kvevents.BlockStoredEvent)
			gomega.Expect(ok).To(gomega.BeTrue())
			storedCount += len(storeEvent.BlockHashes)
		case kvevents.EventTypeBlockRemoved:
			removeEvent, ok := genEvent.(*kvevents.BlockRemovedEvent)
			gomega.Expect(ok).To(gomega.BeTrue())
			removedCount += len(removeEvent.BlockHashes)
		case kvevents.EventTypeAllBlocksCleared:
			_, ok := genEvent.(*kvevents.AllBlocksClearedEvent)
			gomega.Expect(ok).To(gomega.BeTrue())
			allCleared = true
		default:
			ginkgo.Fail("unexpected tag " + string(genEvent.Type()))
		}
	}

	return storedCount, removedCount, allCleared
}
