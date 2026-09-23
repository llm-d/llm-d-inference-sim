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
	"encoding/binary"
	"sync"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// eventAdapter is the router-side decoder for the engine under test, selected
// the same way the simulator selects its engine, so a suite running a different
// engine reads that engine's event format. Resolved on first use, since the
// helpers below only run from tests.
var eventAdapter = sync.OnceValue(func() kvevents.EngineAdapter {
	name, err := common.ResolveEngineName()
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	adapter, err := engineadapter.NewAdapter(name)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	return adapter
})

// StoredEventInfo holds parsed metadata from a single BlockStoredEvent
type StoredEventInfo struct {
	BlockHashes []uint64
	ParentHash  uint64
	LoraName    *string
	LoraID      *int
}

// payloadFrame asserts a published message's [topic, seq, payload] framing and
// returns its payload frame.
func payloadFrame(parts [][]byte, expectedTopic string, expectedSeq uint64) []byte {
	gomega.Expect(parts).To(gomega.HaveLen(3))
	gomega.Expect(string(parts[0])).To(gomega.Equal(expectedTopic))
	gomega.Expect(binary.BigEndian.Uint64(parts[1])).To(gomega.Equal(expectedSeq))
	return parts[2]
}

// parseBatch decodes a published message with the engine adapter, yielding the
// events as the router sees them.
func parseBatch(parts [][]byte, expectedTopic string, expectedSeq uint64) kvevents.EventBatch {
	rawMsg := kvevents.RawMessage{
		Topic:    expectedTopic,
		Sequence: expectedSeq,
		Payload:  payloadFrame(parts, expectedTopic, expectedSeq),
	}

	_, _, batch, err := eventAdapter().ParseMessage(&rawMsg)
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
				ParentHash:  storeEvent.ParentHash,
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

// stubEncoder encodes each Event as itself, so this package's own tests can
// exercise what it owns -- event generation, batching, topic and sequence
// handling -- without depending on any engine's wire format. Engine formats are
// asserted where they are implemented (see pkg/engine/vllm).
type stubEncoder struct{}

// EncodeEvent marshals ev directly, keyed by its Go field names.
func (stubEncoder) EncodeEvent(ev Event) ([]byte, error) {
	return msgpack.Marshal(ev)
}

// decodeStubEvents returns the events carried by one message published by a
// cache wired to stubEncoder.
func decodeStubEvents(parts [][]byte, expectedTopic string, expectedSeq uint64) []Event {
	var batch msgpackEventBatch
	gomega.Expect(msgpack.Unmarshal(payloadFrame(parts, expectedTopic, expectedSeq), &batch)).To(gomega.Succeed())

	events := make([]Event, 0, len(batch.Events))
	for _, raw := range batch.Events {
		var ev Event
		gomega.Expect(msgpack.Unmarshal(raw, &ev)).To(gomega.Succeed())
		events = append(events, ev)
	}
	return events
}

// decodeStubStoredEvents is decodeStubEvents restricted to store events.
func decodeStubStoredEvents(parts [][]byte, expectedTopic string, expectedSeq uint64) []Event {
	stored := make([]Event, 0, 1)
	for _, ev := range decodeStubEvents(parts, expectedTopic, expectedSeq) {
		if ev.Action == ActionStore {
			stored = append(stored, ev)
		}
	}
	return stored
}

// countStubEventBlocks returns the number of stored and removed block hashes
// carried by one message published by a cache wired to stubEncoder.
func countStubEventBlocks(parts [][]byte, expectedTopic string, expectedSeq uint64) (stored int, removed int) {
	for _, ev := range decodeStubEvents(parts, expectedTopic, expectedSeq) {
		switch ev.Action {
		case ActionStore:
			stored += len(ev.Hashes)
		case ActionRemove:
			removed += len(ev.Hashes)
		case ActionAllBlocksCleared:
		}
	}
	return stored, removed
}

// SendReplayRequestAndRecv connects a REQ socket to a KV events replay ROUTER
// endpoint, sends startSeq as an 8-byte big-endian frame, and returns all
// reply messages up to and including the end-of-replay sentinel. The REQ
// socket strips the identity + empty-delimiter frames automatically, so each
// returned message has frames [topic, seq(8B big-endian), payload]; the
// sentinel is distinguished from real batches by its empty topic frame.
func SendReplayRequestAndRecv(ctx context.Context, endpoint string, startSeq uint64) []zmq4.Msg {
	req := zmq4.NewReq(ctx)
	defer req.Close() //nolint:errcheck

	gomega.Expect(req.Dial(endpoint)).To(gomega.Succeed())

	frame := make([]byte, 8)
	binary.BigEndian.PutUint64(frame, startSeq)
	gomega.Expect(req.Send(zmq4.NewMsg(frame))).To(gomega.Succeed())

	var replies []zmq4.Msg
	for {
		msg, err := req.Recv()
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(msg.Frames).To(gomega.HaveLen(3))
		replies = append(replies, msg)
		if len(msg.Frames[0]) == 0 {
			break
		}
	}
	return replies
}

// ParseKVBatchRank decodes the DataParallelRank field from a raw ZMQ message.
func ParseKVBatchRank(frames [][]byte) int {
	gomega.Expect(frames).To(gomega.HaveLen(3))
	var batch msgpackEventBatch
	gomega.Expect(msgpack.Unmarshal(frames[2], &batch)).To(gomega.Succeed())
	gomega.Expect(batch.DataParallelRank).NotTo(gomega.BeNil())
	return *batch.DataParallelRank
}

// verifySingleBlockEviction ensures that exactly one block from the provided list
// has been removed from the cache
func verifySingleBlockEviction(bCache *blockCache, model string, blocks []uint64) {
	evictedCnt := 0
	for _, blockHash := range blocks {
		_, blockExists := bCache.getBlockInfo(blockKey{hash: blockHash, modelName: model})
		if !blockExists {
			evictedCnt++
		}
	}
	gomega.Expect(evictedCnt).To(gomega.Equal(1))
}

// verifyAllBlocksRetained checks that every block in the provided list is
// still present in the cache, confirming that no eviction has occurred for these specific keys
func verifyAllBlocksRetained(bCache *blockCache, model string, blocks []uint64) {
	for _, blockHash := range blocks {
		_, blockExists := bCache.getBlockInfo(blockKey{hash: blockHash, modelName: model})
		gomega.Expect(blockExists).To(gomega.BeTrue())
	}
}
