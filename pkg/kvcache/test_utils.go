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
	"encoding/binary"

	"github.com/onsi/gomega"
	"github.com/vmihailenco/msgpack/v5"
)

// StubEncoder encodes each Event as itself, so tests can exercise what this
// package owns -- event generation, batching, topic and sequence handling --
// without depending on any engine's wire format. Engine formats are asserted
// where they are implemented (see pkg/engine/vllm).
type StubEncoder struct{}

// EncodeEvent marshals ev directly, keyed by its Go field names.
func (StubEncoder) EncodeEvent(ev Event) ([]byte, error) {
	return msgpack.Marshal(ev)
}

// DecodeStubEvents decodes one ZMQ message published by a cache wired to
// StubEncoder, asserting its topic and sequence, and returns the events the
// batch carried.
func DecodeStubEvents(parts [][]byte, expectedTopic string, expectedSeq uint64) []Event {
	// The message should be [topic, seq, payload]
	gomega.Expect(parts).To(gomega.HaveLen(3))
	gomega.Expect(string(parts[0])).To(gomega.Equal(expectedTopic))
	gomega.Expect(binary.BigEndian.Uint64(parts[1])).To(gomega.Equal(expectedSeq))

	var batch struct {
		//nolint:unused
		_msgpack         struct{} `msgpack:",as_array"`
		TS               float64
		Events           []msgpack.RawMessage
		DataParallelRank *int `msgpack:",omitempty"`
	}
	gomega.Expect(msgpack.Unmarshal(parts[2], &batch)).To(gomega.Succeed())

	events := make([]Event, 0, len(batch.Events))
	for _, raw := range batch.Events {
		var ev Event
		gomega.Expect(msgpack.Unmarshal(raw, &ev)).To(gomega.Succeed())
		events = append(events, ev)
	}
	return events
}

// DecodeStubStoredEvents is DecodeStubEvents restricted to store events.
func DecodeStubStoredEvents(parts [][]byte, expectedTopic string, expectedSeq uint64) []Event {
	stored := make([]Event, 0, 1)
	for _, ev := range DecodeStubEvents(parts, expectedTopic, expectedSeq) {
		if ev.Action == ActionStore {
			stored = append(stored, ev)
		}
	}
	return stored
}

// CountStubEventBlocks returns the number of stored and removed block hashes
// carried by one message published by a cache wired to StubEncoder.
func CountStubEventBlocks(parts [][]byte, expectedTopic string, expectedSeq uint64) (stored int, removed int) {
	for _, ev := range DecodeStubEvents(parts, expectedTopic, expectedSeq) {
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
