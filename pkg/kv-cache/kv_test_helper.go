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

func ParseKVEvent(parts [][]byte, expectedTopic string, expectedSeq uint64) ([]uint64, []uint64, bool) {
	// The message should be [topic, seq, payload]
	gomega.Expect(parts).To(gomega.HaveLen(3))

	gomega.Expect(string(parts[0])).To(gomega.Equal(expectedTopic))

	seq := binary.BigEndian.Uint64(parts[1])
	gomega.Expect(seq).To(gomega.Equal(expectedSeq))

	// vllmAdapter := engineadapter.NewVLLMAdapter()
	rawMsg := kvevents.RawMessage{
		Topic:    string(parts[0]),
		Sequence: seq,
		Payload:  parts[2],
	}

	// use vllm adapter from kv-cache to parse the message into a batch of generic events
	_, _, batch, err := vllmAdapter.ParseMessage(&rawMsg)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	removed := make([]uint64, 0)
	stored := make([]uint64, 0)
	allCleared := false

	for _, genEvent := range batch.Events {
		switch genEvent.Type() {
		case kvevents.EventTypeBlockStored:
			var storeEvent *kvevents.BlockStoredEvent
			storeEvent, ok := genEvent.(*kvevents.BlockStoredEvent)
			gomega.Expect(ok).To(gomega.BeTrue())
			stored = append(stored, storeEvent.BlockHashes...)
		case kvevents.EventTypeBlockRemoved:
			var removeEvent *kvevents.BlockRemovedEvent
			removeEvent, ok := genEvent.(*kvevents.BlockRemovedEvent)
			gomega.Expect(ok).To(gomega.BeTrue())
			removed = append(removed, removeEvent.BlockHashes...)
		case kvevents.EventTypeAllBlocksCleared:
			_, ok := genEvent.(*kvevents.AllBlocksClearedEvent)
			gomega.Expect(ok).To(gomega.BeTrue())
			allCleared = true
		default:
			ginkgo.Fail("unexpected tag " + string(genEvent.Type()))
			continue
		}
	}

	return stored, removed, allCleared
}
