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

	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvevents"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/vmihailenco/msgpack/v5"
)

func ParseKVEvent(parts [][]byte, expectedTopic string, expectedSeq uint64) ([]any, []any, bool) {
	// The message should be [topic, seq, payload]
	gomega.Expect(parts).To(gomega.HaveLen(3))

	gomega.Expect(string(parts[0])).To(gomega.Equal(expectedTopic))

	seq := binary.BigEndian.Uint64(parts[1])
	gomega.Expect(seq).To(gomega.Equal(expectedSeq))

	removed := make([]any, 0)
	stored := make([]any, 0)
	allCleared := false

	var eventBatch kvevents.EventBatch
	err := msgpack.Unmarshal(parts[2], &eventBatch)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	for _, rawEvent := range eventBatch.Events {
		var taggedUnion []msgpack.RawMessage
		err := msgpack.Unmarshal(rawEvent, &taggedUnion)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(taggedUnion).ToNot(gomega.BeEmpty())

		payloadBytes, err := msgpack.Marshal(taggedUnion[1:])
		gomega.Expect(err).NotTo(gomega.HaveOccurred())

		var tag string
		err = msgpack.Unmarshal(taggedUnion[0], &tag)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())

		switch tag {
		case kvevents.BlockStoredEventTag:
			var bs kvevents.BlockStored
			err = msgpack.Unmarshal(payloadBytes, &bs)
			stored = append(stored, bs.BlockHashes...)
		case kvevents.BlockRemovedEventTag:
			var br kvevents.BlockRemoved
			err = msgpack.Unmarshal(payloadBytes, &br)
			removed = append(removed, br.BlockHashes...)
		case kvevents.AllBlocksClearedEventTag:
			var ac kvevents.AllBlocksCleared
			err = msgpack.Unmarshal(payloadBytes, &ac)
			allCleared = true

		default:
			ginkgo.Fail("unexpected tag " + tag)
			continue
		}
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
	}
	return stored, removed, allCleared
}
