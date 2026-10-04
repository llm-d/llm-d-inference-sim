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

// Wire-format tests for the vLLM KV-event encoder. The oracle is
// engineadapter.VLLMAdapter -- the router-side parser these events exist to
// feed -- so a format change that the router cannot read fails here.

package vllm

import (
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
)

const eventsTestBlockSize = 16

var eventsTestTopic = kvcache.CreateKVEventsTopic("127.0.0.1", 8000, common.TestModelName)

func newEventEncoder(mapFormat bool) kvcache.EventEncoder {
	cfg := common.Configuration{}
	cfg.KVCache.TokenBlockSize = eventsTestBlockSize
	cfg.KVCache.UseVllmMapEventFormat = mapFormat

	enc, err := Engine{}.NewKVEventEncoder(cfg)
	Expect(err).NotTo(HaveOccurred())
	return enc
}

// publishEvents encodes evs and wraps them in the same batch envelope
// pkg/kvcache publishes, producing the transport message a router receives.
func publishEvents(enc kvcache.EventEncoder, evs ...kvcache.Event) *kvevents.RawMessage {
	raws := make([]msgpack.RawMessage, 0, len(evs))
	for _, ev := range evs {
		encoded, err := enc.EncodeEvent(ev)
		Expect(err).NotTo(HaveOccurred())
		raws = append(raws, encoded)
	}

	rank := 0
	batch := struct {
		//nolint:unused
		_msgpack         struct{} `msgpack:",as_array"`
		TS               float64
		Events           []msgpack.RawMessage
		DataParallelRank *int
	}{TS: float64(time.Now().UnixNano()) / 1e9, Events: raws, DataParallelRank: &rank}

	payload, err := msgpack.Marshal(&batch)
	Expect(err).NotTo(HaveOccurred())

	return &kvevents.RawMessage{Topic: eventsTestTopic, Sequence: 1, Payload: payload}
}

// parseAs decodes msg with the adapter for the named engine.
func parseAs(engineType string, msg *kvevents.RawMessage) kvevents.EventBatch {
	adapter, err := engineadapter.NewAdapter(engineType)
	Expect(err).NotTo(HaveOccurred())
	_, _, batch, err := adapter.ParseMessage(msg)
	Expect(err).NotTo(HaveOccurred())
	return batch
}

// fieldCount returns the number of elements in a positionally encoded event.
func fieldCount(encoded []byte) int {
	var fields []msgpack.RawMessage
	Expect(msgpack.Unmarshal(encoded, &fields)).To(Succeed())
	return len(fields)
}

var _ = Describe("vLLM KV-event encoder", func() {
	loraName := "lora1"
	loraID := 1

	storeEvent := kvcache.Event{
		Action:   kvcache.ActionStore,
		Hashes:   []uint64{1, 2},
		Tokens:   []uint32{10, 20},
		LoraName: &loraName,
		LoraID:   &loraID,
	}

	DescribeTable("round-trips every event through VLLMAdapter",
		func(mapFormat bool) {
			parent := uint64(7)
			withParent := storeEvent
			withParent.ParentHash = &parent

			batch := parseAs(engineadapter.EngineTypeVLLM, publishEvents(newEventEncoder(mapFormat),
				withParent,
				kvcache.Event{Action: kvcache.ActionRemove, Hashes: []uint64{3, 4}},
				kvcache.Event{Action: kvcache.ActionAllBlocksCleared},
			))
			Expect(batch.Events).To(HaveLen(3))

			stored, ok := batch.Events[0].(*kvevents.BlockStoredEvent)
			Expect(ok).To(BeTrue())
			Expect(stored.BlockHashes).To(Equal([]uint64{1, 2}))
			Expect(stored.Tokens).To(Equal([]uint32{10, 20}))
			Expect(stored.ParentHash).To(Equal(parent))
			Expect(stored.BlockSize).To(Equal(eventsTestBlockSize))
			Expect(stored.DeviceTier).To(Equal(gpu))
			Expect(stored.LoraName).To(HaveValue(Equal(loraName)))
			Expect(stored.LoraID).To(HaveValue(Equal(loraID)))

			removed, ok := batch.Events[1].(*kvevents.BlockRemovedEvent)
			Expect(ok).To(BeTrue())
			Expect(removed.BlockHashes).To(Equal([]uint64{3, 4}))

			_, ok = batch.Events[2].(*kvevents.AllBlocksClearedEvent)
			Expect(ok).To(BeTrue())
		},
		Entry("list format", false),
		Entry("map format", true),
	)

	DescribeTable("omits lora metadata for base-model requests",
		func(mapFormat bool) {
			base := storeEvent
			base.LoraName = nil
			base.LoraID = nil

			batch := parseAs(engineadapter.EngineTypeVLLM, publishEvents(newEventEncoder(mapFormat), base))
			stored, ok := batch.Events[0].(*kvevents.BlockStoredEvent)
			Expect(ok).To(BeTrue())
			Expect(stored.LoraName).To(BeNil())
			Expect(stored.LoraID).To(BeNil())
		},
		Entry("list format", false),
		Entry("map format", true),
	)

	// A nil ParentHash is the empty block hash on the wire: 0 positionally,
	// msgpack nil in the map format. The adapter reports 0 for both.
	DescribeTable("reports no parent as the empty block hash",
		func(mapFormat bool) {
			batch := parseAs(engineadapter.EngineTypeVLLM, publishEvents(newEventEncoder(mapFormat), storeEvent))
			stored, ok := batch.Events[0].(*kvevents.BlockStoredEvent)
			Expect(ok).To(BeTrue())
			Expect(stored.ParentHash).To(BeZero())
		},
		Entry("list format", false),
		Entry("map format", true),
	)

	It("rejects an unknown action", func() {
		_, err := newEventEncoder(false).EncodeEvent(kvcache.Event{Action: kvcache.EventAction(42)})
		Expect(err).To(HaveOccurred())
	})

	// The positional array lengths are the one thing sglang differs on -- it
	// omits trailing defaults -- so pin vLLM's full-length arrays here.
	It("encodes positional events at their full field count", func() {
		enc := newEventEncoder(false)

		stored, err := enc.EncodeEvent(storeEvent)
		Expect(err).NotTo(HaveOccurred())
		Expect(fieldCount(stored)).To(Equal(9))

		removed, err := enc.EncodeEvent(kvcache.Event{Action: kvcache.ActionRemove, Hashes: []uint64{3}})
		Expect(err).NotTo(HaveOccurred())
		Expect(fieldCount(removed)).To(Equal(3))
	})

	// Evidence that sglang's format differs only in array length, not layout:
	// its adapter pads short arrays, so it reads vLLM's full-length ones too.
	It("stays readable by the sglang adapter", func() {
		batch := parseAs(engineadapter.EngineTypeSGLang, publishEvents(newEventEncoder(false),
			storeEvent,
			kvcache.Event{Action: kvcache.ActionRemove, Hashes: []uint64{3, 4}},
		))
		Expect(batch.Events).To(HaveLen(2))

		stored, ok := batch.Events[0].(*kvevents.BlockStoredEvent)
		Expect(ok).To(BeTrue())
		Expect(stored.BlockHashes).To(Equal([]uint64{1, 2}))
		Expect(stored.Tokens).To(Equal([]uint32{10, 20}))
		Expect(stored.BlockSize).To(Equal(eventsTestBlockSize))

		removed, ok := batch.Events[1].(*kvevents.BlockRemovedEvent)
		Expect(ok).To(BeTrue())
		Expect(removed.BlockHashes).To(Equal([]uint64{3, 4}))
	})
})
