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

type msgpackBlockRemovedEvent struct {
	//nolint:unused
	_msgpack    struct{} `msgpack:",as_array"`
	Tag         string
	BlockHashes []any
	Medium      *string `msgpack:",omitempty"`
}

type msgpackAllBlocksClearedEvent struct {
	//nolint:unused
	_msgpack struct{} `msgpack:",as_array"`
	Tag      string
}

type EventData struct {
	action   EventAction
	tokens   []uint32
	hashes   []uint64
	loraName *string
	loraID   *int
}

type KVEventSender struct {
	publisher    *common.Publisher
	topic        string
	eventChan    common.Channel[EventData]
	maxBatchSize int
	delay        time.Duration
	batch        []kvevents.GenericEvent
	logger       logr.Logger
}

func NewKVEventSender(publisher *common.Publisher, topic string, ch common.Channel[EventData], maxBatchSize int,
	delay time.Duration, logger logr.Logger) *KVEventSender {
	return &KVEventSender{
		publisher:    publisher,
		topic:        topic,
		eventChan:    ch,
		maxBatchSize: maxBatchSize,
		delay:        delay,
		batch:        make([]kvevents.GenericEvent, 0, maxBatchSize),
		logger:       logger,
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
			var err error
			var event kvevents.GenericEvent

			switch eventData.action {
			case eventActionStore:
				event = &kvevents.BlockStoredEvent{
					BlockHashes: eventData.hashes,
					Tokens:      eventData.tokens,
					DeviceTier:  GPU,
					ParentHash:  uint64(kvblock.EmptyBlockHash),
					LoraID:      eventData.loraID,
					LoraName:    eventData.loraName,
				}
			case eventActionRemove:
				event = &kvevents.BlockRemovedEvent{BlockHashes: eventData.hashes, DeviceTier: GPU}
			case eventActionAllBlocksCleared:
				event = &kvevents.AllBlocksClearedEvent{DeviceTier: GPU}
			default:
				return fmt.Errorf("invalid event action %d", eventData.action)
			}
			if err != nil {
				return fmt.Errorf("failed to marshal value: %w", err)
			}

			s.batch = append(s.batch, event)

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

// helper to publish collected batch if not empty
func (s *KVEventSender) publishHelper(ctx context.Context) error {
	if len(s.batch) == 0 {
		return nil
	}

	events := []msgpack.RawMessage{}

	for _, event := range s.batch {
		var raw interface{}

		switch e := event.(type) {
		case *kvevents.BlockStoredEvent:
			raw = &msgpackBlockStoredEvent{
				Tag:         string(kvevents.EventTypeBlockStored),
				BlockHashes: convertUint64ToAnySlice(e.BlockHashes),
				TokenIds:    e.Tokens,
				Medium:      &GPU,
				LoraID:      e.LoraID,
				LoraName:    e.LoraName,
			}
		case *kvevents.BlockRemovedEvent:
			raw = &msgpackBlockRemovedEvent{
				Tag:         string(kvevents.EventTypeBlockRemoved),
				BlockHashes: convertUint64ToAnySlice(e.BlockHashes),
				Medium:      &GPU,
			}
		case *kvevents.AllBlocksClearedEvent:
			raw = &msgpackAllBlocksClearedEvent{
				Tag: string(kvevents.EventTypeAllBlocksCleared),
			}
		default:
			return fmt.Errorf("unknown generic event type: %T", event)
		}

		eventBytes, err := msgpack.Marshal(raw)
		if err != nil {
			return fmt.Errorf("failed to marshal event: %w", err)
		}
		events = append(events, msgpack.RawMessage(eventBytes))
	}

	dpRank := 0

	batch := msgpackEventBatch{
		TS:               float64(time.Now().UnixNano()) / 1e9,
		Events:           events,
		DataParallelRank: &dpRank,
	}

	err := s.publisher.PublishEvent(ctx, s.topic, batch)

	// reset batch
	s.batch = make([]kvevents.GenericEvent, 0, s.maxBatchSize)

	return err
}

func convertUint64ToAnySlice(input []uint64) []any {
	result := make([]any, len(input))
	for i, v := range input {
		result[i] = v
	}
	return result
}
