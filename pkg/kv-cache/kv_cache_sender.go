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
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvevents"
	"github.com/vmihailenco/msgpack/v5"
)

type EventAction int

const (
	eventActionStore EventAction = iota
	eventActionRemove
)

type EventData struct {
	action     EventAction
	hashValues []uint64
}

type KVEventSender struct {
	publisher    *common.Publisher
	topic        string
	eventChan    chan EventData
	maxBatchSize int
	delay        time.Duration
	batch        []msgpack.RawMessage
	logger       logr.Logger
}

func NewKVEventSender(publisher *common.Publisher, topic string, ch chan EventData, maxBatchSize int,
	delay time.Duration, logger logr.Logger) *KVEventSender {
	return &KVEventSender{
		publisher:    publisher,
		topic:        topic,
		eventChan:    ch,
		maxBatchSize: maxBatchSize,
		delay:        delay,
		batch:        make([]msgpack.RawMessage, 0, maxBatchSize),
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
				s.logger.Info("Existing, discard remaining events", "num of events", len(s.batch))
			}
			return ctx.Err()

		case eventData, ok := <-s.eventChan:
			if !ok {
				// Channel closed, discard remaining events and exit
				if len(s.batch) > 0 {
					s.logger.Info("Channel closed, discard remaining events", "num of events", len(s.batch))
				}
				return nil
			}

			if s.publisher == nil {
				continue
			}

			// Encode eventData's hash value to msgpack.RawMessage
			var payload []byte
			var err error

			switch eventData.action {
			case eventActionStore:
				payload, err = msgpack.Marshal(kvevents.BlockStored{BlockHashes: eventData.hashValues}.ToTaggedUnion())
			case eventActionRemove:
				payload, err = msgpack.Marshal(kvevents.BlockRemoved{BlockHashes: eventData.hashValues}.ToTaggedUnion())
			default:
				return fmt.Errorf("invalid event action %d", eventData.action)
			}
			if err != nil {
				return fmt.Errorf("failed to marshal value: %w", err)
			}

			s.batch = append(s.batch, payload)

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

	dpRank := 0
	eventBatch := kvevents.EventBatch{
		TS:               float64(time.Now().UnixNano()) / 1e9,
		Events:           s.batch,
		DataParallelRank: &dpRank,
	}

	err := s.publisher.PublishEvent(ctx, s.topic, eventBatch)

	// reset batch
	s.batch = make([]msgpack.RawMessage, 0, s.maxBatchSize)

	return err
}
