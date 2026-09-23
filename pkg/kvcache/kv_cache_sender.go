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
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/vmihailenco/msgpack/v5"
)

type EventAction int

const (
	ActionStore EventAction = iota
	ActionRemove
	ActionAllBlocksCleared
)

// Event is one KV-cache event, in the engine-independent form the block cache
// produces. The active engine's EventEncoder turns it into that engine's own
// wire representation.
type Event struct {
	Action EventAction
	Tokens []uint32
	Hashes []uint64
	// ParentHash is the last already-cached block hash, or nil when there is
	// no parent (first block of the sequence). Only meaningful for ActionStore.
	ParentHash *uint64
	LoraName   *string
	LoraID     *int
}

// EventEncoder marshals one Event into the wire form of a particular engine.
// Batching, the batch envelope, the topic, and publishing are common to every
// engine and stay in this package.
type EventEncoder interface {
	EncodeEvent(ev Event) ([]byte, error)
}

// msgpackEventBatch is the published batch envelope. Both vLLM and SGLang use
// this same shape, so it is not part of EventEncoder.
type msgpackEventBatch struct {
	//nolint:unused
	_msgpack         struct{} `msgpack:",as_array"`
	TS               float64
	Events           []msgpack.RawMessage
	DataParallelRank *int `msgpack:",omitempty"`
}

type KVEventSender struct {
	publisher    *common.Publisher
	topic        string
	eventChan    common.Channel[Event]
	maxBatchSize int
	delay        time.Duration
	batch        []msgpack.RawMessage
	encoder      EventEncoder
	logger       logr.Logger
	dpRank       int
	replayer     *kvEventsReplayer // nil when replay is disabled
}

func NewKVEventSender(publisher *common.Publisher, topic string, ch common.Channel[Event], maxBatchSize int,
	delay time.Duration, encoder EventEncoder, dpRank int, logger logr.Logger,
	replayer *kvEventsReplayer) *KVEventSender {
	return &KVEventSender{
		publisher:    publisher,
		topic:        topic,
		eventChan:    ch,
		maxBatchSize: maxBatchSize,
		delay:        delay,
		batch:        make([]msgpack.RawMessage, 0, maxBatchSize),
		encoder:      encoder,
		logger:       logger,
		dpRank:       dpRank,
		replayer:     replayer,
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

		case event, ok := <-s.eventChan.Channel:
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

			encoded, err := s.encoder.EncodeEvent(event)
			if err != nil {
				return err
			}
			s.batch = append(s.batch, encoded)

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

	dpRank := s.dpRank

	batch := msgpackEventBatch{
		TS:               float64(time.Now().UnixNano()) / 1e9,
		Events:           s.batch,
		DataParallelRank: &dpRank,
	}

	seq, payload, err := s.publisher.PublishEvent(ctx, s.topic, batch)
	if err == nil && s.replayer != nil {
		s.replayer.store(seq, payload)
	}

	// reset batch
	s.batch = make([]msgpack.RawMessage, 0, s.maxBatchSize)

	return err
}
