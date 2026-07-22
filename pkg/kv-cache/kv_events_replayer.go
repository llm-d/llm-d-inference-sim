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
	"context"
	"encoding/binary"
	"sync"
	"time"

	"github.com/go-logr/logr"
	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
)

// replayRecvErrorBackoff bounds how fast we retry socket.Recv() after a
// non-context error, so a persistent receive failure can't busy-spin the
// loop or flood the log.
const replayRecvErrorBackoff = 100 * time.Millisecond

// replayEntry holds one published event batch payload together with its
// sequence number for range-based filtering.
type replayEntry struct {
	seq     uint64
	payload []byte
}

// replaySentinel is the sequence value used in the vLLM end-of-replay sentinel frame.
const replaySentinel = ^uint64(0) // 0xFFFFFFFFFFFFFFFF

// replayQueue is a fixed-capacity ring buffer of replayEntry values.
// When full, the oldest entry is silently overwritten (sliding window).
type replayQueue struct {
	mu       sync.Mutex
	buf      []replayEntry
	head     int // next write slot
	size     int // number of valid entries
	capacity int
}

func newReplayQueue(capacity int) *replayQueue {
	return &replayQueue{
		buf:      make([]replayEntry, capacity),
		capacity: capacity,
	}
}

// push adds an entry, overwriting the oldest when full.
func (q *replayQueue) push(entry replayEntry) {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.buf[q.head] = entry
	q.head = (q.head + 1) % q.capacity
	if q.size < q.capacity {
		q.size++
	}
}

// since returns all stored entries with seq >= startSeq, in insertion order.
// Entries are stored in strictly increasing sequence order. The loop short-circuits
// on the first match and appends everything from that point on
func (q *replayQueue) since(startSeq uint64) []replayEntry {
	q.mu.Lock()
	defer q.mu.Unlock()

	// head points one slot past the newest entry, so stepping back size slots
	// reaches the oldest entry.
	// +capacity before % prevents a negative result when
	// head has wrapped around.
	oldest := (q.head - q.size + q.capacity) % q.capacity

	var result []replayEntry
	collecting := false
	for i := 0; i < q.size; i++ {
		e := q.buf[(oldest+i)%q.capacity]
		if !collecting {
			if e.seq >= startSeq {
				collecting = true
			} else {
				continue
			}
		}
		result = append(result, e)
	}
	return result
}

// len returns the number of entries currently stored.
func (q *replayQueue) len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.size
}

// kvEventsReplayer binds a ZMQ ROUTER socket on the replay endpoint.
type kvEventsReplayer struct {
	endpoint string
	topic    string
	queue    *replayQueue
	logger   logr.Logger
}

// newKVEventsReplayer creates a replayer that will bind a ROUTER socket on endpoint.
// topic is included in every replayed batch frame, matching the live PUB stream.
func newKVEventsReplayer(endpoint string, topic string, queueSize int, logger logr.Logger) *kvEventsReplayer {
	return &kvEventsReplayer{
		endpoint: endpoint,
		topic:    topic,
		queue:    newReplayQueue(queueSize),
		logger:   logger,
	}
}

// store is called by KVEventSender each time a batch is published, recording
// the msgpack payload and its sequence number in the sliding queue.
func (r *kvEventsReplayer) store(seq uint64, payload []byte) {
	r.queue.push(replayEntry{seq: seq, payload: payload})
	r.logger.V(logging.DEBUG).Info("KV events replayer stored batch", "seq", seq, "queue_size", r.queue.len())
}

// run binds the ROUTER socket and handles replay requests until ctx is cancelled.
// Each request arrives as [identity, empty-delimiter, 8-byte-seq] (REQ/ROUTER
// pair). Replies are sent back to the same identity.
func (r *kvEventsReplayer) run(ctx context.Context) error {
	socket := zmq4.NewRouter(ctx)
	defer socket.Close() //nolint:errcheck

	if err := socket.Listen(r.endpoint); err != nil {
		return err
	}
	r.logger.V(logging.INFO).Info("KV events replayer listening", "endpoint", r.endpoint)

	for {
		msg, err := socket.Recv()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			r.logger.Error(err, "KV events replayer receive error")
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(replayRecvErrorBackoff):
			}
			continue
		}

		// ROUTER + REQ framing: [identity, empty-delimiter, payload]
		if len(msg.Frames) != 3 {
			r.logger.V(logging.DEBUG).Info("KV events replayer: unexpected frame count, ignoring",
				"frames", len(msg.Frames))
			continue
		}

		// msg.Frames[1] is the empty delimiter inserted by the REQ socket.
		r.handleReplayRequest(ctx, socket, msg.Frames[0], msg.Frames[2])
	}
}

// handleReplayRequest parses the start sequence number, looks up matched
// batches, and sends them (plus the end-of-replay sentinel) back
// to the requesting client over the ROUTER socket.
// Each reply frame follows the same [topic, seq(8B big-endian), payload] wire
// format as the live PUB stream so subscribers can decode them identically.
//
// The send loop runs in a goroutine so a slow client doesn't block the
// ROUTER's receive loop (Recv/Send use independent locks in go-zeromq/zmq4).
// Send() itself blocks for as long as the client takes to drain its socket —
// there is no per-message timeout, since go-zeromq's Send() performs a
// literal blocking OS write with no way to bound or cancel it once in
// flight, so a timeout here couldn't actually free a stuck client's
// connection anyway; a slow-but-still-reading client (the expected case)
// just waits, at the cost of leaking this goroutine if a client vanishes
// without closing its connection.
func (r *kvEventsReplayer) handleReplayRequest(ctx context.Context, socket zmq4.Socket, identity []byte, frame []byte) {
	if len(frame) < 8 {
		r.logger.V(logging.DEBUG).Info("KV events replayer: replay request frame too short, ignoring")
		return
	}

	startSeq := binary.BigEndian.Uint64(frame)
	batches := r.queue.since(startSeq)
	r.logger.V(logging.INFO).Info("KV events replayer replay request",
		"start_seq", startSeq, "batches_to_replay", len(batches))

	// Copy identity so the goroutine closure is safe after run() advances.
	id := make([]byte, len(identity))
	copy(id, identity)

	topic := []byte(r.topic)

	go func() {
		for _, b := range batches {
			if ctx.Err() != nil {
				return
			}
			reply := zmq4.NewMsgFrom(id, []byte{}, topic, common.EncodeSeq(b.seq), b.payload)
			if err := socket.Send(reply); err != nil {
				r.logger.Error(err, "KV events replayer: failed to send replay batch, abandoning remaining replay",
					"seq", b.seq)
				return
			}
		}
		if ctx.Err() != nil {
			return
		}
		// End-of-replay sentinel: empty topic, seq=0xFFFFFFFFFFFFFFFF, empty payload.
		sentinel := zmq4.NewMsgFrom(id, []byte{}, []byte{}, common.EncodeSeq(replaySentinel), []byte{})
		if err := socket.Send(sentinel); err != nil {
			r.logger.Error(err, "KV events replayer: failed to send end-of-replay sentinel")
		}
	}()
}
