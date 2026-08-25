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
	"net"
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

// replayQueue is a bounded FIFO of replayEntry values, holding at most
// capacity entries in insertion order. When full, the oldest entry is
// dropped to make room (sliding window).
type replayQueue struct {
	mu       sync.Mutex
	buf      []replayEntry
	capacity int
}

func newReplayQueue(capacity int) *replayQueue {
	return &replayQueue{
		buf:      make([]replayEntry, 0, capacity),
		capacity: capacity,
	}
}

// push adds an entry, dropping the oldest when full.
func (q *replayQueue) push(entry replayEntry) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.buf) == q.capacity {
		copy(q.buf, q.buf[1:])
		q.buf = q.buf[:len(q.buf)-1]
	}
	q.buf = append(q.buf, entry)
}

// since returns all stored entries with seq >= startSeq, in insertion order.
// Entries are stored in strictly increasing sequence order, so the search
// short-circuits on the first match and the rest is copied out in one go.
func (q *replayQueue) since(startSeq uint64) []replayEntry {
	q.mu.Lock()
	defer q.mu.Unlock()

	start := len(q.buf)
	for i, e := range q.buf {
		if e.seq >= startSeq {
			start = i
			break
		}
	}
	result := make([]replayEntry, len(q.buf)-start)
	copy(result, q.buf[start:])
	return result
}

// len returns the number of entries currently stored.
func (q *replayQueue) len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.buf)
}

// kvEventsReplayer binds a ZMQ ROUTER socket on the replay endpoint.
type kvEventsReplayer struct {
	endpoint string
	topic    string
	queue    *replayQueue
	logger   logr.Logger
	socket   zmq4.Socket // set by listen; nil until then
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
	r.logger.V(logging.TRACE).Info("KV events replayer stored batch", "seq", seq, "queue_size", r.queue.len())
}

// listen creates and binds the ROUTER socket, returning the resolved address.
// This matters when endpoint uses a wildcard port (e.g. "tcp://127.0.0.1:0"):
// the caller can only learn the OS-assigned port after Listen succeeds, which
// is why binding is split out from serve rather than happening inside it.
// The socket is kept open until serve returns; call serve exactly once after
// a successful listen.
func (r *kvEventsReplayer) listen(ctx context.Context) (net.Addr, error) {
	socket := zmq4.NewRouter(ctx)
	if err := socket.Listen(r.endpoint); err != nil {
		_ = socket.Close()
		return nil, err
	}
	r.socket = socket

	addr := socket.Addr()
	r.logger.V(logging.INFO).Info("KV events replayer listening", "endpoint", addr.String())
	return addr, nil
}

// serve handles replay requests on the socket bound by listen, until ctx is
// cancelled. Each request arrives as [identity, empty-delimiter, 8-byte-seq]
// (REQ/ROUTER pair). Replies are sent back to the same identity.
func (r *kvEventsReplayer) serve(ctx context.Context) error {
	socket := r.socket
	defer socket.Close() //nolint:errcheck

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
// ROUTER's receive loop — Recv and Send use independent locks in
// go-zeromq/zmq4, so new incoming replay requests are still parsed while a
// reply is in flight.
//
// This does NOT isolate clients from each other, though: every Send() on a
// ROUTER socket funnels through zmq4's routerMWriter.write(), which holds a
// single mutex for the entire blocking OS write, regardless of which peer
// it's writing to (verified against go-zeromq/zmq4@v0.17.0's router.go). So
// while this goroutine is blocked inside Send() for one client, every other
// client's reply goroutine — call it from a request received concurrently —
// blocks too, waiting on that same mutex, even if its own connection is
// healthy and actively draining. There is no per-message timeout to bound
// this: go-zeromq's Send() performs a literal blocking OS write with no way
// to cancel it once in flight (the context timeout it wraps the write in is
// never wired to a net.Conn deadline), so a single client that stops
// draining its socket without closing the connection can stall replay for
// every other client on this rank until that connection is closed.
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
