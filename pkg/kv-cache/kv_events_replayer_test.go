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
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// sendReplayRequestAndRecv connects a REQ socket to the replayer ROUTER endpoint,
// sends an 8-byte big-endian start sequence number, and returns all reply frames
// received until the end-of-replay sentinel (seq == 0xFFFFFFFFFFFFFFFF).
func sendReplayRequestAndRecv(ctx context.Context, endpoint string, startSeq uint64) []zmq4.Msg {
	req := zmq4.NewReq(ctx)
	defer req.Close() //nolint:errcheck

	err := req.Dial(endpoint)
	Expect(err).NotTo(HaveOccurred())

	frame := make([]byte, 8)
	binary.BigEndian.PutUint64(frame, startSeq)
	err = req.Send(zmq4.NewMsg(frame))
	Expect(err).NotTo(HaveOccurred())

	var replies []zmq4.Msg
	for {
		msg, err := req.Recv()
		Expect(err).NotTo(HaveOccurred())
		// REQ socket strips the identity + delimiter; frames are [topic, seq(8B), payload]
		Expect(msg.Frames).To(HaveLen(3))
		seq := binary.BigEndian.Uint64(msg.Frames[1])
		replies = append(replies, msg)
		if seq == replaySentinel {
			break
		}
	}
	return replies
}

var _ = Describe("kvEventsReplayer", func() {
	const (
		replayEndpoint = "tcp://127.0.0.1:15558"
		replayTopic    = "kv.test-topic"
	)

	Describe("replayQueue", func() {
		It("stores entries and returns them in insertion order", func() {
			q := newReplayQueue(5)
			for i := uint64(1); i <= 3; i++ {
				q.push(replayEntry{seq: i, payload: []byte{byte(i)}})
			}
			Expect(q.len()).To(Equal(3))
			Expect(q.since(1)).To(HaveLen(3))
			Expect(q.since(2)).To(HaveLen(2))
			Expect(q.since(3)).To(HaveLen(1))
			Expect(q.since(4)).To(BeEmpty())
		})

		It("overwrites the oldest entry when full", func() {
			q := newReplayQueue(3)
			for i := uint64(1); i <= 5; i++ {
				q.push(replayEntry{seq: i, payload: []byte{byte(i)}})
			}
			Expect(q.len()).To(Equal(3))
			// seq 1 and 2 have been dropped; only 3, 4, 5 remain
			Expect(q.since(1)).To(HaveLen(3))
			seqs := make([]uint64, 0, 3)
			for _, e := range q.since(1) {
				seqs = append(seqs, e.seq)
			}
			Expect(seqs).To(Equal([]uint64{3, 4, 5}))
		})

		It("returns correct payloads for since", func() {
			q := newReplayQueue(10)
			for i := uint64(1); i <= 5; i++ {
				q.push(replayEntry{seq: i, payload: []byte{byte(i * 10)}})
			}
			entries := q.since(3)
			Expect(entries).To(HaveLen(3))
			Expect(entries[0].seq).To(Equal(uint64(3)))
			Expect(entries[1].seq).To(Equal(uint64(4)))
			Expect(entries[2].seq).To(Equal(uint64(5)))
		})
	})

	Describe("kvEventsReplayer.store", func() {
		It("feeds batches into the queue", func() {
			r := newKVEventsReplayer(replayEndpoint, replayTopic, 10, GinkgoLogr)

			r.store(1, []byte("batch-1"))
			r.store(2, []byte("batch-2"))
			r.store(3, []byte("batch-3"))

			Expect(r.queue.len()).To(Equal(3))
			entries := r.queue.since(2)
			Expect(entries).To(HaveLen(2))
			Expect(entries[0].seq).To(Equal(uint64(2)))
			Expect(entries[0].payload).To(Equal([]byte("batch-2")))
		})

		It("slides out old batches when queue is full", func() {
			r := newKVEventsReplayer(replayEndpoint, replayTopic, 3, GinkgoLogr)

			for i := uint64(1); i <= 5; i++ {
				r.store(i, []byte{byte(i)})
			}

			Expect(r.queue.len()).To(Equal(3))
			// only seq 3, 4, 5 remain
			Expect(r.queue.since(1)).To(HaveLen(3))
			Expect(r.queue.since(1)[0].seq).To(Equal(uint64(3)))
		})
	})

	Describe("kvEventsReplayer.run", func() {
		It("receives replay request and sends matched batches back to the requester", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			r := newKVEventsReplayer(replayEndpoint, replayTopic, 10, GinkgoLogr)

			// Pre-populate the queue with 4 batches
			for i := uint64(1); i <= 4; i++ {
				r.store(i, []byte{byte(i)})
			}

			// Start the replayer in the background
			runDone := make(chan struct{})
			go func() {
				defer close(runDone)
				_ = r.run(ctx)
			}()

			// Give the socket time to bind
			time.Sleep(300 * time.Millisecond)

			// Send a replay request from seq 3; expect 2 batches + sentinel back
			replies := sendReplayRequestAndRecv(ctx, replayEndpoint, 3)

			// Last reply is the sentinel; the rest are the matched batches
			Expect(replies).To(HaveLen(3)) // seq 3, seq 4, sentinel
			Expect(string(replies[0].Frames[0])).To(Equal(replayTopic))
			Expect(binary.BigEndian.Uint64(replies[0].Frames[1])).To(Equal(uint64(3)))
			Expect(string(replies[1].Frames[0])).To(Equal(replayTopic))
			Expect(binary.BigEndian.Uint64(replies[1].Frames[1])).To(Equal(uint64(4)))
			// The sentinel carries an empty topic frame, distinguishing it from real batches.
			Expect(replies[2].Frames[0]).To(BeEmpty())
			Expect(binary.BigEndian.Uint64(replies[2].Frames[1])).To(Equal(replaySentinel))

			cancel()
			Eventually(runDone, 3*time.Second).Should(BeClosed())
		})

		It("ignores messages with unexpected frame counts", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			r := newKVEventsReplayer(replayEndpoint, replayTopic, 10, GinkgoLogr)
			r.store(1, []byte{0x01})

			runDone := make(chan struct{})
			go func() {
				defer close(runDone)
				_ = r.run(ctx)
			}()

			time.Sleep(300 * time.Millisecond)

			// A DEALER socket is a legal raw peer for ROUTER but does NOT add an empty
			// delimiter frame. Sending one frame from DEALER produces [identity, data] at
			// the ROUTER — 2 frames, not the expected 3 — so it must be ignored.
			dealer := zmq4.NewDealer(ctx)
			defer dealer.Close() //nolint:errcheck
			Expect(dealer.Dial(replayEndpoint)).To(Succeed())
			err := dealer.Send(zmq4.NewMsg([]byte("bad-request")))
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(200 * time.Millisecond)
			// Queue should be unchanged
			Expect(r.queue.len()).To(Equal(1))

			cancel()
			Eventually(runDone, 3*time.Second).Should(BeClosed())
		})
	})
})
