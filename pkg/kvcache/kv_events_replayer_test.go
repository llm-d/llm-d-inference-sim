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

var _ = Describe("kvEventsReplayer", func() {
	const (
		// Port 0 lets the OS assign a free port; listen() reports back which one.
		replayEndpoint = "tcp://127.0.0.1:0"
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
	})

	Describe("kvEventsReplayer.run", func() {
		// startReplayer binds r and serves it in the background, returning the
		// resolved endpoint (with the OS-assigned port, if any) and a channel
		// closed once serve returns.
		startReplayer := func(ctx context.Context, r *kvEventsReplayer) (string, chan struct{}) {
			addr, err := r.listen(ctx)
			Expect(err).NotTo(HaveOccurred())

			done := make(chan struct{})
			go func() {
				defer close(done)
				_ = r.serve(ctx)
			}()
			return "tcp://" + addr.String(), done
		}

		It("receives replay request and sends matched batches back to the requester", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			r := newKVEventsReplayer(replayEndpoint, replayTopic, 10, GinkgoLogr)

			// Pre-populate the queue with 4 batches
			for i := uint64(1); i <= 4; i++ {
				r.store(i, []byte{byte(i)})
			}

			endpoint, runDone := startReplayer(ctx, r)

			// Send a replay request from seq 3; expect 2 batches + sentinel back
			replies := SendReplayRequestAndRecv(ctx, endpoint, 3)

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

			endpoint, runDone := startReplayer(ctx, r)

			// A DEALER socket is a legal raw peer for ROUTER but does NOT add an empty
			// delimiter frame. Sending one frame from DEALER produces [identity, data] at
			// the ROUTER — 2 frames, not the expected 3 — so it must be ignored.
			dealer := zmq4.NewDealer(ctx)
			defer dealer.Close() //nolint:errcheck
			Expect(dealer.Dial(endpoint)).To(Succeed())
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
