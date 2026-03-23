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

package common

import (
	"bytes"
	"context"
	"encoding/binary"
	"net"

	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/vmihailenco/msgpack/v5"
)

const (
	topic = "test-topic"
	data  = "Hello"
)

var _ = Describe("Publisher", func() {
	It("should publish and receive correct message", func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		sub, endpoint := CreateSub(ctx, "")
		// nolint
		defer sub.Close()

		time.Sleep(100 * time.Millisecond)

		pub, err := NewPublisher(ctx, endpoint)
		Expect(err).NotTo(HaveOccurred())

		go func() {
			// Make sure that sub.RecvMessageBytes is called before pub.PublishEvent
			time.Sleep(time.Second)
			err := pub.PublishEvent(ctx, topic, data)
			Expect(err).NotTo(HaveOccurred())
		}()

		// The message should be [topic, seq, payload]
		msg, err := sub.Recv()
		Expect(err).NotTo(HaveOccurred())

		// message should have 3 frames: topic, sequence, payload
		Expect(msg.Frames).To(HaveLen(3))

		// check topic
		Expect(string(msg.Frames[0])).To(Equal(topic))

		// check sequence
		seq := binary.BigEndian.Uint64(msg.Frames[1])
		Expect(seq).To(Equal(uint64(1)))

		// check payload
		var payload string
		err = msgpack.Unmarshal(msg.Frames[2], &payload)
		Expect(err).NotTo(HaveOccurred())
		Expect(payload).To(Equal(data))
	})

	It("should connect to zmq listener after delay", func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		freePort, err := getFreePort()
		Expect(err).NotTo(HaveOccurred())
		Expect(freePort).ToNot(BeEmpty())
		endpoint := "tcp://127.0.0.1:" + freePort

		// create publisher - it will try to connect to the listener, but the listener is not started yet
		pub, err := NewPublisher(ctx, endpoint)
		Expect(err).NotTo(HaveOccurred())
		// nolint
		defer pub.Close()

		sub := NewSub(ctx)
		// nolint
		defer sub.Close()

		receivedEvents := 0

		// start listener after delay of 3 seconds
		go func() {
			time.Sleep(3 * time.Second)

			StartSub(sub, endpoint, "")

			for {
				// The message should be [topic, seq, payload]
				msg, err := sub.Recv()
				if err != nil && err.Error() == "context canceled" {
					return
				}
				Expect(err).ToNot(HaveOccurred())
				// message should have 3 frames: topic, sequence, payload
				Expect(msg.Frames).To(HaveLen(3))

				receivedEvents++
			}
		}()

		// send events
		var index uint64 = 0
		for ; index < 6; index++ {
			time.Sleep(time.Second)

			// Create message payload
			var payload bytes.Buffer
			enc := msgpack.NewEncoder(&payload)
			enc.UseArrayEncodedStructs(true)
			err := enc.Encode(index)
			Expect(err).ToNot(HaveOccurred())

			// sequence number for ordering
			seqBytes := make([]byte, 8)
			binary.BigEndian.PutUint64(seqBytes, index)

			// send topic, sequence, payload
			msg := zmq4.NewMsgFrom([]byte(topic), seqBytes, payload.Bytes())

			err = pub.socket.Send(msg)
			Expect(err).ToNot(HaveOccurred())
		}

		time.Sleep(time.Second)

		// first 3 events were sent while listener was not started, so they should not be received. The rest 3 events should be received.
		Expect(receivedEvents).To(Equal(3))

	})
})

func getFreePort() (string, error) {
	var listener net.Listener
	var err error
	if listener, err = net.Listen("tcp", ":0"); err == nil {
		var port string
		_, port, err = net.SplitHostPort(listener.Addr().String())
		defer func() {
			_ = listener.Close()
		}()
		return port, err
	}
	return "", err
}
