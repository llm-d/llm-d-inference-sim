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
	"context"
	"encoding/binary"

	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	zmq "github.com/pebbe/zmq4"
	"github.com/vmihailenco/msgpack/v5"
)

const (
	topic    = "test-topic"
	endpoint = "tcp://localhost:5557"
	data     = "Hello"
)

var _ = Describe("Publisher", func() {
	It("should publish and receive correct message", func() {
		zctx, err := zmq.NewContext()
		Expect(err).NotTo(HaveOccurred())
		sub, err := zctx.NewSocket(zmq.SUB)
		Expect(err).NotTo(HaveOccurred())
		err = sub.Bind(endpoint)
		Expect(err).NotTo(HaveOccurred())
		err = sub.SetSubscribe(topic)
		Expect(err).NotTo(HaveOccurred())
		//nolint
		defer sub.Close()

		time.Sleep(100 * time.Millisecond)

		pub, err := NewPublisher(endpoint)
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		go func() {
			// Make sure that sub.RecvMessageBytes is called before pub.PublishEvent
			time.Sleep(time.Second)
			err := pub.PublishEvent(ctx, topic, data)
			Expect(err).NotTo(HaveOccurred())
		}()

		// The message should be [topic, seq, payload]
		parts, err := sub.RecvMessageBytes(0)
		Expect(err).NotTo(HaveOccurred())
		Expect(parts).To(HaveLen(3))

		Expect(string(parts[0])).To(Equal(topic))

		seq := binary.BigEndian.Uint64(parts[1])
		Expect(seq).To(Equal(uint64(1)))

		var payload string
		err = msgpack.Unmarshal(parts[2], &payload)
		Expect(err).NotTo(HaveOccurred())
		Expect(payload).To(Equal(data))
	})
})
