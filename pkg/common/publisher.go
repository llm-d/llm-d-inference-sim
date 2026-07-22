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
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/vmihailenco/msgpack/v5"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// ParseEndpointPort splits a ZMQ endpoint string (e.g. "tcp://127.0.0.1:5557")
// into the prefix up to and including the last colon, and the trailing port
// number. ok is false if there is no trailing ":<port>".
func ParseEndpointPort(endpoint string) (prefix string, port int, ok bool) {
	lastColon := strings.LastIndex(endpoint, ":")
	if lastColon < 0 {
		return "", 0, false
	}
	port, err := strconv.Atoi(endpoint[lastColon+1:])
	if err != nil {
		return "", 0, false
	}
	return endpoint[:lastColon+1], port, true
}

// OffsetEndpointPort adds the given offset to the port in a ZMQ endpoint
// string (e.g. "tcp://127.0.0.1:5557"). Returns the original
// endpoint unchanged if parsing fails.
func OffsetEndpointPort(endpoint string, offset int) string {
	prefix, port, ok := ParseEndpointPort(endpoint)
	if !ok {
		return endpoint
	}
	return prefix + strconv.Itoa(port+offset)
}

// EncodeSeq encodes a sequence number as an 8-byte big-endian slice, the
// wire format used for the seq frame in both the live PUB stream and KV
// events replay.
func EncodeSeq(seq uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, seq)
	return b
}

// Publisher sends events to a ZMQ endpoint.
type Publisher struct {
	socket   zmq4.Socket
	endpoint string
	seqNum   uint64
}

// NewPublisher creates a new ZMQ publisher.
// endpoint is the ZMQ address to bind to (e.g., "tcp://*:5557").
// retries is the maximum number of connection attempts.
func NewPublisher(ctx context.Context, endpoint string) (*Publisher, error) {
	socket := zmq4.NewPub(ctx,
		// -1 means try forever
		zmq4.WithDialerMaxRetries(-1),
		// reconnect if server restarts
		zmq4.WithAutomaticReconnect(true),
		// wait 1s between attempts
		zmq4.WithDialerRetry(time.Second),
	)

	// 2. Push Dial into a background goroutine
	go func() {
		// wait until the listener is ready
		err := socket.Dial(endpoint)
		if err != nil {
			// Context cancellation during shutdown is expected — don't treat it as an error.
			if ctx.Err() != nil {
				return
			}
			log.FromContext(ctx).Error(err, "ZMQ dialer exited", "endpoint", endpoint)
		} else {
			log.FromContext(ctx).Info("ZMQ dialer connected", "endpoint", endpoint)
		}
	}()

	return &Publisher{
		socket:   socket,
		endpoint: endpoint,
	}, nil
}

// PublishEvent marshals batch, assigns the next sequence number, and sends
// [topic, seq, payload] over ZMQ. Returns the assigned sequence number and
// the marshaled payload so callers can store it without re-encoding.
func (p *Publisher) PublishEvent(ctx context.Context, topic string, batch interface{}) (uint64, []byte, error) {
	logger := klog.FromContext(ctx).V(0)

	payload, err := msgpack.Marshal(batch)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to marshal event batch: %w", err)
	}

	// sequence number for ordering
	seq := atomic.AddUint64(&p.seqNum, 1)

	// send topic, sequence, payload
	msg := zmq4.NewMsgFrom([]byte(topic), EncodeSeq(seq), payload)

	if err = p.socket.Send(msg); err != nil {
		return 0, nil, fmt.Errorf("failed to send message to topic %s: %w", topic, err)
	}

	logger.V(logging.TRACE).Info("Published event batch", "topic", topic, "seq", seq)
	return seq, payload, nil
}

// Close closes the publisher and cleans up resources.
func (p *Publisher) Close() error {
	if p.socket != nil {
		return p.socket.Close()
	}
	return nil
}
