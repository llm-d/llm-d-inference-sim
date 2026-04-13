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

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/onsi/gomega"
)

// constants
const (
	TestModelName    = "testmodel"
	QwenModelName    = "Qwen/Qwen2-0.5B"
	MMModelName      = "Qwen/Qwen2-VL-2B-Instruct"
	wildcardEndpoint = "tcp://*:*"
)

// CreateSub creates a ZMQ sub, subscribes to the provided topic, and returns the
// sub and the endpoint to publish events on
func CreateSub(ctx context.Context, topic string) (zmq4.Socket, string) {
	sub := NewSub(ctx)

	return sub, StartSub(sub, wildcardEndpoint, topic)
}

func NewSub(ctx context.Context) zmq4.Socket {
	return zmq4.NewSub(ctx)
}

// starts the given sub on a random port and subscribes to the given topic. Returns the sub and the real endpoint to publish events on.
func StartSub(sub zmq4.Socket, endpoint string, topic string) string {
	err := sub.Listen(endpoint)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	if topic != "" {
		err = sub.SetOption(zmq4.OptionSubscribe, topic)
	} else {
		err = sub.SetOption(zmq4.OptionSubscribe, "")
	}
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	realEndpoint := sub.Addr()
	gomega.Expect(realEndpoint).NotTo(gomega.BeNil())

	return "tcp://" + realEndpoint.String()
}
