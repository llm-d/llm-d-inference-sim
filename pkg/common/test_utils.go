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
	"github.com/onsi/gomega"
	zmq "github.com/pebbe/zmq4"
)

// CreateSub creates a ZMQ sub, subscribes to the provided topic, and returns the
// sub and the endpoint to publish events on
func CreateSub(topic string) (*zmq.Socket, string) {
	wildcardEndpoint := "tcp://*:*"
	zctx, err := zmq.NewContext()
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	sub, err := zctx.NewSocket(zmq.SUB)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	err = sub.Bind(wildcardEndpoint)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	// get the actual port
	endpoint, err := sub.GetLastEndpoint()
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	err = sub.SetSubscribe(topic)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	return sub, endpoint
}
