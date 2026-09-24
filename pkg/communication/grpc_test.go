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

package communication

import (
	"github.com/buaazp/fasthttprouter"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"google.golang.org/grpc"
)

// fakeTransport is a minimal Transport double for testing bindGRPC's gating on
// the active engine's gRPC support, independent of any real engine.
type fakeTransport struct {
	grpcSupported bool
}

func (fakeTransport) NewRequestValidator() (RequestValidator, error) { return nil, nil }

func (fakeTransport) BindHTTP(*fasthttprouter.Router, *Communication) {}

func (f fakeTransport) BindGRPC(*grpc.Server, *Communication) bool {
	return f.grpcSupported
}

var _ = Describe("bindGRPC", func() {
	var comm *Communication

	BeforeEach(func() {
		comm = &Communication{}
	})

	It("returns no server for an engine with no gRPC surface", func() {
		server, ok := comm.bindGRPC(fakeTransport{grpcSupported: false})
		Expect(ok).To(BeFalse())
		Expect(server).To(BeNil())
	})

	It("returns a bound server for an engine with a gRPC surface", func() {
		server, ok := comm.bindGRPC(fakeTransport{grpcSupported: true})
		Expect(ok).To(BeTrue())
		Expect(server).NotTo(BeNil())
	})
})
