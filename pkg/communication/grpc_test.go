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

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

// fakeTransport is a minimal Transport double for testing the behaviour this
// package gates on the active engine -- gRPC support and the error wire format
// -- independent of any real engine.
type fakeTransport struct {
	grpcSupported bool
}

func (fakeTransport) BindHTTP(*fasthttprouter.Router, *Communication) {}

func (f fakeTransport) BindGRPC(*grpc.Server, *Communication) bool {
	return f.grpcSupported
}

func (fakeTransport) ErrorBody(err api.Error, route api.ErrorRoute) any {
	if route == api.ErrorRouteMessages {
		return api.NewMessagesErrorResponse(err.Type, err.Message)
	}
	return api.ErrorResponse{Error: err}
}

func (f fakeTransport) StreamErrorBody(err api.Error, route api.ErrorRoute) any {
	return f.ErrorBody(err, route)
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
