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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

// The frame an error takes in a stream belongs to the endpoint: the Messages API
// names every event and ends without a terminator, where the OpenAI-shaped
// endpoints send a bare data frame followed by [DONE].
var _ = Describe("streaming error frames", func() {
	body := api.NewMessagesErrorResponse("invalid_request_error", "messages must not be empty")

	It("names the event for the Messages API, and ends the stream without a marker", func() {
		builder := &messagesHTTPRespBuilder{}

		bytes, err := builder.createErrorChunk(body).SSEBytes()
		Expect(err).NotTo(HaveOccurred())
		Expect(string(bytes)).To(Equal("event: error\ndata: " +
			`{"type":"error","error":{"type":"invalid_request_error","message":"messages must not be empty"}}` +
			"\n\n"))
		Expect(builder.createDoneChunk()).To(BeNil())
	})

	It("sends a bare data frame elsewhere, followed by the terminator", func() {
		builder := &chatComplHTTPRespBuilder{}

		bytes, err := builder.createErrorChunk(api.ErrorResponse{Error: api.NewError("boom", 400, nil)}).SSEBytes()
		Expect(err).NotTo(HaveOccurred())
		Expect(string(bytes)).To(Equal("data: " +
			`{"error":{"message":"boom","type":"BadRequestError","param":null,"code":400}}` + "\n\n"))

		done, err := builder.createDoneChunk().SSEBytes()
		Expect(err).NotTo(HaveOccurred())
		Expect(string(done)).To(Equal("data: [DONE]\n\n"))
	})
})
