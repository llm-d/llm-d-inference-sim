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

package llmdinferencesim

import (
	"context"
	"fmt"
	"io"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/grpc/pb"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

var _ = Describe("gRPC", func() {

	It("get model info", func() {
		ctx := context.TODO()
		s, err := startServerWithMode(ctx, common.ModeEcho)
		Expect(err).NotTo(HaveOccurred())

		r, err := s.GetModelInfo(ctx, &pb.GetModelInfoRequest{})
		Expect(err).NotTo(HaveOccurred())
		Expect(r.ModelPath).To(Equal(testModel))
	})

	DescribeTable("generate, no streaming",
		func(maxTokens uint32, finishReason string) {
			ctx := context.TODO()
			s, err := startServerWithMode(ctx, common.ModeEcho)
			Expect(err).NotTo(HaveOccurred())

			req := pb.GenerateRequest{
				RequestId: "123",
				Input: &pb.GenerateRequest_Tokenized{
					Tokenized: &pb.TokenizedInput{
						InputIds: []uint32{32, 45, 78, 13},
					},
				},
				SamplingParams: &pb.SamplingParams{
					MaxTokens: &maxTokens,
				},
			}

			out := mockGenerateServer{}
			err = s.Generate(&req, &out)
			Expect(err).NotTo(HaveOccurred())
			Expect(out.responses).To(HaveLen(1))
			resp := out.responses[0].GetComplete()
			Expect(resp).NotTo(BeNil())
			Expect(resp.OutputIds).To(HaveLen(4))
			Expect(resp.CompletionTokens).To(Equal(uint32(4)))
			Expect(resp.PromptTokens).To(Equal(uint32(4)))
			Expect(resp.FinishReason).To(Equal(finishReason))
		},
		func(maxTokens uint32, finishReason string) string {
			return fmt.Sprintf("max tokens: %d", maxTokens)
		},
		Entry(nil, uint32(128), common.StopFinishReason),
		Entry(nil, uint32(3), common.LengthFinishReason),
	)
})

type mockGenerateServer struct {
	grpc.ServerStream
	ctx       context.Context
	responses []*pb.GenerateResponse
	err       error
}

func (m *mockGenerateServer) Context() context.Context { return m.ctx }
func (m *mockGenerateServer) Send(resp *pb.GenerateResponse) error {
	m.responses = append(m.responses, resp)
	return m.err
}
func (m *mockGenerateServer) SetTrailer(metadata.MD)       {}
func (m *mockGenerateServer) RecvMsg(interface{}) error    { return io.EOF }
func (m *mockGenerateServer) SendHeader(metadata.MD) error { return nil }
func (m *mockGenerateServer) SendMsg(interface{}) error    { return nil }
func (m *mockGenerateServer) SetHeader(metadata.MD) error  { return nil }
