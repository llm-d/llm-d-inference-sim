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
	"time"

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
		s, _, err := startServerHandle(ctx, common.ModeEcho, nil, nil)
		Expect(err).NotTo(HaveOccurred())

		r, err := s.GetModelInfo(ctx, &pb.GetModelInfoRequest{})
		Expect(err).NotTo(HaveOccurred())
		Expect(r.ModelPath).To(Equal(testModel))
	})

	DescribeTable("generate, no streaming",
		func(maxTokens uint32, finishReason string, ttft string, itl string, expectedTime time.Duration) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeEcho,
				"--time-to-first-token", ttft, "--inter-token-latency", itl}
			s, _, err := startServerHandle(ctx, common.ModeEcho, args, nil)
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

			out := mockGenerateServer{start: time.Now()}
			err = s.Generate(&req, &out)
			Expect(err).NotTo(HaveOccurred())
			Expect(out.responses).To(HaveLen(1))
			resp := out.responses[0].GetComplete()
			Expect(resp).NotTo(BeNil())
			Expect(resp.OutputIds).To(HaveLen(4))
			Expect(resp.CompletionTokens).To(Equal(uint32(4)))
			Expect(resp.PromptTokens).To(Equal(uint32(4)))
			Expect(resp.FinishReason).To(Equal(finishReason))

			Expect(out.ttft).To(BeNumerically(">", expectedTime))
			if expectedTime == 0 {
				Expect(out.ttft).To(BeNumerically("<", time.Millisecond))
			}
		},
		func(maxTokens uint32, finishReason string, ttft string, itl string, expectedTime time.Duration) string {
			return fmt.Sprintf("max tokens: %d, ttft: %s, intertoken latency: %s", maxTokens, ttft, itl)
		},
		Entry(nil, uint32(128), common.StopFinishReason, "0", "0", time.Duration(0)),
		Entry(nil, uint32(3), common.LengthFinishReason, "0", "0", time.Duration(0)),
		Entry(nil, uint32(128), common.StopFinishReason, "500", "200", time.Second),
	)

	DescribeTable("generate, streaming",
		func(maxTokens uint32, finishReason string, ttft string, itl string, expectedTTFT time.Duration, expectedITL time.Duration) {
			ctx := context.TODO()
			args := []string{"cmd", "--model", testModel, "--mode", common.ModeEcho,
				"--time-to-first-token", ttft, "--inter-token-latency", itl}
			s, _, err := startServerHandle(ctx, common.ModeEcho, args, nil)
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
				Stream: true,
			}

			out := mockGenerateServer{start: time.Now()}
			err = s.Generate(&req, &out)
			Expect(err).NotTo(HaveOccurred())
			Expect(out.responses).To(HaveLen(5))
			for i, resp := range out.responses {
				if i < 4 {
					chunk := resp.GetChunk()
					Expect(chunk).NotTo(BeNil())
					Expect(chunk.TokenIds).To(HaveLen(1))
					Expect(chunk.PromptTokens).To(Equal(uint32(4)))
				} else {
					complete := resp.GetComplete()
					Expect(complete).NotTo(BeNil())
					Expect(complete.FinishReason).To(Equal(finishReason))
				}
			}

			Expect(out.ttft).To(BeNumerically(">", expectedTTFT))
			if expectedTTFT == 0 {
				Expect(out.ttft).To(BeNumerically("<", time.Millisecond))
			}
			Expect(out.itl).To(BeNumerically(">", expectedITL))
			if expectedITL == 0 {
				Expect(out.ttft).To(BeNumerically("<", time.Millisecond))
			}
		},
		func(maxTokens uint32, finishReason string, ttft string, itl string, expectedTTFT time.Duration, expectedITL time.Duration) string {
			return fmt.Sprintf("max tokens: %d, ttft: %s, intertoken latency: %s", maxTokens, ttft, itl)
		},
		Entry(nil, uint32(128), common.StopFinishReason, "0", "0", time.Duration(0), time.Duration(0)),
		Entry(nil, uint32(3), common.LengthFinishReason, "0", "0", time.Duration(0), time.Duration(0)),
		Entry(nil, uint32(128), common.StopFinishReason, "500", "300", 500*time.Millisecond, 900*time.Millisecond),
	)
})

type mockGenerateServer struct {
	grpc.ServerStream
	ctx       context.Context
	responses []*pb.GenerateResponse
	start     time.Time
	latest    time.Time
	ttft      time.Duration
	itl       time.Duration
	err       error
}

func (m *mockGenerateServer) Context() context.Context { return m.ctx }
func (m *mockGenerateServer) Send(resp *pb.GenerateResponse) error {
	if m.ttft == 0 {
		m.ttft = time.Since(m.start)
	} else {
		m.itl += time.Since(m.latest)
	}
	m.latest = time.Now()

	m.responses = append(m.responses, resp)
	return m.err
}
func (m *mockGenerateServer) SetTrailer(metadata.MD)       {}
func (m *mockGenerateServer) RecvMsg(interface{}) error    { return io.EOF }
func (m *mockGenerateServer) SendHeader(metadata.MD) error { return nil }
func (m *mockGenerateServer) SendMsg(interface{}) error    { return nil }
func (m *mockGenerateServer) SetHeader(metadata.MD) error  { return nil }
