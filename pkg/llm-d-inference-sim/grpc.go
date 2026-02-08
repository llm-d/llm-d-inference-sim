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
	"net"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/grpc/pb"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
)

// Submit a generation request (supports streaming)
func (s *VllmSimulator) Generate(in *pb.GenerateRequest, out grpc.ServerStreamingServer[pb.GenerateResponse]) error {
	req := s.pbRequestToRequest(in)
	sender := &grpcResponseSender{
		baseResponseSender: baseResponseSender{
			sim: &s.context,
		},
		response: make(chan *grpcResponse, 1),
	}

	s.handleRequest(req, sender)

	for response := range sender.response {
		select {
		case <-out.Context().Done():
			return out.Context().Err()
		default:
			// Send error
			if response.err != nil {
				return status.Errorf(extractGRPCCode(response.err), response.err.Message, response.err)
			}
			// Send response
			var resp *pb.GenerateResponse
			if in.Stream {
				s.context.simulateTTFT(response.respCtx)

				startDecode := time.Now()
				for i, token := range response.respCtx.responseTokens().Tokens {
					if i != 0 {
						s.context.simulateInterTokenLatency()
					}
					resp := buildPBResponseChunk([]uint32{token}, response.respCtx)
					if err := out.Send(resp); err != nil {
						return status.Errorf(codes.Internal, "send failed: %v", err)
					}
				}
				common.WriteToChannel(s.context.metrics.reqDecodeTimeChan, time.Since(startDecode).Seconds(), s.context.logger,
					"metrics.reqDecodeTimeChan")

				resp = buildPBResponseComplete(response.respCtx, true)
				sender.responseSentCallback(response.respCtx.requestContext(), response.respCtx.displayModel())
			} else {
				resp = buildPBResponseComplete(response.respCtx, false)
			}
			if err := out.Send(resp); err != nil {
				return status.Errorf(codes.Internal, "send failed: %v", err)
			}
			response.respCtx.done()
			return nil
		}
	}
	return nil
}

// Submit an embedding request
func (s *VllmSimulator) Embed(ctx context.Context, in *pb.EmbedRequest) (*pb.EmbedResponse, error) {
	return nil, nil
}

// Health check
func (s *VllmSimulator) HealthCheck(ctx context.Context, in *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	return nil, nil
}

// Abort a running request
func (s *VllmSimulator) Abort(ctx context.Context, in *pb.AbortRequest) (*pb.AbortResponse, error) {
	return nil, nil
}

// Get model information
func (s *VllmSimulator) GetModelInfo(ctx context.Context, in *pb.GetModelInfoRequest) (*pb.GetModelInfoResponse, error) {
	return &pb.GetModelInfoResponse{
		ModelPath: s.context.config.Model,
	}, nil
}

// Get server information
func (s *VllmSimulator) GetServerInfo(ctx context.Context, in *pb.GetServerInfoRequest) (*pb.GetServerInfoResponse, error) {
	return nil, nil
}

func (s *VllmSimulator) startGRPC(ctx context.Context, listener net.Listener) error {
	server := grpc.NewServer()
	pb.RegisterVllmEngineServer(server, s)
	reflection.Register(server)
	serverErr := make(chan error, 1)
	go func() {
		s.context.logger.V(logging.INFO).Info("Server starting", "protocol", "gRPC", "port", s.context.config.Port)
		serverErr <- server.Serve(listener)
	}()

	select {
	case <-ctx.Done():
		s.context.logger.V(logging.INFO).Info("Shutdown signal received, shutting down gRPC server")
		server.Stop()
		s.context.logger.V(logging.INFO).Info("gRPC server stopped")
		return nil

	case err := <-serverErr:
		if err != nil {
			s.context.logger.Error(err, "gRPC server failed")
		}
		return err
	}
}

func (s *VllmSimulator) pbRequestToRequest(in *pb.GenerateRequest) *textCompletionRequest {
	var maxTokens *int64
	if in.GetSamplingParams() != nil && in.GetSamplingParams().MaxTokens != nil {
		maxTokensValue := int64(*in.GetSamplingParams().MaxTokens)
		maxTokens = &maxTokensValue
	}
	req := openaiserverapi.NewTextCompletionRequest(in.GetRequestId(), in.GetStream(),
		s.context.config.Model, maxTokens)

	if in.GetTokenized() != nil {
		prompt := &openaiserverapi.Tokenized{}
		prompt.Tokens = in.GetTokenized().InputIds
		req.SetTokenizedPrompt(prompt)
	} else {
		req.Prompt = in.GetText()
	}

	return &textCompletionRequest{TextCompletionRequest: *req}
}

func buildPBResponseChunk(tokens []uint32, respCtx responseContext) *pb.GenerateResponse {
	return &pb.GenerateResponse{
		Response: &pb.GenerateResponse_Chunk{
			Chunk: &pb.GenerateStreamChunk{
				TokenIds:         tokens,
				PromptTokens:     uint32(respCtx.usageData().PromptTokens),
				CachedTokens:     uint32(respCtx.numberCachedPromptTokens()),
				CompletionTokens: uint32(len(tokens)),
			},
		},
	}
}

func buildPBResponseComplete(respCtx responseContext, lastChunkInStream bool) *pb.GenerateResponse {
	var outputIds []uint32
	if !lastChunkInStream {
		outputIds = respCtx.responseTokens().Tokens
	}
	var completionTokens uint32
	if !lastChunkInStream {
		completionTokens = uint32(respCtx.usageData().CompletionTokens)
	}
	return &pb.GenerateResponse{
		Response: &pb.GenerateResponse_Complete{
			Complete: &pb.GenerateComplete{
				OutputIds:        outputIds,
				PromptTokens:     uint32(respCtx.usageData().PromptTokens),
				CompletionTokens: completionTokens,
				CachedTokens:     uint32(respCtx.numberCachedPromptTokens()),
				FinishReason:     *respCtx.finishReason(),
			},
		},
	}
}

func extractGRPCCode(err *openaiserverapi.Error) codes.Code {
	switch err.Code {
	case fasthttp.StatusBadRequest:
		return codes.InvalidArgument
	case fasthttp.StatusUnauthorized:
		return codes.Unauthenticated
	case fasthttp.StatusForbidden:
		return codes.PermissionDenied
	case fasthttp.StatusNotFound:
		return codes.NotFound
	case fasthttp.StatusConflict:
		return codes.Aborted
	case fasthttp.StatusTooManyRequests:
		return codes.ResourceExhausted
	case fasthttp.StatusNotImplemented:
		return codes.Unimplemented
	case fasthttp.StatusServiceUnavailable:
		return codes.Unavailable
	case fasthttp.StatusGatewayTimeout:
		return codes.DeadlineExceeded
	case fasthttp.StatusInternalServerError:
		return codes.Internal
	default:
		return codes.Unknown
	}
}
