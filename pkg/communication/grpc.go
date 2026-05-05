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

package communication

import (
	"context"
	"net"

	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication/grpc/pb"
	vllmsim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
)

// Submit a generation request (supports streaming)
func (c *Communication) Generate(in *pb.GenerateRequest, out grpc.ServerStreamingServer[pb.GenerateResponse]) error {
	if c.stopping.Load() {
		return status.Error(codes.Unavailable, "server is shutting down")
	}

	req := c.pbRequestToRequest(in)
	respBuilder := &generationGRPCRespBuilder{}
	_, channel, err, _ := c.simulator.HandleRequest(req)
	if err != nil {
		return status.Errorf(extractGRPCCode(err), err.Message, err)
	}

	c.logger.V(logging.DEBUG).Info("Received", "new gRPC", req.AsString())

	var respCtx vllmsim.ResponseContext
	tokens := openaiserverapi.Tokenized{
		Tokens:  make([]uint32, 0),
		Strings: make([]string, 0),
	}

	for response := range channel.Channel {
		select {
		case <-out.Context().Done():
			return out.Context().Err()
		default:
			// Send error
			if response.Err != nil {
				return status.Errorf(extractGRPCCode(response.Err), response.Err.Message, response.Err)
			}
			respCtx = response.RespCtx

			if in.Stream {
				// Send response chunk
				if response.Tokens != nil {
					chunk := respBuilder.createChunk(respCtx, response.Tokens)
					if err := sendResponse(chunk, out); err != nil {
						return err
					}
				}
			} else if response.Tokens != nil {
				tokens.Append(*response.Tokens)
			}
		}
	}

	var resp any
	if in.Stream {
		resp = respBuilder.createLastChunk(respCtx)
	} else {
		resp = respBuilder.createResponse(respCtx, &tokens)
	}
	if err := sendResponse(resp, out); err != nil {
		return err
	}

	return nil
}

// Submit an embedding request
func (c *Communication) Embed(ctx context.Context, in *pb.EmbedRequest) (*pb.EmbedResponse, error) {
	return nil, nil
}

// Health check
func (c *Communication) HealthCheck(ctx context.Context, in *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	return nil, nil
}

// Abort a running request
func (c *Communication) Abort(ctx context.Context, in *pb.AbortRequest) (*pb.AbortResponse, error) {
	return nil, nil
}

// Get model information
func (c *Communication) GetModelInfo(ctx context.Context, in *pb.GetModelInfoRequest) (*pb.GetModelInfoResponse, error) {
	return &pb.GetModelInfoResponse{
		ModelPath: c.simulator.Context.Config.Model,
	}, nil
}

// Get server information
func (c *Communication) GetServerInfo(ctx context.Context, in *pb.GetServerInfoRequest) (*pb.GetServerInfoResponse, error) {
	return nil, nil
}

// startGRPC starts the gRPC server and returns the server instance and an error channel.
// It does not handle shutdown — callers are responsible for calling server.Stop().
func (c *Communication) startGRPC(listener net.Listener) (*grpc.Server, <-chan error) {
	server := grpc.NewServer()
	pb.RegisterVllmEngineServer(server, c)
	reflection.Register(server)
	errCh := make(chan error, 1)
	go func() {
		c.logger.V(logging.INFO).Info("Server starting", "protocol", "gRPC", "port", c.simulator.Context.Config.Port)
		errCh <- server.Serve(listener)
	}()
	return server, errCh
}

func (c *Communication) pbRequestToRequest(in *pb.GenerateRequest) *vllmsim.GenerationRequest {
	var maxTokens *int64
	if in.GetSamplingParams() != nil && in.GetSamplingParams().MaxTokens != nil {
		maxTokensValue := int64(*in.GetSamplingParams().MaxTokens)
		maxTokens = &maxTokensValue
	}
	req := openaiserverapi.NewGenerationRequest(in.GetRequestId(), in.GetStream(),
		c.simulator.Context.Config.Model, maxTokens)

	if in.GetTokenized() != nil {
		prompt := &openaiserverapi.Tokenized{}
		prompt.Tokens = in.GetTokenized().InputIds
		req.SetTokenizedPrompt(prompt)
	} else {
		req.Prompt = in.GetText()
	}

	return &vllmsim.GenerationRequest{GenerationRequest: *req}
}

func sendResponse(response any, out grpc.ServerStreamingServer[pb.GenerateResponse]) error {
	resp, ok := response.(*pb.GenerateResponse)
	if !ok {
		return status.Error(codes.Internal, "response of invalid type")
	}

	if err := out.Send(resp); err != nil {
		return status.Errorf(codes.Internal, "send failed: %v", err)
	}

	return nil
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
