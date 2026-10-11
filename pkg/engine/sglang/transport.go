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

package sglang

import (
	"github.com/buaazp/fasthttprouter"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/communication"
)

// objectError is the value of the "object" field of an sglang error body.
const objectError = "error"

// BindHTTP registers the sglang-specific HTTP routes on r, on top of the common
// routes comm's own HTTP server already registers. sglang's native routes
// (/generate, /get_model_info, /get_server_info, /flush_cache, /ready) are not
// simulated, so only the engine-neutral OpenAI-compatible surface is served.
// Readiness is therefore probed at the core's /health/ready, not at sglang's
// /ready.
func (Engine) BindHTTP(*fasthttprouter.Router, *communication.Communication) {}

// BindGRPC reports that this engine has no gRPC surface: sglang has no
// equivalent of vLLM's gRPC engine service, so no gRPC listener is opened and
// the port serves HTTP alone.
func (Engine) BindGRPC(*grpc.Server, *communication.Communication) bool { return false }

// errorBody is sglang's error object: the same fields vLLM puts inside its
// "error" envelope, plus the "object" tag that names them an error.
type errorBody struct {
	Object string `json:"object"`
	api.Error
}

// wrappedErrorBody nests an error under an "error" key, the way sglang frames a
// streaming error frame and everything it answers on the Responses API.
type wrappedErrorBody[T any] struct {
	Error T `json:"error"`
}

// responsesErrorType is the error type sglang reports on the Responses API,
// whatever the status: its errors there are built by one helper that spells the
// type this way and does not vary it.
const responsesErrorType = "invalid_request_error"

// ErrorBody frames err the way sglang does on a route of this family. Each of the
// three is framed by its own helper upstream, and no two agree: the OpenAI-shaped
// routes put the error's fields at the top level, the Responses API nests them
// under an "error" key and spells every type the same way, and the Messages API
// answers in the Anthropic envelope.
func (Engine) ErrorBody(err api.Error, route api.ErrorRoute) any {
	switch route {
	case api.ErrorRouteMessages:
		return messagesErrorBody(err)
	case api.ErrorRouteResponses:
		err.Type = responsesErrorType
		return wrappedErrorBody[api.Error]{Error: err}
	default:
		return errorBody{Object: objectError, Error: err}
	}
}

// StreamErrorBody frames a streaming error frame, which sglang wraps under an
// "error" key on the routes whose whole-response bodies it does not wrap. The
// Responses API keeps the error's own type here, unlike the whole-response body
// above, and the Messages API frames a frame the way it frames a body.
func (Engine) StreamErrorBody(err api.Error, route api.ErrorRoute) any {
	switch route {
	case api.ErrorRouteMessages:
		return messagesErrorBody(err)
	case api.ErrorRouteResponses:
		return wrappedErrorBody[api.Error]{Error: err}
	default:
		return wrappedErrorBody[errorBody]{Error: errorBody{Object: objectError, Error: err}}
	}
}

// messagesErrorTypes maps an HTTP status to the error type sglang reports for it
// on /v1/messages. The vocabulary is Anthropic's own, so that a client parsing
// the response into its typed error classes recognizes it; a status that is not
// listed is reported as a plain API error.
var messagesErrorTypes = map[int]string{
	fasthttp.StatusBadRequest:            "invalid_request_error",
	fasthttp.StatusUnauthorized:          "authentication_error",
	fasthttp.StatusForbidden:             "permission_error",
	fasthttp.StatusNotFound:              "not_found_error",
	fasthttp.StatusRequestTimeout:        "request_timeout_error",
	fasthttp.StatusRequestEntityTooLarge: "request_too_large",
	fasthttp.StatusUnprocessableEntity:   "invalid_request_error",
	fasthttp.StatusTooManyRequests:       "rate_limit_error",
	fasthttp.StatusServiceUnavailable:    "overloaded_error",
}

// messagesAPIError is the error type of a status the map above does not name.
const messagesAPIError = "api_error"

// messagesServerErrorMessage replaces the message of a server error, which
// sglang never lets through on this route.
const messagesServerErrorMessage = "Internal server error"

// messagesErrorBody puts the error in the Anthropic envelope, with the type
// mapped to Anthropic's vocabulary and a server error's message replaced.
func messagesErrorBody(err api.Error) any {
	errorType, named := messagesErrorTypes[err.Code]
	if !named {
		errorType = messagesAPIError
	}
	message := err.Message
	if err.Code >= fasthttp.StatusInternalServerError {
		message = messagesServerErrorMessage
	}
	return api.NewMessagesErrorResponse(errorType, message)
}
