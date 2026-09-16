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
	"mime"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/valyala/fasthttp"
)

func validateStrictContentType(contentType string) *api.Error {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err == nil && mediaType == "application/json" {
		return nil
	}

	return &api.Error{
		Message: "1 validation error:\n  Unsupported Media Type: Only 'application/json' is allowed " +
			"[\"Unsupported Media Type: Only 'application/json' is allowed\"]",
		Type: "Bad Request",
		Code: fasthttp.StatusBadRequest,
	}
}

func badRequest(message string, param *string) *api.Error {
	err := api.NewError(message, fasthttp.StatusBadRequest, param)
	return &err
}
