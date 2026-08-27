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

import "github.com/valyala/fasthttp"

// Route pairs an HTTP method and path with the handler func that serves it.
type Route struct {
	Method  string
	Path    string
	Handler fasthttp.RequestHandler
}

// RouteProvider returns the engine-specific HTTP routes to bind once c exists.
// Supplied by the active engine (see pkg/engine).
type RouteProvider func(c *Communication) []Route
