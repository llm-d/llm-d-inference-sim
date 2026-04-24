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

import "net"

// StartHTTPServer starts the HTTP server on the given listener and blocks until it exits.
// Intended for use in tests with a custom listener. Shutdown is driven by closing the listener.
func (c *Communication) StartHTTPServer(listener net.Listener) error {
	_, errCh, err := c.startHTTPServer(listener)
	if err != nil {
		return err
	}
	return <-errCh
}
