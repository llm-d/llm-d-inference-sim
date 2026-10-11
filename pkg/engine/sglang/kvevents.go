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
	"errors"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
)

// NewKVEventEncoder reports that this engine has no KV-event wire format.
//
// sglang encodes its events as tagged msgpack maps, while the router adapter
// that consumes them decodes positional arrays in vLLM's field order; the two
// cannot read each other, so there is no format to emit until the contract is
// settled upstream. Unreachable in practice, since ValidateConfig rejects an
// enabled KV cache before the block cache is built.
func (Engine) NewKVEventEncoder(common.Configuration) (kvcache.EventEncoder, error) {
	return nil, errors.New("the sglang engine does not support KV-cache events")
}
