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

package endpoint

import (
	"github.com/go-logr/logr"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/kvcache"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

// fakeRuntime is a minimal Runtime double for tests that exercise request-context
// behavior (tool-call creation, echo tokenization) without a real simulator engine.
type fakeRuntime struct {
	config    *common.Configuration
	random    *common.Random
	tokenizer tokenizer.Tokenizer
}

var _ Runtime = (*fakeRuntime)(nil)

func (f *fakeRuntime) Config() *common.Configuration     { return f.config }
func (f *fakeRuntime) GetRandom() *common.Random         { return f.random }
func (f *fakeRuntime) GetTokenizer() tokenizer.Tokenizer { return f.tokenizer }
func (f *fakeRuntime) Logger() logr.Logger               { return klog.Background() }
func (f *fakeRuntime) RequestStarted(req api.Request)    {}
func (f *fakeRuntime) GetResponseTokens(req api.Request) (*api.Tokenized, string, error) {
	return nil, "", nil
}
func (f *fakeRuntime) KVCacheOnRequestStart(req api.Request) (kvcache.PrefixCacheStats, *api.Error) {
	return kvcache.PrefixCacheStats{}, nil
}
func (f *fakeRuntime) KVCacheOnRequestEnd(requestID string)            {}
func (f *fakeRuntime) Sleep() bool                                     { return false }
func (f *fakeRuntime) WakeUp(activateKVCache bool)                     {}
func (f *fakeRuntime) IsSleeping() bool                                { return false }
func (f *fakeRuntime) ValidateBaseModel(model string) *api.Error       { return nil }
func (f *fakeRuntime) ShouldSendImage(headerOverride bool) bool        { return headerOverride }
func (f *fakeRuntime) MooncakeEngineMap() map[string]map[string]string { return nil }
