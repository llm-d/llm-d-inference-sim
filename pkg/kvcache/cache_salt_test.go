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
package kvcache

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Cache salt isolation", func() {
	It("isolates salts and models while preserving same-salt prefix reuse", func() {
		helper, err := NewKVCacheHelper(context.Background(), &common.Configuration{
			IP: localhost, Model: common.TestModelName,
			KVCache: common.KVCacheConfig{KVCacheSize: 64, TokenBlockSize: 8, EventBatchSize: 1},
		}, logr.Discard(), nil, nil, nil)
		Expect(err).NotTo(HaveOccurred())
		cases := []struct {
			model, salt string
			extended    bool
			cached      int
		}{
			{"model-a", `"tenant-a"`, false, 0},
			{"model-a", `"tenant-b"`, false, 0},
			{"model-a", `"tenant-a"`, false, 16},
			{"model-a", `"tenant-a"`, true, 16},
			{"model-b", `"tenant-a"`, false, 0},
			{"model-a", `null`, false, 0},
			{"model-a", `""`, false, 16},
			{"model-a", `"tenant-b"`, true, 16},
		}
		for i, tc := range cases {
			var req api.ChatCompletionsRequest
			Expect(json.Unmarshal([]byte(fmt.Sprintf(`{"model":%q,"cache_salt":%s}`, tc.model, tc.salt)), &req)).To(Succeed())
			req.SetDisplayedModel(tc.model)
			req.SetRequestID(strconv.Itoa(i))
			tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
			if tc.extended {
				tokens = append(tokens, 17, 18, 19, 20, 21, 22, 23, 24)
			}
			req.SetTokenizedPrompt(&api.Tokenized{Tokens: tokens})
			stats, err := helper.OnRequestStart(&req)
			Expect(err).NotTo(HaveOccurred())
			Expect(stats.CachedPromptTokens).To(Equal(tc.cached), "case %d", i)
			Expect(helper.OnRequestEnd(req.GetRequestID())).To(Succeed())
		}
	})
})
