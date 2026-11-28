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

package tests

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Named tool_choice not in tools", func() {
	const notInTools = "The tool `get_time` specified in `tool_choice` does not match any of the specified `tools`"
	const chatTools = `"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{}}}}]`
	const messagesTools = `"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{}}}]`

	DescribeTable("is rejected with 400",
		func(path string, fields string) {
			client, err := startServer(context.TODO(), common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{"model":%q,"max_tokens":16,"messages":[{"role":"user","content":%q}],%s}`,
				common.TestModelName, testUserMessage, fields)
			resp, err := client.Post("http://localhost"+path, "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(body)).To(ContainSubstring(notInTools))
		},
		Entry("chat completions", "/v1/chat/completions",
			chatTools+`,"tool_choice":{"type":"function","function":{"name":"get_time"}}`),
		Entry("streaming chat completions", "/v1/chat/completions",
			chatTools+`,"tool_choice":{"type":"function","function":{"name":"get_time"}},"stream":true`),
		Entry("messages", "/v1/messages",
			messagesTools+`,"tool_choice":{"type":"tool","name":"get_time"}`),
	)
})
