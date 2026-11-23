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

var _ = Describe("Responses tool_choice validation", func() {
	const weatherTool = `"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{}}}]`

	DescribeTable("rejects with 400",
		func(fields string, wantMsg string) {
			client, err := startServer(context.TODO(), common.ModeRandom)
			Expect(err).NotTo(HaveOccurred())

			reqBody := fmt.Sprintf(`{"model":%q,"input":%q,%s}`, common.TestModelName, testUserMessage, fields)
			resp, err := client.Post("http://localhost/v1/responses", "application/json", strings.NewReader(reqBody))
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(resp.Body.Close()).To(Succeed()) }()
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(body)).To(ContainSubstring(wantMsg))
			Expect(string(body)).To(ContainSubstring(`"param":"tool_choice"`))
		},
		Entry("a named tool that is not in tools, streaming",
			weatherTool+`,"tool_choice":{"type":"function","name":"get_time"},"stream":true`,
			"Tool choice 'function' not found in 'tools' parameter."),
		Entry("required without tools", `"tool_choice":"required"`,
			"Tool choice 'required' must be specified with 'tools' parameter."),
	)
})
