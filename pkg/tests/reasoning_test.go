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
    "bufio"
    "context"
    "encoding/json"
    "io"
    "net/http"
    "strconv"
    "strings"

    "github.com/llm-d/llm-d-inference-sim/pkg/common"
    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"
)

var _ = Describe("chat completion reasoning simulation", func() {
    for _, percentage := range []int{0, 50, 100} {
        pct := percentage
        It("partitions echo tokens in non-streamed responses", func() {
            client, err := startServerWithArgs(context.TODO(), []string{
                "cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
                "--reasoning-percentage", strconv.Itoa(pct),
            })
            Expect(err).NotTo(HaveOccurred())
            prompt := "alpha beta gamma delta"
            body := `{"model":"` + common.TestModelName + `","messages":[{"role":"user","content":"` + prompt + `"}],"max_tokens":20}`
            resp, err := client.Post(baseURL+"/chat/completions", "application/json", strings.NewReader(body))
            Expect(err).NotTo(HaveOccurred())
            defer resp.Body.Close()
            Expect(resp.StatusCode).To(Equal(http.StatusOK))
            data, err := io.ReadAll(resp.Body)
            Expect(err).NotTo(HaveOccurred())
            var out struct {
                Choices []struct { Message struct {
                    Reasoning string `json:"reasoning"`
                    Content string `json:"content"`
                } `json:"message"` } `json:"choices"`
                Usage struct { CompletionTokens int `json:"completion_tokens"` } `json:"usage"`
            }
            Expect(json.Unmarshal(data, &out)).To(Succeed())
            Expect(out.Choices).To(HaveLen(1))
            Expect(out.Choices[0].Message.Reasoning + out.Choices[0].Message.Content).To(Equal(prompt))
            Expect(out.Usage.CompletionTokens).To(BeNumerically(">", 0))
            if pct == 0 {
                Expect(out.Choices[0].Message.Reasoning).To(BeEmpty())
            } else if pct == 100 {
                Expect(out.Choices[0].Message.Content).To(BeEmpty())
            } else {
                Expect(out.Choices[0].Message.Reasoning).NotTo(BeEmpty())
                Expect(out.Choices[0].Message.Content).NotTo(BeEmpty())
            }
        })
    }

    It("streams reasoning before content with usage unchanged", func() {
        client, err := startServerWithArgs(context.TODO(), []string{
            "cmd", "--model", common.TestModelName, "--mode", common.ModeEcho,
            "--reasoning-percentage", "50", "--max-num-seqs", "4",
        })
        Expect(err).NotTo(HaveOccurred())
        prompt := "alpha beta gamma delta"
        body := `{"model":"` + common.TestModelName + `","messages":[{"role":"user","content":"` + prompt + `"}],"max_tokens":20,"stream":true,"stream_options":{"include_usage":true},"n":2}`
        resp, err := client.Post(baseURL+"/chat/completions", "application/json", strings.NewReader(body))
        Expect(err).NotTo(HaveOccurred())
        defer resp.Body.Close()
        Expect(resp.StatusCode).To(Equal(http.StatusOK))
        scanner := bufio.NewScanner(resp.Body)
        reason := map[int]string{}
        content := map[int]string{}
        seenContent := map[int]bool{}
        var usage int
        for scanner.Scan() {
            line := scanner.Text()
            if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
                continue
            }
            var frame struct {
                Choices []struct {
                    Index int `json:"index"`
                    Delta struct {
                        Reasoning string `json:"reasoning"`
                        Content string `json:"content"`
                    } `json:"delta"`
                } `json:"choices"`
                Usage *struct { CompletionTokens int `json:"completion_tokens"` } `json:"usage"`
            }
            Expect(json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame)).To(Succeed())
            if frame.Usage != nil { usage = frame.Usage.CompletionTokens }
            for _, choice := range frame.Choices {
                if choice.Delta.Reasoning != "" {
                    Expect(seenContent[choice.Index]).To(BeFalse())
                    reason[choice.Index] += choice.Delta.Reasoning
                }
                if choice.Delta.Content != "" {
                    seenContent[choice.Index] = true
                    content[choice.Index] += choice.Delta.Content
                }
            }
        }
        Expect(scanner.Err()).NotTo(HaveOccurred())
        Expect(reason).To(HaveLen(2))
        Expect(content).To(HaveLen(2))
        for i := 0; i < 2; i++ {
            Expect(reason[i] + content[i]).To(Equal(prompt))
        }
        Expect(usage).To(BeNumerically(">", 0))
    })
})
