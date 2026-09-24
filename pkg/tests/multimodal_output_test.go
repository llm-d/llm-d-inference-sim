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

package tests

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

var _ = Describe("POST /v1/audio/speech", func() {
	It("returns binary WAV for a non-streaming request", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		body := `{"model":"` + common.TestModelName + `","input":"hello world","voice":"alloy"}`
		resp, err := client.Post("http://localhost/v1/audio/speech", "application/json", strings.NewReader(body))
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		Expect(resp.Header.Get("Content-Type")).To(Equal("audio/wav"))

		data, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(data).NotTo(BeEmpty())
		// Valid WAV starts with "RIFF"
		Expect(string(data[:4])).To(Equal("RIFF"))
	})

	It("returns 400 when input is missing", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Post("http://localhost/v1/audio/speech", "application/json",
			strings.NewReader(`{"model":"m"}`))
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
	})

	It("streams speech.audio.delta and speech.audio.done SSE events", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		body := `{"model":"` + common.TestModelName + `","input":"hello","stream":true}`
		resp, err := client.Post("http://localhost/v1/audio/speech", "application/json", strings.NewReader(body))
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		Expect(resp.Header.Get("Content-Type")).To(Equal("text/event-stream"))

		var eventTypes []string
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == api.SSEDoneMarker {
				break
			}
			var event map[string]json.RawMessage
			Expect(json.Unmarshal([]byte(payload), &event)).To(Succeed())
			var eventType string
			Expect(json.Unmarshal(event["type"], &eventType)).To(Succeed())
			eventTypes = append(eventTypes, eventType)
		}

		Expect(eventTypes).To(ContainElements("speech.audio.delta", "speech.audio.done"))
	})
})

var _ = Describe("POST /v1/images/generations", func() {
	It("returns a b64_json image for a basic request", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		body := `{"model":"` + common.TestModelName + `","prompt":"a cat","size":"1024x1024"}`
		resp, err := client.Post("http://localhost/v1/images/generations", "application/json", strings.NewReader(body))
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		Expect(resp.Header.Get("Content-Type")).To(Equal("application/json"))

		var result api.ImagesGenerationsResponse
		Expect(json.NewDecoder(resp.Body).Decode(&result)).To(Succeed())
		Expect(result.Data).To(HaveLen(1))
		Expect(result.Data[0].B64JSON).NotTo(BeEmpty())
		Expect(result.Size).To(Equal("1024x1024"))
	})

	It("returns n images when n > 1", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		body := `{"model":"` + common.TestModelName + `","prompt":"dogs","n":3}`
		resp, err := client.Post("http://localhost/v1/images/generations", "application/json", strings.NewReader(body))
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		var result api.ImagesGenerationsResponse
		Expect(json.NewDecoder(resp.Body).Decode(&result)).To(Succeed())
		Expect(result.Data).To(HaveLen(3))
	})

	It("returns 400 when prompt is missing", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Post("http://localhost/v1/images/generations", "application/json",
			strings.NewReader(`{"model":"m"}`))
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
	})
})

var _ = Describe("POST /v1/chat/completions audio modality", func() {
	It("includes message.audio when modalities contains audio", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		body := `{
			"model":"` + common.TestModelName + `",
			"modalities":["text","audio"],
			"messages":[{"role":"user","content":"say hello"}]
		}`
		resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(body))
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck

		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var result map[string]json.RawMessage
		Expect(json.NewDecoder(resp.Body).Decode(&result)).To(Succeed())

		var choices []map[string]json.RawMessage
		Expect(json.Unmarshal(result["choices"], &choices)).To(Succeed())
		Expect(choices).NotTo(BeEmpty())

		var message map[string]json.RawMessage
		Expect(json.Unmarshal(choices[0]["message"], &message)).To(Succeed())
		Expect(message).To(HaveKey("audio"))

		var audio api.ChatAudio
		Expect(json.Unmarshal(message["audio"], &audio)).To(Succeed())
		Expect(audio.ID).NotTo(BeEmpty())
		Expect(audio.Data).NotTo(BeEmpty())
		Expect(audio.ExpiresAt).To(BeNumerically(">", 0))
	})

	It("omits message.audio when modalities does not contain audio", func() {
		ctx := context.TODO()
		client, err := startServer(ctx, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())

		body := `{
			"model":"` + common.TestModelName + `",
			"messages":[{"role":"user","content":"say hello"}]
		}`
		resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(body))
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck

		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var result map[string]json.RawMessage
		Expect(json.NewDecoder(resp.Body).Decode(&result)).To(Succeed())

		var choices []map[string]json.RawMessage
		Expect(json.Unmarshal(result["choices"], &choices)).To(Succeed())
		Expect(choices).NotTo(BeEmpty())

		var message map[string]json.RawMessage
		Expect(json.Unmarshal(choices[0]["message"], &message)).To(Succeed())
		Expect(message).NotTo(HaveKey("audio"))
	})
})
