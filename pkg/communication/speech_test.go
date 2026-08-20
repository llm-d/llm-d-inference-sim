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
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/valyala/fasthttp/fasthttputil"
	"k8s.io/klog/v2"
)

const speechURL = "http://localhost/v1/audio/speech"

var _ = Describe("Speech API", func() {
	var client *http.Client

	BeforeEach(func() {
		ctx := context.Background()
		sim := newRunningSim(ctx)
		comm := New(klog.Background(), sim)
		listener := fasthttputil.NewInmemoryListener()

		go func() {
			_ = comm.StartHTTPServer(ctx, listener)
		}()

		DeferCleanup(func() {
			Expect(listener.Close()).To(Succeed())
			sim.Stop()
		})

		client = &http.Client{
			Transport: &http.Transport{
				DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
					return listener.Dial()
				},
			},
		}
	})

	It("returns deterministic WAV audio and usage headers", func() {
		resp := postSpeech(client, fmt.Sprintf(`{
			"model": %q,
			"input": "Hello from the simulator",
			"instructions": "Speak clearly",
			"voice": "vivian",
			"ref_audio": "data:audio/wav;base64,ignored",
			"ref_text": "ignored"
		}`, common.TestModelName))
		defer closeSpeechBody(resp)

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		Expect(resp.Header.Get("Content-Type")).To(Equal("audio/wav"))
		Expect(resp.Header.Get("x-vllm-omni-input-tokens")).To(Equal("6"))
		Expect(resp.Header.Get("x-vllm-omni-output-tokens")).To(Equal("2"))
		Expect(resp.Header.Get("x-vllm-omni-total-tokens")).To(Equal("8"))
		Expect(resp.Header.Get("x-vllm-omni-input-text-tokens")).To(Equal("6"))
		Expect(resp.Header.Get("x-vllm-omni-input-audio-tokens")).To(Equal("0"))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(body).To(HaveLen(1004))
		Expect(body[:4]).To(Equal([]byte("RIFF")))
		Expect(body[8:12]).To(Equal([]byte("WAVE")))
	})

	DescribeTable("maps response formats to media types",
		func(responseFormat string, expectedMediaType string, expectedPrefix []byte) {
			resp := postSpeech(client, fmt.Sprintf(`{
				"model": %q,
				"input": "format test",
				"response_format": %q
			}`, common.TestModelName, responseFormat))
			defer closeSpeechBody(resp)

			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			Expect(resp.Header.Get("Content-Type")).To(Equal(expectedMediaType))
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(bytes.HasPrefix(body, expectedPrefix)).To(BeTrue())
		},
		Entry("WAV", "wav", "audio/wav", []byte("RIFF")),
		Entry("PCM", "pcm", "audio/pcm", []byte{0xb0, 0x04}),
		Entry("FLAC", "flac", "audio/flac", []byte("fLaC")),
		Entry("MP3", "mp3", "audio/mpeg", []byte("ID3")),
		Entry("Opus", "opus", "audio/ogg", []byte("OggS")),
	)

	It("streams raw audio when stream_format is audio", func() {
		resp := postSpeech(client, fmt.Sprintf(`{
			"model": %q,
			"input": "raw stream",
			"response_format": "pcm",
			"stream_format": "audio"
		}`, common.TestModelName))
		defer closeSpeechBody(resp)

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		Expect(resp.Header.Get("Content-Type")).To(Equal("audio/pcm"))
		Expect(resp.ContentLength).To(Equal(int64(-1)))

		firstHalf := make([]byte, 480)
		_, err := io.ReadFull(resp.Body, firstHalf)
		Expect(err).NotTo(HaveOccurred())

		secondHalf, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(secondHalf).To(HaveLen(480))
	})

	DescribeTable("streams speech.audio SSE events",
		func(requestFields string) {
			resp := postSpeech(client, fmt.Sprintf(`{
				"model": %q,
				"input": "SSE stream",
				"response_format": "pcm",
				%s
			}`, common.TestModelName, requestFields))
			defer closeSpeechBody(resp)

			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			Expect(resp.Header.Get("Content-Type")).To(Equal("text/event-stream"))

			events := readSpeechSSE(resp.Body)
			Expect(events).To(HaveLen(3))
			Expect(events[0].eventType).To(Equal("speech.audio.delta"))
			Expect(events[1].eventType).To(Equal("speech.audio.delta"))
			Expect(events[2].eventType).To(Equal("speech.audio.done"))

			audioBytes := make([]byte, 0, 960)
			for _, event := range events[:2] {
				var delta api.SpeechAudioDeltaEvent
				Expect(json.Unmarshal(event.data, &delta)).To(Succeed())
				Expect(delta.Type).To(Equal("speech.audio.delta"))
				Expect(delta.ResponseFormat).To(Equal("pcm"))
				chunk, err := base64.StdEncoding.DecodeString(delta.Audio)
				Expect(err).NotTo(HaveOccurred())
				audioBytes = append(audioBytes, chunk...)
			}
			Expect(audioBytes).To(HaveLen(960))

			var done api.SpeechAudioDoneEvent
			Expect(json.Unmarshal(events[2].data, &done)).To(Succeed())
			Expect(done.Type).To(Equal("speech.audio.done"))
			Expect(done.Usage.InputTokens).To(Equal(2))
			Expect(done.Usage.OutputTokens).To(Equal(2))
			Expect(done.Usage.TotalTokens).To(Equal(4))
		},
		Entry("with stream=true", `"stream": true`),
		Entry("with stream_format=sse", `"stream_format": "sse"`),
	)

	DescribeTable("rejects invalid speech requests",
		func(requestBody string, expectedStatus int, expectedMessage string) {
			resp := postSpeech(client, requestBody)
			defer closeSpeechBody(resp)

			Expect(resp.StatusCode).To(Equal(expectedStatus))
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(body)).To(ContainSubstring(expectedMessage))
		},
		Entry("missing input",
			fmt.Sprintf(`{"model": %q}`, common.TestModelName),
			http.StatusBadRequest, "input is required"),
		Entry("unknown model",
			`{"model": "unknown", "input": "hello"}`,
			http.StatusNotFound, "does not exist"),
		Entry("invalid response format",
			fmt.Sprintf(`{"model": %q, "input": "hello", "response_format": "aac"}`, common.TestModelName),
			http.StatusBadRequest, "unsupported response_format"),
		Entry("invalid stream format",
			fmt.Sprintf(`{"model": %q, "input": "hello", "stream_format": "json"}`, common.TestModelName),
			http.StatusBadRequest, "unsupported stream_format"),
		Entry("compressed streaming format",
			fmt.Sprintf(`{"model": %q, "input": "hello", "stream": true, "response_format": "mp3"}`,
				common.TestModelName),
			http.StatusBadRequest, "requires response_format"),
		Entry("streaming speed adjustment",
			fmt.Sprintf(`{"model": %q, "input": "hello", "stream": true, "response_format": "pcm", "speed": 1.5}`,
				common.TestModelName),
			http.StatusBadRequest, "does not support speed adjustment"),
	)
})

func postSpeech(client *http.Client, body string) *http.Response {
	req, err := http.NewRequest(http.MethodPost, speechURL, strings.NewReader(body))
	Expect(err).NotTo(HaveOccurred())
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	Expect(err).NotTo(HaveOccurred())
	return resp
}

func closeSpeechBody(resp *http.Response) {
	Expect(resp.Body.Close()).To(Succeed())
}

type speechSSEEvent struct {
	eventType string
	data      []byte
}

func readSpeechSSE(reader io.Reader) []speechSSEEvent {
	scanner := bufio.NewScanner(reader)
	var events []speechSSEEvent
	var eventType string
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			eventType = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			events = append(events, speechSSEEvent{
				eventType: eventType,
				data:      bytes.Clone([]byte(strings.TrimPrefix(line, "data: "))),
			})
		}
	}
	Expect(scanner.Err()).NotTo(HaveOccurred())
	return events
}
