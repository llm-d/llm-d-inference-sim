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

package api

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("AudioSpeechRequest", func() {
	It("unmarshals required input field", func() {
		var req AudioSpeechRequest
		Expect(json.Unmarshal([]byte(`{"input":"hello world"}`), &req)).To(Succeed())
		Expect(req.Input).To(Equal("hello world"))
	})

	It("unmarshals all standard fields", func() {
		raw := `{"input":"hi","model":"m","voice":"alloy","response_format":"wav","speed":1.5,"stream":true}`
		var req AudioSpeechRequest
		Expect(json.Unmarshal([]byte(raw), &req)).To(Succeed())
		Expect(req.Model).To(Equal("m"))
		Expect(req.Voice).To(Equal("alloy"))
		Expect(req.ResponseFormat).To(Equal("wav"))
		Expect(req.Speed).To(Equal(1.5))
		Expect(req.Stream).To(BeTrue())
	})

	It("accepts extended vllm-omni fields without error", func() {
		raw := `{"input":"hi","task_type":"CustomVoice","language":"English","instructions":"speak warmly","seed":42}`
		var req AudioSpeechRequest
		Expect(json.Unmarshal([]byte(raw), &req)).To(Succeed())
		Expect(req.TaskType).To(Equal("CustomVoice"))
		Expect(req.Language).To(Equal("English"))
		Expect(req.Instructions).To(Equal("speak warmly"))
		Expect(*req.Seed).To(Equal(int64(42)))
	})
})

var _ = Describe("ImagesGenerationsRequest", func() {
	It("unmarshals required prompt field", func() {
		var req ImagesGenerationsRequest
		Expect(json.Unmarshal([]byte(`{"prompt":"a cat"}`), &req)).To(Succeed())
		Expect(req.Prompt).To(Equal("a cat"))
	})

	It("unmarshals n and size", func() {
		raw := `{"prompt":"p","n":3,"size":"512x512","response_format":"b64_json"}`
		var req ImagesGenerationsRequest
		Expect(json.Unmarshal([]byte(raw), &req)).To(Succeed())
		Expect(req.N).To(Equal(3))
		Expect(req.Size).To(Equal("512x512"))
		Expect(req.ResponseFormat).To(Equal("b64_json"))
	})

	It("accepts diffusion extension fields without error", func() {
		steps := 20
		raw := `{"prompt":"p","negative_prompt":"blurry","num_inference_steps":20,"seed":7}`
		var req ImagesGenerationsRequest
		Expect(json.Unmarshal([]byte(raw), &req)).To(Succeed())
		Expect(req.NegativePrompt).To(Equal("blurry"))
		Expect(*req.NumInferenceSteps).To(Equal(steps))
		Expect(*req.Seed).To(Equal(int64(7)))
	})
})

var _ = Describe("ChatAudio in Message", func() {
	It("marshals message.audio when Audio is set", func() {
		msg := Message{
			Role:    RoleAssistant,
			Content: ChatComplContent{Raw: "hi"},
			Audio: &ChatAudio{
				ID:         "audio-123",
				Data:       "base64data",
				ExpiresAt:  9999,
				Transcript: "hi",
			},
		}
		b, err := json.Marshal(msg)
		Expect(err).NotTo(HaveOccurred())

		var out map[string]json.RawMessage
		Expect(json.Unmarshal(b, &out)).To(Succeed())
		Expect(out).To(HaveKey("audio"))

		var audio ChatAudio
		Expect(json.Unmarshal(out["audio"], &audio)).To(Succeed())
		Expect(audio.ID).To(Equal("audio-123"))
		Expect(audio.Data).To(Equal("base64data"))
		Expect(audio.ExpiresAt).To(Equal(int64(9999)))
		Expect(audio.Transcript).To(Equal("hi"))
	})

	It("omits message.audio when Audio is nil", func() {
		msg := Message{
			Role:    RoleAssistant,
			Content: ChatComplContent{Raw: "hi"},
		}
		b, err := json.Marshal(msg)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(b)).NotTo(ContainSubstring("audio"))
	})
})

var _ = Describe("ChatCompletionsRequest modalities", func() {
	It("unmarshals modalities field", func() {
		raw := `{"model":"m","messages":[],"modalities":["text","audio"]}`
		var req ChatCompletionsRequest
		Expect(json.Unmarshal([]byte(raw), &req)).To(Succeed())
		Expect(req.Modalities).To(ConsistOf("text", "audio"))
	})

	It("treats absent modalities as nil", func() {
		raw := `{"model":"m","messages":[]}`
		var req ChatCompletionsRequest
		Expect(json.Unmarshal([]byte(raw), &req)).To(Succeed())
		Expect(req.Modalities).To(BeNil())
	})

	It("accepts sampling_params_list without error", func() {
		raw := `{"model":"m","messages":[],"sampling_params_list":[{"temperature":0.9},{"temperature":0.4}]}`
		var req ChatCompletionsRequest
		Expect(json.Unmarshal([]byte(raw), &req)).To(Succeed())
		Expect(req.SamplingParamsList).To(HaveLen(2))
	})
})
