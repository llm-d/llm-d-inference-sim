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

package tokenizer

import (
	"strings"

	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const (
	input = "The purple giraffe sang opera while riding a bicycle through the crowded market."
)

var _ = Describe("tokenizer", func() {
	messages := []openaiserverapi.Message{
		{Role: openaiserverapi.RoleUser, Content: openaiserverapi.ChatComplContent{Raw: "q1"}},
		{Role: openaiserverapi.RoleAssistant, Content: openaiserverapi.ChatComplContent{Raw: "a1"}},
		{Role: openaiserverapi.RoleUser, Content: openaiserverapi.ChatComplContent{Raw: "q2"}},
	}

	It("should tokenize with simple tokenizer", func() {
		tokens, strTokens, err := tokenizerMngr.TestTokenizer().RenderText(input)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))

		output := strings.Join(strTokens, "")
		Expect(output).To(Equal(input))
	})

	It("should tokenize chat with simple tokenizer", func() {
		tokens, strTokens, _, err := tokenizerMngr.TestTokenizer().RenderMessages(messages)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))
	})

	It("should tokenize with real tokenizer", func() {
		tokens, strTokens, err := tokenizerMngr.RealTokenizer().RenderText(input)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		Expect(strTokens).NotTo(BeEmpty())
		Expect(tokens).To(HaveLen(len(strTokens)))

		output := strings.Join(strTokens, "")
		Expect(output).To(Equal(input))
	})

	It("should tokenize chat with real tokenizer", func() {
		// in /chat/completions case the string tokens are not returned
		tokens, _, _, err := tokenizerMngr.RealTokenizer().RenderMessages(messages)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
	})

	Describe("stubMMFeaturesForMessages", func() {
		text := func(s string) openaiserverapi.ChatComplContentBlock {
			return openaiserverapi.ChatComplContentBlock{Type: "text", Text: s}
		}
		image := func(url string) openaiserverapi.ChatComplContentBlock {
			return openaiserverapi.ChatComplContentBlock{
				Type:     "image_url",
				ImageURL: openaiserverapi.ChatComplImageBlock{Url: url},
			}
		}
		audio := func(data, format string) openaiserverapi.ChatComplContentBlock {
			return openaiserverapi.ChatComplContentBlock{
				Type:       "input_audio",
				InputAudio: openaiserverapi.ChatComplAudioBlock{Data: data, Format: format},
			}
		}
		video := func(url string) openaiserverapi.ChatComplContentBlock {
			return openaiserverapi.ChatComplContentBlock{
				Type:     "video_url",
				VideoURL: openaiserverapi.ChatComplVideoBlock{Url: url},
			}
		}
		mkMsg := func(blocks ...openaiserverapi.ChatComplContentBlock) openaiserverapi.Message {
			return openaiserverapi.Message{
				Role:    openaiserverapi.RoleUser,
				Content: openaiserverapi.ChatComplContent{Structured: blocks},
			}
		}

		It("returns nil when no media blocks are present", func() {
			feats := stubMMFeaturesForMessages([]openaiserverapi.Message{mkMsg(text("hello"))}, 100)
			Expect(feats).To(BeNil())
		})

		It("emits an image hash keyed by image", func() {
			feats := stubMMFeaturesForMessages([]openaiserverapi.Message{mkMsg(text("describe"), image("http://x/a.jpg"))}, 100)
			Expect(feats).NotTo(BeNil())
			Expect(feats.MMHashes).To(HaveKey(mmModalityImage))
			Expect(feats.MMHashes[mmModalityImage]).To(HaveLen(1))
			Expect(feats.MMHashes[mmModalityImage][0]).To(HavePrefix("sim_img_"))
			Expect(feats.MMPlaceholders[mmModalityImage]).To(HaveLen(1))
		})

		It("emits an audio hash keyed by audio", func() {
			feats := stubMMFeaturesForMessages([]openaiserverapi.Message{mkMsg(text("transcribe"), audio("base64data", "wav"))}, 100)
			Expect(feats).NotTo(BeNil())
			Expect(feats.MMHashes).To(HaveKey(mmModalityAudio))
			Expect(feats.MMHashes[mmModalityAudio][0]).To(HavePrefix("sim_audio_"))
		})

		It("emits a video hash keyed by video", func() {
			feats := stubMMFeaturesForMessages([]openaiserverapi.Message{mkMsg(text("watch"), video("http://x/v.mp4"))}, 100)
			Expect(feats).NotTo(BeNil())
			Expect(feats.MMHashes).To(HaveKey(mmModalityVideo))
			Expect(feats.MMHashes[mmModalityVideo][0]).To(HavePrefix("sim_video_"))
		})

		It("returns all three modality keys for mixed multimedia", func() {
			feats := stubMMFeaturesForMessages([]openaiserverapi.Message{mkMsg(
				text("mixed"), image("http://x/a.jpg"), audio("data", "wav"), video("http://x/v.mp4"),
			)}, 120)
			Expect(feats).NotTo(BeNil())
			Expect(feats.MMHashes).To(HaveKey(mmModalityImage))
			Expect(feats.MMHashes).To(HaveKey(mmModalityAudio))
			Expect(feats.MMHashes).To(HaveKey(mmModalityVideo))
		})

		It("produces deterministic hashes for identical input", func() {
			msgs := []openaiserverapi.Message{mkMsg(image("http://x/a.jpg"), audio("data", "wav"), video("http://x/v.mp4"))}
			a := stubMMFeaturesForMessages(msgs, 100)
			b := stubMMFeaturesForMessages(msgs, 100)
			Expect(a).To(Equal(b))
		})

		It("skips media blocks with empty identifiers", func() {
			feats := stubMMFeaturesForMessages([]openaiserverapi.Message{mkMsg(
				image(""), audio("", "wav"), video(""),
			)}, 100)
			Expect(feats).To(BeNil())
		})

		It("treats audio format as routing-irrelevant (same data different format collides)", func() {
			wav := stubMMFeaturesForMessages([]openaiserverapi.Message{mkMsg(audio("samebytes", "wav"))}, 100)
			mp3 := stubMMFeaturesForMessages([]openaiserverapi.Message{mkMsg(audio("samebytes", "mp3"))}, 100)
			Expect(wav.MMHashes[mmModalityAudio]).To(Equal(mp3.MMHashes[mmModalityAudio]))
		})
	})

	Describe("splitIntoTokens", func() {
		bt := newBaseTokenizer()
		const text = "I hear it's very cold."
		// Natural split produced by the regex — every entry includes any trailing
		// whitespace, so joining the slice reproduces the original text exactly.
		naturalTokens := []string{"I ", "hear ", "it", "'", "s ", "very ", "cold", "."}

		It("returns the natural split when count is -1", func() {
			Expect(bt.splitIntoTokens(text, -1)).To(Equal(naturalTokens))
		})

		It("returns the natural split when count equals the natural length", func() {
			Expect(bt.splitIntoTokens(text, len(naturalTokens))).To(Equal(naturalTokens))
		})

		It("pads with empty strings when count exceeds the natural length", func() {
			result := bt.splitIntoTokens(text, len(naturalTokens)+3)
			Expect(result).To(HaveLen(len(naturalTokens) + 3))
			Expect(result[:len(naturalTokens)]).To(Equal(naturalTokens))
			Expect(result[len(naturalTokens):]).To(Equal([]string{"", "", ""}))
		})

		It("merges the tail into the last kept token when count is smaller", func() {
			result := bt.splitIntoTokens(text, 3)
			Expect(result).To(HaveLen(3))
			Expect(result[0]).To(Equal(naturalTokens[0]))
			Expect(result[1]).To(Equal(naturalTokens[1]))
			// The third token absorbs the remaining natural tokens.
			Expect(result[2]).To(Equal(strings.Join(naturalTokens[2:], "")))
			Expect(strings.Join(result, "")).To(Equal(text))
		})

		It("returns an empty slice for empty input with count -1", func() {
			Expect(bt.splitIntoTokens("", -1)).To(BeEmpty())
		})

		It("pads empty input when count is positive", func() {
			Expect(bt.splitIntoTokens("", 4)).To(Equal([]string{"", "", "", ""}))
		})
	})
})
