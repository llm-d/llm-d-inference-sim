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
	"context"
	"encoding/base64"
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

const (
	input = "The purple giraffe sang opera while riding a bicycle through the crowded market."
)

func TestSimpleTokenizerOwnsCachedStrings(t *testing.T) {
	st := NewSimpleTokenizer()
	ids, texts, err := st.RenderText(strings.Repeat(" ", 1024*1024) + "token")
	if err != nil || len(ids) != 2 {
		t.Fatalf("expected whitespace and word tokens, got %v, %v", ids, err)
	}
	cached := st.idToEntry[ids[1]].Value.(*reverseMapEntry).str
	if cached != texts[1] || unsafe.StringData(cached) == unsafe.StringData(texts[1]) {
		t.Fatal("cached token must own its text without retaining the source prompt")
	}
}

var _ = Describe("tokenizer", func() {
	messages := []api.Message{
		{Role: api.RoleUser, Content: api.ChatComplContent{Raw: "q1"}},
		{Role: api.RoleAssistant, Content: api.ChatComplContent{Raw: "a1"}},
		{Role: api.RoleUser, Content: api.ChatComplContent{Raw: "q2"}},
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
		tokens, strTokens, _, err := tokenizerMngr.TestTokenizer().RenderMessages(messages, api.RenderTools{}, "")
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
		tokens, _, _, err := tokenizerMngr.RealTokenizer().RenderMessages(messages, api.RenderTools{}, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
	})

	It("should return kwargs_data for multimodal messages via test tokenizer", func() {
		mmMessages := []api.Message{
			{Role: api.RoleUser, Content: api.ChatComplContent{
				Structured: []api.ChatComplContentBlock{
					{Type: "image_url", ImageURL: &api.ChatComplURLBlock{Url: "http://x/a.jpg"}},
				},
			}},
		}
		_, _, features, err := tokenizerMngr.TestTokenizer().RenderMessages(mmMessages, api.RenderTools{}, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(features).NotTo(BeNil())
		Expect(features.KwargsData).To(HaveKey(mmModalityImage))
		Expect(features.KwargsData[mmModalityImage]).To(HaveLen(1))
		_, decodeErr := base64.StdEncoding.DecodeString(features.KwargsData[mmModalityImage][0])
		Expect(decodeErr).NotTo(HaveOccurred())
	})

	It("should omit kwargs_data for mm_payload none via test tokenizer", func() {
		mmMessages := []api.Message{
			{Role: api.RoleUser, Content: api.ChatComplContent{
				Structured: []api.ChatComplContentBlock{
					{Type: "image_url", ImageURL: &api.ChatComplURLBlock{Url: "http://x/a.jpg"}},
				},
			}},
		}
		_, _, features, err := tokenizerMngr.TestTokenizer().RenderMessages(mmMessages, api.RenderTools{}, api.MMPayloadNone)
		Expect(err).NotTo(HaveOccurred())
		Expect(features).NotTo(BeNil())
		Expect(features.MMHashes).To(HaveKey(mmModalityImage))
		Expect(features.KwargsData).To(BeNil())
	})

	It("should detokenize previously tokenized text with simple tokenizer", func() {
		st := NewSimpleTokenizer()
		tokens, _, err := st.RenderText(input)
		Expect(err).NotTo(HaveOccurred())

		output, err := st.Detokenize(tokens)
		Expect(err).NotTo(HaveOccurred())
		Expect(output).To(Equal(input))
	})

	It("should render unknown token ids as placeholders with simple tokenizer", func() {
		st := NewSimpleTokenizer()
		output, err := st.Detokenize([]uint32{12345})
		Expect(err).NotTo(HaveOccurred())
		Expect(output).To(Equal("<unk_12345>"))
	})

	It("should evict the least recently encoded ids when the reverse map is full", func() {
		st := newSimpleTokenizerWithCapacity(2)
		oldIDs, _, err := st.RenderText("aaa")
		Expect(err).NotTo(HaveOccurred())
		Expect(oldIDs).To(HaveLen(1))

		newIDs, _, err := st.RenderText("bbb ccc")
		Expect(err).NotTo(HaveOccurred())
		Expect(newIDs).To(HaveLen(2))
		Expect(st.evictionOrder.Len()).To(Equal(2))

		output, err := st.Detokenize(oldIDs)
		Expect(err).NotTo(HaveOccurred())
		Expect(output).To(Equal(fmt.Sprintf("<unk_%d>", oldIDs[0])))

		output, err = st.Detokenize(newIDs)
		Expect(err).NotTo(HaveOccurred())
		Expect(output).To(Equal("bbb ccc"))
	})

	It("should keep re-encoded ids when the reverse map is full", func() {
		st := newSimpleTokenizerWithCapacity(2)
		keptIDs, _, err := st.RenderText("aaa bbb")
		Expect(err).NotTo(HaveOccurred())
		Expect(keptIDs).To(HaveLen(2))

		// re-encoding "bbb" (string tokens keep trailing whitespace, so it
		// must stay last) refreshes it, making "aaa " the eviction victim
		_, _, err = st.RenderText("ccc bbb")
		Expect(err).NotTo(HaveOccurred())
		Expect(st.evictionOrder.Len()).To(Equal(2))

		output, err := st.Detokenize(keptIDs)
		Expect(err).NotTo(HaveOccurred())
		Expect(output).To(Equal(fmt.Sprintf("<unk_%d>bbb", keptIDs[0])))
	})

	It("should detokenize with real tokenizer", func() {
		tokens, _, err := tokenizerMngr.RealTokenizer().RenderText(input)
		Expect(err).NotTo(HaveOccurred())

		output, err := tokenizerMngr.RealTokenizer().Detokenize(tokens)
		if err != nil && strings.Contains(err.Error(), "status 404") {
			Skip("render container does not serve /derender")
		}
		Expect(err).NotTo(HaveOccurred())
		Expect(output).To(Equal(input))
	})

	It("should return nil kwargs_data for text-only messages via real tokenizer", func() {
		tokens, _, features, err := tokenizerMngr.RealTokenizer().RenderMessages(messages, api.RenderTools{}, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).NotTo(BeEmpty())
		// text-only messages carry no MM features
		if features != nil {
			Expect(features.KwargsData).To(BeEmpty())
		}
	})

	Describe("multimodal render", func() {
		text := func(s string) api.ChatComplContentBlock {
			return api.ChatComplContentBlock{Type: "text", Text: s}
		}
		image := func(url string) api.ChatComplContentBlock {
			return api.ChatComplContentBlock{
				Type:     "image_url",
				ImageURL: &api.ChatComplURLBlock{Url: url},
			}
		}
		audio := func(data, format string) api.ChatComplContentBlock {
			return api.ChatComplContentBlock{
				Type:       "input_audio",
				InputAudio: &api.ChatComplInputAudioBlock{Data: data, Format: format},
			}
		}
		audioURL := func(url string) api.ChatComplContentBlock {
			return api.ChatComplContentBlock{
				Type:     "audio_url",
				AudioURL: &api.ChatComplURLBlock{Url: url},
			}
		}
		video := func(url string) api.ChatComplContentBlock {
			return api.ChatComplContentBlock{
				Type:     "video_url",
				VideoURL: &api.ChatComplURLBlock{Url: url},
			}
		}
		mkMsg := func(blocks ...api.ChatComplContentBlock) api.Message {
			return api.Message{
				Role:    api.RoleUser,
				Content: api.ChatComplContent{Structured: blocks},
			}
		}
		render := func(msgs ...api.Message) ([]uint32, []string, *api.RenderMMFeatures) {
			tokens, strs, feats, err := NewSimpleTokenizer().RenderMessages(msgs, api.RenderTools{}, "")
			Expect(err).NotTo(HaveOccurred())
			return tokens, strs, feats
		}
		expectPadRun := func(tokens []uint32, p api.RenderPlaceholder, modality string) {
			layout := mmLayouts[modality]
			Expect(p.Length).To(Equal(layout.spanLength()))
			Expect(p.Offset + p.Length).To(BeNumerically("<=", len(tokens)))
			for _, id := range tokens[p.Offset : p.Offset+p.Length] {
				Expect(id).To(Equal(layout.padTokenID()))
			}
		}

		It("returns nil features when no media blocks are present", func() {
			_, _, feats := render(mkMsg(text("hello")))
			Expect(feats).To(BeNil())
		})

		It("renders text-only messages as the joined string", func() {
			msgs := []api.Message{mkMsg(text("hello")), mkMsg(text("world"))}
			tokens, strs, _ := render(msgs...)
			expected, expectedStrs, err := NewSimpleTokenizer().RenderText(
				api.StartMessageSeparator + msgs[0].PlainText(true) + api.EndMessageSeparator +
					api.StartMessageSeparator + msgs[1].PlainText(true) + api.EndMessageSeparator)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal(expected))
			Expect(strs).To(Equal(expectedStrs))
		})

		It("emits an image hash keyed by image", func() {
			_, _, feats := render(mkMsg(text("describe"), image("http://x/a.jpg")))
			Expect(feats).NotTo(BeNil())
			Expect(feats.MMHashes).To(HaveKey(mmModalityImage))
			Expect(feats.MMHashes[mmModalityImage]).To(HaveLen(1))
			Expect(feats.MMHashes[mmModalityImage][0]).To(HavePrefix("sim_img_"))
			Expect(feats.MMPlaceholders[mmModalityImage]).To(HaveLen(1))
		})

		It("emits an audio hash keyed by audio", func() {
			_, _, feats := render(mkMsg(text("transcribe"), audio("base64data", "wav")))
			Expect(feats).NotTo(BeNil())
			Expect(feats.MMHashes).To(HaveKey(mmModalityAudio))
			Expect(feats.MMHashes[mmModalityAudio][0]).To(HavePrefix("sim_audio_"))
		})

		It("emits an audio hash for audio_url keyed by audio", func() {
			_, _, feats := render(mkMsg(text("transcribe"), audioURL("http://x/a.flac")))
			Expect(feats).NotTo(BeNil())
			Expect(feats.MMHashes).To(HaveKey(mmModalityAudio))
			Expect(feats.MMHashes[mmModalityAudio]).To(HaveLen(1))
			Expect(feats.MMHashes[mmModalityAudio][0]).To(HavePrefix("sim_audio_"))
			Expect(feats.MMPlaceholders[mmModalityAudio]).To(HaveLen(1))
		})

		It("emits a video hash keyed by video", func() {
			_, _, feats := render(mkMsg(text("watch"), video("http://x/v.mp4")))
			Expect(feats).NotTo(BeNil())
			Expect(feats.MMHashes).To(HaveKey(mmModalityVideo))
			Expect(feats.MMHashes[mmModalityVideo][0]).To(HavePrefix("sim_video_"))
		})

		It("places each item as a uniform run of its modality's pad token", func() {
			tokens, strs, feats := render(mkMsg(
				text("mixed"), image("http://x/a.jpg"), text("and"), audio("data", "wav"), video("http://x/v.mp4"), image("http://x/b.jpg"),
			))
			Expect(feats).NotTo(BeNil())
			Expect(tokens).To(HaveLen(len(strs)))
			Expect(feats.MMPlaceholders[mmModalityImage]).To(HaveLen(2))
			for modality, placeholders := range feats.MMPlaceholders {
				Expect(placeholders).To(HaveLen(len(feats.MMHashes[modality])))
				for _, p := range placeholders {
					expectPadRun(tokens, p, modality)
					Expect(strs[p.Offset]).To(Equal(mmLayouts[modality].padToken))
				}
			}
		})

		It("keeps placeholders sorted and non-overlapping across modalities", func() {
			_, _, feats := render(mkMsg(image("http://x/a.jpg"), audio("data", "wav"), image("http://x/b.jpg"), video("http://x/v.mp4")))
			var all []api.RenderPlaceholder
			for _, placeholders := range feats.MMPlaceholders {
				all = append(all, placeholders...)
			}
			sort.Slice(all, func(i, j int) bool { return all[i].Offset < all[j].Offset })
			for i := 1; i < len(all); i++ {
				Expect(all[i].Offset).To(BeNumerically(">=", all[i-1].Offset+all[i-1].Length))
			}
		})

		It("places the run right after the preceding text", func() {
			msg := mkMsg(text("describe"), image("http://x/a.jpg"))
			tokens, _, feats := render(msg)
			prefix, _, err := NewSimpleTokenizer().RenderText(api.StartMessageSeparator + msg.Role + ": describe\n")
			Expect(err).NotTo(HaveOccurred())
			p := feats.MMPlaceholders[mmModalityImage][0]
			Expect(p.Offset).To(Equal(len(prefix)))
			Expect(tokens[:p.Offset]).To(Equal(prefix))
		})

		It("detokenizes pad runs to the pad token string", func() {
			st := NewSimpleTokenizer()
			tokens, _, feats, err := st.RenderMessages([]api.Message{mkMsg(image("http://x/a.jpg"))}, api.RenderTools{}, "")
			Expect(err).NotTo(HaveOccurred())
			p := feats.MMPlaceholders[mmModalityImage][0]
			out, err := st.Detokenize(tokens[p.Offset : p.Offset+p.Length])
			Expect(err).NotTo(HaveOccurred())
			Expect(out).To(Equal(strings.Repeat(mmLayouts[mmModalityImage].padToken, p.Length)))
		})

		It("derives the image and video span from the grid metadata", func() {
			for _, modality := range []string{mmModalityImage, mmModalityVideo} {
			var gridDims []int64
			for _, v := range MMItemMetadata(modality) {
				gridDims = v
			}
			Expect(gridDims).To(HaveLen(3))
			Expect(mmLayouts[modality].spanLength()).To(Equal(int(gridDims[0] * gridDims[1] * gridDims[2] / 4)))
			}
			Expect(MMItemMetadata(mmModalityImage)).To(HaveKeyWithValue("image_grid_thw", []int64{1, 8, 8}))
			Expect(MMItemMetadata(mmModalityVideo)).To(HaveKeyWithValue("video_grid_thw", []int64{2, 8, 8}))
		})

		It("produces deterministic hashes for identical input", func() {
			msgs := []api.Message{mkMsg(image("http://x/a.jpg"), audio("data", "wav"), video("http://x/v.mp4"))}
			_, _, a := render(msgs...)
			_, _, b := render(msgs...)
			Expect(a).To(Equal(b))
		})

		It("skips media blocks with empty identifiers", func() {
			_, _, feats := render(mkMsg(image(""), audio("", "wav"), video("")))
			Expect(feats).To(BeNil())
		})

		It("treats audio format as routing-irrelevant (same data different format collides)", func() {
			_, _, wav := render(mkMsg(audio("samebytes", "wav")))
			_, _, mp3 := render(mkMsg(audio("samebytes", "mp3")))
			Expect(wav.MMHashes[mmModalityAudio]).To(Equal(mp3.MMHashes[mmModalityAudio]))
		})

		It("emits kwargs_data as valid base64 per modality for each item", func() {
			_, _, feats := render(mkMsg(image("http://x/a.jpg"), audio("data", "wav"), video("http://x/v.mp4")))
			Expect(feats).NotTo(BeNil())
			for _, mod := range []string{mmModalityImage, mmModalityAudio, mmModalityVideo} {
				Expect(feats.KwargsData).To(HaveKey(mod))
				Expect(feats.KwargsData[mod]).To(HaveLen(1))
				for _, s := range feats.KwargsData[mod] {
					_, err := base64.StdEncoding.DecodeString(s)
					Expect(err).NotTo(HaveOccurred(), "kwargs_data[%s] entry is not valid base64", mod)
				}
			}
		})
	})

	Describe("ExpandContentParts", func() {
		imagePad := mmLayouts[mmModalityImage].padTokenID()
		videoPad := mmLayouts[mmModalityVideo].padTokenID()
		imageSpan := mmLayouts[mmModalityImage].spanLength()
		videoSpan := mmLayouts[mmModalityVideo].spanLength()

		It("expands each collapsed pad token to its modality's run, in order", func() {
			parts := []api.GenerateContentPart{
				{Type: "image_url", URL: "http://x/a.jpg", UUID: "img-a"},
				{Type: "video_url", URL: "http://x/v.mp4", UUID: "vid-a"},
				{Type: "image_url", URL: "http://x/b.jpg", UUID: "img-b"},
			}
			tokens, feats, err := ExpandContentParts([]uint32{1, imagePad, 2, videoPad, 3, imagePad, 4}, parts)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(HaveLen(4 + 2*imageSpan + videoSpan))
			Expect(feats.MMHashes).To(Equal(map[string][]string{
				mmModalityImage: {"img-a", "img-b"},
				mmModalityVideo: {"vid-a"},
			}))
			Expect(feats.MMPlaceholders).To(Equal(map[string][]api.RenderPlaceholder{
				mmModalityImage: {{Offset: 1, Length: imageSpan}, {Offset: 3 + imageSpan + videoSpan, Length: imageSpan}},
				mmModalityVideo: {{Offset: 2 + imageSpan, Length: videoSpan}},
			}))
			Expect(tokens[0]).To(Equal(uint32(1)))
			Expect(tokens[1+imageSpan]).To(Equal(uint32(2)))
			Expect(tokens[len(tokens)-1]).To(Equal(uint32(4)))
		})

		It("matches the render layout after the coordinator collapses it", func() {
			st := NewSimpleTokenizer()
			msgs := []api.Message{{Role: api.RoleUser, Content: api.ChatComplContent{Structured: []api.ChatComplContentBlock{
				{Type: "text", Text: "compare"},
				{Type: "image_url", ImageURL: &api.ChatComplURLBlock{Url: "http://x/a.jpg"}},
				{Type: "image_url", ImageURL: &api.ChatComplURLBlock{Url: "http://x/b.jpg"}},
			}}}}
			rendered, _, feats, err := st.RenderMessages(msgs, api.RenderTools{}, api.MMPayloadNone)
			Expect(err).NotTo(HaveOccurred())
			placeholders := feats.MMPlaceholders[mmModalityImage]
			var collapsed []uint32
			pos := 0
			for _, p := range placeholders {
				collapsed = append(collapsed, rendered[pos:p.Offset+1]...)
				pos = p.Offset + p.Length
			}
			collapsed = append(collapsed, rendered[pos:]...)
			parts := []api.GenerateContentPart{
				{Type: "image_url", URL: "http://x/a.jpg", UUID: feats.MMHashes[mmModalityImage][0]},
				{Type: "image_url", URL: "http://x/b.jpg", UUID: feats.MMHashes[mmModalityImage][1]},
			}
			expanded, expandedFeats, err := ExpandContentParts(collapsed, parts)
			Expect(err).NotTo(HaveOccurred())
			Expect(expanded).To(Equal(rendered))
			Expect(expandedFeats.MMHashes).To(Equal(feats.MMHashes))
			Expect(expandedFeats.MMPlaceholders).To(Equal(feats.MMPlaceholders))
		})

		It("keys a part without uuid by the hash render gives its URL", func() {
			st := NewSimpleTokenizer()
			msgs := []api.Message{{Role: api.RoleUser, Content: api.ChatComplContent{Structured: []api.ChatComplContentBlock{
				{Type: "image_url", ImageURL: &api.ChatComplURLBlock{Url: "http://x/a.jpg"}},
			}}}}
			_, _, feats, err := st.RenderMessages(msgs, api.RenderTools{}, "")
			Expect(err).NotTo(HaveOccurred())
			hashes := ContentPartsMMHashes([]api.GenerateContentPart{{Type: "image_url", URL: "http://x/a.jpg"}})
			Expect(hashes).To(Equal(feats.MMHashes))
		})

		It("fails when a part has no matching pad token", func() {
			parts := []api.GenerateContentPart{
				{Type: "image_url", URL: "http://x/a.jpg", UUID: "img-a"},
				{Type: "video_url", URL: "http://x/v.mp4", UUID: "vid-a"},
			}
			_, _, err := ExpandContentParts([]uint32{1, imagePad, 2}, parts)
			Expect(err).To(MatchError(ContainSubstring("<|video_pad|>")))
		})

		It("skips parts of unknown type", func() {
			tokens, feats, err := ExpandContentParts([]uint32{1, 2}, []api.GenerateContentPart{{Type: "file_url", URL: "x", UUID: "f"}})
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(Equal([]uint32{1, 2}))
			Expect(feats).To(BeNil())
		})
	})

	Describe("New tokenizer selection", func() {
		newConfig := func() *common.Configuration {
			return &common.Configuration{
				Model:           common.QwenModelName,
				RenderTimeout:   30 * time.Second,
				MMRenderTimeout: 60 * time.Second,
			}
		}

		It("returns an HF tokenizer when --render-url is set", func() {
			cfg := newConfig()
			cfg.RenderURL = "http://localhost:8082"

			tok, err := New(context.Background(), cfg, klog.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(tok).To(BeAssignableToTypeOf(&HFTokenizer{}))
		})

		It("returns the simulated tokenizer when --render-url is empty", func() {
			cfg := newConfig()
			cfg.Model = "my-fake-model"

			tok, err := New(context.Background(), cfg, klog.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(tok).To(BeAssignableToTypeOf(&SimpleTokenizer{}))
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
