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

package dataset

import (
	"context"
	"fmt"
	"strings"

	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func createDataset() *DefaultDataset {
	ds := DefaultDataset{}
	ctx := context.Background()
	logger := log.FromContext(ctx)
	err := ds.Init(context.Background(), logger, common.NewRandom(time.Now().UnixNano(), 8080), 1024, tokenizerMngr.TestTokenizer())
	Expect(err).ShouldNot(HaveOccurred())

	return &ds
}

var _ = Describe("Default Dataset", Ordered, func() {
	var dataset *DefaultDataset

	BeforeAll(func() {
		dataset = createDataset()
	})

	AfterAll(func() {
		err := dataset.Close()
		Expect(err).ShouldNot(HaveOccurred())
	})

	Context("GetRandomTokens", func() {
		It("should return complete text", func() {
			req := &openaiserverapi.ChatCompletionsRequest{}
			req.SetTokenizedPrompt(&openaiserverapi.Tokenized{})
			req.SetTokenizedPromptForEcho(&openaiserverapi.Tokenized{})
			tokens, finishReason, err := dataset.GetResponseTokens(req)
			Expect(err).ShouldNot(HaveOccurred())
			text := strings.Join(tokens.Strings, "")
			Expect(IsValidText(text)).To(BeTrue())
			Expect(finishReason).Should(Equal(common.StopFinishReason))
		})

		It("should return short text", func() {
			maxCompletionTokens := int64(2)
			req := &openaiserverapi.ChatCompletionsRequest{
				MaxCompletionTokens: &maxCompletionTokens,
			}
			tokens, finishReason, err := dataset.GetResponseTokens(req)
			Expect(err).ShouldNot(HaveOccurred())
			tokensCnt := int64(tokens.Length())
			Expect(tokensCnt).Should(BeNumerically("<=", maxCompletionTokens))
			if tokensCnt == maxCompletionTokens {
				Expect(finishReason).To(Equal(common.LengthFinishReason))
			} else {
				Expect(tokensCnt).To(BeNumerically("<", maxCompletionTokens))
				Expect(finishReason).To(Equal(common.StopFinishReason))
			}
		})

		It("should return long text", func() {
			maxCompletionTokens := int64(1000)
			req := &openaiserverapi.ChatCompletionsRequest{
				MaxTokens: &maxCompletionTokens,
			}
			tokens, finishReason, err := dataset.GetResponseTokens(req)
			Expect(err).ShouldNot(HaveOccurred())
			tokensCnt := int64(tokens.Length())
			Expect(tokensCnt).Should(BeNumerically("<=", maxCompletionTokens))
			text := strings.Join(tokens.Strings, "")
			Expect(IsValidText(text)).To(BeTrue())
			if tokensCnt == maxCompletionTokens {
				Expect(finishReason).To(Equal(common.LengthFinishReason))
			} else {
				Expect(tokensCnt).To(BeNumerically("<", maxCompletionTokens))
				Expect(finishReason).To(Equal(common.StopFinishReason))
			}
		})

		DescribeTable("should return exact num of tokens",
			func(maxCompletionTokens int) {
				n := int64(maxCompletionTokens)
				req := &openaiserverapi.ChatCompletionsRequest{
					MaxTokens: &n,
				}
				req.SetIgnoreEOS(true)
				tokens, finishReason, err := dataset.GetResponseTokens(req)
				Expect(err).ShouldNot(HaveOccurred())
				nGenTokens := int64(tokens.Length())
				Expect(nGenTokens).Should(Equal(n))
				Expect(finishReason).To(Equal(common.LengthFinishReason))
			},
			func(maxCompletionTokens int) string {
				return fmt.Sprintf("maxCompletionTokens: %d", maxCompletionTokens)
			},
			Entry("1", 1),
			Entry("42", 42),
			Entry("99", 99),
			Entry("10000", 10000),
		)
	})

	Context("GetRandomTokens", func() {
		lenArr := []int{5, 20, 50, 150}

		for _, len := range lenArr {
			name := fmt.Sprintf("should return text with %d tokens", len)
			It(name, func() {
				tokens := dataset.generatePresetRandomTokens(len)
				Expect(tokens.Tokens).Should(HaveLen(len))
			})
		}
	})

	Context("IsValidText", func() {
		validTxts := make([]string, 3)
		invalidTxts := make([]string, 5)

		validTxts[0] = completionFakeResponses[0][:4]
		validTxts[1] = completionFakeResponses[1]
		validTxts[2] = completionFakeResponses[1] + completionFakeResponses[2]

		invalidTxts[0] = (completionFakeResponses[1] + " " + completionFakeResponses[2])[3:4]
		invalidTxts[1] = completionFakeResponses[0][4:]
		invalidTxts[2] = completionFakeResponses[1] + "-" + completionFakeResponses[2]
		invalidTxts[3] = completionFakeResponses[1] + " "
		invalidTxts[4] = completionFakeResponses[1] + "   " + completionFakeResponses[2]

		for _, txt := range validTxts {
			It("text should be valid", func() {
				Expect(IsValidText(txt)).To(BeTrue())
			})
		}

		for _, txt := range invalidTxts {
			It("text should be invalid", func() {
				Expect(IsValidText(txt)).To(BeFalse())
			})
		}
	})
})

var _ = Describe("Echo Dataset", Ordered, func() {
	var (
		testPrompt      string
		dataset         EchoDataset
		maxTokens       int64
		smallMaxTokens  int64
		promptTokens    []uint32
		promptStrTokens []string

		theText          string
		theTextTokens    []uint32
		theTextStrTokens []string
	)
	BeforeAll(func() {
		testPrompt = "Hello world!"
		dataset = EchoDataset{}
		maxTokens = int64(20)
		smallMaxTokens = int64(2)

		var err error
		promptTokens, promptStrTokens, err = tokenizerMngr.TestTokenizer().RenderText(testPrompt)
		Expect(err).NotTo(HaveOccurred())

		theText = "Give a man a fish and you feed him for a day; teach a man to fish and you feed him for a lifetime"
		theTextTokens, theTextStrTokens, err = tokenizerMngr.TestTokenizer().RenderText(theText)
		Expect(err).NotTo(HaveOccurred())
	})

	Context("getTokensInEchoMode", func() {
		It("should return the same text, max tokens is not defined", func() {
			req := &openaiserverapi.TextCompletionsRequest{
				Prompt: theText,
			}

			promptTokenized := openaiserverapi.Tokenized{Tokens: theTextTokens, Strings: theTextStrTokens}
			req.SetTokenizedPrompt(&promptTokenized)
			req.SetTokenizedPromptForEcho(&promptTokenized)
			tokens, finishReason, err := dataset.GetResponseTokens(req)
			Expect(err).ShouldNot(HaveOccurred())
			Expect(tokens.Strings).Should(Equal(theTextStrTokens))
			Expect(finishReason).Should(Equal(common.StopFinishReason))
		})
		It("should return the same text, max tokens is higher than the text length", func() {
			maxTokens := int64(1000)
			req := &openaiserverapi.TextCompletionsRequest{
				Prompt:    theText,
				MaxTokens: &maxTokens,
			}
			promptTokenized := openaiserverapi.Tokenized{Tokens: theTextTokens, Strings: theTextStrTokens}
			req.SetTokenizedPrompt(&promptTokenized)
			req.SetTokenizedPromptForEcho(&promptTokenized)

			tokens, finishReason, err := dataset.GetResponseTokens(req)
			Expect(err).ShouldNot(HaveOccurred())
			Expect(tokens.Strings).Should(Equal(theTextStrTokens))
			Expect(finishReason).Should(Equal(common.StopFinishReason))
		})
		It("should return the same text, finish reason is length", func() {
			maxTokens := int64(2)
			req := &openaiserverapi.TextCompletionsRequest{
				Prompt:    theText,
				MaxTokens: &maxTokens,
			}
			promptTokenized := openaiserverapi.Tokenized{Tokens: theTextTokens, Strings: theTextStrTokens}
			req.SetTokenizedPrompt(&promptTokenized)
			req.SetTokenizedPromptForEcho(&promptTokenized)

			tokens, finishReason, err := dataset.GetResponseTokens(req)
			Expect(err).ShouldNot(HaveOccurred())
			Expect(tokens.Strings).Should(Equal(theTextStrTokens))
			Expect(finishReason).Should(Equal(common.LengthFinishReason))
		})
		It("should return the last message in chat completion", func() {
			req := &openaiserverapi.ChatCompletionsRequest{
				Messages: []openaiserverapi.ChatComplMessage{
					{Role: openaiserverapi.RoleUser, Content: openaiserverapi.ChatComplContent{Raw: "user message1"}},
					{Role: openaiserverapi.RoleAssistant, Content: openaiserverapi.ChatComplContent{Raw: "assistant message1"}},
					{Role: openaiserverapi.RoleUser, Content: openaiserverapi.ChatComplContent{Raw: testPrompt}},
				},
			}
			promptTokens, promptStrTokens, _, err := tokenizerMngr.TestTokenizer().RenderChatCompletion(req.Messages)
			Expect(err).ShouldNot(HaveOccurred())
			respTokens, resptStrTokens, err := tokenizerMngr.TestTokenizer().RenderText(testPrompt)
			Expect(err).ShouldNot(HaveOccurred())

			req.SetTokenizedPrompt(&openaiserverapi.Tokenized{Tokens: promptTokens, Strings: promptStrTokens})
			req.SetTokenizedPromptForEcho(&openaiserverapi.Tokenized{Tokens: respTokens, Strings: resptStrTokens})

			tokens, _, err := dataset.GetResponseTokens(req)
			Expect(err).ShouldNot(HaveOccurred())

			Expect(tokens.Strings).Should(Equal(resptStrTokens))
		})
	})

	DescribeTable("should work correctly in echo mode",
		func(maxTokens *int64, ignoreEos bool, isChat bool, expectedFinishReason string) {
			// tests that in echo mode the right response is returned
			var req openaiserverapi.Request
			if isChat {
				chatReq := openaiserverapi.ChatCompletionsRequest{MaxTokens: maxTokens}
				chatReq.Messages = []openaiserverapi.ChatComplMessage{{Role: openaiserverapi.RoleUser, Content: openaiserverapi.ChatComplContent{Raw: testPrompt}}}
				chatReq.IgnoreEOS = ignoreEos
				req = &chatReq
			} else {
				textReq := openaiserverapi.TextCompletionsRequest{Prompt: testPrompt, MaxTokens: maxTokens}
				textReq.IgnoreEOS = ignoreEos
				req = &textReq
			}
			req.SetTokenizedPrompt(&openaiserverapi.Tokenized{Tokens: promptTokens, Strings: promptStrTokens})
			req.SetTokenizedPromptForEcho(&openaiserverapi.Tokenized{Tokens: promptTokens, Strings: promptStrTokens})

			tokens, finishReason, err := dataset.GetResponseTokens(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(finishReason).To(Equal(expectedFinishReason))
			Expect(tokens.Strings).To(Equal(promptStrTokens))
		},
		func(maxTokens *int64, ignoreEos bool, isChat bool, expectedFinishReason string) string {
			return fmt.Sprintf("maxTokens: %s, ignoreEos: %t, isChat: %t, expectedFinishReason: %s", maxTokensToStr(maxTokens), ignoreEos, isChat, expectedFinishReason)
		},
		Entry(nil, nil, false, false, common.StopFinishReason),
		Entry(nil, &maxTokens, false, false, common.StopFinishReason),
		Entry(nil, &maxTokens, true, false, common.StopFinishReason),
		Entry(nil, nil, false, true, common.StopFinishReason),
		Entry(nil, &maxTokens, false, true, common.StopFinishReason),
		Entry(nil, &maxTokens, true, true, common.StopFinishReason),
		Entry(nil, &smallMaxTokens, false, false, common.LengthFinishReason),
	)

})

var _ = Describe("cumulativeBucketsProbabilities", Ordered, func() {
	type bucketBoundaries struct {
		start int
		end   int
	}

	var (
		dataset *DefaultDataset
	)

	BeforeAll(func() {
		dataset = createDataset()
	})

	DescribeTable("calcBucketBoundaries",
		func(maxTokens int, expectedBuckets []bucketBoundaries) {
			Expect(expectedBuckets).To(HaveLen(len(dataset.histogramHelper.cumulativeBucketsProbabilities) - 1))

			for i := range len(dataset.histogramHelper.cumulativeBucketsProbabilities) - 1 {
				start, end := dataset.histogramHelper.calcBucketBoundaries(maxTokens, i)
				Expect(start).To(Equal(expectedBuckets[i].start))
				Expect(end).To(Equal(expectedBuckets[i].end))
			}

		},
		func(maxTokens int, expectedBuckets []bucketBoundaries) string {
			var builder strings.Builder

			for _, b := range expectedBuckets {
				fmt.Fprintf(&builder, "<%d, %d>, ", b.start, b.end)
			}
			return fmt.Sprintf("maxTokens: %d,expectedBuckets: %s", maxTokens, builder.String())
		},
		Entry(nil, 500, []bucketBoundaries{{1, 20}, {21, 40}, {41, 60}, {61, 480}, {481, 499}}),
		Entry(nil, 47, []bucketBoundaries{{1, 9}, {10, 18}, {19, 27}, {28, 36}, {37, 46}}),
		Entry(nil, 50, []bucketBoundaries{{1, 9}, {10, 19}, {20, 29}, {30, 39}, {40, 49}}),
	)
})
