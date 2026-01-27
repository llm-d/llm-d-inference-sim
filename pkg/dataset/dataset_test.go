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
	err := ds.Init(context.Background(), logger, common.NewRandom(time.Now().UnixNano(), 8080), 1024)
	Expect(err).ShouldNot(HaveOccurred())

	return &ds
}

var _ = Describe("Dataset", Ordered, func() {
	var (
		dataset *DefaultDataset
	)

	BeforeEach(func() {
		dataset = createDataset()
	})

	AfterEach(func() {
		err := dataset.Close()
		Expect(err).ShouldNot(HaveOccurred())
	})

	Context("GetRandomTokens", func() {
		It("should return complete text", func() {
			req := &openaiserverapi.ChatCompletionRequest{}
			tokens, finishReason, err := dataset.GetTokens(req, common.ModeRandom)
			Expect(err).ShouldNot(HaveOccurred())
			text := strings.Join(tokens, "")
			Expect(IsValidText(text)).To(BeTrue())
			Expect(finishReason).Should(Equal(common.StopFinishReason))
		})

		It("should return short text", func() {
			maxCompletionTokens := int64(2)
			req := &openaiserverapi.ChatCompletionRequest{
				MaxCompletionTokens: &maxCompletionTokens,
			}
			tokens, finishReason, err := dataset.GetTokens(req, common.ModeRandom)
			Expect(err).ShouldNot(HaveOccurred())
			tokensCnt := int64(len(tokens))
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
			req := &openaiserverapi.ChatCompletionRequest{
				MaxTokens: &maxCompletionTokens,
			}
			tokens, finishReason, err := dataset.GetTokens(req, common.ModeRandom)
			Expect(err).ShouldNot(HaveOccurred())
			tokensCnt := int64(len(tokens))
			Expect(tokensCnt).Should(BeNumerically("<=", maxCompletionTokens))
			text := strings.Join(tokens, "")
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
				req := &openaiserverapi.ChatCompletionRequest{
					MaxTokens: &n,
				}
				req.SetIgnoreEOS(true)
				tokens, finishReason, err := dataset.GetTokens(req, common.ModeRandom)
				Expect(err).ShouldNot(HaveOccurred())
				nGenTokens := int64(len(tokens))
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

	Context("getTokensInEchoMode", func() {
		theText := "Give a man a fish and you feed him for a day; teach a man to fish and you feed him for a lifetime"
		theTokens := common.Tokenize(theText)

		It("should return the same text, max tokens is not defined", func() {
			req := &openaiserverapi.TextCompletionRequest{
				Prompt: theText,
			}
			req.SetTokenizedPrompt(&openaiserverapi.Tokenized{Strings: theTokens})
			tokens, finishReason, err := dataset.getTokensInEchoMode(req)
			Expect(err).ShouldNot(HaveOccurred())
			Expect(tokens).Should(Equal(theTokens))
			Expect(finishReason).Should(Equal(common.StopFinishReason))
		})
		It("should return the same text, max tokens is higher than the text length", func() {
			maxTokens := int64(1000)
			req := &openaiserverapi.TextCompletionRequest{
				Prompt:    theText,
				MaxTokens: &maxTokens,
			}
			req.SetTokenizedPrompt(&openaiserverapi.Tokenized{Strings: theTokens})

			tokens, finishReason, err := dataset.getTokensInEchoMode(req)
			Expect(err).ShouldNot(HaveOccurred())
			Expect(tokens).Should(Equal(theTokens))
			Expect(finishReason).Should(Equal(common.StopFinishReason))
		})
		It("should return the same text, finish reason is stop", func() {
			maxTokens := int64(2)
			req := &openaiserverapi.TextCompletionRequest{
				Prompt:    theText,
				MaxTokens: &maxTokens,
			}
			req.SetTokenizedPrompt(&openaiserverapi.Tokenized{Strings: theTokens})

			tokens, finishReason, err := dataset.getTokensInEchoMode(req)
			Expect(err).ShouldNot(HaveOccurred())
			Expect(tokens).Should(Equal(theTokens))
			Expect(finishReason).Should(Equal(common.LengthFinishReason))
		})
	})

	Context("GetRandomTokens", func() {
		lenArr := []int{5, 20, 50, 150}

		for _, len := range lenArr {
			name := fmt.Sprintf("should return text with %d tokens", len)
			It(name, func() {
				tokens := dataset.generatePresetRandomTokens(len)
				Expect(tokens).Should(HaveLen(len))
			})
		}
	})

	Context("IsValidText", func() {
		validTxts := make([]string, 0)
		invalidTxts := make([]string, 0)

		validTxts = append(validTxts, completionFakeResponses[0][:4])
		validTxts = append(validTxts, completionFakeResponses[1])
		validTxts = append(validTxts, completionFakeResponses[1]+" "+completionFakeResponses[2])

		invalidTxts = append(invalidTxts, (completionFakeResponses[1] + " " + completionFakeResponses[2])[3:4])
		invalidTxts = append(invalidTxts, completionFakeResponses[0][4:])
		invalidTxts = append(invalidTxts, completionFakeResponses[1]+"-"+completionFakeResponses[2])
		invalidTxts = append(invalidTxts, completionFakeResponses[1]+" ")
		invalidTxts = append(invalidTxts, completionFakeResponses[1]+"   "+completionFakeResponses[2])

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

var _ = Describe("cumulativeBucketsProbabilities", Ordered, func() {
	type bucketBoundaries struct {
		start int
		end   int
	}

	dataset := createDataset()

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
			bucketsStr := ""
			for _, b := range expectedBuckets {
				bucketsStr += fmt.Sprintf("<%d, %d>, ", b.start, b.end)
			}
			return fmt.Sprintf("maxTokens: %d,expectedBuckets: %s", maxTokens, bucketsStr)
		},
		Entry(nil, 500, []bucketBoundaries{{1, 20}, {21, 40}, {41, 60}, {61, 480}, {481, 499}}),
		Entry(nil, 47, []bucketBoundaries{{1, 9}, {10, 18}, {19, 27}, {28, 36}, {37, 46}}),
		Entry(nil, 50, []bucketBoundaries{{1, 9}, {10, 19}, {20, 29}, {30, 39}, {40, 49}}),
	)
})
