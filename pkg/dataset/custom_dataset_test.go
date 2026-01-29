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
	"encoding/json"
	"fmt"
	"os"
	"time"

	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"

	. "github.com/onsi/gomega"

	_ "github.com/mattn/go-sqlite3"
)

const (
	testPrompt = "Hello world!"
)

var _ = Describe("CustomDataset", Ordered, func() {
	var (
		sqliteHelper          *sqliteHelper
		dsDownloader          *CustomDatasetDownloader
		file_folder           string
		path                  string
		validDBPath           string
		pathToInvalidDB       string
		pathNotExist          string
		pathToInvalidTableDB  string
		pathToInvalidColumnDB string
		pathToInvalidTypeDB   string
		random                *common.Random
		tknzr                 tokenizer.Tokenizer
	)

	BeforeAll(func() {
		random = common.NewRandom(time.Now().UnixNano(), 8080)
		file_folder = ".llm-d"
		path = file_folder + "/test.sqlite3"
		err := os.MkdirAll(file_folder, os.ModePerm)
		Expect(err).NotTo(HaveOccurred())
		validDBPath = file_folder + "/test.valid.sqlite3"
		pathNotExist = file_folder + "/test.notexist.sqlite3"
		pathToInvalidDB = file_folder + "/test.invalid.sqlite3"
		pathToInvalidTableDB = file_folder + "/test.invalid.table.sqlite3"
		pathToInvalidColumnDB = file_folder + "/test.invalid.column.sqlite3"
		pathToInvalidTypeDB = file_folder + "/test.invalid.type.sqlite3"
		tknzr, err = tokenizer.New("", false, "")
		Expect(err).ShouldNot(HaveOccurred())
	})

	BeforeEach(func() {
		sqliteHelper = newSqliteHelper(klog.Background())
		dsDownloader = NewDsDownloader(klog.Background())
	})

	It("should return error for invalid DB path", func() {
		err := sqliteHelper.connectToDB("/invalid/path/to/db.sqlite", false, true)
		Expect(err).To(HaveOccurred())
	})

	It("should download file from url", func() {
		// remove file if it exists
		_, err := os.Stat(path)
		if err == nil {
			err = os.Remove(path)
			Expect(err).NotTo(HaveOccurred())
		}
		url := "https://llm-d.ai"
		err = dsDownloader.DownloadDataset(context.Background(), url, path)
		Expect(err).NotTo(HaveOccurred())
		_, err = os.Stat(path)
		Expect(err).NotTo(HaveOccurred())
		err = os.Remove(path)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should not download file from url", func() {
		url := "https://256.256.256.256" // invalid url
		err := dsDownloader.DownloadDataset(context.Background(), url, path)
		Expect(err).To(HaveOccurred())
	})

	It("should successfully init dataset", func() {
		dataset := &CustomDataset{}
		err := dataset.Init(context.Background(), klog.Background(), random, validDBPath, false, 1024, tknzr)
		Expect(err).NotTo(HaveOccurred())

		row := dataset.sqliteHelper.db.QueryRow("SELECT n_gen_tokens FROM llmd WHERE prompt_hash=X'74bf14c09c038321cba39717dae1dc732823ae4abd8e155959367629a3c109a8';")
		var n_gen_tokens int
		err = row.Scan(&n_gen_tokens)
		Expect(err).NotTo(HaveOccurred())
		Expect(n_gen_tokens).To(Equal(4))

		var jsonStr string
		row = dataset.sqliteHelper.db.QueryRow("SELECT gen_tokens FROM llmd WHERE prompt_hash=X'74bf14c09c038321cba39717dae1dc732823ae4abd8e155959367629a3c109a8';")
		err = row.Scan(&jsonStr)
		Expect(err).NotTo(HaveOccurred())
		var tokens []string
		err = json.Unmarshal([]byte(jsonStr), &tokens)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).To(Equal([]string{"Hello", " llm-d ", "world", "!"}))

		err = dataset.sqliteHelper.db.Close()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should return error for non-existing DB path", func() {
		err := sqliteHelper.connectToDB(pathNotExist, false, true)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("database file does not exist"))
	})

	It("should return error for invalid DB file", func() {
		err := sqliteHelper.connectToDB(pathToInvalidDB, false, true)
		Expect(err).To(HaveOccurred())
	})

	It("should return error for DB with invalid table", func() {
		err := sqliteHelper.connectToDB(pathToInvalidTableDB, false, true)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failed to verify database"))
	})

	It("should return error for DB with invalid column", func() {
		err := sqliteHelper.connectToDB(pathToInvalidColumnDB, false, true)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("missing expected column"))
	})

	It("should return error for DB with invalid column type", func() {
		err := sqliteHelper.connectToDB(pathToInvalidTypeDB, false, true)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("incorrect type"))
	})

	It("should return correct prompt hash in bytes", func() {
		// b't\xbf\x14\xc0\x9c\x03\x83!\xcb\xa3\x97\x17\xda\xe1\xdcs(#\xaeJ\xbd\x8e\x15YY6v)\xa3\xc1\t\xa8'
		expectedHashBytes := []byte{0x74, 0xbf, 0x14, 0xc0, 0x9c, 0x03, 0x83, 0x21, 0xcb, 0xa3, 0x97, 0x17, 0xda, 0xe1, 0xdc, 0x73, 0x28, 0x23, 0xae, 0x4a, 0xbd, 0x8e, 0x15, 0x59, 0x59, 0x36, 0x76, 0x29, 0xa3, 0xc1, 0x09, 0xa8}

		req := &openaiserverapi.TextCompletionRequest{
			Prompt: testPrompt,
		}
		dataset := &CustomDataset{}
		hashBytes := dataset.getPromptHash(req)
		Expect(hashBytes).To(Equal(expectedHashBytes))
	})

	It("should return correct prompt hash in hex", func() {
		expectedHashHex := "74bf14c09c038321cba39717dae1dc732823ae4abd8e155959367629a3c109a8"

		req := &openaiserverapi.TextCompletionRequest{
			Prompt: testPrompt,
		}
		dataset := &CustomDataset{}
		hashBytes := dataset.getPromptHash(req)
		hashHex := dataset.getPromptHashHex(hashBytes)
		Expect(hashHex).To(Equal(expectedHashHex))
	})

	Context("custom dataset", func() {
		dataset := &CustomDataset{}
		longPrompt := "1, 2, 3, 4, 5, 6"
		maxTokens := int64(20)
		smallMaxTokens := int64(2)
		exactMaxToken := int64(4)
		var promptTokens []string

		BeforeAll(func() {
			var err error
			_, promptTokens, err = tknzr.Encode(testPrompt, "")
			Expect(err).NotTo(HaveOccurred())
			err = dataset.Init(context.Background(), klog.Background(), random, validDBPath, false, 1024, tknzr)
			Expect(err).NotTo(HaveOccurred())

		})

		AfterAll(func() {
			err := dataset.sqliteHelper.db.Close()
			Expect(err).NotTo(HaveOccurred())
		})

		DescribeTable("should work correctly in random mode with ignore eos",
			func(prompt string, maxTokens *int64, isChat bool, expectedFinishReason string) {
				var req openaiserverapi.Request
				if isChat {
					chatReq := openaiserverapi.ChatCompletionRequest{MaxTokens: maxTokens}
					chatReq.Messages = []openaiserverapi.Message{{Role: openaiserverapi.RoleUser, Content: openaiserverapi.Content{Raw: prompt}}}
					chatReq.IgnoreEOS = true
					req = &chatReq
				} else {
					textReq := openaiserverapi.TextCompletionRequest{Prompt: prompt, MaxTokens: maxTokens}
					textReq.IgnoreEOS = true
					req = &textReq
				}

				tokens, finishReason, err := dataset.GetTokens(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(finishReason).To(Equal(expectedFinishReason))
				if maxTokens != nil {
					Expect(tokens.Strings).To(HaveLen(int(*maxTokens)))
				}
			},
			func(prompt string, maxTokens *int64, isChat bool, expectedFinishReason string) string {
				return fmt.Sprintf("prompt: '%s', maxTokens: %s, isChat: %t, expectedFinishReason: %s", prompt, maxTokensToStr(maxTokens), isChat, expectedFinishReason)
			},
			Entry(nil, longPrompt, &maxTokens, false, common.LengthFinishReason),
			Entry(nil, longPrompt, &maxTokens, true, common.LengthFinishReason),
			Entry(nil, longPrompt, &smallMaxTokens, false, common.LengthFinishReason),
			Entry(nil, longPrompt, &smallMaxTokens, true, common.LengthFinishReason),
			Entry(nil, testPrompt, &maxTokens, false, common.LengthFinishReason),
			Entry(nil, testPrompt, &maxTokens, true, common.LengthFinishReason),
			Entry(nil, testPrompt, &smallMaxTokens, false, common.LengthFinishReason),
			Entry(nil, testPrompt, &smallMaxTokens, true, common.LengthFinishReason),
			Entry(nil, testPrompt, &exactMaxToken, false, common.LengthFinishReason),
			Entry(nil, testPrompt, &exactMaxToken, true, common.LengthFinishReason),
			Entry(nil, longPrompt, &exactMaxToken, true, common.LengthFinishReason),
		)

		It("should return tokens for existing prompt", func() {
			req := &openaiserverapi.TextCompletionRequest{
				Prompt: testPrompt,
			}
			req.SetTokenizedPrompt(&openaiserverapi.Tokenized{Strings: promptTokens})

			tokens, finishReason, err := dataset.GetTokens(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(finishReason).To(Equal(common.StopFinishReason))
			Expect(tokens.Strings).To(Equal([]string{"Hello", " llm-d ", "world", "!"}))
		})

		It("should return at most 2 tokens for existing prompt", func() {
			req := &openaiserverapi.TextCompletionRequest{
				Prompt:    testPrompt,
				MaxTokens: &smallMaxTokens,
			}
			tokens, _, err := dataset.GetTokens(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens.Length()).To(BeNumerically("<=", smallMaxTokens))
		})

		It("should successfully init dataset with in-memory option", func() {
			req := &openaiserverapi.TextCompletionRequest{
				Prompt: testPrompt,
			}
			req.SetTokenizedPrompt(&openaiserverapi.Tokenized{Strings: promptTokens})

			tokens, finishReason, err := dataset.GetTokens(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(finishReason).To(Equal(common.StopFinishReason))
			Expect(tokens.Strings).To(Equal([]string{"Hello", " llm-d ", "world", "!"}))
		})

		It("should work correctly for chat request with multiple messages", func() {
			req := openaiserverapi.ChatCompletionRequest{MaxTokens: &maxTokens}
			req.Messages = []openaiserverapi.Message{
				{Role: openaiserverapi.RoleUser, Content: openaiserverapi.Content{Raw: testPrompt}},
				{Role: openaiserverapi.RoleAssistant, Content: openaiserverapi.Content{Raw: "this is assistant response"}},
				{Role: openaiserverapi.RoleUser, Content: openaiserverapi.Content{Raw: testPrompt}},
			}

			req.SetTokenizedPrompt(&openaiserverapi.Tokenized{Strings: promptTokens})

			tokens, finishReason, err := dataset.GetTokens(&req)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens.Length()).To(BeNumerically("<=", maxTokens))
			Expect((tokens.Length() == int(maxTokens) && finishReason == common.LengthFinishReason) ||
				(tokens.Length() < int(maxTokens) && finishReason == common.StopFinishReason)).To(BeTrue())
		})
	})
})

var _ = Describe("custom dataset for multiple simulators", Ordered, func() {
	It("should not fail on custom datasets initialization", func() {
		file_folder := ".llm-d"
		validDBPath := file_folder + "/test.valid.sqlite3"

		tokenizer, err := tokenizer.New("", false, "")
		Expect(err).ShouldNot(HaveOccurred())

		random1 := common.NewRandom(time.Now().UnixNano(), 8081)
		dataset1 := &CustomDataset{}
		err = dataset1.Init(context.Background(), klog.Background(), random1, validDBPath, false, 1024, tokenizer)
		Expect(err).NotTo(HaveOccurred())

		random2 := common.NewRandom(time.Now().UnixNano(), 8082)
		dataset2 := &CustomDataset{}
		err = dataset2.Init(context.Background(), klog.Background(), random2, validDBPath, false, 1024, tokenizer)
		Expect(err).NotTo(HaveOccurred())
	})
})

var _ = Describe("download custom dataset from HF", Ordered, func() {
	// currently there is only one dataset which is too large
	// once we will create a small sample dataset - restore this test
	XIt("should download and save ds", func() {
		url := "https://huggingface.co/datasets/hf07397/inference-sim-datasets/resolve/91ffa7aafdfd6b3b1af228a517edc1e8f22cd274/huggingface/ShareGPT_Vicuna_unfiltered/conversations.sqlite3"
		downloader := NewDsDownloader(klog.Background())
		tempFile := "./ds1.sqlite3"

		if _, err := os.Stat(tempFile); err == nil {
			err := os.Remove(tempFile)
			Expect(err).NotTo(HaveOccurred())
		}
		err := downloader.DownloadDataset(context.Background(), url, tempFile)
		Expect(err).NotTo(HaveOccurred())

		err = os.Remove(tempFile)
		Expect(err).NotTo(HaveOccurred())
	})
})
