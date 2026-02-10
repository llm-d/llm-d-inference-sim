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

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	"k8s.io/klog/v2"

	. "github.com/onsi/gomega"

	_ "github.com/mattn/go-sqlite3"
)

const (
	tokenizerTmpDir = "./test_tokenizers"
)

type validDBElement struct {
	input          string
	tokenizedInput openaiserverapi.Tokenized
	chatMessages   []string
	hexa           string
	strRespTokens  []string
	intRespTokens  []uint32
}

var _ = Describe("CustomDataset", Ordered, func() {
	var (
		sqliteHelper          *sqliteHelper
		dsDownloader          *CustomDatasetDownloader
		file_folder           string
		path                  string
		validDBPath           string
		tableName             string
		pathToInvalidDB       string
		pathNotExist          string
		pathToInvalidTableDB  string
		pathToInvalidColumnDB string
		pathToInvalidTypeDB   string
		random                *common.Random
		tknzr                 tokenizer.Tokenizer
		validDB               []validDBElement
	)

	BeforeAll(func() {
		random = common.NewRandom(time.Now().UnixNano(), 8080)
		file_folder = ".llm-d"
		path = file_folder + "/test.sqlite3"
		err := os.MkdirAll(file_folder, os.ModePerm)
		Expect(err).NotTo(HaveOccurred())
		validDBPath = file_folder + "/test.valid.sqlite3"
		tableName = "llmd"
		pathNotExist = file_folder + "/test.notexist.sqlite3"
		pathToInvalidDB = file_folder + "/test.invalid.sqlite3"
		pathToInvalidTableDB = file_folder + "/test.invalid.table.sqlite3"
		pathToInvalidColumnDB = file_folder + "/test.invalid.column.sqlite3"
		pathToInvalidTypeDB = file_folder + "/test.invalid.type.sqlite3"
		tknzr, err = tokenizer.New("Qwen/Qwen3-0.6B", true, tokenizerTmpDir)
		Expect(err).ShouldNot(HaveOccurred())

		validDB = make([]validDBElement, 3)

		// #1 in db: intput1, completions, short response
		validDB[0].input = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
		validDB[0].hexa = "73205d2e432e6b117e0b75cdddeac019ee863f4b524f75bf57c15c5a47a445e4"
		validDB[0].strRespTokens = []string{"Hello", " human", "!"}
		validDB[0].intRespTokens = []uint32{9707, 3738, 0}

		// #3 in db: intput2, message1, completions, long response
		validDB[1].input = "Hello world!"
		validDB[1].hexa = "90db35b48bf168f20fa36537861e1d64fac6af372267aec9d10437a3f83f8bec"
		validDB[1].strRespTokens = []string{"this", " is", " assistant", " long", " response", ",", " it", " should", " contain", " at",
			" least", " ", "1", "0", " tokens"}
		validDB[1].intRespTokens = []uint32{574, 374, 17847, 1293, 2033, 11, 432, 1265, 6644, 518, 3245, 220, 16, 15, 11211}

		// #6 in db: intput2, message2, chat completions, short response
		validDB[2].input = "### user:\nHello world!\n### assistant:\nthis is assistant long response, it should contain at least 10 tokens\n### user:\nHello world again\n"
		validDB[2].hexa = "067b89152dee047c66e53926f47d65366509729ad2c5a8e1d1e2dbb05f2eab41"
		validDB[2].strRespTokens = []string{"short", " response"}
		validDB[2].intRespTokens = []uint32{8676, 2033}
		validDB[2].chatMessages = []string{"Hello world!",
			"this is assistant long response, it should contain at least 10 tokens",
			"Hello world again"}

		for i := range validDB {
			tokens, strTokens, err := tknzr.Encode(validDB[i].input, "")
			Expect(err).ToNot(HaveOccurred())
			Expect(tokens).ToNot(BeEmpty())
			Expect(tokens).ToNot(BeNil())
			Expect(strTokens).ToNot(BeEmpty())
			Expect(strTokens).ToNot(BeNil())
			validDB[i].tokenizedInput = openaiserverapi.Tokenized{Tokens: tokens, Strings: strTokens}
		}

	})

	BeforeEach(func() {
		sqliteHelper = newSqliteHelper("llmd", klog.Background())
		dsDownloader = NewDsDownloader(klog.Background())
	})

	AfterAll(func() {
		// remove temp test db
		err := os.Remove(path)
		Expect(err).NotTo(HaveOccurred())
		// remova test tokenizer directory
		err = os.RemoveAll(tokenizerTmpDir)
		Expect(err).NotTo(HaveOccurred())

	})

	It("should return error for invalid DB path", func() {
		err := sqliteHelper.connectToDB("/invalid/path/to/db.sqlite", false)
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
		err := dataset.Init(context.Background(), klog.Background(), random, validDBPath, tableName, false, 1024, tknzr)
		Expect(err).NotTo(HaveOccurred())

		row := dataset.sqliteHelper.db.QueryRow(fmt.Sprintf("SELECT n_gen_tokens FROM llmd WHERE prompt_hash=X'%s';", validDB[0].hexa))
		var n_gen_tokens int
		err = row.Scan(&n_gen_tokens)
		Expect(err).NotTo(HaveOccurred())
		Expect(n_gen_tokens).To(Equal(len(validDB[0].intRespTokens)))

		var jsonStr string
		row = dataset.sqliteHelper.db.QueryRow(fmt.Sprintf("SELECT gen_tokens FROM llmd WHERE prompt_hash=X'%s';", validDB[0].hexa))
		err = row.Scan(&jsonStr)
		Expect(err).NotTo(HaveOccurred())
		var tokenized openaiserverapi.Tokenized
		err = json.Unmarshal([]byte(jsonStr), &tokenized)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokenized.Strings).To(Equal(validDB[0].strRespTokens))

		err = dataset.sqliteHelper.db.Close()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should return error for non-existing DB path", func() {
		err := sqliteHelper.connectToDB(pathNotExist, false)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("database file does not exist"))
	})

	It("should return error for invalid DB file", func() {
		err := sqliteHelper.connectToDB(pathToInvalidDB, false)
		Expect(err).To(HaveOccurred())
	})

	It("should return error for DB with invalid table", func() {
		err := sqliteHelper.connectToDB(pathToInvalidTableDB, false)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failed to verify database"))
	})

	It("should return error for DB with invalid column", func() {
		err := sqliteHelper.connectToDB(pathToInvalidColumnDB, false)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("missing expected column"))
	})

	It("should return error for DB with missing column", func() {
		err := sqliteHelper.connectToDB(pathToInvalidTypeDB, false)
		Expect(err).To(HaveOccurred())
		fmt.Printf("<><> %#v\n", err)
		Expect(err.Error()).To(ContainSubstring("has incorrect type"))
	})

	It("should return correct prompt hash in bytes", func() {
		req := &openaiserverapi.TextCompletionRequest{}
		req.SetTokenizedPrompt(&validDB[0].tokenizedInput)
		dataset := &CustomDataset{}
		hashBytes := dataset.getPromptHash(req)
		hashHex := dataset.getPromptHashHex(hashBytes)
		Expect(hashHex).To(Equal(validDB[0].hexa))
	})

	It("should return correct prompt hash in hex", func() {
		tokens, strTokens, err := tknzr.Encode(validDB[0].input, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).To(Equal(validDB[0].tokenizedInput.Tokens))
		Expect(strTokens).To(Equal(validDB[0].tokenizedInput.Strings))

		prompt := openaiserverapi.Tokenized{Tokens: tokens, Strings: strTokens}
		Expect(prompt).To(Equal(validDB[0].tokenizedInput))

		req := &openaiserverapi.TextCompletionRequest{}
		req.SetTokenizedPrompt(&prompt)
		dataset := &CustomDataset{}
		hashBytes := dataset.getPromptHash(req)
		hashHex := dataset.getPromptHashHex(hashBytes)
		Expect(hashHex).To(Equal(validDB[0].hexa))
	})

	Context("custom dataset", func() {
		dataset := &CustomDataset{}
		maxTokens := int64(20)
		smallMaxTokens := int64(2)
		exactMaxToken := int64(4)

		BeforeAll(func() {
			err := dataset.Init(context.Background(), klog.Background(), random, validDBPath, tableName, false, 1024, tknzr)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterAll(func() {
			err := dataset.sqliteHelper.db.Close()
			Expect(err).NotTo(HaveOccurred())
		})

		DescribeTable("should work correctly in random mode with ignore eos",
			func(index int, maxTokens *int64, isChat bool, expectedFinishReason string) {
				var req openaiserverapi.Request
				if isChat {
					chatReq := openaiserverapi.ChatCompletionRequest{MaxTokens: maxTokens}
					chatReq.IgnoreEOS = true
					req = &chatReq
				} else {
					textReq := openaiserverapi.TextCompletionRequest{MaxTokens: maxTokens}
					textReq.IgnoreEOS = true
					req = &textReq
				}
				req.SetTokenizedPrompt(&validDB[index].tokenizedInput)
				tokens, finishReason, err := dataset.GetResponseTokens(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(finishReason).To(Equal(expectedFinishReason))
				if maxTokens != nil {
					Expect(tokens.Strings).To(HaveLen(int(*maxTokens)))
				}
			},
			func(index int, maxTokens *int64, isChat bool, expectedFinishReason string) string {
				return fmt.Sprintf("validDB index: '%d', maxTokens: %s, isChat: %t, expectedFinishReason: %s", index, maxTokensToStr(maxTokens), isChat, expectedFinishReason)
			},
			Entry(nil, 1, &maxTokens, false, common.LengthFinishReason),
			Entry(nil, 1, &maxTokens, true, common.LengthFinishReason),
			Entry(nil, 1, &smallMaxTokens, false, common.LengthFinishReason),
			Entry(nil, 1, &smallMaxTokens, true, common.LengthFinishReason),
			Entry(nil, 1, &exactMaxToken, true, common.LengthFinishReason),
			Entry(nil, 0, &maxTokens, false, common.LengthFinishReason),
			Entry(nil, 0, &maxTokens, true, common.LengthFinishReason),
			Entry(nil, 0, &smallMaxTokens, false, common.LengthFinishReason),
			Entry(nil, 0, &smallMaxTokens, true, common.LengthFinishReason),
			Entry(nil, 0, &exactMaxToken, false, common.LengthFinishReason),
			Entry(nil, 0, &exactMaxToken, true, common.LengthFinishReason),
		)

		It("should return tokens for existing prompt", func() {
			req := &openaiserverapi.TextCompletionRequest{
				Prompt: validDB[1].input,
			}
			req.SetTokenizedPrompt(&validDB[1].tokenizedInput)

			tokens, finishReason, err := dataset.GetResponseTokens(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(finishReason).To(Equal(common.StopFinishReason))
			Expect(tokens.Strings).To(Equal(validDB[1].strRespTokens))
		})

		It("should return at most 2 tokens for existing prompt", func() {
			req := &openaiserverapi.TextCompletionRequest{
				MaxTokens: &smallMaxTokens,
			}
			req.SetTokenizedPrompt(&validDB[1].tokenizedInput)
			tokens, _, err := dataset.GetResponseTokens(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens.Length()).To(BeNumerically("<=", smallMaxTokens))
		})

		It("should successfully init dataset with in-memory option", func() {
			req := &openaiserverapi.TextCompletionRequest{
				Prompt: validDB[1].input,
			}
			req.SetTokenizedPrompt(&validDB[1].tokenizedInput)

			tokens, finishReason, err := dataset.GetResponseTokens(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(finishReason).To(Equal(common.StopFinishReason))
			Expect(tokens.Strings).To(Equal(validDB[1].strRespTokens))
		})

		It("should work correctly for chat request with multiple messages", func() {
			req := openaiserverapi.ChatCompletionRequest{MaxTokens: &maxTokens}
			req.Messages = []openaiserverapi.Message{
				{Role: openaiserverapi.RoleUser, Content: openaiserverapi.Content{Raw: validDB[2].chatMessages[0]}},
				{Role: openaiserverapi.RoleAssistant, Content: openaiserverapi.Content{Raw: validDB[2].chatMessages[1]}},
				{Role: openaiserverapi.RoleUser, Content: openaiserverapi.Content{Raw: validDB[2].chatMessages[2]}},
			}

			req.SetTokenizedPrompt(&validDB[2].tokenizedInput)

			tokens, finishReason, err := dataset.GetResponseTokens(&req)
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
		tableName := "llmd"

		tokenizer, err := tokenizer.New("", false, "")
		Expect(err).ShouldNot(HaveOccurred())

		random1 := common.NewRandom(time.Now().UnixNano(), 8081)
		dataset1 := &CustomDataset{}
		err = dataset1.Init(context.Background(), klog.Background(), random1, validDBPath, tableName, false, 1024, tokenizer)
		Expect(err).NotTo(HaveOccurred())

		random2 := common.NewRandom(time.Now().UnixNano(), 8082)
		dataset2 := &CustomDataset{}
		err = dataset2.Init(context.Background(), klog.Background(), random2, validDBPath, tableName, false, 1024, tokenizer)
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
