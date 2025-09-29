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
	"os"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"

	_ "github.com/mattn/go-sqlite3"
)

const (
	testPrompt = "Hello world!"
)

var _ = Describe("CustomDataset", Ordered, func() {
	var (
		dataset               *CustomDataset
		file_folder           string
		path                  string
		validDBPath           string
		pathToInvalidDB       string
		pathNotExist          string
		pathToInvalidTableDB  string
		pathToInvalidColumnDB string
		pathToInvalidTypeDB   string
	)

	BeforeAll(func() {
		common.InitRandom(time.Now().UnixNano())
	})

	BeforeEach(func() {
		dataset = &CustomDataset{}
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
	})

	AfterEach(func() {
		if dataset.db != nil {
			err := dataset.db.Close()
			Expect(err).NotTo(HaveOccurred())
		}
	})

	It("should return error for invalid DB path", func() {
		err := dataset.connectToDB("/invalid/path/to/db.sqlite", false)
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
		err = dataset.downloadDataset(context.Background(), url, path)
		Expect(err).NotTo(HaveOccurred())
		_, err = os.Stat(path)
		Expect(err).NotTo(HaveOccurred())
		err = os.Remove(path)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should not download file from url", func() {
		url := "https://256.256.256.256" // invalid url
		err := dataset.downloadDataset(context.Background(), url, path)
		Expect(err).To(HaveOccurred())
	})

	It("should successfully init dataset", func() {
		err := dataset.Init(context.Background(), klog.Background(), validDBPath, "", false)
		Expect(err).NotTo(HaveOccurred())

		row := dataset.db.QueryRow("SELECT n_gen_tokens FROM llmd WHERE prompt_hash=X'74bf14c09c038321cba39717dae1dc732823ae4abd8e155959367629a3c109a8';")
		var n_gen_tokens int
		err = row.Scan(&n_gen_tokens)
		Expect(err).NotTo(HaveOccurred())
		Expect(n_gen_tokens).To(Equal(4))

		var jsonStr string
		row = dataset.db.QueryRow("SELECT gen_tokens FROM llmd WHERE prompt_hash=X'74bf14c09c038321cba39717dae1dc732823ae4abd8e155959367629a3c109a8';")
		err = row.Scan(&jsonStr)
		Expect(err).NotTo(HaveOccurred())
		var tokens []string
		err = json.Unmarshal([]byte(jsonStr), &tokens)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).To(Equal([]string{"Hello", " llm-d ", "world", "!"}))

	})

	It("should return error for non-existing DB path", func() {
		err := dataset.connectToDB(pathNotExist, false)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("database file does not exist"))
	})

	It("should return error for invalid DB file", func() {
		err := dataset.connectToDB(pathToInvalidDB, false)
		Expect(err).To(HaveOccurred())
	})

	It("should return error for DB with invalid table", func() {
		err := dataset.connectToDB(pathToInvalidTableDB, false)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failed to verify database"))
	})

	It("should return error for DB with invalid column", func() {
		err := dataset.connectToDB(pathToInvalidColumnDB, false)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("missing expected column"))
	})

	It("should return error for DB with invalid column type", func() {
		err := dataset.connectToDB(pathToInvalidTypeDB, false)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("incorrect type"))
	})

	It("should return correct prompt hash in bytes", func() {
		// b't\xbf\x14\xc0\x9c\x03\x83!\xcb\xa3\x97\x17\xda\xe1\xdcs(#\xaeJ\xbd\x8e\x15YY6v)\xa3\xc1\t\xa8'
		expectedHashBytes := []byte{0x74, 0xbf, 0x14, 0xc0, 0x9c, 0x03, 0x83, 0x21, 0xcb, 0xa3, 0x97, 0x17, 0xda, 0xe1, 0xdc, 0x73, 0x28, 0x23, 0xae, 0x4a, 0xbd, 0x8e, 0x15, 0x59, 0x59, 0x36, 0x76, 0x29, 0xa3, 0xc1, 0x09, 0xa8}

		req := &openaiserverapi.TextCompletionRequest{
			Prompt: testPrompt,
		}

		hashBytes := dataset.GetPromptHash(req)
		Expect(hashBytes).To(Equal(expectedHashBytes))
	})

	It("should return correct prompt hash in hex", func() {
		expectedHashHex := "74bf14c09c038321cba39717dae1dc732823ae4abd8e155959367629a3c109a8"

		req := &openaiserverapi.TextCompletionRequest{
			Prompt: testPrompt,
		}

		hashBytes := dataset.GetPromptHash(req)
		hashHex := dataset.GetPromptHashHex(hashBytes)
		Expect(hashHex).To(Equal(expectedHashHex))
	})

	It("should return tokens for existing prompt", func() {
		err := dataset.Init(context.Background(), klog.Background(), validDBPath, "", false)
		Expect(err).NotTo(HaveOccurred())

		req := &openaiserverapi.TextCompletionRequest{
			Prompt: testPrompt,
		}
		tokens, finishReason, err := dataset.GetTokens(req, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())
		Expect(finishReason).To(Equal(StopFinishReason))
		Expect(tokens).To(Equal([]string{"Hello", " llm-d ", "world", "!"}))
	})

	It("should return at most 2 tokens for existing prompt", func() {
		err := dataset.Init(context.Background(), klog.Background(), validDBPath, "", false)
		Expect(err).NotTo(HaveOccurred())
		n := int64(2)
		req := &openaiserverapi.TextCompletionRequest{
			Prompt:    testPrompt,
			MaxTokens: &n,
		}
		tokens, _, err := dataset.GetTokens(req, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(tokens)).To(BeNumerically("<=", 2))
	})

	It("should successfully init dataset with in-memory option", func() {
		err := dataset.Init(context.Background(), klog.Background(), validDBPath, "", true)
		Expect(err).NotTo(HaveOccurred())

		req := &openaiserverapi.TextCompletionRequest{
			Prompt: testPrompt,
		}
		tokens, finishReason, err := dataset.GetTokens(req, common.ModeRandom)
		Expect(err).NotTo(HaveOccurred())
		Expect(finishReason).To(Equal(StopFinishReason))
		Expect(tokens).To(Equal([]string{"Hello", " llm-d ", "world", "!"}))
	})
})
