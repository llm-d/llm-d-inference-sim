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
	"encoding/json"
	"os"

	"github.com/go-logr/logr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	_ "github.com/mattn/go-sqlite3"
)

var _ = Describe("CustomDataset", func() {
	var (
		dataset               *CustomDataset
		file_folder           string
		savePath              string
		validDBPath           string
		pathToInvalidDB       string
		pathNotExist          string
		pathToInvalidTableDB  string
		pathToInvalidColumnDB string
		pathToInvalidTypeDB   string
	)

	BeforeEach(func() {
		dataset = &CustomDataset{
			BaseDataset: BaseDataset{
				Logger: logr.Discard(),
			},
		}
		file_folder = ".llm-d"
		savePath = file_folder + "/test.sqlite3"
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
		err := dataset.connectToDB("/invalid/path/to/db.sqlite")
		Expect(err).To(HaveOccurred())
	})

	It("should download file from url", func() {
		url := "https://llm-d.ai"
		err := dataset.downloadDataset(url, savePath)
		Expect(err).NotTo(HaveOccurred())
		_, err = os.Stat(savePath)
		Expect(err).NotTo(HaveOccurred())
		err = os.Remove(savePath)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should not download file from url", func() {
		url := "https://256.256.256.256" // invalid url
		err := dataset.downloadDataset(url, savePath)
		Expect(err).To(HaveOccurred())
	})

	It("should successfully init dataset", func() {
		err := dataset.Init(validDBPath, "", "")
		Expect(err).NotTo(HaveOccurred())

		row := dataset.db.QueryRow("SELECT n_gen_tokens FROM llmd WHERE prompt_hash=X'b94d27b9934d041c52e5b721d7373f13a07ed5e79179d63c5d8a0c102a9d00b2';")
		var n_gen_tokens int
		err = row.Scan(&n_gen_tokens)
		Expect(err).NotTo(HaveOccurred())
		Expect(n_gen_tokens).To(Equal(3))

		var jsonStr string
		row = dataset.db.QueryRow("SELECT gen_tokens FROM llmd WHERE prompt_hash=X'b94d27b9934d041c52e5b721d7373f13a07ed5e79179d63c5d8a0c102a9d00b2';")
		err = row.Scan(&jsonStr)
		Expect(err).NotTo(HaveOccurred())
		var tokens []string
		err = json.Unmarshal([]byte(jsonStr), &tokens)
		Expect(err).NotTo(HaveOccurred())
		Expect(tokens).To(Equal([]string{"Hello", "world", "!"}))

	})

	It("should return error for non-existing DB path", func() {
		err := dataset.connectToDB(pathNotExist)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("database file does not exist"))
	})

	It("should return error for invalid DB file", func() {
		err := dataset.connectToDB(pathToInvalidDB)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("file is not a database"))
	})

	It("should return error for DB with invalid table", func() {
		err := dataset.connectToDB(pathToInvalidTableDB)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failed to verify database"))
	})

	It("should return error for DB with invalid column", func() {
		err := dataset.connectToDB(pathToInvalidColumnDB)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("missing expected column"))
	})

	It("should return error for DB with invalid column type", func() {
		err := dataset.connectToDB(pathToInvalidTypeDB)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("incorrect type"))
	})
})
