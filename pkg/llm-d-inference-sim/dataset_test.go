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

package llmdinferencesim

import (
	"os"

	"github.com/go-logr/logr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	_ "github.com/mattn/go-sqlite3"
)

var _ = Describe("Dataset", func() {
	var (
		dataset     *Dataset
		file_folder string
		savePath    string
	)

	BeforeEach(func() {
		dataset = &Dataset{
			logger: logr.Discard(),
		}
		file_folder = "./.llm-d"
		savePath = file_folder + "/test.sqlite3"
		err := os.MkdirAll(file_folder, os.ModePerm)
		Expect(err).NotTo(HaveOccurred())
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
		validDBPath := file_folder + "/test.valid.sqlite3"
		err := dataset.Init(validDBPath, "", "")
		Expect(err).NotTo(HaveOccurred())

		// read from the db to verify it's valid
		row := dataset.db.QueryRow("SELECT * FROM t;")
		var value string
		err = row.Scan(&value)
		Expect(err).NotTo(HaveOccurred())
		Expect(value).To(Equal("llm-d"))
	})

	It("should raise err with invalid DB content", func() {
		err := dataset.connectToDB(file_folder)
		Expect(err).NotTo(HaveOccurred())

		// read from the db to verify it's not valid
		row := dataset.db.QueryRow("SELECT * FROM t;")
		var value string
		err = row.Scan(&value)
		Expect(err).To(HaveOccurred())
	})
})
