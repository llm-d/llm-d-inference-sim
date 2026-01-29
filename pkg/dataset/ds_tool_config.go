/*
Copyright 2026 The llm-d-inference-sim Authors.

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
	"errors"
	"fmt"
	"os"
	"path"

	"github.com/spf13/pflag"
)

type DSToolConfiguration struct {
	hfRepo             string
	localPath          string
	file               string
	token              string
	maxRecords         int
	tokenizersCacheDir string
	model              string
	outputPath         string
	outputFile         string
}

func NewDefaultDSToolConfiguration() *DSToolConfiguration {
	return &DSToolConfiguration{
		hfRepo:             "",
		localPath:          "",
		file:               "",
		outputFile:         "inference-sim-dataset",
		outputPath:         "",
		token:              "",
		maxRecords:         10000,
		tokenizersCacheDir: "hf_token",
		model:              "",
	}
}

func (c *DSToolConfiguration) LoadConfig() error {
	f := pflag.NewFlagSet("ds_tool flags", pflag.ContinueOnError)

	f.StringVar(&c.hfRepo, "hf-repo", "", "HuggingFace dataset (e.g. 'anon8231489123/ShareGPT_Vicuna_unfiltered')")
	f.StringVar(&c.localPath, "local-path", "", "Local directory")
	f.StringVar(&c.file, "file", "", "File name, relevant both for HF and local")
	f.StringVar(&c.outputFile, "output-file", "inference-sim-dataset",
		"Output file name without extension, two files will be created: <output-file>.json and <output-file>.csv")
	f.StringVar(&c.outputPath, "output-path", "", "Output path")
	f.IntVar(&c.maxRecords, "max-records", 10000, "Max records to process")

	f.StringVar(&c.model, "model", "", "Model name")
	f.StringVar(&c.tokenizersCacheDir, "tokenizers-cache-dir", "hf_cache", "Directory for caching tokenizers, default is hf_cache")

	if err := f.Parse(os.Args[1:]); err != nil {
		if err == pflag.ErrHelp {
			// --help - exit without printing an error message
			os.Exit(0)
		}
		return err
	}

	c.token = os.Getenv("HF_TOKEN")

	return c.validate()
}

func (c *DSToolConfiguration) validate() error {
	if c.model == "" {
		return errors.New("--model is not defined")
	}
	if c.tokenizersCacheDir == "" {
		return errors.New("--tokenizers-cache-dir cannot be empty")
	}
	if c.hfRepo == "" && c.localPath == "" {
		return errors.New("either --hf-repo or --local-path must be specified")
	}
	if c.hfRepo != "" && c.localPath != "" {
		return errors.New("specify only one of --hf-repo or --local-path")
	}
	if c.hfRepo != "" && c.file == "" {
		return errors.New("--hf-repo defined but --file is empty")
	}
	if c.localPath != "" && c.file == "" {
		return errors.New("--local-path defined but --file is empty")
	}

	if err := c.validateFileNotExists(c.getOutputDBFullFileName()); err != nil {
		return err
	}
	if err := c.validateFileNotExists(c.getOutputJsonFullFileName()); err != nil {
		return err
	}

	return nil
}

// validateDbNotExists checks if an output database file already exists at the given path
// Returns an error if the file exists or if there's an issue checking the file
func (c *DSToolConfiguration) validateFileNotExists(path string) error {
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("output file already exists: %s", path)
	} else if !os.IsNotExist(err) {
		// Some other error occurred (permissions, etc.)
		return fmt.Errorf("error checking output file: %w", err)
	}
	return nil
}

func (c *DSToolConfiguration) getOutputDBFullFileName() string {
	return path.Join(c.outputPath, c.outputFile+".sqlite3")
}

func (c *DSToolConfiguration) getOutputJsonFullFileName() string {
	return path.Join(c.outputPath, c.outputFile+".json")
}
