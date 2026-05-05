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
	"os"
	"path"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/spf13/pflag"
)

type DSToolConfiguration struct {
	hfRepo          string
	localPath       string
	inputFile       string
	token           string
	maxRecords      int
	renderURL       string
	renderTimeout   time.Duration
	mmRenderTimeout time.Duration
	model           string
	outputPath      string
	outputFile      string
	tableName       string
}

func NewDefaultDSToolConfiguration() *DSToolConfiguration {
	return &DSToolConfiguration{
		hfRepo:          "",
		localPath:       "",
		inputFile:       "",
		outputFile:      "inference-sim-dataset",
		outputPath:      "",
		token:           "",
		maxRecords:      10000,
		renderURL:       "",
		renderTimeout:   5 * time.Second,
		mmRenderTimeout: 30 * time.Second,
		model:           "",
		tableName:       "llmd",
	}
}

func (c *DSToolConfiguration) LoadConfig() error {
	f := pflag.NewFlagSet("ds_tool flags", pflag.ContinueOnError)

	f.StringVar(&c.hfRepo, "hf-repo", "", "HuggingFace dataset (e.g. 'anon8231489123/ShareGPT_Vicuna_unfiltered')")
	f.StringVar(&c.localPath, "local-path", "", "Local directory")
	f.StringVar(&c.inputFile, "input-file", "", "File name, relevant both for HF and local")
	f.StringVar(&c.outputFile, "output-file", "inference-sim-dataset",
		"Output file name without extension, two files will be created: <output-file>.json and <output-file>.csv")
	f.StringVar(&c.outputPath, "output-path", "", "Output path")
	f.StringVar(&c.outputPath, "table-name", common.DefaultDSTableName, "Table name, default is 'llmd'")
	f.IntVar(&c.maxRecords, "max-records", 10000, "Maximum number of source dataset records to process; if the dataset contains more, the rest are discarded")
	f.StringVar(&c.renderURL, "render-url", c.renderURL, "URL of the tokenizer render service")
	f.DurationVar(&c.renderTimeout, "render-timeout", c.renderTimeout, "Timeout for tokenizer render requests")
	f.DurationVar(&c.mmRenderTimeout, "mm-render-timeout", c.mmRenderTimeout, "Timeout for multi-modal tokenizer render requests")

	f.StringVar(&c.model, "model", "", "Model name")

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
	if c.hfRepo == "" && c.localPath == "" {
		return errors.New("either --hf-repo or --local-path must be specified")
	}
	if c.hfRepo != "" && c.localPath != "" {
		return errors.New("specify only one of --hf-repo or --local-path")
	}
	if c.hfRepo != "" && c.inputFile == "" {
		return errors.New("--hf-repo defined but --file is empty")
	}
	if c.localPath != "" && c.inputFile == "" {
		return errors.New("--local-path defined but --file is empty")
	}

	if c.renderURL == "" {
		return errors.New("--render-url is not defined")
	}

	if err := validateFileNotExist(c.getOutputDBFullFileName()); err != nil {
		return err
	}
	if err := validateFileNotExist(c.getOutputJsonFullFileName()); err != nil {
		return err
	}
	if err := validateFileNotExist(c.getOutputCardFullFileName()); err != nil {
		return err
	}

	return nil
}

func (c *DSToolConfiguration) getOutputDBFullFileName() string {
	return c.getOutputFileName(".sqlite3")
}

func (c *DSToolConfiguration) getOutputJsonFullFileName() string {
	return c.getOutputFileName(".json")
}

func (c *DSToolConfiguration) getOutputCardFullFileName() string {
	return c.getOutputFileName(".md")
}

func (c *DSToolConfiguration) getOutputFileName(extension string) string {
	return path.Join(c.outputPath, c.outputFile+extension)
}
