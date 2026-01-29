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
	"errors"
	"hash/fnv"
	"os"
	"regexp"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization"
)

const hfTokenEnvVar = "HF_TOKEN"

type Tokenizer interface {
	// Encode tokenizes the input, modelName is optional, if not provided, the model from the configuration will be used
	Encode(input, modelName string) ([]uint32, []string, error)
}

type HFTokenizer struct {
	tokenizer tokenization.Tokenizer
	model     string
}

type SimpleTokenizer struct {
	re *regexp.Regexp
}

// Simple Tokenizer
func NewSimpleTokenizer() *SimpleTokenizer {
	re := regexp.MustCompile(`(\{|\}|:|,|-|\.|\?|\!|;|@|#|\$|%|\^|&|\*|\(|\)|\+|\-|_|~|/|\\|>|<|\[|\]|=|"|\w+)(\s*)`)
	return &SimpleTokenizer{re: re}
}

func stringsToUint32sHash(strings []string) []uint32 {
	hashes := make([]uint32, len(strings))
	for i, s := range strings {
		h := fnv.New32a()
		h.Write([]byte(s))
		hashes[i] = h.Sum32()
	}
	return hashes
}

func (st *SimpleTokenizer) Encode(input, modelName string) ([]uint32, []string, error) {
	strTokens := st.re.FindAllString(input, -1)

	return stringsToUint32sHash(strTokens), strTokens, nil
}

// HF Tokenizer
func NewHFTokenizer(config common.Configuration) (*HFTokenizer, error) {
	hfConfig := tokenization.DefaultHFTokenizerConfig()
	if config.TokenizersCacheDir != "" {
		hfConfig.TokenizersCacheDir = config.TokenizersCacheDir
	}

	hfToken := os.Getenv(hfTokenEnvVar)
	if hfToken != "" {
		hfConfig.HuggingFaceToken = hfToken
	}

	hftTokenizer, err := tokenization.NewCachedHFTokenizer(hfConfig)
	if err != nil {
		return nil, errors.Join(err, errors.New("failed to create hf tokenizer"))
	}

	return &HFTokenizer{tokenizer: hftTokenizer, model: config.Model}, nil
}

func (hft *HFTokenizer) Encode(input, modelName string) ([]uint32, []string, error) {
	model := modelName
	if model == "" {
		model = hft.model
	}
	tokens, offsets, err := hft.tokenizer.Encode(input, model)
	textTokens := make([]string, len(tokens))
	for i, offset := range offsets {
		textTokens[i] = input[offset[0]:offset[1]]
	}

	return tokens, textTokens, err
}
