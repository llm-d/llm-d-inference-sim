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
	Init(config common.Configuration) error
	// Tokenize(text string) []uint64
	// Encode tokenizes the input, modelName is optional, if not set, the model from the configuration will be used
	Encode(input, modelName string) ([]uint32, error)
}

type HFTokenizer struct {
	tokenizer tokenization.Tokenizer
	model     string
}

type SimpleTokenizer struct {
	re *regexp.Regexp
}

// Simple Tokenizer
func CreateSimpleTokenizer() *SimpleTokenizer {
	return &SimpleTokenizer{}
}

func (st *SimpleTokenizer) Init(_ common.Configuration) error {
	st.re = regexp.MustCompile(`(\{|\}|:|,|-|\.|\?|\!|;|@|#|\$|%|\^|&|\*|\(|\)|\+|\-|_|~|/|\\|>|<|\[|\]|=|"|\w+)(\s*)`)
	return nil
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

func (st *SimpleTokenizer) Encode(input, modelName string) ([]uint32, error) {
	strTokens := st.re.FindAllString(input, -1)
	return stringsToUint32sHash(strTokens), nil
}

// HF Tokenizer
func CreateHFTokenizer() *HFTokenizer {
	return &HFTokenizer{}
}

func (hft *HFTokenizer) Init(config common.Configuration) error {
	tokenizationConfig, err := tokenization.DefaultConfig()
	if err != nil {
		return errors.Join(err, errors.New("failed to create default tokenization configuration"))
	}

	if tokenizationConfig.HFTokenizerConfig == nil {
		tokenizationConfig.HFTokenizerConfig = &tokenization.HFTokenizerConfig{}
	}

	if config.TokenizersCacheDir != "" {
		tokenizationConfig.HFTokenizerConfig.TokenizersCacheDir = config.TokenizersCacheDir
	}

	hfToken := os.Getenv(hfTokenEnvVar)
	if hfToken != "" {
		tokenizationConfig.HFTokenizerConfig.HuggingFaceToken = hfToken
	}

	hft.tokenizer, err = tokenization.NewCachedHFTokenizer(tokenizationConfig.HFTokenizerConfig)
	if err != nil {
		return errors.Join(err, errors.New("failed to create hf tokenizer"))
	}
	hft.model = config.Model

	return nil
}

func (hft *HFTokenizer) Encode(input, modelName string) ([]uint32, error) {
	model := modelName
	if model == "" {
		model = hft.model
	}
	tokens, _, err := hft.tokenizer.Encode(input, model)
	return tokens, err
}
