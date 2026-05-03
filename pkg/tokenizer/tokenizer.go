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
	"context"
	"fmt"
	"hash/fnv"
	"regexp"
	"strings"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/valyala/fasthttp"
)

type Tokenizer interface {
	// Converts input text to tokens
	RenderText(input string) ([]uint32, []string, error)
	// Converts input to tokens in two steps: templatization and tokenization
	RenderChatCompletion(messages []openaiserverapi.ChatComplMessage) ([]uint32, []string, *tokenization.MultiModalFeatures, error)
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

// Converts input to tokens
func (st *SimpleTokenizer) RenderText(input string) ([]uint32, []string, error) {
	tokens, textTokens := st.tokenize(input)
	return tokens, textTokens, nil
}

// Converts input to tokens in two steps: templatization and tokenization
func (st *SimpleTokenizer) RenderChatCompletion(messages []openaiserverapi.ChatComplMessage) ([]uint32, []string, *tokenization.MultiModalFeatures, error) {
	input := FlattenChatRequest(messages)
	tokens, textTokens := st.tokenize(input)
	return tokens, textTokens, nil, nil
}

func (st *SimpleTokenizer) tokenize(input string) ([]uint32, []string) {
	strTokens := st.re.FindAllString(input, -1)

	return stringsToUint32sHash(strTokens), strTokens
}

// Creates a string representing the given chat completions request
func FlattenChatRequest(messages []openaiserverapi.ChatComplMessage) string {
	var builder strings.Builder
	for _, msg := range messages {
		builder.WriteString(fmt.Sprintf("### %s:\n%s\n", msg.Role, msg.Content.Raw))
	}
	return builder.String()
}

func modelExists(model string) bool {
	url := "https://huggingface.co/api/models/" + model

	statusCode, _, err := fasthttp.Get(nil, url)
	if err != nil {
		return false
	}

	return statusCode == fasthttp.StatusOK
}

func New(ctx context.Context, config *common.Configuration, logger logr.Logger) (Tokenizer, error) {
	var err error
	var tokenizer Tokenizer

	if modelExists(config.Model) {
		tokenizer, err = NewHFTokenizer(ctx, logger, config.UDSSocketPath, config.Model)
	} else {
		logger.Info("Model is not a real HF model, using simulated tokenizer", "model", config.Model)
		tokenizer = NewSimpleTokenizer()
	}

	return tokenizer, err
}
