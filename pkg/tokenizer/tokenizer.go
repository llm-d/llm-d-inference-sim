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
	"hash/fnv"
	"regexp"
	"strings"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/valyala/fasthttp"
)

type Tokenizer interface {
	// RenderText renders plain text and returns token IDs and string tokens
	RenderText(text string) ([]uint32, []string, error)
	// RenderMessages renders chat messages and returns token IDs, string tokens, and multimodal features
	RenderMessages(messages []openaiserverapi.Message) ([]uint32, []string, *openaiserverapi.RenderMMFeatures, error)
}

type baseTokenizer struct {
	re *regexp.Regexp
}

type SimpleTokenizer struct {
	baseTokenizer
}

func New(ctx context.Context, config *common.Configuration, logger logr.Logger) (Tokenizer, error) {
	var err error
	var tokenizer Tokenizer

	if modelExists(config.Model) {
		tokenizer, err = NewHFTokenizer(ctx, logger, config.RenderURL, config.Model, config.RenderTimeout, config.MMRenderTimeout)
	} else {
		logger.Info("Model is not a real HF model, using simulated tokenizer", "model", config.Model)
		tokenizer = NewSimpleTokenizer()
	}

	return tokenizer, err
}

func newBaseTokenizer() baseTokenizer {
	re := regexp.MustCompile(`(\{|\}|:|,|-|\.|\?|\!|;|@|#|\$|%|\^|&|\*|\(|\)|\+|\-|_|~|/|\\|>|<|\[|\]|=|"|\w+)(\s*)`)
	return baseTokenizer{re: re}
}

func (bt *baseTokenizer) splitIntoTokens(input string, count int) []string {
	// separate the given string into sub-strings simulating tokens
	tokens := bt.re.FindAllString(input, -1)

	// if tokens length is ok - return the textual tokens
	if count == -1 || count == len(tokens) {
		return tokens
	}
	// there are not enough tokens to return, pad with empty strings
	if count > len(tokens) {
		return append(tokens, make([]string, count-len(tokens))...)
	}

	// there are too many tokens, merge tail into the last kept token, and return the required number of tokens
	tokens[count-1] = strings.Join(tokens[count-1:], "")
	return tokens[:count]
}

// Simple Tokenizer
func NewSimpleTokenizer() *SimpleTokenizer {
	return &SimpleTokenizer{baseTokenizer: newBaseTokenizer()}
}

func (st *baseTokenizer) tokenize(input string) ([]uint32, []string) {
	strTokens := st.splitIntoTokens(input, -1)

	return stringsToUint32sHash(strTokens), strTokens
}

func (st *SimpleTokenizer) RenderText(text string) ([]uint32, []string, error) {
	tokens, textTokens := st.tokenize(text)
	return tokens, textTokens, nil
}

func (st *SimpleTokenizer) RenderMessages(messages []openaiserverapi.Message) ([]uint32, []string, *openaiserverapi.RenderMMFeatures, error) {
	var builder strings.Builder
	for _, msg := range messages {
		builder.WriteString(openaiserverapi.StartMessageSeparator)
		text := msg.PlainText(true)
		builder.WriteString(text)
		builder.WriteString(openaiserverapi.EndMessageSeparator)
	}
	tokens, textTokens := st.tokenize(builder.String())
	return tokens, textTokens, nil, nil
}

func modelExists(model string) bool {
	url := "https://huggingface.co/api/models/" + model

	statusCode, _, err := fasthttp.Get(nil, url)
	if err != nil {
		return false
	}

	return statusCode == fasthttp.StatusOK
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

func FlattenMessages(messages []openaiserverapi.Message) string {
	var builder strings.Builder
	for _, msg := range messages {
		builder.WriteString(msg.PlainText(true))
	}
	return builder.String()
}
