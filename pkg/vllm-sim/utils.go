/*
Copyright 2025 The vLLM-Sim Authors.

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

package vllmsim

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/valyala/fasthttp"
)

// list of responses to use in random mode for comepltion requests
var chatCompletionFakeResponses = []string{
	`Testing, testing 1,2,3.`,
	`I am fine, how are you today?`,
	`I am your AI assistant, how can I help you today?`,
	`Today is a nice sunny day.`,
	`The temperature here is twenty-five degrees centigrade.`,
	`Today it is partially cloudy and raining.`,
	`To be or not to be that is the question.`,
	`Alas, poor Yorick! I knew him, Horatio: A fellow of infinite jest`,
	`The rest is silence. `,
	`Give a man a fish and you feed him for a day; teach a man to fish and you feed him for a lifetime`,
}

// getRandomResponseText returns random response text from the pre-defined list of responses
// considering max completion tokens if it is not nil
func getRandomResponseText(max_completion_tokens *int64) string {
	index := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(chatCompletionFakeResponses))
	text := chatCompletionFakeResponses[index]

	return getResponseText(max_completion_tokens, text)
}

// getResponseText returns response text, from a given text
// considering max completion tokens if it is not nil
func getResponseText(max_completion_tokens *int64, text string) string {
	// should not happen
	if max_completion_tokens != nil && *max_completion_tokens <= 0 {
		return ""
	}

	// no max completion tokens, return entire text
	if max_completion_tokens == nil {
		return text
	}
	// create tokens from text, splitting by spaces
	tokens := strings.Fields(text)

	// return entire text
	if *max_completion_tokens >= int64(len(tokens)) {
		return text
	}
	// return truncated text
	return strings.Join(tokens[0:*max_completion_tokens], " ")
}

func formatBody(src []byte, contentType string) string {
	if strings.HasPrefix(contentType, "application/json") {
		var prettyJSON bytes.Buffer
		error := json.Indent(&prettyJSON, src, "", "\t")
		if error != nil {
			return fmt.Sprintf("JSON parse error: '%s' %s", error, string(src))
		}
		return prettyJSON.String()
	}
	if strings.HasPrefix(contentType, "text/event-stream") {
		texts := strings.Split(string(src), "\n\n")
		finalText := ""
		for _, text := range texts {
			text = strings.TrimSpace(text)
			if text == "" {
				continue
			}
			if strings.HasSuffix(text, "[DONE]") {
				finalText += "\n\t" + text
				continue
			}
			hasData := false
			if strings.HasPrefix(text, "data:") {
				hasData = true
				text = "{" + strings.Replace(text, "data:", "\"data\":", 1) + "}"
			}
			var prettyJSON bytes.Buffer
			error := json.Indent(&prettyJSON, []byte(text), "", "\t")
			if error != nil {
				return fmt.Sprintf("JSON parse error: '%s' %s", error, string(text))
			}
			text = prettyJSON.String()
			if hasData {
				text = strings.Replace(text, "\"data\":", "data:", 1)
				text = text[1 : len(text)-2]
			}
			finalText += text
		}
		return finalText
	}
	return string(src)
}

func loggingRequestHandler(next fasthttp.RequestHandler, logger logr.Logger) fasthttp.RequestHandler {
	return func(ctx *fasthttp.RequestCtx) {
		// Log request details
		contentType := string(ctx.Request.Header.ContentType())
		logger.Info("Request", "method", ctx.Method(), "uri", ctx.URI(), "type", contentType)
		logger.Info("Request", "body", formatBody(ctx.Request.Body(), contentType))

		// Call the next handler
		next(ctx)

		// Log response details
		contentType = string(ctx.Response.Header.ContentType())
		logger.Info("Response", "status", ctx.Response.StatusCode(), "type", contentType)
		logger.Info("Response", "body", formatBody(ctx.Response.Body(), contentType))
	}
}

// Given a partial string, access the full string
func getFullTextFromPartialString(partial string) string {
	for _, str := range chatCompletionFakeResponses {
		if strings.Contains(str, partial) {
			return str
		}
	}
	return ""
}
