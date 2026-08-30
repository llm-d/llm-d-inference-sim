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

package communication

import (
	"bufio"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/valyala/fasthttp"
)

const (
	defaultSpeechResponseFormat = "wav"
	speechStreamChunkDelay      = 100 * time.Millisecond

	speechInputTokensHeader      = "x-vllm-omni-input-tokens"
	speechOutputTokensHeader     = "x-vllm-omni-output-tokens"
	speechTotalTokensHeader      = "x-vllm-omni-total-tokens"
	speechInputTextTokensHeader  = "x-vllm-omni-input-text-tokens"
	speechInputAudioTokensHeader = "x-vllm-omni-input-audio-tokens"
)

var speechMediaTypes = map[string]string{
	"wav":  "audio/wav",
	"pcm":  "audio/pcm",
	"flac": "audio/flac",
	"mp3":  "audio/mpeg",
	"opus": "audio/ogg",
}

// HandleSpeech handles OpenAI-compatible POST /v1/audio/speech requests.
func (c *Communication) HandleSpeech(ctx *fasthttp.RequestCtx) {
	if c.stopping.Load() {
		c.sendSpeechError(ctx, "server is shutting down", fasthttp.StatusServiceUnavailable)
		return
	}

	var req api.CreateSpeechRequest
	if err := json.Unmarshal(ctx.Request.Body(), &req); err != nil {
		c.sendSpeechError(ctx, "Failed to read and parse request body, "+err.Error(), fasthttp.StatusBadRequest)
		return
	}
	if req.Input == nil {
		c.sendSpeechError(ctx, "input is required", fasthttp.StatusBadRequest)
		return
	}

	model := req.Model
	if model == "" {
		model = c.simulator.Context.Config().Model
	}
	if err := c.simulator.ValidateBaseModel(model); err != nil {
		c.sendError(ctx, err, false)
		return
	}

	responseFormat := defaultSpeechResponseFormat
	if req.ResponseFormat != nil {
		responseFormat = *req.ResponseFormat
	}
	mediaType, ok := speechMediaTypes[responseFormat]
	if !ok {
		c.sendSpeechError(ctx, fmt.Sprintf("unsupported response_format %q", responseFormat), fasthttp.StatusBadRequest)
		return
	}

	streamFormat := ""
	if req.StreamFormat != nil {
		streamFormat = *req.StreamFormat
		if streamFormat != "sse" && streamFormat != "audio" {
			c.sendSpeechError(ctx, fmt.Sprintf("unsupported stream_format %q", streamFormat), fasthttp.StatusBadRequest)
			return
		}
	}

	streaming := req.Stream || streamFormat != ""
	if streaming && responseFormat != "pcm" && responseFormat != "wav" {
		c.sendSpeechError(ctx,
			fmt.Sprintf("streaming speech requires response_format \"pcm\" or \"wav\", got %q", responseFormat),
			fasthttp.StatusBadRequest)
		return
	}
	if req.Speed != nil {
		if *req.Speed < 0.25 || *req.Speed > 4.0 {
			c.sendSpeechError(ctx, "speed must be between 0.25 and 4.0", fasthttp.StatusBadRequest)
			return
		}
		if streaming && *req.Speed != 1.0 {
			c.sendSpeechError(ctx, "streaming speech does not support speed adjustment", fasthttp.StatusBadRequest)
			return
		}
	}

	requestID := c.getRequestID(ctx)
	c.addResponseHeaders(ctx, requestID)
	ctx.SetStatusCode(fasthttp.StatusOK)

	payload := syntheticSpeechPayload(responseFormat)
	chunks := splitSpeechPayload(payload)
	usage := buildSpeechUsage(*req.Input, req.Instructions, len(chunks))

	if streamFormat == "audio" {
		ctx.SetContentType(mediaType)
		setRawSpeechStream(ctx, chunks)
		return
	}
	if req.Stream || streamFormat == "sse" {
		ctx.SetContentType("text/event-stream")
		ctx.Response.Header.Set("Cache-Control", "no-cache")
		setSSESpeechStream(ctx, chunks, responseFormat, usage)
		return
	}

	ctx.SetContentType(mediaType)
	setSpeechUsageHeaders(ctx, usage)
	ctx.SetBody(payload)
}

func (c *Communication) sendSpeechError(ctx *fasthttp.RequestCtx, message string, statusCode int) {
	err := api.NewError(message, statusCode, nil)
	c.sendError(ctx, &err, false)
}

func buildSpeechUsage(input string, instructions string, outputTokens int) api.SpeechTokenUsage {
	textTokens := len(strings.Fields(input)) + len(strings.Fields(instructions))
	return api.SpeechTokenUsage{
		InputTokens:  textTokens,
		OutputTokens: outputTokens,
		TotalTokens:  textTokens + outputTokens,
		InputTokenDetails: api.SpeechInputTokenDetails{
			TextTokens: textTokens,
		},
	}
}

func setSpeechUsageHeaders(ctx *fasthttp.RequestCtx, usage api.SpeechTokenUsage) {
	ctx.Response.Header.Set(speechInputTokensHeader, strconv.Itoa(usage.InputTokens))
	ctx.Response.Header.Set(speechOutputTokensHeader, strconv.Itoa(usage.OutputTokens))
	ctx.Response.Header.Set(speechTotalTokensHeader, strconv.Itoa(usage.TotalTokens))
	ctx.Response.Header.Set(speechInputTextTokensHeader, strconv.Itoa(usage.InputTokenDetails.TextTokens))
	ctx.Response.Header.Set(speechInputAudioTokensHeader, strconv.Itoa(usage.InputTokenDetails.AudioTokens))
}

func syntheticSpeechPayload(responseFormat string) []byte {
	pcm := make([]byte, 960)
	for i := 0; i < len(pcm)/2; i++ {
		sample := int16(1200)
		if (i/60)%2 != 0 {
			sample = -sample
		}
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(sample))
	}

	switch responseFormat {
	case "wav":
		return wavPayload(pcm)
	case "pcm":
		return pcm
	case "flac":
		return append([]byte("fLaC"), pcm...)
	case "mp3":
		return append([]byte("ID3\x04\x00\x00\x00\x00\x00\x00"), pcm...)
	case "opus":
		return append([]byte("OggS\x00\x02simulated-OpusHead"), pcm...)
	default:
		return nil
	}
}

func wavPayload(pcm []byte) []byte {
	wav := make([]byte, 44+len(pcm))
	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], uint32(36+len(pcm)))
	copy(wav[8:12], "WAVE")
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)
	binary.LittleEndian.PutUint16(wav[20:22], 1)
	binary.LittleEndian.PutUint16(wav[22:24], 1)
	binary.LittleEndian.PutUint32(wav[24:28], 24000)
	binary.LittleEndian.PutUint32(wav[28:32], 48000)
	binary.LittleEndian.PutUint16(wav[32:34], 2)
	binary.LittleEndian.PutUint16(wav[34:36], 16)
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], uint32(len(pcm)))
	copy(wav[44:], pcm)
	return wav
}

func splitSpeechPayload(payload []byte) [][]byte {
	midpoint := len(payload) / 2
	return [][]byte{payload[:midpoint], payload[midpoint:]}
}

func setRawSpeechStream(ctx *fasthttp.RequestCtx, chunks [][]byte) {
	pr, pw := io.Pipe()
	ctx.Response.SetBodyStream(pr, -1)

	go func() {
		writer := bufio.NewWriter(pw)
		defer func() {
			_ = writer.Flush()
			_ = pw.Close()
		}()

		for i, chunk := range chunks {
			if _, err := writer.Write(chunk); err != nil {
				_ = pw.CloseWithError(err)
				return
			}
			if err := writer.Flush(); err != nil {
				_ = pw.CloseWithError(err)
				return
			}
			if i < len(chunks)-1 {
				time.Sleep(speechStreamChunkDelay)
			}
		}
	}()
}

func setSSESpeechStream(
	ctx *fasthttp.RequestCtx,
	chunks [][]byte,
	responseFormat string,
	usage api.SpeechTokenUsage,
) {
	pr, pw := io.Pipe()
	ctx.Response.SetBodyStream(pr, -1)

	go func() {
		writer := bufio.NewWriter(pw)
		defer func() {
			_ = writer.Flush()
			_ = pw.Close()
		}()

		for i, chunk := range chunks {
			event := api.SpeechAudioDeltaEvent{
				Type:           "speech.audio.delta",
				Audio:          base64.StdEncoding.EncodeToString(chunk),
				ResponseFormat: responseFormat,
			}
			if err := writeSpeechSSEEvent(writer, event.Type, event); err != nil {
				_ = pw.CloseWithError(err)
				return
			}
			if i < len(chunks)-1 {
				time.Sleep(speechStreamChunkDelay)
			}
		}

		done := api.SpeechAudioDoneEvent{
			Type:  "speech.audio.done",
			Usage: usage,
		}
		if err := writeSpeechSSEEvent(writer, done.Type, done); err != nil {
			_ = pw.CloseWithError(err)
		}
	}()
}

func writeSpeechSSEEvent(writer *bufio.Writer, eventType string, event any) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", eventType, data); err != nil {
		return err
	}
	return writer.Flush()
}
