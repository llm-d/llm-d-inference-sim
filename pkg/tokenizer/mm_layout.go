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

package tokenizer

import (
	"fmt"
	"slices"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
)

// mmLayout is the simulated tokenizer's fixed shape for one modality's items.
// Qwen-VL merges each 2x2 block of a temporal-height-width patch grid into one
// token, so an image or video spans t*h*w/4 placeholder tokens.
type mmLayout struct {
	padToken      string
	hashPrefix    string
	metadataKey   string
	metadata      []int64
	patchesPerPad int64
}

var mmLayouts = map[string]mmLayout{
	mmModalityImage: {padToken: "<|image_pad|>", hashPrefix: "img", metadataKey: "image_grid_thw", metadata: []int64{1, 8, 8}, patchesPerPad: 4},
	mmModalityVideo: {padToken: "<|video_pad|>", hashPrefix: "video", metadataKey: "video_grid_thw", metadata: []int64{2, 8, 8}, patchesPerPad: 4},
	mmModalityAudio: {padToken: "<|audio_pad|>", hashPrefix: "audio", metadataKey: "audio_num_tokens", metadata: []int64{16}, patchesPerPad: 1},
}

var blockModalities = map[string]string{
	"image_url":   mmModalityImage,
	"input_audio": mmModalityAudio,
	"audio_url":   mmModalityAudio,
	"video_url":   mmModalityVideo,
}

var contentPartModalities = map[string]string{
	"image_url": mmModalityImage,
	"audio_url": mmModalityAudio,
	"video_url": mmModalityVideo,
}

func KnownContentPartType(t string) bool {
	_, ok := contentPartModalities[t]
	return ok
}

func (l mmLayout) padTokenID() uint32 {
	return fnv32(l.padToken)
}

func (l mmLayout) spanLength() int {
	patches := int64(1)
	for _, v := range l.metadata {
		patches *= v
	}
	return int(patches / l.patchesPerPad)
}

func (l mmLayout) padRun() []uint32 {
	run := make([]uint32, l.spanLength())
	for i := range run {
		run[i] = l.padTokenID()
	}
	return run
}

// Deterministic so repeats hit cache.
func simMMHash(modality string, modalIndex int, identifier string) string {
	return fmt.Sprintf("sim_%s_%d_%x", mmLayouts[modality].hashPrefix, modalIndex, fnv32(identifier))
}

// MMItemMetadata returns the placeholder metadata vLLM publishes in an item's
// ec_transfer_params descriptor, for the simulated layout of modality.
func MMItemMetadata(modality string) map[string][]int64 {
	layout, ok := mmLayouts[modality]
	if !ok {
		return nil
	}
	return map[string][]int64{layout.metadataKey: slices.Clone(layout.metadata)}
}

// ContentPartsMMHashes returns the mm_hashes of a generate request's
// content_parts, keyed by modality. A part is identified by its uuid, or
// otherwise by the hash render gives its URL. Parts of unknown type are skipped.
func ContentPartsMMHashes(parts []api.GenerateContentPart) map[string][]string {
	hashes := map[string][]string{}
	for _, part := range parts {
		if modality, ok := contentPartModalities[part.Type]; ok {
			hashes[modality] = append(hashes[modality], contentPartID(modality, len(hashes[modality]), part))
		}
	}
	if len(hashes) == 0 {
		return nil
	}
	return hashes
}

func contentPartID(modality string, modalIndex int, part api.GenerateContentPart) string {
	if part.UUID != "" {
		return part.UUID
	}
	return simMMHash(modality, modalIndex, part.URL)
}

// ExpandContentParts restores the placeholder runs a coordinator collapsed to
// one pad token each, matching content_parts to pad tokens in order. A part
// without a matching pad token is an error, matching vLLM's placeholder-count
// validation.
func ExpandContentParts(tokenIDs []uint32, parts []api.GenerateContentPart) ([]uint32, *api.RenderMMFeatures, error) {
	expanded := make([]uint32, 0, len(tokenIDs))
	hashes := map[string][]string{}
	placeholders := map[string][]api.RenderPlaceholder{}
	modalIndexes := map[string]int{}
	pos := 0
	for _, part := range parts {
		modality, ok := contentPartModalities[part.Type]
		if !ok {
			continue
		}
		modalIndex := modalIndexes[modality]
		modalIndexes[modality]++
		layout := mmLayouts[modality]
		idx := slices.Index(tokenIDs[pos:], layout.padTokenID())
		if idx < 0 {
			return nil, nil, fmt.Errorf("expected a %s placeholder for %s content part %d, but none found in token_ids", layout.padToken, modality, modalIndex)
		}
		expanded = append(expanded, tokenIDs[pos:pos+idx]...)
		hashes[modality] = append(hashes[modality], contentPartID(modality, modalIndex, part))
		placeholders[modality] = append(placeholders[modality],
			api.RenderPlaceholder{Offset: len(expanded), Length: layout.spanLength()})
		expanded = append(expanded, layout.padRun()...)
		pos += idx + 1
	}
	if len(hashes) == 0 {
		return tokenIDs, nil, nil
	}
	expanded = append(expanded, tokenIDs[pos:]...)
	return expanded, &api.RenderMMFeatures{MMHashes: hashes, MMPlaceholders: placeholders}, nil
}
