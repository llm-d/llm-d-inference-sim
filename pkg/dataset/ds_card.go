package dataset

import (
	"os"
	"strconv"
	"strings"
)

const (
	modelNamePlaceholder          = "<MODEL_NAME>"
	hfDSRepoPlaceholder           = "<HF_DS_REPO>"
	hfDSUrlPlaceholder            = "<HF_DS_URL>"
	hfFileNamePlaceholder         = "<HF_FILE_NAME>"
	sourceRecordsCountPlaceholder = "<SOURCE_RECORDS_COUNT>"
	genRecordsCountPlaceholder    = "<GEN_RECORDS_COUNT>"
)

const cardTemplate = `
# Dataset Card

## Overview

This dataset is derived from conversational data and has been processed into a tokenized format suitable for LLM inference simulation. The dataset contains pre-tokenized prompts and responses, enabling efficient testing of inference systems without requiring live model execution.

## Tokenization Model
` + modelNamePlaceholder + `

## Source Dataset

The original dataset consists of multi-turn conversations between humans and AI assistants. <br>
Dataset: [` + hfDSRepoPlaceholder + `](` + hfDSUrlPlaceholder + `), file ` + hfFileNamePlaceholder + `

### Dataset Formats

This dataset is available in two formats:

- **JSON:** Human-readable format ideal for debugging and reference.
- **SQLite:** Optimized for efficient querying, used by the simulator.

### Data Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| ` + "`" + `prompt_hash` + "`" + ` | string | SHA-256 hash uniquely identifying the input prompt |
| ` + "`" + `input_text` + "`" + ` | string | The prompt text (raw or chat-templated) |
| ` + "`" + `generated` + "`" + ` | string | The response text from the assistant |
| ` + "`" + `n_gen_tokens` + "`" + ` | integer | Total count of tokens in the generated response |
| ` + "`" + `gen_tokens` + "`" + ` | object | Tokenized response containing ` + "`" + `strings` + "`" + ` (token text) and ` + "`" + `numbers` + "`" + ` (token IDs) |

### Data Example

` + "`'''" + `json
{
  "prompt_hash": "OZ5Edy+9rw0CsSMabW2TwSxR78jJGYRVRWtz8SXRm6U=",
  "n_gen_tokens": 4,
  "gen_tokens": {
    "strings": ["g", "pt", " a", "1"],
    "numbers": [70, 417, 264, 16]
  },
  "input_text": "human q1",
  "generated": "gpt a1"
}
` + "```" + `

## SQLite Database Schema

The SQLite version provides efficient querying capabilities and used by the simulator, it has the following schema:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| ` + "`" + `id` + "`" + ` | INTEGER PRIMARY KEY AUTOINCREMENT | Auto-incrementing primary key |
| ` + "`" + `prompt_hash` + "`" + ` | BLOB NOT NULL | Binary hash identifier for the input prompt |
| ` + "`" + `gen_tokens` + "`" + ` | JSON NOT NULL | JSON object containing tokenized response data |
| ` + "`" + `n_gen_tokens` + "`" + ` | INTEGER NOT NULL | Count of generated tokens |

### Example Query

Calculate the average response length:
` + "```" + `sql
SELECT AVG(n_gen_tokens) FROM llmd;
` + "```" + `

## Dataset Characteristics

- **Tokenization**: All responses are pre-tokenized using a specified language model tokenizer
- **Dual Format**: Each conversation generates both completion and chat-completion variants
- **Hash-Based Indexing**: Prompts are indexed by SHA-256 hash for efficient lookup
- **Token Details**: Both string representations and numerical token IDs are preserved
- **Scalable**: SQLite format supports efficient querying of large datasets


## Dataset Statistics

- **Source Dataset Record Count**: ` + sourceRecordsCountPlaceholder + `
- **Dataset Record Count**: ` + genRecordsCountPlaceholder + `
`

func generateCardFile(modelName, hfDSRepo, hfFileName, cardFilePath string, sourceDSRecsCount, genRecordsCount int) error {
	hfDSUrl := "https://huggingface.co/datasets/" + hfDSRepo

	replacer := strings.NewReplacer(
		modelNamePlaceholder, modelName,
		hfDSRepoPlaceholder, hfDSRepo,
		hfDSUrlPlaceholder, hfDSUrl,
		hfFileNamePlaceholder, hfFileName,
		sourceRecordsCountPlaceholder, strconv.Itoa(sourceDSRecsCount),
		genRecordsCountPlaceholder, strconv.Itoa(genRecordsCount),
	)

	result := replacer.Replace(cardTemplate)
	err := os.WriteFile(cardFilePath, []byte(result), 0644)

	return err
}
