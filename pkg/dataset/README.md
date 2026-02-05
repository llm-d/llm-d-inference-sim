# Dataset Convertion Tool

The `ds-tool` is used to convert conversation datasets into the format required by llm-d-inference-sim. It processes source datasets (from HuggingFace or local files) and generates both JSON and SQLite outputs with tokenized data. In addition, a dataset card is generated too


## Prerequisites

1. **HuggingFace Token (Optional):** If downloading from HuggingFace, set the `HF_TOKEN` environment variable:
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```

2. **Model and Tokenizer:** Ensure you have access to the model you want to use for tokenization.

## Command Line Options

| Option | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | string | Yes | - | Model name for tokenization |
| `--hf-repo` | string | No* | - | HuggingFace dataset repository (e.g., `anon8231489123/ShareGPT_Vicuna_unfiltered`) |
| `--local-path` | string | No* | - | Local directory containing the dataset file |
| `--file` | string | Yes | - | File name (relevant for both HF and local modes) |
| `--output-path` | string | No | - | Output directory path, by default current folder |
| `--output-file` | string | No | `inference-sim-dataset` | Output file name without extension (creates `.json`, `.sqlite3` and `.md` files) |
| `--table-name` | string | No | `llmd` | Name of the table created in the SQLite DB |
| `--max-records` | int | No | `10000` | Maximum number of records to read |
| `--tokenizers-cache-dir` | string | No | `hf_cache` | Directory for caching tokenizers |

**Note:** Either `--hf-repo` or `--local-path` must be specified, but not both.
**Note:** `--max-records` defines number of records to read, since original dataset contains conversations, number of records in the output file/db will be larger.

### Usage Examples

#### Example 1: Download from HuggingFace
```bash
export HF_TOKEN=your_token_here
./ds-tool \
  --hf-repo anon8231489123/ShareGPT_Vicuna_unfiltered \
  --file ShareGPT_V3_unfiltered_cleaned_split.json \
  --model meta-llama/Llama-3-8B \
  --output-path ./output \
  --output-file my-dataset \
  --max-records 5000
```

#### Example 2: Process Local Dataset
```bash
./ds-tool \
  --local-path ./data \
  --file conversations.json \
  --model meta-llama/Llama-3-8B \
  --output-file local-dataset \
  --tokenizers-cache-dir ./tokenizer_cache
```

#### Example 3: Minimal Configuration
```bash
./dataset-tool \
  --model meta-llama/Llama-3-8B
  --local-path ./data \
  --file dataset.json \
```

## Input Dataset Structure

The dataset tool expects input files in a specific JSON format containing conversation records. Each record represents a multi-turn conversation between a human and an assistant (GPT).

### Required Fields

The input JSON file must be an array of objects, where each object contains:

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | string | Yes | Unique identifier for the conversation |
| `conversations` | array | Yes | Array of conversation turns |

### Conversation Turn Structure

Each element in the `conversations` array must have:

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `from` | string | Yes | Role of the speaker: `"human"` or `"gpt"` |
| `value` | string | Yes | The text content of the message |

### Input Format Requirements

1. **Conversation Pairs:** Conversations must alternate between `"human"` and `"gpt"` roles
2. **Even Number of Turns:** Each conversation should have pairs of human questions and GPT responses
3. **Valid Roles:** Only `"human"` and `"gpt"` roles are recognized

### Input Example

```json
[
  {
    "id": "conversation_001",
    "conversations": [
      {
        "from": "human",
        "value": "What is the capital of France?"
      },
      {
        "from": "gpt",
        "value": "The capital of France is Paris."
      },
      {
        "from": "human",
        "value": "What is its population?"
      },
      {
        "from": "gpt",
        "value": "Paris has a population of approximately 2.2 million people in the city proper."
      }
    ]
  },
  {
    "id": "conversation_002",
    "conversations": [
      {
        "from": "human",
        "value": "Explain quantum computing"
      },
      {
        "from": "gpt",
        "value": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations."
      }
    ]
  }
]
```

## Dataset Structure

The dataset is provided in two formats:
- **JSON:** Ideal for debugging and referense.
- **SQLite:** Used by the simulator.

The tool generates two files:
1. **`<output-file>.json`**: Human-readable JSON format with all fields
2. **`<output-file>.sqlite3`**: SQLite database for efficient querying

In addition output contains the dataset card file which contains information about the created dataset.

### Database Schema (SQLite)

The SQLite version of this dataset is used by the simulator for efficient querying of token counts and prompt lookups. The table is structured as follows:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Primary Key. |
| `prompt_hash` | `BLOB NOT NULL` | Hash identifier for the input. |
| `gen_tokens` | `JSON NOT NULL` | JSON containing `strings` and `numbers` arrays. |
| `n_gen_tokens`| `INTEGER NOT NULL` | The count of generated tokens. |


### SQL Query Example
To find the average token length of generated responses:
```sql
SELECT AVG(n_gen_tokens) FROM llmd;
```

### JSON Data Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `prompt_hash` | `string` | A unique SHA-256 (or similar) hash of the input prompt. |
| `gen_tokens` | `struct/json` | Contains response text in two formats, as numerical tokens and as strings: `strings` (token text) and `numbers` (token IDs). |
| `n_gen_tokens`| `int` | The total number of tokens in the generated response. |
| `input_text` | `string` | The text fed to the model (raw or chat-templated). |
| `generated` | `string` | The plain response text. |

### Data Example (JSON)
```json
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
```

### Processing Behavior

For each conversation pair (human question + GPT response), the tool generates **two output records**:

1. **Completions Format:** Uses the raw human prompt as input
2. **Chat Completions Format:** Uses the full conversation history with chat templates (e.g., `### user:\n...\n### assistant:\n...`)

This dual-format approach ensures compatibility with both `/completions` and `/chat/completions` API endpoints in the inference simulator.
