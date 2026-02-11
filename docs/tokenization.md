# Tokenization

The simulator offers flexible tokenization to balance **accuracy** vs. **performance**. The mode is automatically selected based on the model name provided in the `--model` argument.

## HuggingFace Mode (Real Models)
This mode is activated when the `--model` parameter specifies a valid HuggingFace model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`).

* **Behavior:** The simulator downloads the actual tokenizer from HuggingFace and uses it for incoming requests.
* **Accuracy:** Ensures exact token counts and boundaries.
* **Requirements:**
    * **Network Access:** Required on the first run to download the tokenizer.
    * **Authentication:** Requires the `HF_TOKEN` environment variable if the model is gated or private.
* **Storage:** Tokenizers are stored in the directory specified by `--tokenizers-cache-dir` (default: `hf_cache`).
* **Performance:** Adds startup overhead (downloading/loading) and runtime overhead (tokenization logic).

## Simulated Mode (Dummy Models)
This mode is activated when the `--model` parameter specifies a name that does not exist on HuggingFace (e.g., `my-fake-model`, `test-cluster-1`, `gpt-4-turbo-sim`).

* **Behavior:** Uses a simple regex-based tokenizer to split text and generates token hashes using the FNV-32a algorithm.
* **Accuracy:** Approximate. Token boundaries will not match real models.
* **Pros:**
    * **Zero Startup Overhead:** No downloads or heavy model loading.
    * **High Throughput:** Ideal for infrastructure testing where exact token boundaries are irrelevant.
    * **No Network/Auth:** Works completely offline without API tokens.

## Performance Considerations
**Important:** If you want to avoid the time and network overhead of HuggingFace tokenization:
- Use a "fake" or non-existent model name (e.g., `--model fake-model`)
- This is recommended for testing scenarios where exact tokenization accuracy is not required
- HuggingFace tokenization is only necessary when you need accurate token counts matching actual HuggingFace models

## Configuration
| Parameter | Description | Default |
|---|---|---|
| --model | The model name. Determines the tokenization mode.(Mandatory)| |
| --tokenizers-cache-dir | Directory for caching HuggingFace tokenizers | hf_cache |
| HF_TOKEN | Environment variable for authenticating with HuggingFace | |


## Examples
Running with HuggingFace Tokenization:
```bash
export HF_TOKEN=hf_... ./bin/llm-d-inference-sim --model meta-llama/Llama-3.1-8B-Instruct --tokenizers-cache-dir /tmp/hf_cache
```

Running with Simulated Tokenization:
```bash
./bin/llm-d-inference-sim --model test-sim-model
```
