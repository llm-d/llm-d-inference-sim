# Tokenization

The simulator offers flexible tokenization to balance **accuracy** vs. **performance**. The mode is automatically selected based on the model name provided in the `--model` argument: at startup the simulator queries `https://huggingface.co/api/models/<model>` and, if the model exists on HuggingFace, switches to **HuggingFace Mode**; otherwise it falls back to **Simulated Mode**.

## HuggingFace Mode (Real Models)
This mode is activated when the `--model` parameter specifies a model ID that is reachable on HuggingFace (e.g., `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`).

* **Behavior:** The simulator does **not** load a tokenizer in-process. Instead, every tokenization call is forwarded over HTTP to an external **vLLM render service** (vLLM running with `vllm launch render`). The simulator POSTs to `<render-url>/render` and consumes the returned token IDs and string tokens.
* **Accuracy:** Ensures exact token counts and boundaries — tokenization is performed by a real vLLM/HuggingFace tokenizer in the render service.
* **Requirements:**
    * **Network access** at startup to `https://huggingface.co/api/models/<model>` so the simulator can detect that the model is a real HuggingFace model.
    * **A running vLLM render service** reachable at `--render-url`. See the [README](../README.md#standalone-testing) for instructions on starting the render container, or use the `make run-render` helper. Without this service the simulator will start, but tokenization requests will fail.
* **Performance:** Each tokenization call is a network round-trip to the render service.

## Simulated Mode (Dummy Models)
This mode is activated when the `--model` parameter specifies a name that does not exist on HuggingFace (e.g., `my-fake-model`, `test-cluster-1`, `gpt-4-turbo-sim`).

* **Behavior:** Uses an in-process regex-based tokenizer to split text and generates token hashes using the FNV-32a algorithm. No external service or network access is needed.
* **Accuracy:** Approximate. Token boundaries will not match real models.
* **Pros:**
    * **Zero startup overhead:** No render service, no downloads.
    * **High throughput:** Ideal for infrastructure testing where exact token boundaries are irrelevant.
    * **No network dependency:** Works completely offline.

## Performance Considerations
**Important:** If you want to avoid the cost and operational overhead of running the render service:
- Use a "fake" or non-existent model name (e.g., `--model fake-model`)
- Or use the `--force-dummy-tokenizer` flag with any model name
- This is recommended for testing scenarios where exact tokenization accuracy is not required
- The render service is only required when you need accurate token counts matching actual HuggingFace models

## Configuration
| Parameter | Description | Default |
|---|---|---|
| `--model` | The model name. Determines the tokenization mode. (Mandatory) | |
| `--render-url` | URL of the vLLM render service. Used only in HuggingFace Mode. | `http://localhost:8082` |
| `--render-timeout` | Timeout for tokenizer render requests (Go duration, e.g. `30s`). | `30s` |
| `--mm-render-timeout` | Timeout for multi-modal tokenizer render requests. | `60s` |
| `--force-dummy-tokenizer` | Force the use of dummy tokenizer even if a real model name is provided | `false` |


## Examples
Running with HuggingFace tokenization (requires a vLLM render service):
```bash
./bin/llm-d-inference-sim \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --render-url http://localhost:8082
```

Running with simulated tokenization (no render service required):
```bash
./bin/llm-d-inference-sim --model test-sim-model
```

Running with simulated tokenization (forcing dummy tokenizer with a real model name):
```bash
./bin/llm-d-inference-sim --model meta-llama/Llama-3.1-8B-Instruct --force-dummy-tokenizer
```
