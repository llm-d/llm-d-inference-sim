# Tokenization

The simulator offers flexible tokenization to balance **accuracy** vs. **performance**. The mode is selected by the `--render-url` flag: if a URL is provided the simulator uses **HuggingFace Mode**, otherwise it uses **Simulated Mode**.

## HuggingFace Mode (Real Models)
This mode is activated when `--render-url` points at a running vLLM render service. Typical usage is a real model such as `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct` served by that render service.

* **Behavior:** The simulator does **not** load a tokenizer in-process. Instead, every tokenization call is forwarded over HTTP to an external **vLLM render service** (vLLM running with `vllm launch render`). The simulator POSTs to `<render-url>/render` and consumes the returned token IDs and string tokens.
* **Accuracy:** Ensures exact token counts and boundaries — tokenization is performed by a real vLLM/HuggingFace tokenizer in the render service.
* **Requirements:**
    * **A running vLLM render service** reachable at `--render-url`. See the [README](../README.md#standalone-testing) for instructions on starting the render container, or use the `make run-render` helper.
* **Performance:** Each tokenization call is a network round-trip to the render service.

## Simulated Mode (Dummy Models)
This mode is activated when `--render-url` is not set.

* **Behavior:** Uses an in-process regex-based tokenizer to split text and generates token hashes using the FNV-32a algorithm. No external service or network access is needed.
* **Accuracy:** Approximate. Token boundaries will not match real models.
* **Pros:**
    * **Zero startup overhead:** No render service, no downloads.
    * **High throughput:** Ideal for infrastructure testing where exact token boundaries are irrelevant.
    * **No network dependency:** Works completely offline.

When `--render-url` is not set but the model name looks like a HuggingFace repo id (contains `/`), the simulator logs a WARN and still falls back to the simulated tokenizer. This surfaces the fact that token ids will be pseudo-hashes and will not match a real vLLM tokenizer — a mismatch that breaks KV-cache block hashing and prefix-cache-aware routing.

## Performance Considerations
**Important:** If you want to avoid the cost and operational overhead of running the render service:
- Omit `--render-url`
- This is recommended for testing scenarios where exact tokenization accuracy is not required
- The render service is only required when you need accurate token counts matching actual HuggingFace models

## Configuration
| Parameter | Description | Default |
|---|---|---|
| `--model` | The model name. (Mandatory) | |
| `--render-url` | URL of the vLLM render service. When set, HuggingFace Mode is used; when unset, Simulated Mode is used. | (empty) |
| `--render-timeout` | Timeout for tokenizer render requests (Go duration, e.g. `30s`). | `30s` |
| `--mm-render-timeout` | Timeout for multi-modal tokenizer render requests. | `60s` |
| `--force-dummy-tokenizer` | (deprecated) Force the simulated tokenizer even when `--render-url` is set. Omit `--render-url` instead. | `false` |


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

