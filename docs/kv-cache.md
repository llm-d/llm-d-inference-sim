# KV Cache

The simulator includes a prefix-cache subsystem that mimics vLLM's KV cache behavior without requiring a GPU. It tracks which token blocks are resident in cache, emits ZMQ block-lifecycle events consumed by the llm-d routing layer, and exposes Prometheus metrics compatible with the vLLM metric schema.

## What the KV cache simulation does

- **Prefix cache tracking**: incoming prompts are tokenized and split into fixed-size blocks. Blocks already present in the cache are counted as cache hits and do not need to be prefilled.
- **ZMQ event emission**: `BlockStored`, `BlockRemoved`, and `AllBlocksCleared` events are published over ZMQ so that external consumers (e.g., the llm-d KV cache router) can maintain an accurate view of which blocks each pod holds.
- **Latency simulation**: when `prefill-time-per-token` is configured, cache hits reduce the simulated TTFT proportionally — only non-cached tokens contribute to prefill time.
- **Prometheus metrics**: `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_hits`, and `vllm:prefix_cache_queries` are reported.

## Enabling KV cache

Add `--enable-kvcache true` (CLI) or `enable-kvcache: true` (YAML config):

```bash
./bin/llm-d-inference-sim --model Qwen/Qwen2.5-1.5B-Instruct --enable-kvcache true
```

**Prerequisites**:

1. `POD_IP` environment variable must be set — it is embedded in the ZMQ topic name so subscribers can identify which pod published the event.
2. The vLLM render sidecar must be running — the KV cache relies on actual token IDs to compute block hashes. See [manifests/deployment_kvcache.yaml](../manifests/deployment_kvcache.yaml) for a complete Kubernetes example.
3. KV cache is **not** supported in `--mm-encoder-only` mode.

## Configuration options

All KV cache parameters can be set via CLI flags or the equivalent YAML keys (names are identical).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable-kvcache` | bool | `false` | Enable KV cache simulation |
| `kv-cache-size` | int | `1024` | Maximum number of token blocks the cache can hold |
| `block-size` | int | `16` | Tokens per block; valid values: `8`, `16`, `32`, `64`, `128` |
| `hash-seed` | string | value of `PYTHONHASHSEED` env var | Seed for block key hash generation; must match the seed used by real vLLM instances to ensure identical block hashes |
| `zmq-endpoint` | string | `tcp://127.0.0.1:5557` | ZMQ address to publish events (the simulator dials this address) |
| `event-batch-size` | int | `16` | Maximum number of events bundled into a single ZMQ message |
| `global-cache-hit-threshold` | float | `0` (disabled) | Server-wide minimum cache hit rate `[0, 1]`. Requests whose hit rate falls below this threshold return immediately with finish reason `cache_threshold`. Individual requests can override this with the `cache_hit_threshold` field. |

### Example YAML config

```yaml
model: "Qwen/Qwen2.5-1.5B-Instruct"
enable-kvcache: true
kv-cache-size: 2048
block-size: 16
hash-seed: "42"
zmq-endpoint: "tcp://127.0.0.1:5557"
event-batch-size: 32
global-cache-hit-threshold: 0.5
# latency parameters that interact with the KV cache
prefill-overhead: 10ms
prefill-time-per-token: 50us
prefill-time-std-dev: 5us
```

## Latency and the KV cache

The KV cache interacts with Time To First Token (TTFT) simulation. Behavior depends on the configured **latency calculator** (`latency-calculator` flag) and whether the request specifies disaggregated P/D.

### Aggregated (local) prefill — normal requests

When `do_remote_prefill` is not set in the request, the simulator performs local prefill. How cache hits affect TTFT:

| Calculator | TTFT formula |
|-----------|--------------|
| `default` (empty, no `time-to-first-token` set) | `prefill-overhead + (prompt_tokens − cached_tokens) × prefill-time-per-token` |
| `default` (with `time-to-first-token` set) | fixed `time-to-first-token` — cache hits have no effect |
| `constant` | fixed `time-to-first-token` — cache hits have no effect |
| `per-token` | `prefill-overhead + (prompt_tokens − cached_tokens) × prefill-time-per-token` |

Cache hits reduce TTFT only when using the token-aware path (i.e., `prefill-time-per-token` is set and `time-to-first-token` is not).

### Disaggregated P/D — decode pods

When a request sets `kv_transfer_params.do_remote_prefill: true`, the simulator models the time to transfer pre-computed KV blocks from a remote prefill pod rather than running a local prefill:

| Calculator | TTFT formula |
|-----------|--------------|
| `default` (no `kv-cache-transfer-latency`) | `prompt_tokens × kv-cache-transfer-time-per-token` |
| `default` (with `kv-cache-transfer-latency` set) | fixed `kv-cache-transfer-latency` |
| `constant` | fixed `kv-cache-transfer-latency` |
| `per-token` | `prompt_tokens × kv-cache-transfer-time-per-token` |

Relevant latency flags for disaggregated P/D:

| Flag | Description |
|------|-------------|
| `kv-cache-transfer-latency` | Fixed KV block transfer time |
| `kv-cache-transfer-latency-std-dev` | Std dev for the fixed transfer time (max 30% of base value) |
| `kv-cache-transfer-time-per-token` | Per-token KV block transfer time |
| `kv-cache-transfer-time-std-dev` | Std dev for the per-token transfer time |

See [configuration.md](configuration.md) for all latency parameters.

## ZMQ event emission

When `enable-kvcache` is true and `zmq-endpoint` is set, the simulator publishes KV cache events over a ZMQ PUB socket.

### Topic format

```
kv@<POD_IP>@<model-name>
```

Example: `kv@10.0.0.1@Qwen/Qwen2.5-1.5B-Instruct`

The model name used is the base model name (`--model`), not a served-model-name alias.

### Message format

Each message has three ZMQ frames:

| Frame | Content |
|-------|---------|
| 0 | Topic string (UTF-8) |
| 1 | 8-byte big-endian sequence number (monotonically increasing per simulator instance) |
| 2 | msgpack-encoded event batch |

The event batch is a msgpack array with three fields:
1. `ts` (float64) — Unix timestamp in seconds (`nanoseconds / 1e9`)
2. `events` (array) — list of msgpack-encoded event arrays
3. `data_parallel_rank` (int, optional) — always `0`

### Event types

Events are msgpack arrays where the first element is a string tag.

**BlockStored** — emitted when new blocks are allocated for a request:

| Position | Field | Type | Notes |
|----------|-------|------|-------|
| 0 | tag | string | `"BlockStored"` |
| 1 | block_hashes | array | uint64 hashes of the stored blocks |
| 2 | parent_block_hash | uint64 | hash of the preceding block (currently `0`) |
| 3 | token_ids | array of uint32 | tokens contained in these blocks |
| 4 | block_size | int | tokens per block (mirrors `--block-size`) |
| 5 | lora_id | int (optional) | LoRA adapter ID, omitted for base model requests |
| 6 | medium | string (optional) | always `"GPU"` |
| 7 | lora_name | string (optional) | LoRA adapter name, omitted for base model requests |

**BlockRemoved** — emitted when a block is evicted to make room for a new request:

| Position | Field | Type | Notes |
|----------|-------|------|-------|
| 0 | tag | string | `"BlockRemoved"` |
| 1 | block_hashes | array | uint64 hashes of the evicted blocks |
| 2 | medium | string (optional) | always `"GPU"` |

**AllBlocksCleared** — emitted when the cache is fully discarded (see [Sleep mode](#sleep-mode-integration)):

| Position | Field | Type | Notes |
|----------|-------|------|-------|
| 0 | tag | string | `"AllBlocksCleared"` |

### Event batching

Events are buffered and sent together for efficiency. A batch is flushed when either:
- The number of pending events reaches `event-batch-size`, or
- 1 second elapses since the last flush.

## Block eviction policy

When the cache is full and a new block must be stored, the simulator evicts one **unused** block (a block whose last request has already completed). Active blocks — those belonging to a currently running request — are never evicted.

Eviction priority:
1. Oldest unused block belonging to an **unloaded LoRA adapter**
2. Oldest unused block belonging to any model (base or loaded LoRA)

This mirrors vLLM's behavior of preferring to keep blocks for loaded adapters resident.

## LoRA awareness

Block keys incorporate the model name, so a prompt cached under the base model is distinct from the same prompt cached under a LoRA adapter. When a LoRA adapter is loaded or unloaded, its blocks shift eviction priority accordingly:

- **Loaded adapter**: its blocks are treated the same as base model blocks (lower eviction priority)
- **Unloaded adapter**: its blocks become the first eviction candidates

`BlockStored` events include `lora_id` and `lora_name` fields when the request targets a LoRA adapter, allowing subscribers to associate blocks with the correct adapter.

## Cache hit threshold

The `cache_hit_threshold` feature allows routing decisions based on whether the current pod has enough of the prompt already cached.

### Per-request field

Include `cache_hit_threshold` in the request body (value in `[0, 1]`):

```json
{
  "model": "my-model",
  "messages": [...],
  "cache_hit_threshold": 0.8
}
```

If the fraction of cached prompt tokens is below the threshold, the simulator returns an empty response immediately with finish reason `cache_threshold`. The caller can then route the request to a pod with a higher cache hit rate.

### Server-wide default

Set `global-cache-hit-threshold` to apply a threshold to all requests that do not specify one:

```yaml
global-cache-hit-threshold: 0.5
```

A per-request `cache_hit_threshold` always takes precedence over the global value.

## Response usage fields

When KV cache is enabled, the response `usage` object includes cached token information:

```json
{
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "prompt_tokens_detail": {
      "cached_tokens": 64
    }
  }
}
```

`cached_tokens` is the number of prompt tokens that were found in the local KV cache.

## Prometheus metrics

| Metric | Description |
|--------|-------------|
| `vllm:kv_cache_usage_perc` | Fraction of KV cache blocks currently in use (0–1). Updated after each request starts and finishes. |
| `vllm:prefix_cache_hits` | Cumulative number of prompt tokens served from cache across all requests |
| `vllm:prefix_cache_queries` | Cumulative number of prompt tokens checked against the cache across all requests |
| `vllm:cache_config_info` | Static info gauge with labels `cache_dtype`, `num_gpu_blocks`, `num_cpu_blocks`, `block_size` |

The hit rate at any point is `prefix_cache_hits / prefix_cache_queries`.

## Sleep mode integration

When [sleep mode](http-enpdpoints.md) is enabled (`--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`), the KV cache is integrated with the `/sleep` and `/wake_up` endpoints:

- **`POST /sleep`**: disables the cache and emits an `AllBlocksCleared` event so that subscribers know all blocks have been released.
- **`POST /wake_up`** (no `tags` parameter, or `tags=kv_cache`): re-enables the cache.

## Kubernetes deployment

[manifests/deployment_kvcache.yaml](../manifests/deployment_kvcache.yaml) provides a complete example including:

- The vLLM render sidecar (`vllm/vllm-openai-cpu`) as a native sidecar init container
- `POD_IP` injected from the pod's downward API

Key environment variable setup:

```yaml
env:
  - name: POD_IP
    valueFrom:
      fieldRef:
        fieldPath: status.podIP
```

Without `POD_IP`, the simulator will fail to start when `enable-kvcache: true`.
