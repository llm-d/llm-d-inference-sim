# Latency Simulation

This document describes how `llm-d-inference-sim` models inference latency and provides a
reference for each latency parameter. The simulator does not run a real model — these
parameters are what make its responses look like a real LLM serving stack.

All duration fields use Go duration strings: `100ms`, `1.5s`, `750us`, etc.

If you just need a working config, jump straight to
[Latency Reference Tables and Profiles](latency-profiles.md) — the YAML blocks there are
ready to drop in. The sections here explain how the simulator models inference time and
walk through each parameter, useful when you want to understand or tune values for your own
model and hardware.

---

## How the Simulator Models Time

A request's wall-clock time in a vLLM-style engine is dominated by two phases:

```
total_time ≈ prefill_time + (output_tokens − 1) × inter_token_latency
```

- **Prefill** runs the model once over the full prompt to populate the KV (Key/Value
  attention) cache and produce the first token. Cost grows with prompt length.
- **Decode** generates output tokens one at a time. Cost per token is roughly constant for
  a given batch.

In prefill/decode disaggregation (P/D), the KV cache is transferred over the network before
decode starts. From the client's perspective, `prefill_time` is replaced by `kv_transfer_time`.

The simulator exposes two ways to model each phase:

**Prefill:**
1. **Coarse (`time-to-first-token`)** — a single duration for the whole prefill; does not
   scale with prompt length.
2. **Fine (`prefill-overhead` + `prefill-time-per-token`)** — total prefill is
   `prefill-overhead + (n − n_cached) × prefill-time-per-token`. Use this when TTFT should
   scale with prompt size, which matters for routing/scheduling experiments.

**KV-cache transfer (P/D only):**
- `kv-cache-transfer-latency` — a constant duration per request.
- `kv-cache-transfer-time-per-token` — scales with prompt length as `n × kv-cache-transfer-time-per-token`.

`time-to-first-token` takes precedence over the prefill decomposition; `prefill-overhead`/
`prefill-time-per-token` are used only when both `time-to-first-token` and its std-dev are zero.

`kv-cache-transfer-latency` takes precedence over the per-token KV form;
`kv-cache-transfer-time-per-token` is used only when both `kv-cache-transfer-latency` and
its std-dev are zero.

`time-factor-under-load` multiplies the GPU-bound latency components when the worker pool
is saturated. KV-cache transfer latencies are network-bound and are **not** scaled.

Each duration has a `…-std-dev` companion that adds Gaussian jitter (capped at ±70% of the
mean) to make traces less synthetic.

### Calculator selection

`latency-calculator` selects which calculator the simulator uses:

- `""` (default) — precedence rules above apply.
- `constant` — always use `time-to-first-token` and `kv-cache-transfer-latency`; per-token
  fields are ignored regardless of whether the constant field is zero.
- `per-token` — always use prefill decomposition and per-token KV transfer; constant fields
  are ignored.

Use the explicit calculators when you want to force a particular shape regardless of which
fields are set.

---

## Parameter-by-Parameter Reference

### `time-to-first-token` / `time-to-first-token-std-dev`

The total time before the first decoded token is emitted. In a real engine this includes
the prefill forward pass, queueing, and tokenization.

The std-dev is capped at 30% of the mean; sampled values are clamped to ±70% of the mean.

### `inter-token-latency` / `inter-token-latency-std-dev`

The time between consecutive decoded tokens (also called ITL or TPOT). The dominant cost
for long generations; users perceive it as tokens per second (`1 / inter_token_latency`).
Mostly a function of model size, batch size, and memory bandwidth — not prompt length.

A rough lower bound: `model_weights_bytes / memory_bandwidth`. A 7B FP16 model (~14 GB) on
an H100 (~3 TB/s HBM bandwidth) gives ~5 ms minimum; real systems land at 10–15 ms once
attention and overhead are included.

Quantization scales ITL roughly proportionally with bytes-per-parameter: **FP8** ≈ 2×
faster, **INT4** ≈ 3–4× faster (slightly less than theoretical due to dequantization and
attention overhead). Scale rows in the ITL table by the appropriate factor.

The std-dev is capped at 30% of the mean; sampled values are clamped to ±70% of the mean.

### `prefill-overhead`, `prefill-time-per-token`, `prefill-time-std-dev`

Decomposes prefill into a constant overhead plus a per-token cost:

```
prefill_time = prefill-overhead + (n − n_cached) × prefill-time-per-token
```

`n` is the prompt length; `n_cached` is the number of prefix-cache hits (only uncached
tokens are re-computed). The overhead represents fixed per-request engine cost (kernel
launches, scheduling, bookkeeping); the per-token cost represents the linear component of
attention and FFN compute.

Note the asymmetry with KV transfer: KV transfer scales with the full prompt `n` (all KV
must travel across the wire), while prefill scales only with the uncached tokens `n − n_cached`.

The linear model is a good approximation for typical prompt sizes — FlashAttention and
chunking make observed latency scale close to linear in practice. `prefill-time-std-dev` is
applied to the total prefill time; sampled values are clamped to [0.3, 1.7] × mean.

### `kv-cache-transfer-latency` / `kv-cache-transfer-latency-std-dev`

The constant per-request overhead of moving KV cache from a prefill node to a decode node
in P/D disaggregation. Includes RPC handshake, scheduling, and setup that does not scale
with KV size. Only relevant when P/D is enabled. The std-dev is capped at 30% of the mean;
sampled values are clamped to ±70% of the mean.

### `kv-cache-transfer-time-per-token` / `kv-cache-transfer-time-std-dev`

The per-token cost of KV transfer, used only when `kv-cache-transfer-latency` and its
std-dev are both zero. Total transfer time is:

```
kv_transfer_time = n × kv-cache-transfer-time-per-token
```

where `n` is the full prompt length (unlike prefill, the entire KV history must be present
on the decode node). This is a bandwidth calculation: KV bytes per token divided by
interconnect bandwidth. KV bytes grow with layers, KV heads, and head dimension — larger
models have heavier KV caches.

### `time-factor-under-load`

A multiplicative slowdown applied to `time-to-first-token`, `prefill-overhead`,
`prefill-time-per-token`, and `inter-token-latency` when the request queue is saturated.
Must be `>= 1.0`. **Not** applied to KV-cache transfer parameters, which are network-bound.

- `1.0`: no slowdown — useful for unit tests or single-request benchmarks.
- `1.5–2.0`: realistic for latency-optimized deployments under typical load.
- `2.5–3.5`: realistic for throughput-optimized deployments near `max-num-seqs`.

The factor scales linearly between 1.0 (one request in flight) and the configured value
(at `max-num-seqs`). When `max-num-seqs <= 1`, the factor is forced to `1.0`.

---

## Caveats

- All numbers are **order-of-magnitude estimates**. Real performance varies with engine
  version, kernel availability, CUDA graphs, chunked prefill, batch composition, and
  quantization scheme.
- Decode latency can change by 2–3× depending on KV cache occupancy and whether continuous
  batching is enabled.
- Numbers for MI300X and other non-NVIDIA accelerators are less comprehensively documented
  publicly — treat those rows as lower-confidence.
- For accurate calibration, measure `time_to_first_token` and `inter_token_latency` directly
  from the real engine you want to mimic, then plug those values in.
- The simulator quantizes sampled durations to integer milliseconds. A `…-std-dev` value
  below `1ms` becomes effectively zero. If you want sub-millisecond jitter, widen the
  std-dev or change the engine's time resolution.

---

## Sources and Further Reading

- **[vLLM blog](https://blog.vllm.ai/)** — performance posts benchmarking popular models on H100/A100/MI300X.
- **[vLLM documentation](https://docs.vllm.ai/)** — performance and benchmarking guides, plus runnable scripts in `benchmarks/`.
- **[MLPerf Inference results](https://mlcommons.org/benchmarks/inference-datacenter/)** — industry-standard results broken down by model, accelerator, and submission.
- **[NVIDIA developer blog](https://developer.nvidia.com/blog/)** — search "TensorRT-LLM" or "inference" for per-model latency tables published with each major release.

When you need a more authoritative number than the ranges here, measure against the real
engine rather than copying a published benchmark whose batch size, prompt length, or engine
version may not match yours.
