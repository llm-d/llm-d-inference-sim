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

A request's wall-clock time is dominated by two phases:

```
total_time ≈ prefill_time + decode_time
```

- **Prefill** — runs the model over the prompt to populate the KV cache and produce the
  first token. Cost grows with prompt length.
- **Decode** — generates output tokens one at a time. Cost per token is roughly constant
  for a given batch.

In prefill/decode disaggregation (P/D), the KV cache is transferred over the network before
decode starts; `prefill_time` is replaced by `kv_transfer_time`.

A **calculator** is a pluggable strategy that controls how the simulator turns configuration
parameters into a concrete duration for each request. It owns the prefill formula — which
parameters it reads, and whether it scales with prompt length. Select one via
`latency-calculator`. Decode uses the same formula for all three calculators.

Every duration parameter in the formulas below accepts a `-std-dev` companion for Gaussian
jitter; see the [Parameter Reference](#parameter-by-parameter-reference) for details on
each field.

### Prefill

**`constant`** — TTFT is a single flat duration, independent of prompt length. Use when
prompt size doesn't affect routing or scheduling decisions.

```
prefill_time     = time-to-first-token
kv_transfer_time = kv-cache-transfer-latency        (P/D only)
```

**`per-token`** — TTFT scales with prompt length. Use when routing or scheduling
experiments require latency to vary with prompt size. `n` is the prompt length; `n_cached`
is the number of prefix-cache hits (only uncached tokens are re-computed). KV transfer
uses the full prompt `n` — all KV must be present on the decode node regardless of caching.

```
prefill_time     = prefill-overhead + (n − n_cached) × prefill-time-per-token
kv_transfer_time = n × kv-cache-transfer-time-per-token                (P/D only)
```

**unset / empty string (not recommended)** — Activated by leaving `latency-calculator`
unset (or set to `""`). Retained for backward compatibility; prefer `constant` or
`per-token` explicitly. Applies precedence rules: constant form when
`time-to-first-token` or its std-dev is non-zero, per-token decomposition otherwise.
The same rule applies independently for KV transfer.

---

### Decode

The same formula applies regardless of which calculator is selected.

```
decode_time = (output_tokens − 1) × inter-token-latency
```

### Load factor

`time-factor-under-load` multiplies all GPU-bound latency components
(`time-to-first-token`, `prefill-overhead`, `prefill-time-per-token`, `inter-token-latency`)
when the worker pool is saturated. KV-cache transfer latencies are network-bound and are
**not** scaled.

---

## Parameter-by-Parameter Reference

Every duration parameter has a `…-std-dev` companion that adds Gaussian jitter (capped at
±70% of the mean) to make traces less synthetic. The sections below describe each parameter
and its std-dev together.

### `time-to-first-token` / `time-to-first-token-std-dev`

Used by the `constant` calculator. With an unset calculator, takes precedence over the
prefill decomposition when either field is non-zero.

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

Used by the `per-token` calculator. With an unset calculator, used when
`time-to-first-token` and its std-dev are both zero.

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
applied to the total prefill time; sampled values are clamped to ±70% of the mean.

### `kv-cache-transfer-latency` / `kv-cache-transfer-latency-std-dev`

Used by the `constant` calculator (P/D only). With an unset calculator, takes precedence
over the per-token KV form when either field is non-zero.

The constant per-request overhead of moving KV cache from a prefill node to a decode node.
Includes RPC handshake, scheduling, and setup that does not scale with KV size. The std-dev
is capped at 30% of the mean; sampled values are clamped to ±70% of the mean.

### `kv-cache-transfer-time-per-token` / `kv-cache-transfer-time-std-dev`

Used by the `per-token` calculator (P/D only). With an unset calculator, used when
`kv-cache-transfer-latency` and its std-dev are both zero.

The per-token cost of KV transfer. Total transfer time is:

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

