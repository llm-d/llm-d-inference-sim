# Latency Configuration Reference

This document describes the latency-related configuration parameters of `llm-d-inference-sim`
and provides reference tables with realistic defaults gathered from publicly available
benchmarks of vLLM,
MLPerf, and GPU vendors. The
simulator does not run a real model, so these values are what make its responses look like
a real LLM serving stack - pick values close to the hardware and
model you want to mimic.

All duration fields use Go duration strings: `100ms`, `1.5s`, `750us`, etc.

If you just need a working config, jump straight to
[Suggested Default Profiles](#suggested-default-profiles) - the YAML blocks there are
ready to drop in. The sections in between explain how the simulator models inference time
and walk through each parameter, useful when you want to tune values for your own model
and hardware rather than copy them.

---

## How the Simulator Models Time

A request's wall-clock time in a vLLM-style engine is dominated by two phases:

```
total_time ≈ prefill_time + (output_tokens − 1) × inter_token_latency
```

- **Prefill** runs the model once over the full prompt to populate the KV (Key/Value
  attention) cache and produce the first token. Cost grows with prompt length.
- **Decode** generates output tokens one at a time, each conditioned on the KV cache. Cost
  per token is roughly constant for a given batch.

In prefill/decode disaggregation mode (P/D), prefill runs on a separate node and only the
KV cache is sent over the network to the decode node. The decode-side simulator therefore
substitutes `kv_transfer_time` for `prefill_time` in the formula above, from the client's
perspective `total_time ≈ kv_transfer_time + (output_tokens − 1) × inter_token_latency`.

The simulator exposes two ways to model the prefill phase:

1. **Coarse (`time-to-first-token`)** - a single duration for the whole prefill. Fast to
   configure, but it does not scale with prompt length.
2. **Fine (`prefill-overhead` + `prefill-time-per-token`)** - total prefill is computed as
   `prefill-overhead + (n - n_cached) × prefill-time-per-token`, where `n` is the prompt
   length and `n_cached` is the number of prompt tokens served from the prefix cache. Use
   this when you want TTFT to scale with prompt size, which matters for routing/scheduling
   experiments.

When prefill/decode disaggregation (P/D) is active, KV cache produced on the prefill node
must be transferred to the decode node before decode can start. The simulator offers two
alternative ways to model this transfer (only one is used per request):

- `kv-cache-transfer-latency` - a single constant duration regardless of prompt length.
- `kv-cache-transfer-time-per-token` - the transfer time scales with prompt length as
  `n × kv-cache-transfer-time-per-token` (no separate overhead term).

`kv-cache-transfer-latency` takes precedence: the per-token form is used only when both
`kv-cache-transfer-latency` and its std-dev are zero.

`time-factor-under-load` multiplies the GPU-bound latency contributions (`time-to-first-token`,
`prefill-overhead`, `prefill-time-per-token`, and `inter-token-latency`) when the worker
pool is saturated - it approximates the throughput collapse real systems see at high
concurrency. KV-cache transfer latencies are network-bound and are **not** scaled by this
factor.

Each duration also has a `…StdDev` (standard deviation) companion that adds Gaussian
jitter (capped at ±70% of the mean) to make traces less synthetic.

### Calculator selection

The precedence rules above describe the **default** latency calculator. The
`latency-calculator` config option selects which calculator the simulator uses:

- `latency-calculator: ""` (empty, the default) - the precedence rules above apply: the
  per-token form is used when the corresponding constant field is zero, otherwise the
  constant form wins.
- `latency-calculator: constant` - always use the constant `time-to-first-token` and
  constant `kv-cache-transfer-latency`. The per-token fields (`prefill-overhead`,
  `prefill-time-per-token`, `kv-cache-transfer-time-per-token`) are ignored regardless of
  whether the constant field is zero.
- `latency-calculator: per-token` - always use the prefill decomposition and per-token KV
  transfer. The constant fields (`time-to-first-token`, `kv-cache-transfer-latency`) are
  ignored.

Use the explicit `constant`/`per-token` calculators when you want to force a particular
shape regardless of which fields happen to be set.

---

## Parameter-by-Parameter Reference

### `time-to-first-token` / `time-to-first-token-std-dev`

The total time before the first decoded token is emitted, measured from when the request
enters the engine. In a real engine this is dominated by the prefill forward pass, plus
queueing and tokenization. Use this when you want a single, fixed TTFT (time-to-first-token)
regardless of prompt length. `time-to-first-token` takes precedence over the prefill
decomposition: if either `time-to-first-token` or `time-to-first-token-std-dev` is non-zero,
`prefill-overhead`/`prefill-time-per-token` are ignored. To use the prefill decomposition,
leave both `time-to-first-token` and its std-dev at zero.

The std-dev is capped at 30% of the mean and the sampled value is clamped so it cannot
deviate by more than 70% - this prevents pathological negatives or absurdly long tails.

### `inter-token-latency` / `inter-token-latency-std-dev`

The time between two consecutive decoded tokens (also called ITL - inter-token latency, or
TPOT - time per output token). This is the dominant cost for long generations and is what
users perceive as "tokens per second" (`1 / inter_token_latency`). It is mostly a function
of model size, KV cache length, batch size, and memory bandwidth - not prompt length.

A useful rule of thumb: for memory-bound decode (the common case), `inter-token-latency` is
roughly `model_weights_bytes / memory_bandwidth`, real systems also read KV cache and incur 
overhead. A 7B model in FP16 (16-bit floating-point,
2 bytes per parameter) has ~14 GB of weights, on an H100 (NVIDIA Hopper-generation GPU)
with ~3 TB/s HBM (high-bandwidth memory) bandwidth that's ~5 ms minimum, with real-world
values closer to 10–15 ms once attention and overhead are included.

Quantization scales `inter-token-latency` roughly proportionally with bytes-per-parameter,
since the bottleneck is reading weights from memory. Compared to FP16 (2 bytes/param):
**FP8** (1 byte/param) ≈ ~2× faster, **INT4** (0.5 byte/param) ≈ ~3–4× faster in practice
(a bit less than the 4× theoretical because of dequantization overhead and the fact that
attention and KV-cache reads remain at higher precision). Pick a row in the ITL table and
divide by the appropriate factor to estimate quantized latencies.

### `prefill-overhead`, `prefill-time-per-token`, `prefill-time-std-dev`

Decomposes prefill into a constant overhead plus a per-token cost. Total prefill time is:

```
prefill_time = prefill-overhead + (n - n_cached) × prefill-time-per-token
```

where `n` is the prompt length in tokens and `n_cached` is the number of prompt tokens
that hit the prefix cache (only those that were *not* cached are re-computed). The
overhead represents the fixed cost per prefill (kernel launches, scheduling, per-request
engine bookkeeping) and the per-token cost represents the linear part of the attention
and FFN compute.

Note the asymmetry with KV-cache transfer: KV transfer per-token cost scales with the full
prompt length `n` (every token's KV must travel across the wire regardless of where it was
computed), while prefill scales only with the *uncached* tokens.

The linear model is a good approximation for typical prompt sizes (up to a few thousand tokens). While attention is theoretically quadratic, modern kernels (e.g. FlashAttention) and chunking make observed latency scale close to linear in practice, with deviations appearing only at very large sequence lengths.

`prefill-time-std-dev` is applied to the total prefill time as Gaussian jitter. Because the
total depends on the prompt length `n`, the validator does not impose an upper bound on
this field. Sampled durations are clamped at runtime to [0.3, 1.7] × mean.

### `kv-cache-transfer-latency` / `kv-cache-transfer-latency-std-dev`

The constant per-request overhead of moving KV cache from a prefill node to a decode node
in P/D disaggregation. Includes RPC (remote procedure call) handshake, scheduling, and any
setup that does not scale with KV size. Only matters when P/D is enabled.

### `kv-cache-transfer-time-per-token` / `kv-cache-transfer-time-std-dev`

The per-token cost of the KV transfer. Used only when both `kv-cache-transfer-latency` and
`kv-cache-transfer-latency-std-dev` are zero (otherwise the constant `kv-cache-transfer-latency`
takes precedence). Total transfer time is:

```
kv_transfer_time = n × kv-cache-transfer-time-per-token
```

where `n` is the full prompt length in tokens (unlike prefill, the entire KV history must
be present on the destination decode node). There is no separate overhead term. This is
fundamentally a bandwidth calculation: bytes per token divided by interconnect bandwidth.
KV bytes per token grows with the number of layers, KV heads, and head dimension - larger
models have heavier KV caches.

Like `prefill-time-std-dev`, `kv-cache-transfer-time-std-dev` is applied to the total
transfer time (which depends on `n`), so the validator does not impose an upper bound on
it. Runtime clamping handles oversized std-devs.

### `time-factor-under-load`

A multiplicative slowdown applied to the GPU-bound latency components when the request
queue is saturated. Must be `>= 1.0`. Applied to `time-to-first-token`, `prefill-overhead`,
`prefill-time-per-token`, and `inter-token-latency`. **Not** applied to
`kv-cache-transfer-latency` or `kv-cache-transfer-time-per-token`, which model
network-bound work that the GPU load does not affect.

- `1.0` (default): no slowdown - useful for unit tests or single-request benchmarks.
- `1.5–2.0`: realistic for latency-optimized deployments under typical load.
- `2.5–3.5`: realistic for throughput-optimized deployments running close to `max-num-seqs`.

The factor scales linearly with concurrency: it equals `1.0` when one request is in flight
and reaches the configured value when `max-num-seqs` requests are in flight. As an edge
case, when `max-num-seqs <= 1` the factor is forced to `1.0` regardless of the configured
value, since "fully saturated" and "single request" are the same state.

---

## Reference Tables

The numbers below are approximate and assume FP16/BF16 (16-bit floating-point and 16-bit
"brain float", both 2 bytes per parameter) weights, batch size 1, and reasonable production
settings (PagedAttention, FlashAttention, no CPU offload). Quantized models - FP8 (8-bit
float), INT8 (8-bit integer), or INT4 (4-bit integer) - are roughly 1.5–3× faster on decode.

GPU column headers below: H100 and A100 are NVIDIA datacenter GPUs (Hopper and Ampere
generations), L40S is NVIDIA's cheaper Ada-generation datacenter GPU, MI300X is AMD's
flagship inference GPU. The notation "TP=N" means tensor parallelism across N GPUs - the
model's weights are split across N GPUs that work together on each forward pass, which is
the standard way to fit a large model and to reduce per-GPU bandwidth pressure.

### `inter-token-latency` (per output token, batch size 1)

| Model size | H100 (80GB) | A100 (80GB) | L40S | MI300X |
|---|---|---|---|---|
| 1–3B | 5–8 ms | 8–12 ms | 12–18 ms | 6–10 ms |
| 7–8B | 10–15 ms | 15–25 ms | 25–35 ms | 12–18 ms |
| 13B | 15–22 ms | 22–35 ms | 40–55 ms | 18–28 ms |
| 30–34B (TP=2) | 20–30 ms | 30–45 ms | - | 25–35 ms |
| 70B (TP=4) | 25–40 ms | 40–60 ms | - | 30–45 ms |
| 70B (TP=8) | 18–28 ms | 28–45 ms | - | 22–32 ms |
| 405B (TP=8) | 60–100 ms | infeasible¹ | - | 70–120 ms |

`StdDev` for ITL is typically small in well-behaved engines - 5–15% of the mean
(`1–3 ms` for the 8B/H100 row).

¹ A 405B model (like Llama-3.1 405B) in FP16/BF16 needs ~810 GB of weight memory, which exceeds 8×A100-80GB (640 GB total) even with TP=8. The cell is feasible only with quantization (FP8/INT4) or
pipeline parallelism beyond a single node.

### `time-to-first-token` (256-token prompt, batch size 1)

If you do not use the prefill decomposition, set this directly. Values include both prefill
compute and engine overhead.

| Model size | H100 | A100 | L40S |
|---|---|---|---|
| 1–3B | 30–60 ms | 50–90 ms | 80–130 ms |
| 7–8B | 60–120 ms | 100–180 ms | 180–280 ms |
| 13B | 120–200 ms | 180–300 ms | 350–500 ms |
| 70B (TP=4) | 250–400 ms | 400–700 ms | - |
| 70B (TP=8) | 150–250 ms | 250–400 ms | - |

Real-world TTFT also includes scheduling, tokenization, and request queueing - at high load
TTFT can grow several-fold even though the prefill itself is unchanged.

### Prefill Decomposition (`prefill-overhead` + (n − n_cached) × `prefill-time-per-token`)

| Model size | `prefill-overhead` | `prefill-time-per-token` (H100) | `prefill-time-per-token` (A100) |
|---|---|---|---|
| 1–3B | 15–25 ms | 0.10–0.20 ms | 0.20–0.35 ms |
| 7–8B | 25–40 ms | 0.15–0.30 ms | 0.30–0.55 ms |
| 13B | 40–60 ms | 0.55–0.85 ms | 0.90–1.40 ms |
| 70B (TP=8) | 60–100 ms | 0.40–0.60 ms | 0.80–1.30 ms |
| 405B (TP=8) | 150–250 ms | 6–10 ms | - |

**Sanity check:** for a 256-token prompt, the row above should land near the corresponding
TTFT row (e.g. 8B/H100: `30 + 256 × 0.25 = 94 ms` ≈ 60–120 ms TTFT row).

### KV Cache Transfer

KV bytes per token depend on the model's architecture:

```
kv_bytes_per_token = 2 (K and V) × num_layers × num_kv_heads × head_dim × bytes_per_elem
```

The "GQA" annotation marks models that use Grouped-Query Attention - an architecture
where multiple query heads share a single KV head, which dramatically shrinks the KV cache
compared to vanilla multi-head attention. The table below maps common model names to their
architecture parameters (so you can plug them into the formula above) and the resulting KV
size per token in FP16:

| Model | Layers | KV heads | head_dim | Attention | KV bytes/token (FP16) |
|---|---|---|---|---|---|
| Llama-3 / 3.1 8B | 32 | 8 | 128 | GQA (4:1) | ~128 KB |
| Llama-3 / 3.1 70B | 80 | 8 | 128 | GQA (8:1) | ~320 KB |
| Llama-3.1 405B | 126 | 16 | 128 | GQA (8:1) | ~1 MB |
| Mistral 7B (v0.3) | 32 | 8 | 128 | GQA (4:1) | ~128 KB |
| Mixtral 8×7B (MoE) | 32 | 8 | 128 | GQA (4:1) | ~128 KB |

The "Attention" column shows the GQA query-to-KV-head ratio (e.g. `4:1` means 4 query
heads share each KV head). For other models, look up `num_hidden_layers`,
`num_key_value_heads`, and `head_dim` (or `hidden_size / num_attention_heads`) in the
model's HuggingFace `config.json`.

Per-token transfer time = bytes / effective bandwidth. Real bandwidth is typically 60–80%
of link rated speed.

Interconnect terms used below: **NVLink** is NVIDIA's high-bandwidth GPU-to-GPU link inside
a single server. **InfiniBand (IB)** is a low-latency datacenter fabric, with **NDR** being
its 400 Gb/s generation. **RoCE** (RDMA over Converged Ethernet) brings RDMA (Remote Direct
Memory Access - kernel-bypass networking) semantics to standard Ethernet hardware.

| Interconnect | Effective bandwidth | 8B (~128 KB/tok) | 70B (~320 KB/tok) | 405B (~1 MB/tok) |
|---|---|---|---|---|
| NVLink (intra-node) | ~600 GB/s | ~0.2 µs | ~0.5 µs | ~2 µs |
| InfiniBand NDR 400G | ~40 GB/s | ~3 µs | ~8 µs | ~25 µs |
| RoCE v2 200G | ~20 GB/s | ~6 µs | ~16 µs | ~50 µs |
| Ethernet 100G | ~10 GB/s | ~12 µs | ~32 µs | ~100 µs |
| Ethernet 25G | ~2.5 GB/s | ~50 µs | ~125 µs | ~400 µs |

**`kv-cache-transfer-latency`** (the constant setup overhead) is dominated by RPC and
scheduling, not bandwidth:

| Transport | Typical setup overhead |
|---|---|
| NVLink + shared memory | 100–500 µs |
| RDMA (IB / RoCE) | 0.5–2 ms |
| TCP/Ethernet | 2–10 ms |
| Cross-AZ / cross-region | 10–50+ ms |

---

## Suggested Default Profiles

Three ready-to-use profiles. Drop one into your YAML config and adjust as needed.

For each profile, pick **one** of two forms per request-type:

- **Coarse form** - uses the constant fields (`time-to-first-token`,
  `kv-cache-transfer-latency`). Simpler, latency is independent of prompt length.
- **Fine form** - uses the per-token fields (`prefill-overhead`/`prefill-time-per-token`,
  `kv-cache-transfer-time-per-token`). Latency scales with prompt size, which matters for
  routing/scheduling experiments.

The forms are mutually exclusive: if you set the constant field, the per-token form is
ignored (see the parameter reference above). The fine-form values below are calibrated so
that the prefill cost for a ~256-token prompt is in the same range as the coarse-form TTFT.

### Profile 1: 8B-class model on H100, balanced load

Mirrors a production Llama-3-8B deployment on a single H100, moderate concurrency.

Coarse form:

```yaml
time-to-first-token: 100ms
time-to-first-token-std-dev: 20ms
inter-token-latency: 12ms
inter-token-latency-std-dev: 2ms

kv-cache-transfer-latency: 2ms
kv-cache-transfer-latency-std-dev: 400us

time-factor-under-load: 2.0
```

Fine form:

```yaml
inter-token-latency: 12ms
inter-token-latency-std-dev: 2ms

prefill-overhead: 30ms
prefill-time-per-token: 250us
prefill-time-std-dev: 5ms

kv-cache-transfer-time-per-token: 3us
kv-cache-transfer-time-std-dev: 200us

time-factor-under-load: 2.0
```

### Profile 2: 70B model on 8×H100 (TP=8), throughput-optimized

Mirrors a Llama-3-70B deployment using tensor parallelism (TP=8) on H100 nodes, 
running close to `max-num-seqs` saturation. The fine-form KV values assume an InfiniBand 
interconnect for cross-node disaggregated serving.

Coarse form:

```yaml
time-to-first-token: 200ms
time-to-first-token-std-dev: 40ms
inter-token-latency: 25ms
inter-token-latency-std-dev: 4ms

kv-cache-transfer-latency: 2ms
kv-cache-transfer-latency-std-dev: 400us

time-factor-under-load: 3.0
```

Fine form:

```yaml
inter-token-latency: 25ms
inter-token-latency-std-dev: 4ms

prefill-overhead: 80ms
prefill-time-per-token: 500us
prefill-time-std-dev: 15ms

kv-cache-transfer-time-per-token: 8us
kv-cache-transfer-time-std-dev: 500us

time-factor-under-load: 3.0
```

### Profile 3: Small model (1–3B) on L40S, low-latency edge

Mirrors a small (1–3B) model on a single L40S at the edge, tuned for low concurrency and
quick responses. KV-transfer values assume a modest network for cross-node P/D.

Coarse form:

```yaml
time-to-first-token: 80ms
time-to-first-token-std-dev: 15ms
inter-token-latency: 15ms
inter-token-latency-std-dev: 2ms

kv-cache-transfer-latency: 5ms
kv-cache-transfer-latency-std-dev: 1ms

time-factor-under-load: 1.5
```

Fine form:

```yaml
inter-token-latency: 15ms
inter-token-latency-std-dev: 2ms

prefill-overhead: 20ms
prefill-time-per-token: 230us
prefill-time-std-dev: 3ms

kv-cache-transfer-time-per-token: 12us
kv-cache-transfer-time-std-dev: 500us

time-factor-under-load: 1.5
```

---

## Caveats

- All numbers are **order-of-magnitude estimates**. Real performance varies with engine
  version, kernel availability (FlashAttention-2/3), CUDA Graph use, chunked prefill, batch
  composition, prompt length distribution, and quantization scheme.
- Decode latency on the same model can change by 2–3× depending on KV cache occupancy and
  whether continuous batching is enabled.
- Numbers for AMD MI300X, Intel Gaudi, and other accelerators are less comprehensively
  documented publicly - treat those rows as lower-confidence.
- For the most accurate calibration, run a few requests against the real engine you want to
  mimic and measure `time_to_first_token` and `inter_token_latency` from the response
  stream - then plug those into the simulator.
- The simulator quantizes sampled durations to integer milliseconds inside
  `RandomNormDuration`. Per-token values in microseconds work because they multiply up to
  millisecond-scale totals before sampling, but a `…-std-dev` value below `1ms` becomes
  effectively zero (no observable jitter). If you want sub-millisecond jitter, you'll need
  to widen your std-dev or change the engine's time resolution.

---

## Sources and Further Reading

The numbers in this document were synthesized from publicly available benchmarks:

- **[vLLM blog](https://blog.vllm.ai/)** - the project's official blog, with periodic
  performance posts that benchmark popular models on H100/A100/MI300X.
- **[vLLM documentation](https://docs.vllm.ai/)** - performance and benchmarking guides,
  plus the `benchmarks/` directory in the vLLM repo for runnable scripts.
- **[MLPerf Inference results](https://mlcommons.org/benchmarks/inference-datacenter/)** -
  the industry-standard benchmark, results are submitted by NVIDIA, AMD, Intel, and others
  and broken down by model, accelerator, and submission.
- **[NVIDIA developer blog](https://developer.nvidia.com/blog/)** - search for "TensorRT-LLM"
  or "inference" tags, NVIDIA publishes per-model latency tables with each major release.

When you need a number that is more authoritative than the ranges here, prefer measuring
against the real engine you want to mimic (see the last bullet of *Caveats*) over copying
a published number whose batch size, prompt length, or engine version may not match yours.
