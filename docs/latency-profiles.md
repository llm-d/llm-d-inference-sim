# Latency Reference Tables and Profiles

This document provides reference tables with realistic latency values gathered from publicly
available benchmarks of vLLM, MLPerf, and GPU vendors, along with ready-to-use YAML
profiles. Pick values close to the hardware and model you want to mimic.

For a detailed explanation of how the simulator models inference time and what each
parameter does, see [Latency Simulation](latency-simulation.md).

---

## Reference Tables

The numbers below are approximate and assume FP16/BF16 (16-bit floating-point and 16-bit
"brain float", both 2 bytes per parameter) weights, batch size 1, and reasonable production
settings (PagedAttention, FlashAttention, no CPU offload). Quantized models - FP8 (8-bit
float), INT8 (8-bit integer), or INT4 (4-bit integer) - are roughly 1.5–3× faster on decode.
Exception: 405B rows for H100 assume FP8 quantization because the model does not fit in
FP16 on 8×80 GB; see footnote ¹.

GPU column headers below: H100 and A100 are NVIDIA datacenter GPUs (Hopper and Ampere
generations), L40S is NVIDIA's cheaper Ada-generation datacenter GPU, MI300X is AMD's
flagship inference GPU. The notation "TP=N" means tensor parallelism across N GPUs - the
model's weights are split across N GPUs that work together on each forward pass, which is
the standard way to fit a large model and to reduce per-GPU bandwidth pressure.

### `time-to-first-token` (256-token prompt, batch size 1)

Use directly with the `constant` calculator. Values include both prefill compute and engine
overhead.

| Model size | H100 (80GB) | A100 (80GB) | L40S (48GB) | MI300X (192GB) |
|---|---|---|---|---|
| 1–3B | 30–60 ms | 50–90 ms | 80–130 ms | 25–50 ms |
| 7–8B | 60–120 ms | 100–180 ms | 180–280 ms | 50–100 ms |
| 13B | 120–200 ms | 180–300 ms | 350–500 ms | 100–170 ms |
| 30–34B (TP=2) | 160–280 ms | 280–450 ms | — | 130–230 ms |
| 70B (TP=4) | 250–400 ms | 400–700 ms | — | 200–330 ms |
| 70B (TP=8) | 150–250 ms | 250–400 ms | — | 120–215 ms |
| 405B (TP=8) | 0.6–1.4 s | infeasible¹ | — | 0.9–2.0 s |

Real-world TTFT also includes scheduling, tokenization, and request queueing - at high load
TTFT can grow several-fold even though the prefill itself is unchanged.

¹ A 405B model in FP16/BF16 needs ~810 GB, exceeding 8×A100-80GB (640 GB) even at TP=8. H100 values assume native FP8 quantization (Hopper hardware feature), which halves weight memory to ~405 GB and fits within 640 GB aggregate memory of an 8×80GB cluster. A100 lacks native FP8 and INT8 quantization incurs significant quality/throughput penalties, making it impractical. MI300X (8×192 GB = 1,536 GB) fits FP16 without quantization.

### `inter-token-latency` (per output token, batch size 1)

| Model size | H100 (80GB) | A100 (80GB) | L40S (48GB) | MI300X (192GB) |
|---|---|---|---|---|
| 1–3B | 5–8 ms | 8–12 ms | 12–18 ms | 6–10 ms |
| 7–8B | 10–15 ms | 15–25 ms | 25–35 ms | 12–18 ms |
| 13B | 15–22 ms | 22–35 ms | 40–55 ms | 18–28 ms |
| 30–34B (TP=2) | 20–30 ms | 30–45 ms | — | 25–35 ms |
| 70B (TP=4) | 25–40 ms | 40–60 ms | — | 30–45 ms |
| 70B (TP=8) | 18–28 ms | 28–45 ms | — | 22–32 ms |
| 405B (TP=8) | 60–100 ms | infeasible¹ | — | 70–120 ms |

`StdDev` for ITL is typically small in well-behaved engines - 5–15% of the mean
(`1–3 ms` for the 8B/H100 row). Note that if calculating StdDev for ultra-fast models 
where the value falls below 1ms (e.g., 10% of a 5ms latency), you must round up and keep the 
value at or above `1ms` to prevent `RandomNormDuration` from truncating your jitter to zero.

¹ See footnote under `time-to-first-token` above.


### Per-token calculator: `prefill-overhead` + (n − n_cached) × `prefill-time-per-token`

| Model size | `prefill-overhead` | `prefill-time-per-token` (H100) | `prefill-time-per-token` (A100) | `prefill-time-per-token` (L40S) | `prefill-time-per-token` (MI300X) |
|---|---|---|---|---|---|
| 1–3B | 15–25 ms | 0.10–0.20 ms | 0.20–0.35 ms | 0.35–0.55 ms | 0.08–0.15 ms |
| 7–8B | 25–40 ms | 0.15–0.30 ms | 0.30–0.55 ms | 0.65–1.00 ms | 0.11–0.23 ms |
| 13B | 40–60 ms | 0.55–0.85 ms | 0.90–1.40 ms | 1.50–2.20 ms | 0.42–0.65 ms |
| 30–34B (TP=2) | 50–80 ms | 0.30–0.50 ms | 0.60–1.00 ms | — | 0.22–0.38 ms |
| 70B (TP=4) | 70–110 ms | 0.60–0.90 ms | 1.20–1.80 ms | — | 0.45–0.68 ms |
| 70B (TP=8) | 60–100 ms | 0.40–0.60 ms | 0.80–1.30 ms | — | 0.30–0.45 ms |
| 405B (TP=8) | 150–250 ms | 2–4 ms | infeasible¹ | — | 3–6 ms |

**Sanity check:** for a 256-token prompt, each row above should land near the corresponding
TTFT row. Examples: 8B/H100: `30 + 256 × 0.25 = 94 ms` ≈ 60–120 ms; 70B (TP=8)/H100:
`80 + 256 × 0.50 = 208 ms` ≈ 150–250 ms; 1–3B/L40S: 
`20 + 256 × 0.35 = 110 ms` ≈ 80–130 ms; 405B (TP=8)/H100:
`200 + 256 × 3 = 968 ms` ≈ 0.6–1.4 s.

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

Three ready-to-use profiles. Each is provided as a complete YAML file under
[`manifests/latency-profiles/`](../manifests/latency-profiles/) - pass it directly with
`--config`, or copy the latency fields into your existing config.

Each profile provides two YAML files, one per calculator:

- **`constant` calculator** — uses `time-to-first-token` and `kv-cache-transfer-latency`.
  Simpler; TTFT is independent of prompt length.
- **`per-token` calculator** — uses `prefill-overhead`/`prefill-time-per-token` and
  `kv-cache-transfer-time-per-token`. TTFT scales with prompt length, which matters for
  routing and scheduling experiments.

The `per-token` values below are calibrated so that prefill cost for a ~256-token prompt
is in the same range as the `constant` TTFT. See the
[latency simulation reference](latency-simulation.md#how-the-simulator-models-time) for
details on each calculator.

### Profile 1: 8B-class model on H100, balanced load

Mirrors a production Llama-3-8B deployment on a single H100, moderate concurrency.

Full configs: [constant](../manifests/latency-profiles/8b-h100-balanced-constant.yaml) |
[per-token](../manifests/latency-profiles/8b-h100-balanced-per-token.yaml).

`constant` calculator:

```yaml
latency-calculator: constant
time-to-first-token: 100ms
time-to-first-token-std-dev: 20ms
inter-token-latency: 12ms
inter-token-latency-std-dev: 2ms

kv-cache-transfer-latency: 2ms
kv-cache-transfer-latency-std-dev: 400us

time-factor-under-load: 2.0
```

`per-token` calculator:

```yaml
latency-calculator: per-token
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
running close to `max-num-seqs` saturation. The per-token calculator KV values assume an
InfiniBand interconnect for cross-node disaggregated serving.

Full configs: [constant](../manifests/latency-profiles/70b-h100-tp8-throughput-constant.yaml) |
[per-token](../manifests/latency-profiles/70b-h100-tp8-throughput-per-token.yaml).

`constant` calculator:

```yaml
latency-calculator: constant
time-to-first-token: 200ms
time-to-first-token-std-dev: 40ms
inter-token-latency: 25ms
inter-token-latency-std-dev: 4ms

kv-cache-transfer-latency: 2ms
kv-cache-transfer-latency-std-dev: 400us

time-factor-under-load: 3.0
```

`per-token` calculator:

```yaml
latency-calculator: per-token
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
quick responses. KV-transfer values assume ~Ethernet 100G (~10 GB/s effective) for cross-node P/D.

Full configs: [constant](../manifests/latency-profiles/small-l40s-edge-constant.yaml) |
[per-token](../manifests/latency-profiles/small-l40s-edge-per-token.yaml).

`constant` calculator:

```yaml
latency-calculator: constant
time-to-first-token: 110ms
time-to-first-token-std-dev: 15ms
inter-token-latency: 15ms
inter-token-latency-std-dev: 2ms

kv-cache-transfer-latency: 5ms
kv-cache-transfer-latency-std-dev: 1ms

time-factor-under-load: 1.5
```

`per-token` calculator:

```yaml
latency-calculator: per-token
inter-token-latency: 15ms
inter-token-latency-std-dev: 2ms

prefill-overhead: 20ms
prefill-time-per-token: 350us
prefill-time-std-dev: 3ms

kv-cache-transfer-time-per-token: 12us
kv-cache-transfer-time-std-dev: 500us

time-factor-under-load: 1.5
```

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
