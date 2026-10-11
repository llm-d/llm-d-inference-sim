# SGLang metrics: comparison with the simulator's vLLM surface

Sources for the SGLang side: `python/sglang/srt/observability/metrics_collector.py` on
sgl-project/sglang `main` (153 `sglang:`-prefixed metrics) and
`docs/docs/references/production_metrics.mdx`. Router references are against
`github.com/llm-d/llm-d-router v0.10.0`, resolvable under `$(go env GOMODCACHE)`.

## 1. What the simulator supports

### vLLM engine

24 metrics, names defined in [pkg/engine/vllm/metrics.go:40-62](pkg/engine/vllm/metrics.go#L40-L62)
and documented in [docs/metrics.md](docs/metrics.md). Every metric carries `model_name`
([pkg/engine/vllm/metrics.go:65](pkg/engine/vllm/metrics.go#L65)); three carry more.

| Metric | Type | Extra labels | Value recorded | Source |
|---|---|---|---|---|
| vllm:num_requests_running | Gauge | — | ±1 on `OnRequestRunning` / terminal event | [metrics.go:337](pkg/engine/vllm/metrics.go#L337) |
| vllm:num_requests_waiting | Gauge | — | ±1 on queue/dequeue | [metrics.go:321](pkg/engine/vllm/metrics.go#L321) |
| vllm:kv_cache_usage_perc | Gauge | — | fraction 0..1, block-based, delta-applied | [metrics.go:430](pkg/engine/vllm/metrics.go#L430) |
| vllm:lora_requests_info | Gauge | `max_lora`, `running_lora_adapters`, `waiting_lora_adapters` | Unix timestamp as the value; adapter names in labels | [metrics.go:936](pkg/engine/vllm/metrics.go#L936) |
| vllm:cache_config_info | Gauge | `block_size`, `cache_dtype`, `num_cpu_blocks`, `num_gpu_blocks` | constant 1, stamped once at startup | [metrics.go:944](pkg/engine/vllm/metrics.go#L944), [metrics.go:1122](pkg/engine/vllm/metrics.go#L1122) |
| vllm:request_success_total | Counter | `finish_reason` | +1 per successful request | [metrics.go:1093](pkg/engine/vllm/metrics.go#L1093) |
| vllm:prompt_tokens_total, vllm:generation_tokens_total | Counter | — | += per-request token counts on success | [metrics.go:401](pkg/engine/vllm/metrics.go#L401) |
| vllm:prefix_cache_hits_total, vllm:prefix_cache_queries_total | Counter | — | += cached / queried prompt tokens | [metrics.go:440](pkg/engine/vllm/metrics.go#L440) |
| vllm:request_prompt_tokens, vllm:request_generation_tokens, vllm:max_num_generation_tokens, vllm:request_params_max_tokens | Histogram | — | 1-2-5 buckets up to `max-model-len` | [metrics.go:1038](pkg/engine/vllm/metrics.go#L1038), [pkg/metrics/utils.go:26](pkg/metrics/utils.go#L26) |
| vllm:time_to_first_token_seconds | Histogram | — | prefill duration | [metrics.go:355](pkg/engine/vllm/metrics.go#L355) |
| vllm:time_per_output_token_seconds, vllm:inter_token_latency_seconds | Histogram | — | both observe the same per-token latency | [metrics.go:365](pkg/engine/vllm/metrics.go#L365) |
| vllm:request_time_per_output_token_seconds | Histogram | — | decode duration / generation tokens | [metrics.go:380](pkg/engine/vllm/metrics.go#L380) |
| vllm:e2e_request_latency_seconds, vllm:request_inference_time_seconds | Histogram | — | observed on both success and failure | [metrics.go:414](pkg/engine/vllm/metrics.go#L414) |
| vllm:request_queue_time_seconds, vllm:request_prefill_time_seconds, vllm:request_decode_time_seconds | Histogram | — | per-phase durations | [metrics.go:328](pkg/engine/vllm/metrics.go#L328) |

Bucket boundaries for TTFT, TPOT, and request latency: [pkg/common/utils.go:36-43](pkg/common/utils.go#L36-L43).

### SGLang engine

No metrics. [pkg/engine/sglang/metrics.go](pkg/engine/sglang/metrics.go) registers no collectors
and drops every bus event, so `/metrics` serves an empty body. `fake-metrics` is rejected at
validation.

## 2. SGLang equivalents of the simulator's vLLM metrics

One structural difference applies to the whole table. SGLang's base label set is `model_name`,
`engine_type`, `tp_rank`, `pp_rank`, `moe_ep_rank`, plus `dp_rank` when DP is on, `priority` under
priority scheduling, and anything from `--extra-metric-labels` (collector.py:1171-1183). The
simulator's vLLM surface has only `model_name`. SGLang also requires `--enable-metrics`, and most
scheduler gauges are written once per `log_stats` interval rather than per event.

| SGLang metric | Type | Value reported | vLLM equivalent | Difference |
|---|---|---|---|---|
| sglang:num_running_reqs | Gauge | running requests in the batch | vllm:num_requests_running | Name only; `multiprocess_mode="mostrecent"`, sampled at log interval |
| sglang:num_queue_reqs | Gauge | requests in waiting queue | vllm:num_requests_waiting | Grammar-blocked requests are counted separately in `sglang:num_grammar_queue_reqs`, not folded in |
| sglang:token_usage | Gauge | fraction of KV token slots in use | vllm:kv_cache_usage_perc | Token-based, not block-based. Split further into `full_token_usage` / `swa_token_usage` / `mamba_usage` for hybrid models |
| sglang:prompt_tokens_total | Counter | prefill tokens processed | vllm:prompt_tokens_total | Extra `is_streaming` label, so a single-series query no longer works |
| sglang:generation_tokens_total | Counter | generation tokens | vllm:generation_tokens_total | Extra `is_streaming` label |
| sglang:prompt_tokens_histogram | Histogram | prompt length | vllm:request_prompt_tokens | Different name (`_histogram` suffix); fixed 100..1.1M buckets instead of 1-2-5 up to `max_model_len`. `sglang:uncached_prompt_tokens_histogram` additionally reports compute-only prompt length, which vLLM has no counterpart for |
| sglang:generation_tokens_histogram | Histogram | generation length | vllm:request_generation_tokens | Same naming and bucket differences; reuses the prompt default bucket list |
| sglang:num_requests_total | Counter | requests processed | vllm:request_success_total | Counts all processed requests labeled by `is_streaming`, not successes by `finish_reason`. Aborts are a separate counter, `sglang:num_aborted_requests_total` |
| sglang:time_to_first_token_seconds | Histogram | TTFT | vllm:time_to_first_token_seconds | Extra `is_streaming` label; coarser default buckets (0.1..400 vs 0.001..2560); semantics differ by streaming mode, streaming ends at first detokenized output, non-streaming at the scheduler's first token |
| sglang:e2e_request_latency_seconds | Histogram | end-to-end latency | vllm:e2e_request_latency_seconds | Extra `is_streaming`; buckets 0.1..2400 vs 0.3..7680 |
| sglang:inter_token_latency_seconds | Histogram | token-weighted ITL | vllm:inter_token_latency_seconds, vllm:time_per_output_token_seconds | SGLang has one metric where the simulator emits two identical ones; buckets 0.002..8 vs 0.01..80 |
| sglang:request_time_per_output_token_seconds | Histogram | `(finish - first_token) / (completion_tokens - 1)` | vllm:request_time_per_output_token_seconds | Identical name. Excludes the first token, requests with fewer than two output tokens, aborts, and PD prefill workers; the simulator uses `decode_duration / generation_tokens` with no exclusions. Extra `is_streaming` label |
| sglang:queue_time_seconds | Histogram | queue wait | vllm:request_queue_time_seconds | Name only |
| sglang:per_stage_req_latency_seconds{stage} | Histogram | per-stage latency | vllm:request_prefill_time_seconds, vllm:request_decode_time_seconds, vllm:request_inference_time_seconds | One metric with a `stage` label replaces three separate metrics; exponential buckets, 1ms to ~1191s |
| sglang:cached_tokens_total{cache_source} | Counter | cached prompt tokens by tier (device/host/storage) | vllm:prefix_cache_hits_total | Tiered by `cache_source`; vLLM is single-tier |
| sglang:cache_hit_rate | Gauge | prefix cache hit rate | derived from vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total | SGLang publishes the ratio directly as a gauge instead of two counters |
| sglang:page_size, sglang:num_pages, sglang:context_len | Gauge | page size in tokens, page count, max context | vllm:cache_config_info | Plain gauge values, not an info gauge with label-encoded config. No `cache_dtype` or CPU-block equivalent |
| sglang:lora_pool_slots_used, sglang:lora_pool_slots_total, sglang:lora_pool_utilization | Gauge | occupied slots, `max_loras_per_batch`, ratio | vllm:lora_requests_info | No adapter names at all, only slot counts. vLLM encodes running and waiting adapter names in labels with a timestamp as the value |
| none | — | — | vllm:request_params_max_tokens, vllm:max_num_generation_tokens | No SGLang equivalent |
| none | — | — | vllm:prefix_cache_queries_total | No direct counter; the denominator must be derived from `sglang:prefill_effective_tokens_total{mode}` |

## 3. SGLang metrics with no vLLM counterpart

| SGLang metric | Type | Value reported |
|---|---|---|
| sglang:gen_throughput | Gauge | generation throughput, tokens/s |
| sglang:num_used_tokens | Gauge | absolute used token count (vLLM exposes only the fraction) |
| sglang:max_total_num_tokens, sglang:max_total_num_tokens_swa | Gauge | KV pool capacity in tokens |
| sglang:kv_available_tokens, sglang:kv_evictable_tokens, sglang:kv_used_tokens | Gauge | free / radix-cached / actively used token slots, which separates reclaimable from pinned capacity |
| sglang:swa_*_tokens, sglang:mamba_*_tokens | Gauge | the same breakdown for SWA and Mamba pools (hybrid models) |
| sglang:utilization, sglang:fwd_occupancy | Gauge | scheduler utilization, forward-pass GPU occupancy percentage |
| sglang:new_token_ratio | Gauge | scheduler's new-token admission ratio |
| sglang:max_running_requests_under_SLO | Gauge | admission ceiling the scheduler believes holds SLO |
| sglang:num_retracted_reqs, sglang:num_paused_reqs | Gauge | requests retracted under memory pressure, requests paused by async weight sync |
| sglang:num_retracted_requests_total, sglang:num_retracted_input_tokens_total, sglang:num_retracted_output_tokens_total | Counter | cumulative retraction cost |
| sglang:num_grammar_queue_reqs, sglang:grammar_compilation_time_seconds, sglang:num_grammar_cache_hit_total, sglang:num_grammar_aborted_total, sglang:num_grammar_timeout_total, sglang:num_grammar_total | Gauge/Histogram/Counter | structured-output pipeline health |
| sglang:num_so_requests_total | Counter | structured-output request count |
| sglang:num_prefill_bootstrap_queue_reqs, sglang:num_prefill_inflight_queue_reqs, sglang:num_decode_prealloc_queue_reqs, sglang:num_decode_transfer_queue_reqs, sglang:num_decode_host_receive_queue_reqs | Gauge | PD-disaggregation queue depths, per-stage visibility unavailable from vLLM |
| sglang:pending_prealloc_token_usage | Gauge | tokens promised to decode but not yet allocated |
| sglang:kv_transfer_latency_ms, sglang:kv_transfer_speed_gb_s, sglang:kv_transfer_total_mb, sglang:kv_transfer_bootstrap_ms, sglang:kv_transfer_alloc_ms | Histogram | KV transfer performance between prefill and decode |
| sglang:num_bootstrap_failed_reqs_total, sglang:num_transfer_failed_reqs_total, sglang:num_prefill_retries_total | Counter | PD failure counters |
| sglang:spec_accept_length, sglang:spec_accept_rate, sglang:spec_num_steps, sglang:spec_num_draft_tokens, sglang:spec_verify_calls_total | Gauge/Counter | speculative decoding effectiveness |
| sglang:hicache_host_used_tokens, sglang:hicache_host_total_tokens, sglang:hicache_backup_*, sglang:load_back_*, sglang:evicted_tokens_total, sglang:eviction_duration_seconds | Gauge/Counter/Histogram | hierarchical (L2 host, L3 storage) cache traffic and eviction |
| sglang:prefetched_tokens_total, sglang:storage_prefetch_hit_tokens_total, sglang:storage_prefetch_unfulfilled_tokens_total{reason}, sglang:backuped_tokens_total | Counter | L3 storage prefetch outcomes |
| sglang:prefill_effective_tokens_total{mode}, sglang:realtime_tokens_total{mode} | Counter | token accounting split into prefill_compute / prefill_cache / decode, with retraction re-counts excluded |
| sglang:weight_memory_usage_gb, sglang:kv_cache_memory_usage_gb, sglang:graph_memory_usage_gb, sglang:startup_available_gpu_memory_gb | Gauge | GPU memory breakdown |
| sglang:estimated_flops_per_gpu_total, sglang:estimated_read_bytes_per_gpu_total, sglang:estimated_write_bytes_per_gpu_total | Counter | MFU inputs, requires `--enable-mfu-metrics` |
| sglang:forward_execution_seconds_total, sglang:scheduler_idle_seconds_total, sglang:scheduler_process_cpu_seconds_total, sglang:scheduler_stage_seconds_total | Counter | scheduler and GPU time accounting |
| sglang:startup_time_seconds{phase}, sglang:startup_cuda_graph_time_seconds{phase}, sglang:weight_load_duration_seconds{source} | Gauge | lifecycle timings |
| sglang:is_cuda_graph, sglang:cuda_graph_passes_total{mode} | Gauge/Counter | CUDA-graph coverage |
| sglang:num_streaming_sessions, sglang:streaming_session_held_tokens, sglang:num_unique_running_routing_keys | Gauge | session and routing-key cardinality |
| sglang:encoder_* (14 metrics) | Gauge/Counter/Histogram | multimodal encoder cache and latency, by `modality` |
| sglang:eplb_balancedness, sglang:eplb_gpu_physical_count | Summary/Histogram | MoE expert-parallel balance |

## 4. What llm-d-router consumes

The router reads five metrics per engine and never touches latency or token histograms. The mapping
lives in the core metrics extractor at
`pkg/epp/framework/plugins/datalayer/extractor/metrics/factories.go:88-106`.

| Router attribute | vLLM spec | SGLang spec |
|---|---|---|
| `WaitingQueueSize` | vllm:num_requests_waiting | sglang:num_queue_reqs |
| `RunningRequestsSize` | vllm:num_requests_running | sglang:num_running_reqs |
| `KVCacheUsagePercent` | vllm:kv_cache_usage_perc | sglang:token_usage |
| `MaxActiveModels`, `ActiveModels`, `WaitingModels` | vllm:lora_requests_info | empty, LoRA routing is disabled for SGLang |
| `CacheBlockSize`, `CacheNumBlocks` | vllm:cache_config_info, labels `block_size` / `num_gpu_blocks` | sglang:cache_config_info, labels `page_size` / `num_pages` |

Supporting code, all paths relative to the `llm-d-router@v0.10.0` module root:

- `pkg/epp/framework/plugins/datalayer/extractor/metrics/extractor.go:105-183` — extraction logic,
  including the empty-spec skip and the cache-config label-name override.
- `pkg/epp/framework/plugins/datalayer/extractor/metrics/mapping_registry.go:29` — engine selection
  by the pod label `llm-d.ai/engine-type`; `factories.go:157-176` defaults it to `vllm`.
- `pkg/epp/framework/plugins/datalayer/extractor/metrics/loraspec.go:47-52` — LoRA label parsing
  (`running_lora_adapters`, `waiting_lora_adapters`, `max_lora`).
- `pkg/epp/server/options.go:143-147` — legacy GIE top-level flags, vLLM-only, no SGLang counterpart.
- `pkg/epp/framework/plugins/datalayer/extractor/metrics/README.md` — plugin documentation.

### Two implications for the simulator

1. `sglang:cache_config_info` does not exist upstream. The router expects an info-style gauge with
   `page_size` / `num_pages` labels, but SGLang `main` emits `sglang:page_size` and
   `sglang:num_pages` as independent gauge values. The router's `CacheBlockSizeSpec` /
   `CacheNumBlocksSpec` fields, used for trtllm, are the shape that matches the real engine. A
   simulator that emits what SGLang emits leaves the router's built-in SGLang cache-config
   extraction finding nothing; a simulator that emits `cache_config_info` matches the router but not
   the engine.
2. The router's five-metric set is the minimum viable scope for routing and EPP integration testing:
   `num_queue_reqs`, `num_running_reqs`, `token_usage`, plus whichever cache-config shape gets
   settled. Everything else in section 2 serves load-testing and dashboard fidelity rather than
   router integration.
