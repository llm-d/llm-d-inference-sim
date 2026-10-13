# Prometheus metrics

The simulator supports a subset of standard vLLM Prometheus metrics.<br>
These metrics are exposed via the `/metrics` HTTP REST endpoint. 

Metric names are engine-specific (see [Engines](engine-backends.md)). The [vLLM engine](#vllm) and the [SGLang engine](#sglang) each expose their own set.

## vLLM

Currently supported are the following metrics:
| Metric | Description |
|---|---|
| vllm:num_requests_running | Number of requests currently running on GPU |
| vllm:num_requests_waiting | Prometheus metric for the number of queued requests |
| vllm:request_params_max_tokens | Histogram of the max_tokens request parameter | 
| vllm:request_prompt_tokens | Number of prefill tokens processed |
| vllm:prompt_tokens_total	 | Total number of prompt tokens processed |
| vllm:request_success_total | Count of successfully processed requests |
| vllm:request_generation_tokens | Number of generation tokens processed |
| vllm:generation_tokens_total	 | Total number of generated tokens. |
| vllm:max_num_generation_tokens | Maximum number of requested generation tokens. Currently same as `vllm:request_generation_tokens` since always only one choice is returned |
| vllm:e2e_request_latency_seconds | Histogram of end to end request latency in seconds |
| vllm:request_inference_time_seconds | Histogram of time spent in RUNNING phase for request |
| vllm:request_queue_time_seconds | Histogram of time spent in WAITING phase for request |
| vllm:request_prefill_time_seconds | Histogram of time spent in PREFILL phase for request |
| vllm:request_decode_time_seconds | Histogram of time spent in DECODE phase for request |
| vllm:request_time_per_output_token_seconds | Histogram of time_per_output_token_seconds per request |
| vllm:time_to_first_token_seconds | Histogram of time to first token in seconds |
| vllm:time_per_output_token_seconds | Histogram of time per output token in seconds |
| vllm:inter_token_latency_seconds | Histogram of inter-token latency in seconds |
| vllm:lora_requests_info | Running stats on LoRA requests |
| vllm:kv_cache_usage_perc | The fraction of KV-cache blocks currently in use (from 0 to 1) |
| vllm:cache_config_info | Information of the LLMEngine CacheConfig |
| vllm:prefix_cache_hits_total | Prefix cache hits, in terms of number of cached tokens |
| vllm:prefix_cache_queries_total | Prefix cache queries, in terms of number of queried tokens |
| process_* | Standard process metrics such as `process_start_time_seconds`, as vLLM exports them |

## SGLang

Metrics are opt-in under this engine, as they are in SGLang: `/metrics` is served only when
[`--enable-metrics`](configuration.md#metrics) is given, and a scrape without it gets a 404 rather than
an empty body.

Currently supported are the following metrics:
| Metric | Description |
|---|---|
| sglang:num_running_reqs | The number of running requests |
| sglang:num_queue_reqs | The number of requests in the waiting queue |
| sglang:context_len | Maximum context length |
| sglang:queue_time_seconds | Histogram of queueing time in seconds |
| sglang:time_to_first_token_seconds | Histogram of time to first token in seconds |
| sglang:inter_token_latency_seconds | Histogram of inter-token latency in seconds |
| sglang:request_time_per_output_token_seconds | Per-request TPOT in seconds: `(finished_time - first_token_time) / (completion_tokens - 1)` |
| sglang:e2e_request_latency_seconds | Histogram of end-to-end request latency in seconds |
| sglang:prompt_tokens_total | Number of prefill tokens processed |
| sglang:generation_tokens_total | Number of generation tokens processed |
| sglang:num_requests_total | Number of requests processed |
| sglang:prompt_tokens_histogram | Histogram of prompt token length |
| sglang:uncached_prompt_tokens_histogram | Histogram of uncached (compute) prompt token length |
| sglang:generation_tokens_histogram | Histogram of generation token length |
| sglang:token_usage | The token usage |
| sglang:num_used_tokens | The number of used tokens |
| sglang:kv_used_tokens | Number of actively used token slots in the KV cache pool |
| sglang:kv_available_tokens | Number of free token slots in the KV cache pool |
| sglang:max_total_num_tokens | Maximum total number of tokens in the KV cache pool |
| sglang:page_size | KV cache page size in tokens |
| sglang:num_pages | Number of KV cache pages |
| sglang:cache_hit_rate | The prefix cache hit rate |
| sglang:cached_tokens_total | Number of cached prompt tokens, labeled by `cache_source` |

The bucket boundaries of the three request-latency histograms are the defaults of SGLang's
`--bucket-time-to-first-token`, `--bucket-inter-token-latency`, and `--bucket-e2e-request-latency`; the
simulator does not accept the flags that override them. `sglang:queue_time_seconds` and the three
token-length histograms take SGLang's fixed boundaries for them, so unlike the vLLM engine's token
histograms they do not scale with `max-model-len`.

Every metric carries a `model_name` label, and the six per-request metrics are additionally split by
`is_streaming`, as SGLang splits them. `engine_type` and the `tp_rank`/`pp_rank`/`moe_ep_rank` labels real
SGLang adds are left out: the simulator models neither disaggregation roles nor device topology, so the
labels would carry one constant value. The vLLM surface above is reduced the same way, dropping real
vLLM's `engine`.

A request that fails contributes to no metric but `sglang:num_running_reqs`, since SGLang observes a
request's latency and token counts only once it has finished. `sglang:request_time_per_output_token_seconds`
is narrower still: a request that generated a single token, or that stopped for any reason other than
`stop` or `length`, reports no time per output token, as it reports none upstream.
`sglang:uncached_prompt_tokens_histogram` equals `sglang:prompt_tokens_histogram` while no KV cache is
simulated, since no prompt token can be served from a prefix cache.

`sglang:queue_time_seconds` is observed when a request leaves the waiting queue, so it stays absent
while a free worker is always available to take a request immediately.

### The KV-cache metrics

This engine simulates no KV cache, so the metrics measuring one report an empty cache of the configured
geometry: `token_usage`, `num_used_tokens`, `kv_used_tokens`, and `cache_hit_rate` read zero,
`kv_available_tokens` reads the full capacity, and `cached_tokens_total` stays absent. The ones describing
the cache rather than measuring it carry real values: `max_total_num_tokens` is `kv-cache-size` x
`block-size`, `page_size` is `block-size`, and `num_pages` is `kv-cache-size`. The handlers behind the
measurements are wired, so they begin reporting as soon as the engine gains a KV cache, without a change
here. `cached_tokens_total` is labeled `cache_source="total"`, SGLang's own fallback label when it has no
per-tier breakdown to report, which the simulator's single-tier block cache never will.

### Fake metrics

[Fake metrics](configuration.md#fake-metrics-1) replace every load metric of this surface: the request
gauges, the token-usage gauges, the prefix-cache gauge and counter, the three token and request counters,
and all eight histograms. The keys are this engine's own, since the fakeable set mirrors its own metrics,
and the six metrics SGLang labels by `is_streaming` take both series, so a scrape can break down by the
label the way a real one does.

The gauges describing the configuration are reported from the configuration either way:
`sglang:context_len`, `sglang:page_size`, `sglang:num_pages` and `sglang:max_total_num_tokens`. Fake
metrics do not replace them, since they report the geometry the simulator was given rather than anything a
request changes.

A metric the configuration names nothing for reports nothing: real and fake metrics are never served
together, so a scrape carries what the configuration sets and the gauges above, and no other series at
all. For a histogram split by `is_streaming`, that holds per series.

A faked histogram's samples are observed at their bucket's upper boundary, so its `_sum` follows from
which buckets are filled rather than being configurable on its own. No total is derived from a histogram
the way the vLLM engine derives its token counters: SGLang splits `sglang:prompt_tokens_total` by
`is_streaming` but reports `sglang:prompt_tokens_histogram` without the label, so neither carries a
per-series sum for the other, and the totals are set on their own.
