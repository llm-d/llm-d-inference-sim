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
| vllm:request_max_num_generation_tokens | Maximum number of requested generation tokens. Currently same as `vllm:request_generation_tokens` since always only one choice is returned |
| vllm:e2e_request_latency_seconds | Histogram of end to end request latency in seconds |
| vllm:request_inference_time_seconds | Histogram of time spent in RUNNING phase for request |
| vllm:request_queue_time_seconds | Histogram of time spent in WAITING phase for request |
| vllm:request_prefill_time_seconds | Histogram of time spent in PREFILL phase for request |
| vllm:request_decode_time_seconds | Histogram of time spent in DECODE phase for request |
| vllm:request_time_per_output_token_seconds | Histogram of time_per_output_token_seconds per request |
| vllm:time_to_first_token_seconds | Histogram of time to first token in seconds |
| vllm:time_per_output_token_seconds | Histogram of time per output token in seconds |
| vllm:inter_token_latency_seconds | Histogram of inter-token latency in seconds |
| vllm:lora_requests_info | Running stats on LoRA requests. Exposed only when LoRA modules are configured, and only after the first LoRA request |
| vllm:kv_cache_usage_perc | The fraction of KV-cache blocks currently in use (from 0 to 1) |
| vllm:cache_config_info | Information of the LLMEngine CacheConfig |
| vllm:prefix_cache_hits_total | Prefix cache hits, in terms of number of cached tokens |
| vllm:prefix_cache_queries_total | Prefix cache queries, in terms of number of queried tokens |
| process_* | Standard process metrics such as `process_start_time_seconds`, as vLLM exports them |

A request is aborted when its response is no longer read: for a streamed HTTP response, at the first failed write to the client, for example after the client disconnected; for a gRPC `Generate` call, when the call is cancelled; and for the other sub-requests of a request (n > 1, a prompt array in `/v1/completions`, or several Decision API questions) when one of them fails. A non-streamed HTTP request runs to completion even if its client disconnects.

A request aborted before a worker starts it is dropped when a worker picks it up. It never enters `vllm:num_requests_running` or the KV cache; if it was queued, it stays in `vllm:num_requests_waiting` until then and `vllm:request_queue_time_seconds` is its only observation.

Once a running request is aborted, it stops at once, also during its time-to-first-token, inter-token or image wait: it generates no further tokens, leaves `vllm:num_requests_running` and releases its KV cache blocks. It is counted like a failed request: `vllm:e2e_request_latency_seconds`, `vllm:request_inference_time_seconds` and the queue, prefill, time-to-first-token and inter-token latencies recorded before the abort are observed, while `vllm:request_success_total`, `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`, `vllm:request_params_max_tokens` and `vllm:request_max_num_generation_tokens` are not updated (`vllm:request_decode_time_seconds` and `vllm:request_time_per_output_token_seconds` are observed only when it is aborted during the image wait).

## SGLang

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

The metrics of a feature this engine does not simulate are absent rather than reported as zero. The
KV-cache group (`token_usage`, `cache_hit_rate`, `cached_tokens_total`, `num_used_tokens`, `kv_*`,
`max_total_num_tokens`, `page_size`, `num_pages`, `swa_*`, `mamba_*`, `hicache_*`) follows the KV-cache
simulation, and the LoRA, speculative-decoding, grammar, hierarchical-cache, encoder-cache, MoE, and
prefill/decode-disaggregation groups follow theirs. `sglang:gen_throughput` is a rolling token-rate the
simulator does not compute, `sglang:num_aborted_requests_total` counts client aborts, which the
simulator does not distinguish, and `sglang:func_latency_seconds` times the engine's own Python entry
points. The scheduler's own instrumentation (`per_stage_req_latency_seconds`, `utilization`,
`fwd_occupancy`, `new_token_ratio`, `is_cuda_graph`, the startup and GPU-memory gauges) measures a
runtime the simulator has none of.

[Fake metrics](configuration.md) are a vLLM-engine feature; a configuration asking for them under SGLang
is rejected.
