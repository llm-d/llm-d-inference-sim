# Prometheus metrics

The simulator supports a subset of standard vLLM Prometheus metrics.<br>
These metrics are exposed via the `/metrics` HTTP REST endpoint. 

Currently supported are the following metrics:
| Metric | Description |
|---|---|
| vllm:kv_cache_usage_perc | The fraction of KV-cache blocks currently in use (from 0 to 1) |
| vllm:lora_requests_info | Running stats on LoRA requests |
| vllm:num_requests_running | Number of requests currently running on GPU |
| vllm:num_requests_waiting | Prometheus metric for the number of queued requests |
| vllm:e2e_request_latency_seconds | Histogram of end to end request latency in seconds |
| vllm:request_inference_time_seconds | Histogram of time spent in RUNNING phase for request |
| vllm:request_queue_time_seconds | Histogram of time spent in WAITING phase for request |
| vllm:request_prefill_time_seconds | Histogram of time spent in PREFILL phase for request |
| vllm:request_decode_time_seconds | Histogram of time spent in DECODE phase for request |
| vllm:time_to_first_token_seconds | Histogram of time to first token in seconds |
| vllm:time_per_output_token_seconds | Histogram of time per output token in seconds |
| vllm:inter_token_latency_seconds | Histogram of inter-token latency in seconds |
| vllm:request_generation_tokens | Number of generation tokens processed |
| vllm:generation_tokens_total	 | Total number of generated tokens. |
| vllm:max_num_generation_tokens | Maximum number of requested generation tokens. Currently same as `vllm:request_generation_tokens` since always only one choice is returned |
| vllm:request_params_max_tokens | Histogram of the max_tokens request parameter | 
| vllm:request_prompt_tokens | Number of prefill tokens processed |
| vllm:prompt_tokens_total	 | Total number of prompt tokens processed |
| vllm:request_success_total | Count of successfully processed requests |
