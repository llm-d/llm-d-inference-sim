# API Endpoints
The simulator supports both HTTP (OpenAI-compatible) and gRPC (vLLM-compatible) interfaces on the same port.<br>
HTTP requests are routed to the HTTP server, and HTTP2 requests are routed to the gRPC server. 

## HTTP Endpoints
Currently, the simulator supports a partial OpenAI-compatible API:
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/responses`
- `/v1/embeddings`
- `/v1/models`

For details see the [HTTP Endpoints Guide](http-endpoints.md)

In addition, a set of the vLLM HTTP endpoints are supported:

| Endpoint | Description |
|---|---|
| /v1/messages            | Anthropic Messages API, converted to the chat completions format internally |
| /inference/v1/generate  | vLLM-specific generation endpoint |
| /v1/load_lora_adapter   | Simulates the dynamic registration of a LoRA adapter |
| /v1/unload_lora_adapter | Simulates the dynamic unloading and unregistration of a LoRA adapter |
| /tokenize               | Tokenizes input text and returns token information |
| /v1/completions/render       | Returns the token IDs for a `/v1/completions` request body without generating a response |
| /v1/chat/completions/render  | Returns the token IDs (and multimodal features when applicable) for a `/v1/chat/completions` request body without generating a response |
| /sleep                  | Puts the simulator into sleep mode. Requires both the `enable-sleep-mode` flag and the `VLLM_SERVER_DEV_MODE=1` environment variable; otherwise the request is accepted (HTTP 200) but ignored. |
| /wake_up                | Wakes up the simulator from sleep mode. Accepts an optional `?tags=kv_cache` query parameter; when set (or when no `tags` parameter is provided), the KV cache is re-activated on wake-up. Any other `tags` value wakes up the simulator without re-activating the KV cache. |
| /is_sleeping            | Checks if the simulator is currently in sleep mode |
| /metrics                | Exposes Prometheus metrics (see [Metrics Guide](metrics.md)) |
| /health                 | Standard health check endpoint |
| /health/ready           | Ensure the GPU is "actually working" before allowing the system to send it live traffic |

> **Deprecated:** the simulator also provides a `POST /fake_metrics` endpoint that accepts a [fake metric](configuration.md#fake-metrics) partial-update body directly (only available when started with a `--fake-metrics` configuration). The body is a JSON object containing the metrics to update; unspecified metrics are left unchanged. This endpoint is preserved for backward compatibility and will be removed in release v0.12.0; new callers should use `POST /admin/config` with a `fake-metrics` field instead.

### `/admin/config`

The simulator exposes `GET` and `POST` on `/admin/config` for runtime configuration introspection and partial updates. This endpoint is simulator-specific and is intended for adjusting behavior (currently failure injection, fake metrics, and request latencies) during a test run without restarting the process.

- **`GET /admin/config`** returns the current configuration as JSON. Internal helper fields (`LoraModulesString`, `LorasString`) are stripped, `LoraModules` is exposed as `lora-modules`, and the per-rank `port` is omitted when running with `--data-parallel-size > 1`.
- **`POST /admin/config`** applies a partial JSON update and returns the new configuration. The request body must be a JSON object whose keys are a subset of the admin-configurable fields:
  - `failure-injection-rate` — integer in `[0, 100]`
  - `failure-types` — array of strings from `rate_limit`, `invalid_api_key`, `context_length`, `server_error`, `invalid_request`, `model_not_found`
  - `image-emission-rate` — integer in `[0, 100]`; probability that each `/v1/chat/completions` request emits a synthetic image chunk when the simulator is running with `--omni`. 0 disables the rate mechanism (the `X-Send-Image` header still works); 100 means every request gets an image.
  - `fake-metrics` — partial update of [fake metric](configuration.md#fake-metrics) values. The value is itself a JSON object containing only the metrics to change; any metrics not specified are left unchanged. Available only when the simulator was started with a `--fake-metrics` configuration.

    Absent fields and fields explicitly set to `null` are treated identically — both mean "leave unchanged". To clear a metric whose value is a slice or map (e.g. `ttft-buckets-values`, `request-success-total`), send an empty value: `[]` or `{}`. There is no way to clear a scalar metric (e.g. `running-requests`, `total-prompt-tokens`) via partial update — assign a new value instead.
  - Latency-related fields: `time-to-first-token`, `time-to-first-token-std-dev`, `inter-token-latency`, `inter-token-latency-std-dev`, `kv-cache-transfer-latency`, `kv-cache-transfer-latency-std-dev`, `prefill-overhead`, `prefill-time-per-token`, `prefill-time-std-dev`, `kv-cache-transfer-time-per-token`, `kv-cache-transfer-time-std-dev`, `time-factor-under-load` (float), `latency-calculator` (string; same accepted values as the `--latency-calculator` flag). Duration fields accept a Go duration string (e.g. `"250ms"`, `"1s"`). The same validation rules apply as at startup (no negative values; std-dev ≤ 30 % of base; `time-factor-under-load` ≥ 1.0). Updates take effect on subsequent requests.

  Bodies containing any other field, or values that fail validation, are rejected with `400 Bad Request` and the configuration is left unchanged. Updates are atomic and serialized: concurrent in-flight requests observe either the previous or the new configuration in full, never a mix.

  Examples:
  ```bash
  # Enable 100% rate-limit failures
  curl -X POST http://localhost:8000/admin/config \
    -H 'Content-Type: application/json' \
    -d '{"failure-injection-rate": 100, "failure-types": ["rate_limit"]}'

  # Disable failures again
  curl -X POST http://localhost:8000/admin/config \
    -H 'Content-Type: application/json' \
    -d '{"failure-injection-rate": 0}'

  # Update a subset of fake metrics
  curl -X POST http://localhost:8000/admin/config \
    -H 'Content-Type: application/json' \
    -d '{"fake-metrics": {"running-requests": 7, "kv-cache-usage": 0.5}}'

  # Slow down responses: 500 ms TTFT, 20 ms inter-token latency
  curl -X POST http://localhost:8000/admin/config \
    -H 'Content-Type: application/json' \
    -d '{"time-to-first-token": "500ms", "inter-token-latency": "20ms"}'

  # Emit synthetic images in 30% of omni-mode chat completion responses
  curl -X POST http://localhost:8000/admin/config \
    -H 'Content-Type: application/json' \
    -d '{"image-emission-rate": 30}'
  ```


### Request headers

In addition to standard HTTP headers, the simulator recognizes a few simulator-specific request headers:

| Header | Description |
|---|---|
| `X-Request-Id` | Read on incoming requests to all generation endpoints (including `/v1/embeddings`) and echoed back as a response header when `--enable-request-id-headers` is set. Used to correlate client requests with server logs. |
| `X-Return-Error` | Deterministic failure injection. When set to a numeric HTTP status code (e.g. `429`, `500`), the simulator immediately returns a synthetic error response with that status code, bypassing the probabilistic `--failure-injection-rate` mechanism. A non-integer value yields HTTP 400. Honored on `/v1/chat/completions`, `/v1/completions`, and `/inference/v1/generate`; not honored by `/v1/embeddings`. |
| `X-Cache-Threshold-Finish-Reason` | Deterministic forcing of the `cache_threshold` finish reason. When set to `true`, the response is forced to use the `cache_threshold` finish reason regardless of the actual cache hit rate or the configured `cache_hit_threshold` / `global-cache-hit-threshold` values. Any other value (including `false`, missing, or unparsable) leaves the normal cache-threshold logic in place. The header is parsed for all generation endpoints, but only takes effect on `/v1/chat/completions` and `/v1/completions` (the other endpoints' request types do not implement the cache-threshold override). |
| `X-Send-Image` | Image output for omni mode. Requires the simulator to be started with `--omni`. When set to `true`, a synthetic 1×1 transparent PNG (`data:image/png;base64,…`) is appended to `/v1/chat/completions` responses. In non-streaming responses the assistant message `content` becomes a structured array — a `text` block with the generated tokens followed by an `image_url` block. In streaming responses an additional SSE chunk with `"modality":"image"` is emitted after the token stream, carrying the image in its delta `content`. Has no effect when `--omni` is not set or when the header value is not parseable as `true`. Images can also be emitted probabilistically without this header via the `--image-emission-rate` flag (or the `image-emission-rate` field on `POST /admin/config`). |

## gRPC Endpoints
The simulator implements the `vllm.grpc.engine.VllmEngine` service definition. 
It is available on the same port as the HTTP server.
Only `Generate` and `GetModelInfo` methods are currently implemented. <br>
The `Generate` submits a generation request. Supports streaming responses and standard sampling parameters.<br>
The `GetModelInfo` retrieves metadata about the currently loaded model.


