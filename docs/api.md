# API Endpoints
The simulator supports both HTTP (OpenAI-compatible) and gRPC (vLLM-compatible) interfaces on the same port.<br>
HTTP requests are routed to the HTTP server, and HTTP2 requests are routed to the gRPC server. 

## HTTP Endpoints
Currently, the simulator supports a partial OpenAI-compatible API:
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/models`

For details see the [HTTP Endpoints Guide](http-enpdpoints.md)

In addition, a set of the vLLM HTTP endpoints are suppored:

| Endpoint | Description |
|---|---|
| /inference/v1/generate  | vLLM-specific generation endpoint |
| /v1/load_lora_adapter   | Simulates the dynamic registration of a LoRA adapter |
| /v1/unload_lora_adapter | Simulates the dynamic unloading and unregistration of a LoRA adapter |
| /tokenize               | Tokenizes input text and returns token information |
| /sleep                  | Puts the simulator into sleep mode. Requires both the `enable-sleep-mode` flag and the `VLLM_SERVER_DEV_MODE=1` environment variable; otherwise the request is accepted (HTTP 200) but ignored. |
| /wake_up                | Wakes up the simulator from sleep mode. Accepts an optional `?tags=kv_cache` query parameter; when set (or when no `tags` parameter is provided), the KV cache is re-activated on wake-up. Any other `tags` value wakes up the simulator without re-activating the KV cache. |
| /is_sleeping            | Checks if the simulator is currently in sleep mode |
| /metrics                | Exposes Prometheus metrics (see [Metrics Guide](metrics.md)) |
| /health                 | Standard health check endpoint |
| /health/ready           | Ensure the GPU is "actually working" before allowing the system to send it live traffic |

The simulator also provides a `POST /fake_metrics` endpoint that supports partial updates to [fake metric](configuration.md#fake-metrics) values at runtime. This endpoint is specific to the simulator and is available only when a `--fake-metrics` configuration is provided at startup. The request body must be a JSON object containing the metrics to update; any metrics not specified are left unchanged.

### `/admin/config`

The simulator exposes `GET` and `POST` on `/admin/config` for runtime configuration introspection and partial updates. This endpoint is simulator-specific and is intended for adjusting behavior (currently only failure injection) during a test run without restarting the process.

- **`GET /admin/config`** returns the current configuration as JSON. Internal helper fields (`LoraModulesString`, `LorasString`) are stripped, `LoraModules` is exposed as `lora-modules`, and the per-rank `port` is omitted when running with `--data-parallel-size > 1`.
- **`POST /admin/config`** applies a partial JSON update and returns the new configuration. The request body must be a JSON object whose keys are a subset of the admin-configurable fields:
  - `failure-injection-rate` — integer in `[0, 100]`
  - `failure-types` — array of strings from `rate_limit`, `invalid_api_key`, `context_length`, `server_error`, `invalid_request`, `model_not_found`

  Bodies containing any other field, or values that fail validation, are rejected with `400 Bad Request` and the configuration is left unchanged. Updates are atomic and serialized: concurrent in-flight requests observe either the previous or the new configuration in full, never a mix.

  Example:
  ```bash
  # Enable 100% rate-limit failures
  curl -X POST http://localhost:8000/admin/config \
    -H 'Content-Type: application/json' \
    -d '{"failure-injection-rate": 100, "failure-types": ["rate_limit"]}'

  # Disable failures again
  curl -X POST http://localhost:8000/admin/config \
    -H 'Content-Type: application/json' \
    -d '{"failure-injection-rate": 0}'
  ```

### Request headers

In addition to standard HTTP headers, the simulator recognizes a few simulator-specific request headers:

| Header | Description |
|---|---|
| `X-Request-Id` | Read on incoming requests to all generation endpoints (including `/v1/embeddings`) and echoed back as a response header when `--enable-request-id-headers` is set. Used to correlate client requests with server logs. |
| `X-Return-Error` | Deterministic failure injection. When set to a numeric HTTP status code (e.g. `429`, `500`), the simulator immediately returns a synthetic error response with that status code, bypassing the probabilistic `--failure-injection-rate` mechanism. A non-integer value yields HTTP 400. Honored on `/v1/chat/completions`, `/v1/completions`, and `/inference/v1/generate`; not honored by `/v1/embeddings`. |
| `X-Cache-Threshold-Finish-Reason` | Deterministic forcing of the `cache_threshold` finish reason. When set to `true`, the response is forced to use the `cache_threshold` finish reason regardless of the actual cache hit rate or the configured `cache_hit_threshold` / `global-cache-hit-threshold` values. Any other value (including `false`, missing, or unparseable) leaves the normal cache-threshold logic in place. The header is parsed for all generation endpoints, but only takes effect on `/v1/chat/completions` and `/v1/completions` (the other endpoints' request types do not implement the cache-threshold override). |

## gRPC Endpoints
The simulator implements the `vllm.grpc.engine.VllmEngine` service definition. 
It is available on the same port as the HTTP server.
Only `Generate` and `GetModelInfo` methods are currenlty implemented. <br>
The `Generate` submits a generation request. Supports streaming responses and standard sampling parameters.<br>
The `GetModelInfo` retrieves metadata about the currently loaded model.


