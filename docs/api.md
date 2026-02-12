# API Endpoints
The simulator supports both HTTP (OpenAI-compatible) and gRPC (vLLM-compatible) interfaces on the same port.<br>
HTTP requests are routed to the HTTP server, and HTTP2 requests are routed to the gRPC server. 

## HTTP Endpoints
Currently, the simulator supports a partial OpenAI-compatible API:
- `/v1/chat/completions` 
- `/v1/completions` 
- `/v1/models`

For details see the [HTTP Endpoints Guide](http-enpdpoints.md)

In addition, a set of the vLLM HTTP endpoints are suppored:

| Endpoint | Description |
|---|---|
| /v1/load_lora_adapter   | Simulates the dynamic registration of a LoRA adapter |
| /v1/unload_lora_adapter | Simulates the dynamic unloading and unregistration of a LoRA adapter |
| /tokenize               | Tokenizes input text and returns token information |
| /sleep                  | Puts the simulator into sleep mode (requires `enable-sleep-mode` flag) |
| /wake_up                | Wakes up the simulator from sleep mode |
| /is_sleeping            | Checks if the simulator is currently in sleep mode |
| /metrics                | Exposes Prometheus metrics (see [Metrics Guide](metrics.md)) |
| /health                 | Standard health check endpoint |
| /ready                  | Standard readiness endpoint |


## gRPC Endpoints
The simulator implements the `vllm.grpc.engine.VllmEngine` service definition. 
It is available on the same port as the HTTP server.
Only `Generate` and `GetModelInfo` methods are currenlty implemented. <br>
The `Generate` submits a generation request. Supports streaming responses and standard sampling parameters.<br>
The `GetModelInfo` retrieves metadata about the currently loaded model.


