# API Endpoints
The simulator supports both HTTP (OpenAI-compatible) and gRPC (vLLM-compatible) interfaces on the same port.

## HTTP Endpoints
Currently, it supports a partial OpenAI-compatible API:
- `/v1/chat/completions` 
- `/v1/completions` 
- `/v1/models`

In addition, a set of the vLLM HTTP endpoints are suppored:

| Endpoint | Description |
|---|---|
| /v1/load_lora_adapter   | Simulates the dynamic registration of a LoRA adapter |
| /v1/unload_lora_adapter | Simulates the dynamic unloading and unregistration of a LoRA adapter |
| /tokenize               | Tokenizes input text and returns token information |
| /sleep                  | Puts the simulator into sleep mode (requires `enable-sleep-mode` flag) |
| /wake_up                | Wakes up the simulator from sleep mode |
| /is_sleeping            | Checks if the simulator is currently in sleep mode |
| /metrics                | Exposes Prometheus metrics (see metrics table below) |
| /health                 | Standard health check endpoint |
| /ready                  | Standard readiness endpoint |


## gRPC Endpoints
The simulator implements the `vllm.grpc.engine.VllmEngine` service definition. 
It is available on the same port as the HTTP server.
Only `Generate` method is currenlty implemented. It submits a generation request. Supports streaming responses and standard sampling parameters.


