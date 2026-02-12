[![Go Report Card](https://goreportcard.com/badge/github.com/llm-d/llm-d-inference-sim)](https://goreportcard.com/report/github.com/llm-d/llm-d-inference-sim)
[![License](https://img.shields.io/github/license/llm-d/llm-d-inference-sim)](/LICENSE)
[![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://llm-d.slack.com/archives/C097SUE2HSL)

# vLLM Simulator

`llm-d-inference-sim` is a lightweight, configurable, and real-time simulator designed to mimic the behavior of vLLM without the need for GPUs or running actual heavy models. It operates as a fully OpenAI-compliant server, allowing developers to test clients, schedulers, and infrastructure using realistic request-response cycles, token streaming, and latency patterns.

## Why is this required?

Running full LLM inference requires significant GPU resources and introduces non-deterministic latency, making it difficult to isolate infrastructure bugs or iterate quickly on control-plane logic. This simulator decouples development from heavy inference, offering a controlled environment to:

- **Accelerate Infrastructure Development**: Test routing, scheduling, and KV cache locality logic without waiting for slow, expensive GPU operations.
- **Ensure Deterministic Testing**: simulate precise token timing and latency to isolate performance regressions and bugs in a way that is impossible with non-deterministic real models.
- **Validate Observability**: Mirror vLLM’s Prometheus metrics to ensure monitoring and alerting systems are functioning correctly before deploying to production.
- **Test Advanced Features**: Safely develop complex logic such as LoRA adapter lifecycles (loading, unloading, and switching) and Disaggregated Prefill integrations.

## How it Works

The simulator is designed to act as a drop-in replacement for vLLM, sitting between your client/infrastructure and the void where the GPU usually resides. It processes requests through a configurable simulation engine that governs what is returned and when it is returned.

For detailed configuraiton definitions see the [Configuration Guide](docs/configuration.md)

### Modes of Operation
The simulator decides the content of the response based on two primary modes:

- **Echo Mode** (--mode echo): <br>
Acts as a loopback. The response content mirrors the input (e.g., the last user message in a chat request).
Useful for network throughput testing where content validity is irrelevant.
- **Random Mode** (--mode random): <br>
The default mode. Generates synthetic responses based on requested parameters (like max_tokens).
Utilizes probabilistic histograms to determine response length.
Content is sourced from either a set of pre-defined sentences or a custom dataset (see below).

### Dual Protocol Support
Natively supports both HTTP (OpenAI-compatible) and gRPC (vLLM-compatible) interfaces on the same port, allowing for versatile integration testing across different client architectures. 

For detailed API definitions see the [APIs Guide](docs/api.md).

### Response Generation & Datasets
In Random Mode, the simulator can generate content in two ways:

- **Predefined Text**: By default, it constructs responses by concatenating random sentences from a built-in list until the target token length is met.

- **Real Datasets**: If a dataset is provided (via --dataset-path or --dataset-url), the simulator attempts to match the hash of the incoming prompt to a conversation history in the database.
If a match is found, it returns the stored response.
If no match is found, it falls back to a random response from the dataset or predefined text.<br>
Supports downloading SQLite datasets directly from HuggingFace.

For response generation algorithms details see [Response Generation Guide](docs/response_generation.md).

### Latency Simulation
Unlike simple mock servers that just "sleep" for a fixed time, this simulator models the physics of LLM inference:

- **Time to first token (TTFT)**: Simulates the prefill phase latency, including configurable standard deviation (jitter) for realism.

- **Inter-token latency**: Simulates the decode phase, adding delays between every subsequent token generation.

- **Load Simulation**: The simulator automatically increases latency as the number of concurrent requests becomes higher.

- **Disaggregated Prefill (PD)**: Can simulate KV-cache transfer latency instead of standard TTFT when mimicking Prefill/Decode disaggregation architectures.

### Tokenization
The simulator offers flexible tokenization to balance accuracy vs. performance. The simulator automatically selects between two tokenization modes based on the provided `--model` name:
* **HuggingFace Mode:** Used for real models (e.g., `meta-llama/Llama-3.1-8B-Instruct`). Downloads actual tokenizers for exact accuracy.
* **Simulated Mode:** Used for dummy/non-existent model names. Uses a fast regex tokenizer for maximum performance with zero startup overhead.

For details on caching, environment variables (`HF_TOKEN`), and performance tuning, see the [Tokenization Guide](docs/tokenization.md).

### LoRA Management
Simulates the lifecycle (loading/unloading) of LoRA adapters without occupying actual memory. Reports LoRA related Prometheus metrics.

### KV Cache Simulation
Tracks simulated memory usage and publishes ZMQ events for cache block allocation and eviction.

### Failure Injection
Can randomly inject specific errors (e.g., rate_limit, model_not_found) to test client resilience.

### Deployment Options
The simulator is designed to run either as a standalone binary or within a Kubernetes Pod (e.g., for testing with Kind).

### Observability
The simulator supports a subset of standard vLLM Prometheus metrics.<br>

For detailes see the [Metrics Guide](docs/metrics.md)

## Working with docker image

### Building
To build a Docker image of the vLLM Simulator, run:
```bash
make image-build
```
Please note that the default image tag is `ghcr.io/llm-d/llm-d-inference-sim:dev`. <br>

The following environment variables can be used to change the image tag
| Variable | Descriprtion| Default Value|
| --- | --- | --- |
| IMAGE_REGISTRY | Name of the repo | ghcr.io/llm-d |
| IMAGE_TAG_BASE | Image base name | \$(IMAGE_REGISTRY)/llm-d-inference-sim |
| SIM_TAG | Image tag | dev |
| IMG | The full image specification | \$(IMAGE_TAG_BASE):\$(SIM_TAG) |

### Running
To run the vLLM Simulator image under Docker, run:
```bash
docker run --rm --publish 8000:8000 ghcr.io/llm-d/llm-d-inference-sim:dev  --port 8000 --model "Qwen/Qwen2.5-1.5B-Instruct"  --lora-modules '{"name":"tweet-summary-0"}' '{"name":"tweet-summary-1"}'
```
**Note:** To run the vLLM Simulator with the latest release version, in the above docker command replace `dev` with the current release which can be found on [GitHub](https://github.com/llm-d/llm-d-inference-sim/pkgs/container/llm-d-inference-sim).

**Note:** The above command exposes the simulator on port 8000, and serves the Qwen/Qwen2.5-1.5B-Instruct model.

## Standalone testing

### Building
To build the vLLM simulator to run locally as an executable, run:
```bash
make build
```

### Running
To run the vLLM simulator in a standalone test environment:

1. Set the PYTHONPATH environment variable (needed for the tokenization code) by running:
```bash
. env-setup.sh
```
2. Start the simulator:
```bash
./bin/llm-d-inference-sim --model my_model --port 8000
```

## Kubernetes testing

To run the vLLM simulator in a Kubernetes cluster, run:
```bash
kubectl apply -f manifests/deployment.yaml
```

When testing locally with kind, build the docker image with `make build-image` then load into the cluster:
```shell
kind load --name kind docker-image ghcr.io/llm-d/llm-d-inference-sim:dev
```

Update the `deployment.yaml` file to use the dev tag. 

To verify the deployment is available, run:
```bash
kubectl get deployment vllm-llama3-8b-instruct
kubectl get service vllm-llama3-8b-instruct-svc
```

Use `kubectl port-forward` to expose the service on your local machine:

```bash
kubectl port-forward svc/vllm-llama3-8b-instruct-svc 8000:8000
```

Test the API with curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Prefill/Decode (P/D) Separation Example
An example configuration for P/D (Prefill/Decode) disaggregation deployment can be found in [manifests/disaggregation](manifests/disaggregation).

