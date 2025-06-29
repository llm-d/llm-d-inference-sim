# vLLM Simulator
To help with development and testing we have developed a light weight vLLM simulator. It does not truly
run inference, but it does emulate responses to the HTTP REST endpoints of vLLM. 
Currently it supports partial OpenAI-compatible API:
- /v1/chat/completions 
- /v1/completions 
- /v1/models

In addition, a set of the vLLM HTTP endpoints are suppored as well. These include:
| Endpoint | Description |
|---|---|
| /v1/load_lora_adapter   | simulates the dynamic registration of a LoRA adapter |
| /v1/unload_lora_adapter | simulates the dynamic unloading and unregistration of a LoRA adapter |
| /metrics                | exposes Prometheus metrics. See the table below for details |
| /health                 | standard health check endpoint |
| /ready                  | standard readiness endpoint |

In addition, it supports a subset of vLLM's Prometheus metrics. These metrics are exposed via the /metrics HTTP REST endpoint. Currently supported are the following metrics:
| Metric | Description |
|---|---|
| vllm:gpu_cache_usage_perc | The fraction of KV-cache blocks currently in use (from 0 to 1). Currently this value will always be zero. |
| vllm:lora_requests_info | Running stats on LoRA requests |
| vllm:num_requests_running | Number of requests currently running on GPU |
| vllm:num_requests_waiting | Prometheus metric for the number of queued requests |

The simulated inference has no connection with the model and LoRA adapters specified in the command line parameters or via the /v1/load_lora_adapter HTTP REST endpoint. The /v1/models endpoint returns simulated results based on those same command line parameters and those loaded via the /v1/load_lora_adapter HTTP REST endpoint.

The simulator supports two modes of operation:
- `echo` mode: the response contains the same text that was received in the request. For `/v1/chat/completions` the last message for the role=`user` is used.
- `random` mode: the response is randomly chosen from a set of pre-defined sentences.

Timing of the response is defined by two parameters: `time-to-first-token` and `inter-token-latency`. 

For a request with `stream=true`: `time-to-first-token` defines the delay before the first token is returned, `inter-token-latency` defines the delay between subsequent tokens in the stream. 

For a requst with `stream=false`: the response is returned after delay of `<time-to-first-token> + (<inter-token-latency> * (<number_of_output_tokens> - 1))`

It can be run standalone or in a Pod for testing under packages such as Kind.

## Limitations
API responses contains a subset of the fields provided by the OpenAI API.

<details>
  <summary>Click to show the structure of requests/responses</summary>  

- `/v1/chat/completions`
    - **request**
        - stream
        - model
        - messages
            - role
            - content
    - **response**
        - id
        - created
        - model
        - choices
            - index
            - finish_reason
            - message
- `/v1/completions`
    - **request**
        - stream
        - model
        - prompt
        - max_tokens (for future usage)
    - **response**
        - id
        - created
        - model
        - choices
            - text
- `/v1/models`
    - **response**
        - object (list)
        - data
            - id
            - object (model)
            - created
            - owned_by
            - root
            - parent
</details>
<br/>
For more details see the <a href="https://docs.vllm.ai/en/stable/getting_started/quickstart.html#openai-completions-api-with-vllm">vLLM documentation</a>

## Command line parameters
- `port`: the port the simulator listents on, mandatory
- `model`: the currently 'loaded' model, mandatory
- `lora`: a list of available LoRA adapters, separated by commas, optional, by default empty
- `mode`: the simulator mode, optional, by default `random`
 - `echo`: returns the same text that was sent in the request
 - `random`: returns a sentence chosen at random from a set of pre-defined sentences
- `time-to-first-token`: the time to the first token (in milliseconds), optional, by default zero
- `inter-token-latency`: the time to 'generate' each additional token (in milliseconds), optional, by default zero
- `max-loras`: maximum number of LoRAs in a single batch, optional, default is one
- `max-cpu-loras`: maximum number of LoRAs to store in CPU memory, optional, must be >= than max_loras, default is max_loras
- `max-running-requests`: maximum number of inference requests that could be processed at the same time


## Working with docker image

### Building
To build a Docker image of the vLLM Simulator, run:
```bash
make image-build
```
Please note that the default image tag is `ghcr.io/llm-d/llm-d-inference-sim:dev`. <br>
The following environment variables can be used to change the image tag: `REGISTRY`, `SIM_TAG`, `IMAGE_TAG_BASE` or `IMG`.

### Running
To run the vLLM Simulator image under Docker, run:
```bash
docker run --rm --publish 8000:8000 ghcr.io/llm-d/llm-d-inference-sim:dev  --port 8000 --model "Qwen/Qwen2.5-1.5B-Instruct" --lora "tweet-summary-0,tweet-summary-1"
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
To run the vLLM simulator in a standalone test environment, run:
```bash
./bin/llm-d-inference-sim --model my_model --port 8000
```

## Kubernetes testing

To run the vLLM simulator in a Kubernetes cluster, run:
```bash
kubectl apply -f manifests/deployment.yaml
```

To verify the deployment is available, run:
```bash
kubectl get deployment vllm-llama3-8b-instruct
```
