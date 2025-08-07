[![Go Report Card](https://goreportcard.com/badge/github.com/llm-d/llm-d-inference-sim)](https://goreportcard.com/report/github.com/llm-d/llm-d-inference-sim)
[![License](https://img.shields.io/github/license/llm-d/llm-d-inference-sim)](/LICENSE)
[![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://llm-d.slack.com/archives/C097SUE2HSL)

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

Timing of the response is defined by the `time-to-first-token` and `inter-token-latency` parameters. In case P/D is enabled for a request, `kv-cache-transfer-latency` will be used instead of `time-to-first-token`.

For a request with `stream=true`: `time-to-first-token` or `kv-cache-transfer-latency` defines the delay before the first token is returned, `inter-token-latency` defines the delay between subsequent tokens in the stream. 

For a requst with `stream=false`: the response is returned after delay of `<time-to-first-token> + (<inter-token-latency> * (<number_of_output_tokens> - 1))` or `<kv-cache-transfer-latency> + (<inter-token-latency> * (<number_of_output_tokens> - 1))` in P/D case

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
- `config`: the path to a yaml configuration file that can contain the simulator's command line parameters. If a parameter is defined in both the config file and the command line, the command line value overwrites the configuration file value. An example configuration file can be found at `manifests/config.yaml`
- `port`: the port the simulator listents on, default is 8000
- `model`: the currently 'loaded' model, mandatory
- `served-model-name`: model names exposed by the API (a list of space-separated strings)
- `lora-modules`: a list of LoRA adapters (a list of space-separated JSON strings): '{"name": "name", "path": "lora_path", "base_model_name": "id"}', optional, empty by default
- `max-loras`: maximum number of LoRAs in a single batch, optional, default is one
- `max-cpu-loras`: maximum number of LoRAs to store in CPU memory, optional, must be >= than max-loras, default is max-loras
- `max-model-len`: model's context window, maximum number of tokens in a single request including input and output, optional, default is 1024
- `max-num-seqs`: maximum number of sequences per iteration (maximum number of inference requests that could be processed at the same time), default is 5
- `mode`: the simulator mode, optional, by default `random`
    - `echo`: returns the same text that was sent in the request
    - `random`: returns a sentence chosen at random from a set of pre-defined sentences
- `time-to-first-token`: the time to the first token (in milliseconds), optional, by default zero
- `time-to-first-token-std-dev`: standard deviation for time before the first token will be returned, in milliseconds, optional, default is 0, can't be more than 30% of `time-to-first-token`, will not cause the actual time to first token to differ by more than 70% from `time-to-first-token`
- `inter-token-latency`: the time to 'generate' each additional token (in milliseconds), optional, by default zero
- `inter-token-latency-std-dev`: standard deviation for time between generated tokens, in milliseconds, optional, default is 0, can't be more than 30% of `inter-token-latency`, will not cause the actual inter token latency to differ by more than 70% from `inter-token-latency`
- `kv-cache-transfer-latency`: time for KV-cache transfer from a remote vLLM (in milliseconds), by default zero. Usually much shorter than `time-to-first-token`
- `kv-cache-transfer-latency-std-dev`: standard deviation for time to "transfer" kv-cache from another vLLM instance in case P/D is activated, in milliseconds, optional, default is 0, can't be more than 30% of `kv-cache-transfer-latency`, will not cause the actual latency to differ by more than 70% from `kv-cache-transfer-latency`
- `seed`: random seed for operations (if not set, current Unix time in nanoseconds is used)
- `max-tool-call-integer-param`: the maximum possible value of integer parameters in a tool call, optional, defaults to 100
- `min-tool-call-integer-param`: the minimum possible value of integer parameters in a tool call, optional, defaults to 0
- `max-tool-call-number-param`: the maximum possible value of number (float) parameters in a tool call, optional, defaults to 100
- `min-tool-call-number-param`: the minimum possible value of number (float) parameters in a tool call, optional, defaults to 0
- `max-tool-call-array-param-length`: the maximum possible length of array parameters in a tool call, optional, defaults to 5
- `min-tool-call-array-param-length`: the minimum possible length of array parameters in a tool call, optional, defaults to 1
- `tool-call-not-required-param-probability`: the probability to add a parameter, that is not required, in a tool call, optional, defaults to 50
- `object-tool-call-not-required-field-probability`: the probability to add a field, that is not required, in an object in a tool call, optional, defaults to 50
- `enable-kvcache`: if true, the KV cache support will be enabled in the simulator. In this case, the KV cache will be simulated, and ZQM events will be published when a KV cache block is added or evicted.
- `kv-cache-size`: the maximum number of token blocks in kv cache
- `block-size`: token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128
- `tokenizers-cache-dir`: the directory for caching tokenizers
- `hash-seed`: seed for hash generation (if not set, is read from PYTHONHASHSEED environment variable)
- `zmq-endpoint`: ZMQ address to publish events

In addition, as we are using klog, the following parameters are available:
- `add_dir_header`: if true, adds the file directory to the header of the log messages
- `alsologtostderr`: log to standard error as well as files (no effect when -logtostderr=true)
- `log_backtrace_at`: when logging hits line file:N, emit a stack trace (default :0)
- `log_dir`: if non-empty, write log files in this directory (no effect when -logtostderr=true)
- `log_file`: if non-empty, use this log file (no effect when -logtostderr=true)
- `log_file_max_size`: defines the maximum size a log file can grow to (no effect when -logtostderr=true). Unit is megabytes. If the value is 0, the maximum file size is unlimited. (default 1800)
- `logtostderr`: log to standard error instead of files (default true)
- `one_output`: if true, only write logs to their native severity level (vs also writing to each lower severity level; no effect when -logtostderr=true)
- `skip_headers`: if true, avoid header prefixes in the log messages
- `skip_log_headers`: if true, avoid headers when opening log files (no effect when -logtostderr=true)
- `stderrthreshold`: logs at or above this threshold go to stderr when writing to files and stderr (no effect when -logtostderr=true or -alsologtostderr=true) (default 2)
- `v`: number for the log level verbosity
- `vmodule`: comma-separated list of pattern=N settings for file-filtered logging

---

## Migrating from releases prior to v0.2.0
- `max-running-requests` was replaced by `max-num-seqs`
- `lora` was replaced by `lora-modules`, which is now a list of JSON strings, e.g, '{"name": "name", "path": "lora_path", "base_model_name": "id"}'

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
