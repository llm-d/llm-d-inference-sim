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
| vllm:e2e_request_latency_seconds | Histogram of end to end request latency in seconds |
| vllm:request_inference_time_seconds | Histogram of time spent in RUNNING phase for request |
| vllm:request_queue_time_seconds | Histogram of time spent in WAITING phase for request |
| vllm:request_prefill_time_seconds | Histogram of time spent in PREFILL phase for request |
| vllm:request_decode_time_seconds | Histogram of time spent in DECODE phase for request |
| vllm:time_to_first_token_seconds | Histogram of time to first token in seconds |
| vllm:time_per_output_token_seconds | Histogram of time per output token in seconds |
| vllm:request_generation_tokens | Number of generation tokens processed |
| vllm:max_num_generation_tokens | Maximum number of requested generation tokens. Currently same as `vllm:request_generation_tokens` since always only one choice is returned |
| vllm:request_params_max_tokens | Histogram of the max_tokens request parameter | 
| vllm:request_prompt_tokens | Number of prefill tokens processed |
| vllm:request_success_total | Count of successfully processed requests |
  
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
            - tool_calls
              - function
                - name
                - arguments
	            - id
              - type
              - index
        - max_tokens
        - max_completion_tokens
        - tools 
          - type
          - function
            - name
            - arguments
        - tool_choice
        - logprobs
        - top_logprobs
        - stream_options
          - include_usage
        - do_remote_decode
        - do_remote_prefill
        - remote_block_ids
        - remote_engine_id
        - remote_host
        - remote_port
        - ignore_eos
    - **response**
        - id
        - created
        - model
        - choices
          - index
          - finish_reason
          - message
          - logprobs
            - content
              - token
              - logprob
              - bytes
              - top_logprobs
        - usage
        - object
        - do_remote_decode
        - do_remote_prefill
        - remote_block_ids
        - remote_engine_id
        - remote_host
        - remote_port
- `/v1/completions`
    - **request**
        - stream
        - model
        - prompt
        - max_tokens
        - stream_options
          - include_usage
        - do_remote_decode
        - do_remote_prefill
        - remote_block_ids
        - remote_engine_id
        - remote_host
        - remote_port
        - ignore_eos
        - logprobs
    - **response**
        - id
        - created
        - model
        - choices
          - index
          - finish_reason
          - text
          - logprobs
            - tokens
            - token_logprobs
            - top_logprobs
            - text_offset
        - usage
        - object
        - do_remote_decode
        - do_remote_prefill
        - remote_block_ids
        - remote_engine_id
        - remote_host
        - remote_port
- `/v1/models`
    - **response**
        - object
        - data
            - id
            - object
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
- `max-waiting-queue-length`: maximum length of inference requests waiting queue, default is 1000
- `mode`: the simulator mode, optional, by default `random`
    - `echo`: returns the same text that was sent in the request
    - `random`: returns a sentence chosen at random from a set of pre-defined sentences
---
- `time-to-first-token`: the time to the first token (in milliseconds), optional, by default zero
- `time-to-first-token-std-dev`: standard deviation for time before the first token will be returned, in milliseconds, optional, default is 0, can't be more than 30% of `time-to-first-token`, will not cause the actual time to first token to differ by more than 70% from `time-to-first-token`
- `inter-token-latency`: the time to 'generate' each additional token (in milliseconds), optional, by default zero
- `inter-token-latency-std-dev`: standard deviation for time between generated tokens, in milliseconds, optional, default is 0, can't be more than 30% of `inter-token-latency`, will not cause the actual inter token latency to differ by more than 70% from `inter-token-latency`
- `kv-cache-transfer-latency`: time for KV-cache transfer from a remote vLLM (in milliseconds), by default zero. Usually much shorter than `time-to-first-token`
- `kv-cache-transfer-latency-std-dev`: standard deviation for time to "transfer" kv-cache from another vLLM instance in case P/D is activated, in milliseconds, optional, default is 0, can't be more than 30% of `kv-cache-transfer-latency`, will not cause the actual latency to differ by more than 70% from `kv-cache-transfer-latency`
---
- `prefill-overhead`: constant overhead time for prefill (in milliseconds), optional, by default zero, used in calculating time to first token, this will be ignored if `time-to-first-token` is not `0`
- `prefill-time-per-token`: time taken to generate each token during prefill (in milliseconds), optional, by default zero, this will be ignored if `time-to-first-token` is not `0`
- `prefill-time-std-dev`: similar to `time-to-first-token-std-dev`, but is applied on the final prefill time, which is calculated by `prefill-overhead`, `prefill-time-per-token`, and number of prompt tokens, this will be ignored if `time-to-first-token` is not `0`
- `kv-cache-transfer-time-per-token`: time taken to transfer cache for each token in case P/D is enabled (in milliseconds), optional, by default zero, this will be ignored if `kv-cache-transfer-latency` is not `0`
- `kv-cache-transfer-time-std-dev`: similar to `time-to-first-token-std-dev`, but is applied on the final kv cache transfer time in case P/D is enabled (in milliseconds), which is calculated by `kv-cache-transfer-time-per-token` and number of prompt tokens, this will be ignored if `kv-cache-transfer-latency` is not `0`
---
- `time-factor-under-load`: a multiplicative factor that affects the overall time taken for requests when parallelrequests are being processed. The value of this factor must be >= 1.0, with a default of 1.0. If this factor is 1.0, no extra time is added.  When the factor is x (where x > 1.0) and there are `max-num-seqs` requests, the total time will be multiplied by x. The extra time then decreases multiplicatively to 1.0 when the number of requests is less than MaxNumSeqs.
- `seed`: random seed for operations (if not set, current Unix time in nanoseconds is used)
---
- `max-tool-call-integer-param`: the maximum possible value of integer parameters in a tool call, optional, defaults to 100
- `min-tool-call-integer-param`: the minimum possible value of integer parameters in a tool call, optional, defaults to 0
- `max-tool-call-number-param`: the maximum possible value of number (float) parameters in a tool call, optional, defaults to 100
- `min-tool-call-number-param`: the minimum possible value of number (float) parameters in a tool call, optional, defaults to 0
- `max-tool-call-array-param-length`: the maximum possible length of array parameters in a tool call, optional, defaults to 5
- `min-tool-call-array-param-length`: the minimum possible length of array parameters in a tool call, optional, defaults to 1
- `tool-call-not-required-param-probability`: the probability to add a parameter, that is not required, in a tool call, optional, defaults to 50
- `object-tool-call-not-required-field-probability`: the probability to add a field, that is not required, in an object in a tool call, optional, defaults to 50
---
- `enable-kvcache`: if true, the KV cache support will be enabled in the simulator. In this case, the KV cache will be simulated, and ZQM events will be published when a KV cache block is added or evicted. 
- `kv-cache-size`: the maximum number of token blocks in kv cache
- `block-size`: token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128
- `tokenizers-cache-dir`: the directory for caching tokenizers
- `hash-seed`: seed for hash generation (if not set, is read from PYTHONHASHSEED environment variable)
- `zmq-endpoint`: ZMQ address to publish events
- `zmq-max-connect-attempts`: the maximum number of ZMQ connection attempts, defaults to 0, maximum: 10
- `event-batch-size`: the maximum number of kv-cache events to be sent together, defaults to 16
---
- `failure-injection-rate`: probability (0-100) of injecting failures, optional, default is 0
- `failure-types`: list of specific failure types to inject (rate_limit, invalid_api_key, context_length, server_error, invalid_request, model_not_found), optional, if empty all types are used
---
- `fake-metrics`: represents a predefined set of metrics to be sent to Prometheus as a substitute for the real metrics. When specified, only these fake metrics will be reported — real metrics and fake metrics will never be reported together. The set should include values for 
    - `running-requests`
    - `waiting-requests`
    - `kv-cache-usage`
    - `loras` - an array containing LoRA information objects, each with the fields: `running` (a comma-separated list of LoRAs in use by running requests), `waiting` (a comma-separated list of LoRAs to be used by waiting requests), and `timestamp` (seconds since Jan 1 1970, the timestamp of this metric). 
    - `ttft-buckets-values` - array of values for time-to-first-token buckets, each value in this array is a value for the corresponding bucket. Array may contain less values than number of buckets, all trailing missing values assumed as 0. Buckets upper boundaries are: 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0, +Inf.
    - `tpot-buckets-values` - array of values for time-per-output-token buckets, each value in this array is a value for the corresponding bucket. Array may contain less values than number of buckets, all trailing missing values assumed as 0. Buckets upper boundaries are: 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, +Inf.
    - `e2erl-buckets-values` - array of values for e2e request latency buckets, each value in this array is a value for the corresponding bucket. Array may contain less values than number of buckets, all trailing missing values assumed as 0. Buckets upper boundaries are: 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 
    960.0, 1920.0, 7680.0, +Inf.
    - `queue-time-buckets-values` - array of values for request queue time buckets, each value in this array is a value for the corresponding bucket. Array may contain less values than number of buckets, all trailing missing values assumed as 0. Buckets upper boundaries are: 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 
    960.0, 1920.0, 7680.0, +Inf.
    - `inf-time-buckets-values` - array of values for request inference time buckets, each value in this array is a value for the corresponding bucket. Array may contain less values than number of buckets, all trailing missing values assumed as 0. Buckets upper boundaries are: 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 
    960.0, 1920.0, 7680.0, +Inf.
    - `prefill-time-buckets-values` -  array of values for request prefill time buckets, each value in this array is a value for the corresponding bucket. Array may contain less values than number of buckets, all trailing missing values assumed as 0. Buckets upper boundaries are: 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 
    960.0, 1920.0, 7680.0, +Inf.
    - `decode-time-buckets-values` - array of values for request decode time buckets, each value in this array is a value for the corresponding bucket. Array may contain less values than number of buckets, all trailing missing values assumed as 0. Buckets upper boundaries are: 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 
    960.0, 1920.0, 7680.0, +Inf.
    - `request-prompt-tokens` - array of values for prompt-length buckets
    - `request-generation-tokens` - array of values for generation-length buckets
    - `request-max-generation-tokens` - array of values for max_num_generation_tokens buckets
    - `request-params-max-tokens` - array of values for  max_tokens parameter buckets
    - `request-success-total` - number of successful requests per finish reason, key: finish-reason (stop, length, etc.).
    <br>
    **Example:**<br>
      --fake-metrics '{"running-requests":10,"waiting-requests":30,"kv-cache-usage":0.4,"loras":[{"running":"lora4,lora2","waiting":"lora3","timestamp":1257894567},{"running":"lora4,lora3","waiting":"","timestamp":1257894569}]}'
---
- `data-parallel-size`: number of ranks to run in Data Parallel deployment, from 1 to 8, default is 1. The ports will be assigned as follows: rank 0 will run on the configured `port`, rank 1 on `port`+1, etc.      
---
- `dataset-path`: Optional local file path to the SQLite database file used for generating responses from a dataset.
  - If not set, hardcoded preset responses will be used.
  - If set but the file does not exist the `dataset-url` will be used to download the database to the path specified by `dataset-path`.
  - If the file exists but is currently occupied by another process, responses will be randomly generated from preset text (the same behavior as if the path were not set).
  - Responses are retrieved from the dataset by the hash of the conversation history, with a fallback to a random dataset response, constrained by the maximum output tokens and EoS token handling, if no matching history is found.
  - Refer to [llm-d converted ShareGPT](https://huggingface.co/datasets/hf07397/inference-sim-datasets/blob/0b60737c2dd2c570f486cef2efa7971b02e3efde/README.md) for detailed information on the expected format of the SQLite database file.
- `dataset-url`: Optional URL for downloading the SQLite database file used for response generation.
  - This parameter is only used if the `dataset-path` is also set and the file does not exist at that path.
  - If the file needs to be downloaded, it will be saved to the location specified by `dataset-path`.
  - If the file already exists at the `dataset-path`, it will not be downloaded again
  - Example URL `https://huggingface.co/datasets/hf07397/inference-sim-datasets/resolve/91ffa7aafdfd6b3b1af228a517edc1e8f22cd274/huggingface/ShareGPT_Vicuna_unfiltered/conversations.sqlite3`
- `dataset-in-memory`: If true, the entire dataset will be loaded into memory for faster access. This may require significant memory depending on the size of the dataset. Default is false.
---
- `ssl-certfile`: Path to SSL certificate file for HTTPS (optional)
- `ssl-keyfile`: Path to SSL private key file for HTTPS (optional)
- `self-signed-certs`: Enable automatic generation of self-signed certificates for HTTPS
---
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
- `v`: number for the log level verbosity. Supported levels:
  - Warning (1) - warning messages
  - Info (2) - general application messages, e.g., loaded configuration content, which responses dataset was loaded, etc.
  - Debug (4) - debugging messages, e.g. /completions and /chat/completions request received, load/unload lora request processed, etc.
  - Trace (5) - highest verbosity, e.g. detailed messages on completions request handling and request queue processing, etc.
- `vmodule`: comma-separated list of pattern=N settings for file-filtered logging

## Environment variables
- `POD_NAME`: the simulator pod name. If defined, the response will contain the HTTP header `x-inference-pod` with this value, and the HTTP header `x-inference-port` with the port that the request was received on 
- `POD_NAMESPACE`: the simulator pod namespace. If defined, the response will contain the HTTP header `x-inference-namespace` with this value

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

Note: On macOS, use `make image-build TARGETOS=linux` to pull the correct base image.

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

### Prefill/Decode (P/D) Separation Example
An example configuration for P/D (Prefill/Decode) disaggregation deployment can be found in [manifests/disaggregation](manifests/disaggregation).

## Response generation

The `/v1/completions` and `/v1/chat/completions` endpoints produce responses based on simulator configurations and the specific request parameters.

### Echo mode
In `echo` mode, responses always mirror the request content. 
In case of /v1/completions the prompt field is returned.
In case of /v1/chat/completions the last message is returned.

Parameters `max_tokens`, `max_completions_tokens` and `ignore_eos` are ignored in this mode.

### Random mode
In `random` mode, the fields `max_tokens`, `max_completions_tokens` and `ignore_eos` from the request are used during response generation.

#### Use predefined texts for response generation
The simulator can generate responses from a predefined list of sentences.
If `max_tokens` or `max_completions_tokens` is specified, the response length is caclulated using a histogram with six buckets and the following probabilities: 20%, 30%, 20%, 5%, 10%, 15%.
For a maximum length ≤ 120, bucket sizes are equal.
For a maximum length > 120, all buckets except the fourth are of size 20;
the fourth bucket covers the remaining range.
After the buckets are set, response length is sampled according to these probabilities.


Examples: <br>
max-len = 120: the buckets are 1-20, 21-40, 41-60, 61-80, 81-100, 101-120. <br>
max-len = 200: the buckets are 1-20, 21-40, 41-60, 61-160, 161-180, 181-200. <br>

If the maximum response length is not specified, it defaults to `<model length>-<input-length>`.
In this case, response length is sampled from a Gaussian distribution with mean 40 and standard deviation 20.


After determining the response length:

A random sentence from the predefined list is chosen and trimmed if it exceeds the required length.
If the sentence is shorter, additional random sentences are concatenated until the required token count is met.

If `ignore_eos` is true, the response always reaches the maximum allowed length.

The finish_reason is set to LENGTH if the response length equals the maximum; otherwise, it is set to STOP.


#### Use responses dataset for response generation
If `dataset-url` is set in command line, the dataset is downloaded to the location specified by `dataset-path`.

If a valid dataset exists in the `dataset-path`, it is used for response selection.
The request prompt is hashed, and this value is matched against dataset entries.
If all matches are longer, a random match is selected and then trimmed.

If `ignore_eos` is true and no match meets the required length, the response is completed with random tokens from the predefined list.

If the prompt hash is not present in the dataset, a random response of length ≤ maximum is selected;
if all responses are longer, a random response is chosen and trimmed.

