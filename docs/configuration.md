# Command line parameters
The simulator can be configured using either command-line arguments or a YAML file. Parameter names are consistent across both methods.

## General
- `config`: the path to a yaml configuration file that can contain the simulator's command line parameters. If a parameter is defined in both the config file and the command line, the command line value overwrites the configuration file value. An example configuration file can be found at [manifests/config.yaml](../manifests/config.yaml)
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
    - `random`: returns a sentence chosen at random from a set of pre-defined sentences or a given dataset
- `enable-sleep-mode`: Enable sleep mode feature. When enabled, the simulator can be put to sleep via the `/sleep` endpoint and woken up via the `/wake_up` endpoint
- `enable-request-id-headers`: Enable including X-Request-Id header in responses. When enabled, the simulator will include the request ID in response headers

## Latency 
All latency-related parameters are defined in duration format, e.g., 100ms. Integer format is deprecated.

- `latency-calculator`: specifies the latency calculator to be used to simulate response times. By default, the latency is computed based on the simulator’s current load and the configured latency parameters, such as `time-to-first-token` and `prefill-time-per-token`. Supported values are `per-token` and `constant`, indicating whether or not the calculation accounts for the prompt size.
- `time-to-first-token`: the time to the first token, optional, by default zero
- `time-to-first-token-std-dev`: standard deviation for time before the first token will be returned, optional, default is zero. Can't be more than 30% of `time-to-first-token`, will not cause the actual time to first token to differ by more than 70% from `time-to-first-token`
- `inter-token-latency`: the time to 'generate' each additional token, optional, by default zero
- `inter-token-latency-std-dev`: standard deviation for time between generated tokens, optional, default is zero. Can't be more than 30% of `inter-token-latency`, will not cause the actual inter token latency to differ by more than 70% from `inter-token-latency`
- `kv-cache-transfer-latency`: time for KV-cache "transfer" from a remote vLLM, optional, by default zero. Usually much shorter than `time-to-first-token`
- `kv-cache-transfer-latency-std-dev`: standard deviation for time to "transfer" kv-cache from another vLLM instance in case P/D is activated, optional, default is zero. Can't be more than 30% of `kv-cache-transfer-latency`, will not cause the actual latency to differ by more than 70% from `kv-cache-transfer-latency`
---
- `prefill-overhead`: constant overhead time for prefill, optional, by default zero. Used in calculating time to first token, this will be ignored if `time-to-first-token` is not zero
- `prefill-time-per-token`: time taken to generate each token during prefill, optional, by default zero, this will be ignored if `time-to-first-token` is not zero
- `prefill-time-std-dev`: similar to `time-to-first-token-std-dev`, but is applied on the final prefill time, which is calculated by `prefill-overhead`, `prefill-time-per-token`, and number of prompt tokens, this will be ignored if `time-to-first-token` is not zero
- `kv-cache-transfer-time-per-token`: time taken to transfer cache for each token in case disaggregated P/D is enabled, optional, by default zero. This will be ignored if `kv-cache-transfer-latency` is not zero
- `kv-cache-transfer-time-std-dev`: similar to `time-to-first-token-std-dev`, but is applied on the final kv cache transfer time in case disaggregated P/D is enabled, which is calculated by `kv-cache-transfer-time-per-token` and number of prompt tokens, this will be ignored if `kv-cache-transfer-latency` is not zero
---
- `time-factor-under-load`: a multiplicative factor that affects the overall time taken for requests when parallel requests are being processed. The value of this factor must be >= 1.0, with a default of 1.0. If this factor is 1.0, no extra time is added.  When the factor is x (where x > 1.0) and there are `max-num-seqs` requests, the total time will be multiplied by x. The extra time then decreases multiplicatively to 1.0 when the number of requests is less than `max-num-seqs`.
- `seed`: random seed for operations (if not set, current Unix time in nanoseconds is used)

## Tools 
- `max-tool-call-integer-param`: the maximum possible value of integer parameters in a tool call, optional, defaults to 100
- `min-tool-call-integer-param`: the minimum possible value of integer parameters in a tool call, optional, defaults to 0
- `max-tool-call-number-param`: the maximum possible value of number (float) parameters in a tool call, optional, defaults to 100
- `min-tool-call-number-param`: the minimum possible value of number (float) parameters in a tool call, optional, defaults to 0
- `max-tool-call-array-param-length`: the maximum possible length of array parameters in a tool call, optional, defaults to 5
- `min-tool-call-array-param-length`: the minimum possible length of array parameters in a tool call, optional, defaults to 1
- `tool-call-not-required-param-probability`: the probability to add a parameter, that is not required, in a tool call, optional, defaults to 50
- `object-tool-call-not-required-field-probability`: the probability to add a field, that is not required, in an object in a tool call, optional, defaults to 50


## KV cache
- `enable-kvcache`: if true, the KV cache support will be enabled in the simulator. In this case, the KV cache will be simulated, and ZMQ events will be published when a KV cache block is added or evicted.
- `kv-cache-size`: the maximum number of token blocks in kv cache
- `global-cache-hit-threshold`: default cache hit threshold [0, 1] for all requests. If a request specifies cache_hit_threshold, it takes precedence over this global value
- `block-size`: token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128
- `tokenizers-cache-dir`: the directory for caching tokenizers, default is hf_cache
- `hash-seed`: seed for hash generation (if not set, is read from PYTHONHASHSEED environment variable)
- `zmq-endpoint`: ZMQ address to publish events
- `zmq-max-connect-attempts`: the maximum number of ZMQ connection attempts, defaults to 0, maximum: 10
- `event-batch-size`: the maximum number of kv-cache events to be sent together, defaults to 16

## Failure injection
- `failure-injection-rate`: probability (0-100) of injecting failures, optional, default is 0
- `failure-types`: list of specific failure types to inject (rate_limit, invalid_api_key, context_length, server_error, invalid_request, model_not_found), optional, if empty all types are used

## Data parallel
- `data-parallel-size`: number of ranks to run in Data Parallel deployment, from 1 to 8, default is 1. The ports will be assigned as follows: rank 0 will run on the configured `port`, rank 1 on `port`+1, etc.  
- `data-parallel-rank`: the rank of this instance, used only when running Data Parallel ranks as separate processes

## Datasets
- `dataset-path`: Optional local file path to the SQLite database file used for generating responses from a dataset.
  - If not set, hardcoded preset responses will be used.
  - If set but the file does not exist the `dataset-url` will be used to download the database to the path specified by `dataset-path`.
  - Responses are retrieved from the dataset by the hash of the conversation history, with a fallback to a random dataset response, constrained by the maximum output tokens and EoS token handling, if no matching history is found.
  - Refer to [llm-d converted ShareGPT](https://huggingface.co/datasets/hf07397/inference-sim-datasets/blob/0b60737c2dd2c570f486cef2efa7971b02e3efde/README.md) for detailed information on the expected format of the SQLite database file.
- `dataset-url`: Optional URL for downloading the SQLite database file used for response generation.
  - This parameter is only used if the `dataset-path` is also set and the file does not exist at that path.
  - If the file needs to be downloaded, it will be saved to the location specified by `dataset-path`.
  - If the file already exists at the `dataset-path`, it will not be downloaded again
  - Example URL `https://huggingface.co/datasets/hf07397/inference-sim-datasets/resolve/91ffa7aafdfd6b3b1af228a517edc1e8f22cd274/huggingface/ShareGPT_Vicuna_unfiltered/conversations.sqlite3`
- `dataset-in-memory`: If true, the entire dataset will be loaded into memory for faster access. This may require significant memory depending on the size of the dataset. Default is false.
- `dataset-table-name`: Table name for custom dataset, optional, default is 'llmd'

## SSL
- `ssl-certfile`: Path to SSL certificate file for HTTPS (optional)
- `ssl-keyfile`: Path to SSL private key file for HTTPS (optional)
- `self-signed-certs`: Enable automatic generation of self-signed certificates for HTTPS

## Fake metrics
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

## Klog
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

# Environment variables
- `HF_TOKEN`: HuggingFace access token
- `POD_NAME`: the simulator pod name. If defined, the response will contain the HTTP header `x-inference-pod` with this value, and the HTTP header `x-inference-port` with the port that the request was received on 
- `POD_NAMESPACE`: the simulator pod namespace. If defined, the response will contain the HTTP header `x-inference-namespace` with this value
- `POD_IP`: the simulator pod IP address. Used in kv-events topic name.
Example of definition in yaml: 
  ```yaml
  env:
    - name: POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
  ```
