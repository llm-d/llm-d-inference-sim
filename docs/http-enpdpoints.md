# Overview 
The simulator supports a subset of fields from the standard OpenAI API in both requests and responses. Any fields not listed below may be ignored or not fully supported.


## Request & Response Structure
The following outline details the specific fields accepted in requests and returned in responses:

Structure of requests/responses

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
            - parameters
            - description
        - tool_choice
        - logprobs
        - top_logprobs
        - stream_options
          - include_usage
        - ignore_eos
        - cache_hit_threshold
        - kv_transfer_params
          - do_remote_decode
          - do_remote_prefill
          - remote_engine_id
          - remote_block_ids
          - remote_host
          - remote_port
          - tp_size
    - **response**
        - id
        - created
        - model        
        - choices
          - index
          - finish_reason
          - message
            - role
            - content
            - tool_calls
              - function
                - name
                - arguments
	            - id
              - type
              - index
          - logprobs
            - content
              - token
              - logprob
              - bytes
              - top_logprobs
        - usage
        - object
        - kv_transfer_params
          - do_remote_decode
          - do_remote_prefill
          - remote_engine_id
          - remote_block_ids
          - remote_host
          - remote_port
          - tp_size
- `/v1/completions`
    - **request**
        - stream
        - model
        - prompt
        - max_tokens
        - stream_options
          - include_usage
        - ignore_eos
        - logprobs
        - cache_hit_threshold
        - kv_transfer_params
          - do_remote_decode
          - do_remote_prefill
          - remote_engine_id
          - remote_block_ids
          - remote_host
          - remote_port
          - tp_size
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
        - kv_transfer_params
          - do_remote_decode
          - do_remote_prefill
          - remote_engine_id
          - remote_block_ids
          - remote_host
          - remote_port
          - tp_size
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

### `/v1/completions` prompt forms

The `prompt` field accepts four wire forms, matching the OpenAI spec:

| Form | JSON example | Result |
|---|---|---|
| string | `"hello"` | one prompt, one choice in the response |
| array of strings | `["a", "b"]` | one sub-request per element; one choice per element, indexed in input order |
| array of token ids | `[1, 2, 3]` | one prompt already tokenized; the simulator skips tokenization and uses the ids directly |
| array of arrays of token ids | `[[1,2], [3,4]]` | one sub-request per inner array, each already tokenized |

Notes:

- An empty top-level array (`[]`), an empty string element (`""`), or an empty token-id element (`[]` inside the outer array) are rejected with `400 Bad Request`.
- For pre-tokenized prompts, `prompt_tokens` in the usage equals the number of input ids — the tokenizer is never invoked on the prompt.
- In `--mode echo`, a token-id prompt is replayed back to the client as the comma-separated decimal of the ids (e.g. `[1,2,3]` → `"1,2,3"`); a string prompt is replayed verbatim.

For full details on the expected API behavior and specification, please refer to the [vLLM OpenAI Compatibility Documentation](https://docs.vllm.ai/en/stable/getting_started/quickstart.html#openai-completions-api-with-vllm).