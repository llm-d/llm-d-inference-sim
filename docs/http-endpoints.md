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
            - content (string, or array of content blocks)
              - type (`text` or `image_url`)
              - text
              - image_url
                - url
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
          - prompt_tokens
          - completion_tokens
          - total_tokens
          - prompt_tokens_details
            - cached_tokens
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
            - max_model_len
- `/v1/embeddings`
    - **request**
        - model
        - input (string, array of strings, array of token ids, or array of arrays of token ids)
        - dimensions
        - encoding_format (`float` (default) or `base64`)
        - user
    - **response**
        - object (`list`)
        - model
        - data
            - object (`embedding`)
            - index
            - embedding (array of floats when `encoding_format` is `float`, base64 string when `base64`)
        - usage
            - prompt_tokens
            - total_tokens
- `/v1/messages`
    - **request**
        - stream
        - model
        - messages (required)
            - role (`user` or `assistant`)
            - content (string or array of content blocks)
              - type (`text`, `image`, `tool_use`, `tool_result`)
              - text (for `text` blocks)
              - source (for `image` blocks)
                - type (`base64` or `url`)
                - media_type
                - data
                - url
              - type, id, name, input (for `tool_use` blocks)
              - type, tool_use_id, content (for `tool_result` blocks)
        - system
        - max_tokens (required)
        - tools
          - name
          - description
          - input_schema
        - tool_choice
          - type (`auto`, `any`, `tool`, `none`)
          - name (when type is `tool`)
    - **response**
        - id
        - type
        - role
        - content (array of content blocks)
          - type (`text` or `tool_use`)
          - text (for `text` blocks)
          - id, name, input (for `tool_use` blocks)
        - model
        - stop_reason (`end_turn`, `max_tokens`, `tool_use`)
        - stop_sequence
        - usage
          - input_tokens
          - cache_creation_input_tokens
          - cache_read_input_tokens
          - output_tokens
- `/v1/completions/render`
    - **request** — same shape as `/v1/completions`; only `model` and `prompt` are inspected
        - model
        - prompt (string, array of strings, array of token ids, or array of arrays of token ids — see [`/v1/completions` prompt forms](#v1completions-prompt-forms))
    - **response** — JSON array, one entry per prompt
        - token_ids (array of token ids; for token-id prompts the input ids are returned verbatim)
        - features (omitted; multimodal features are only produced by the chat render endpoint)
- `/v1/chat/completions/render`
    - **request** — same shape as `/v1/chat/completions`; only `model` and `messages` are inspected
        - model
        - messages (same structure as `/v1/chat/completions`, including `image_url` content blocks)
    - **response** — single JSON object
        - token_ids
        - features (present only when at least one message contains an `image_url` block)
            - mm_hashes (map keyed by modality, e.g. `image`, to an array of opaque hash strings)
            - mm_placeholders (map keyed by modality to an array of placeholder regions)
                - offset (token index where the multimodal region begins)
                - length (number of tokens the region spans)
            - kwargs_data (map keyed by modality to an array of strings, one per multimodal item; content is tokenizer-dependent — see [Render endpoints](#render-endpoints))
- `/v1/responses`
    - **request**
        - stream
        - model
        - input (array of input items)
            - type (`message`)
            - role (`user`, `system`, `developer`)
            - content (string or array of content blocks)
              - type (`input_text`, `input_image`, or `input_audio`)
              - text (for `input_text`)
              - image_url (for `input_image` — a URL string)
              - data (for `input_audio` — base64-encoded audio data)
              - format (for `input_audio` — e.g. `wav`, `mp3`)
        - instructions
        - max_output_tokens
        - text
          - format
            - type (`text`, `json_object`, `json_schema`)
        - include (array of strings, e.g. `["message.output_text.logprobs"]`)
        - top_logprobs
    - **response**
        - id
        - model
        - object (`response`)
        - created_at
        - status (`completed`, `in_progress`)
        - instructions
        - output (array of output items)
            - type (`message`)
            - id
            - role (`assistant`)
            - status
            - content
              - type (`output_text`)
              - text
              - logprobs (when `include` contains `message.output_text.logprobs`)
                - token
                - logprob
                - bytes
                - top_logprobs
        - text
          - format
            - type
        - usage
          - input_tokens
          - output_tokens
          - total_tokens
- `/inference/v1/generate`
    - **request**
        - stream
        - model
        - token_ids
        - sampling_params
            - max_tokens
        - features
            - mm_hashes
        - ignore_eos
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
        - model
        - object
        - request_id
        - choices
            - index
            - finish_reason
            - token_ids
        - kv_transfer_params
          - do_remote_decode
          - do_remote_prefill
          - remote_engine_id
          - remote_block_ids
          - remote_host
          - remote_port
          - tp_size
        - ec_transfer_params (map keyed by remote engine id)
            - peer_host
            - peer_port
            - size_bytes
            - nixl_agent_metadata_b64

### `/v1/responses` examples

#### Text-only request

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "test-model",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {"type": "input_text", "text": "What is the capital of France?"}
        ]
      }
    ]
  }'
```

#### Image input

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "test-model",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Describe what you see in this image."},
          {"type": "input_image", "image_url": "https://example.com/photo.jpg"}
        ]
      }
    ]
  }'
```

#### Audio input

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "test-model",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Transcribe this audio clip."},
          {"type": "input_audio", "data": "BASE64_ENCODED_AUDIO_DATA", "format": "wav"}
        ]
      }
    ]
  }'
```

#### Mixed content (text + image + audio)

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "test-model",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Analyze the following media."},
          {"type": "input_image", "image_url": "https://example.com/diagram.png"},
          {"type": "input_audio", "data": "BASE64_ENCODED_AUDIO_DATA", "format": "mp3"}
        ]
      }
    ]
  }'
```

#### Streaming responses

```bash
curl -N -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "test-model",
    "stream": true,
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Tell me a story."},
          {"type": "input_image", "image_url": "https://example.com/scene.jpg"}
        ]
      }
    ]
  }'
```

The streaming response uses Server-Sent Events (SSE) and emits the following event types in order: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, one or more `response.output_text.delta`, `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, `response.completed`.

## `finish_reason` values

The `finish_reason` field in choices may be one of:

- `stop` — generation finished normally (EOS reached or generation budget exhausted).
- `length` — generation stopped because the `max_tokens` / `max_completion_tokens` limit was reached.
- `tool_calls` — generation produced tool calls (chat completions only).
- `remote_decode` — used when `kv_transfer_params.do_remote_decode` is set; signals that decode is to be performed on a remote pod.
- `cache_threshold` — the request's effective KV-cache hit rate fell below `cache_hit_threshold` (or the global `global-cache-hit-threshold`), or the `X-Cache-Threshold-Finish-Reason: true` header was set. See [KV Cache Guide](kv-cache.md).

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

### Render endpoints

`/v1/completions/render` and `/v1/chat/completions/render` mirror vLLM's `/render` behavior — they return the tokenized form of a request without running generation. They are useful for debugging tokenization, pre-computing prompt token counts, and exercising multimodal feature handling.

Pre-tokenized prompts on `/v1/completions/render` (a token-id array, or an array of token-id arrays) are copied through verbatim — the tokenizer is not invoked for those entries — regardless of which tokenizer is active.

For everything else, behavior depends on the active tokenizer (selected automatically based on `--model`):

- **HuggingFace tokenizer** (real model): each text prompt and chat-completions request is forwarded to the upstream vLLM render service at `--render-url`. For chat requests, `mm_features` returned by the upstream are passed through.
- **Simulated tokenizer** (dummy model): the simulator tokenizes locally using its regex-based splitter. For chat requests containing `image_url` blocks, synthetic `mm_features` are produced so multimodal-aware downstream code paths can be exercised without a real renderer.