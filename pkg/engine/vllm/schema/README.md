# vLLM request schema

`openapi.json` contains the complete validation schemas for `ChatCompletionRequest`
and `CompletionRequest`, including their nested types, from vLLM 0.21.0, commit
[`ad7125a431e176d4161099480a66f0169609a690`](https://github.com/vllm-project/vllm/tree/ad7125a431e176d4161099480a66f0169609a690).
The source is licensed under Apache-2.0 by the vLLM contributors.

The exporter evaluates the source's data declarations with Pydantic, without
importing the model engine. It omits method bodies, which are not represented in
JSON Schema. It uses `model_json_schema()` directly to preserve integer bounds;
FastAPI's OpenAPI model converts numeric bounds to floating point. The two request
schemas and their references are stored in an OpenAPI 3.1 container. No fields or
nested types are selected by hand.

Regenerate from a clean checkout at the pinned commit:

```sh
python -m venv /tmp/vllm-schema-env
/tmp/vllm-schema-env/bin/pip install pydantic==2.12.5 openai==2.38.0 \
  openai-harmony==0.0.8 pillow==12.3.0 typing-extensions==4.16.0
/tmp/vllm-schema-env/bin/python scripts/export_strict_request_schema.py \
  /path/to/vllm pkg/engine/vllm/schema/openapi.json
```

Update the snapshot, source pin, semantic checks, and tests together. The
simulator embeds this file and performs no downloads or file reads at startup.

## Validation audit

The audit covers every field in both request models, their `model_validator`
methods, `to_sampling_params`, and the validation methods in `SamplingParams`,
`StructuredOutputsParams`, and `RepetitionDetectionParams` at the pinned commit.
All declared fields receive schema validation. The following table records the
additional semantic rules and the limits of the simulator's validation.

| Fields or source checks | Owner and behavior |
| --- | --- |
| Required fields, types, unions, nested messages/content/tools, seed and priority int64 bounds, nonnegative input token IDs, truncate-prompt bounds | Embedded schema; unknown fields follow vLLM's `extra="allow"`. |
| `n < 1`, explicit output-token limit below one | Existing common endpoint validation, with the same messages in strict and lenient mode. |
| `n > VLLM_MAX_N_SEQUENCES` | Engine sampling validation; environment default 16384 and the complete vLLM message. |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `temperature`, `top_p`, `top_k`, `min_p` | Engine sampling ranges. Zero temperature requires `n=1`; small positive temperatures remain sampling requests because vLLM clamps them. |
| `min_tokens`, `max_tokens`, `max_completion_tokens` | Engine checks the minimum against the explicit maximum or the text-completion default of 16. Chat's non-null `max_completion_tokens` takes precedence. Null maximum does not use the omitted-field default. No validation state is added to the parsed request. |
| `stream_options`, `stream` | Nonempty options require streaming. Null and an empty options object are accepted, matching the request model's truthiness check. |
| `logprobs`, `top_logprobs`, `prompt_logprobs` | Endpoint-specific lower bounds and cross-field requirements; positive/all prompt logprobs are rejected with streaming. Chat `top_logprobs=0` does not require `logprobs=true`. |
| `stop`, `stop_token_ids` | Nonempty stop strings; schema checks token-ID element types. The request models do not expose `detokenize`, so SamplingParams' stop/detokenize conflict is not reachable here. |
| `logit_bias`, `allowed_token_ids` | Integer-key conversion and nonempty allowed-token list. Bias values outside [-100, 100] are accepted because vLLM clamps them. Vocabulary-dependent limits are not assumed. |
| `response_format`, `structured_outputs` | Required JSON-schema object, exactly one structured constraint, conflicting format/constraint combinations, empty choices and blank grammars. Constraint-count errors use concise simulator messages rather than Python dataclass reprs. |
| `tools`, `tool_choice` | Empty tool list, missing tools, empty named-tool name, and named-tool membership. Type/shape failures are schema errors. |
| `continue_final_message`, `add_generation_prompt` | Reject explicit simultaneous true values, matching the request model's before-validator. |
| `cache_salt` | Reject an explicitly empty string. |
| `prompt`, `prompt_embeds` | Completion requires a nonempty prompt or embeddings as defined by the request model. Embedding tensor decoding remains outside this validator. |
| `prompt`, `prompt_embeds` list length > `VLLM_MAX_COMPLETION_PROMPTS` | Engine request-model before-validator; environment default 1024 and the complete vLLM message. A prompt list of token IDs is one prompt. |
| `repetition_detection` | Pattern-size ordering/nonnegativity and minimum repetition count when enabled. |
| `use_beam_search` | Model/schema checks still apply; sampling-only rules are skipped because vLLM constructs `BeamSearchParams` instead of `SamplingParams`. The simulator does not perform real beam search. |
| `echo`, `seed=-1`, `include_stop_str_in_output`, `ignore_eos`, special-token/spacing flags, `length_penalty`, `thinking_token_budget`, `bad_words` | Schema checks; these have no additional model-independent rejection in the audited request/sampling path. Existing simulator generation behavior remains authoritative. |
| Template, RAG, multimodal/rendering, reasoning, tool-loading, token-return, request-ID, user, KV-transfer and `vllm_xargs` fields | Schema checks. The request-model iterable normalization, tool defer-loading propagation and system-message warning do not add JSON request rejection rules. |
| Model/engine-dependent `SamplingParams.verify()` | No assumed vocabulary size, `max_logprobs=20`, logits-processor configuration, speculative-decoding configuration, structured-output compiler or backend/tokenizer compatibility. These require the actual serving model/configuration. |
| Tokenization-dependent context capacity | Existing common context validation remains in the generation path. Strict mode does not recompute `min_tokens` against a tokenized prompt or promise parity for vLLM's effective context-limited maximum. |

Schema errors use the simulator's OpenAI error envelope, not Pydantic's formatted
error lists or coercion behavior. Semantic messages mirror the pinned source where
tests assert exact text; structured constraint-count/merge errors and integer-key
conversion diagnostics use simulator formatting. Error precedence across multiple
invalid fields is not a parity guarantee.
