# Engines

The simulator separates behavior that every inference engine shares from behavior an engine puts its own
name on. If a client could tell one engine from another by it, it is engine-specific, everything else is
the simulator core. That boundary is the `engine.Engine` interface in
[pkg/engine/engine.go](../pkg/engine/engine.go). Two engines are registered: `vllm`, in
[pkg/engine/vllm/](../pkg/engine/vllm/), and `sglang`, in [pkg/engine/sglang/](../pkg/engine/sglang/), whose
support is experimental; see [SGLang support (experimental)](#sglang-support-experimental).

## Selecting an engine

The `engine` setting selects which one runs: see [Configuration](configuration.md#general) for the flag and
[Configuration precedence](configuration.md#configuration-precedence) for how the flag, `SIM_ENGINE`, and
a YAML value rank against each other.

`common.ResolveEngineName` resolves the name before configuration parsing begins, because the chosen
engine registers its own flags and validates its own fields as part of that parse. `engine.Select` then
maps the name to an implementation, rejecting one that is not registered.

## SGLang support (experimental)

SGLang support is experimental and incomplete: it is not ready for production use. For now it simulates
the engine-neutral surface: the OpenAI-compatible endpoints, the latency model, and dataset-backed
response generation, with `owned_by` reported as `sglang`, plus LoRA adapters (below). It has no other
native HTTP routes, no gRPC service, no metrics, and no KV cache. Because the settings behind those
features are engine-owned, a configuration file asking for one is rejected rather than silently ignored,
and the corresponding flags do not exist. `/metrics` serves an empty body.

### LoRA adapters

LoRA adapters follow SGLang's tokenizer manager and LoRA registry, under SGLang's flag names (see
[Configuration](configuration.md#sglang)):

- A request runs on an adapter named as `base:adapter` in its `model` field (split on the first colon), or
  else in its `lora_path` field, a string or one entry per prompt (or per embeddings input); a `lora_path`
  of neither form is answered with 400. vLLM has no `lora_path` field and ignores it, whatever its
  value. A bare adapter name in `model` runs the
  base model. The `model` field is checked only on `/v1/responses`, and only when set (an empty string
  counts as set, an absent or null one does not), against the served model names and the
  loaded adapters, by the whole name or its base part; elsewhere any model is accepted. A response reports
  `model` as the request sent it. `/v1/responses` runs every request on the base model.
- A request naming an adapter while LoRA is disabled, or naming one never loaded, is answered with 400.
  An adapter that was loaded once and since unloaded or evicted is loaded again for the request naming it.
- `/load_lora_adapter` and `/unload_lora_adapter` answer with SGLang's `LoRAUpdateOutput` body
  (`success`, `error_message`, `loaded_adapters`), with status 200 on success and 400 on failure. A body
  that fails validation is answered with 400 and type `Bad Request`, as SGLang's validation error handler
  answers it; the message lists pydantic's errors, without the endpoint's source location that SGLang
  appends. Loading a name already loaded, or unloading one
  that is not, fails. An unload stops new requests from using the adapter at once, then waits for the
  requests holding it to complete.
- With `max-loaded-loras` set, each load past the limit evicts the least recently used unpinned adapter,
  where each request naming an adapter counts as a use. At most `max-loras-per-batch - 1` adapters can be
  pinned.
- `/v1/models` lists the loaded adapters least recently used first, with a null `max_model_len`, and none
  while LoRA is disabled.

Which requests run together is not SGLang's: the batch slots adapters occupy are the core's (see
[What stays engine-neutral](#what-stays-engine-neutral)), so a request on an adapter not in the running
batch waits the way it does under vLLM.

The experimental SGLang backend does not apply strict request validation. Its
`NewRequestValidator` hook returns no validator, so the completion endpoints retain their
ordinary request checks even when `strict` is enabled in the configuration.

Every completion choice carries SGLang's `matched_stop` field, naming which `stop` string or
`stop_token_ids` entry ended generation. It is always `null` today: matching a stop condition requires
`stop`/`stop_token_ids` request handling, which neither engine implements yet. This gap is expected to
close as SGLang support matures, not a permanent design boundary the way the vLLM/SGLang split in
[What an engine owns](#what-an-engine-owns) is.


## What an engine owns

| Hook | Owns | vLLM implementation |
| --- | --- | --- |
| `Name` | The engine's identifier, also reported as `owned_by` in `/v1/models` | [vllm.go](../pkg/engine/vllm/vllm.go) |
| `ApplyDefaults` | Default values of the configuration groups the engine owns | [defaults.go](../pkg/engine/vllm/defaults.go) |
| `BindFlags` | The engine's CLI flags, and unmarshaling its own YAML blocks out of the raw config tree | [flags.go](../pkg/engine/vllm/flags.go) |
| `ApplyEnv` | The engine's environment variables | [env.go](../pkg/engine/vllm/env.go) |
| `ValidateConfig` | Validation rules for the engine's own fields | [validate.go](../pkg/engine/vllm/validate.go) |
| `NewRequestValidator` | Strict request schemas and semantic validation rules | [strict_schema.go](../pkg/engine/vllm/strict_schema.go), [strict_validation.go](../pkg/engine/vllm/strict_validation.go) |
| `BindHTTP` | HTTP routes beyond the OpenAI-compatible set | [transport.go](../pkg/engine/vllm/transport.go) |
| `BindGRPC` | The gRPC service, and whether the engine has one at all | [transport.go](../pkg/engine/vllm/transport.go) |
| `ErrorBody`, `StreamErrorBody` | How an error object is framed in a response body | [transport.go](../pkg/engine/vllm/transport.go) |
| `DecorateChatChoice`, `DecorateChatChunkChoice`, `DecorateTextChoice` | Fields a completion choice carries beyond the OpenAI-compatible core | [transport.go](../pkg/engine/vllm/transport.go) |
| `NewMetricsAdapter` | Prometheus metric names, labels, and the fake-metrics schema | [metrics.go](../pkg/engine/vllm/metrics.go), [fakemetrics.go](../pkg/engine/vllm/fakemetrics.go) |
| `NewKVEventEncoder` | The wire format of a single KV-cache event | [kvevents.go](../pkg/engine/vllm/kvevents.go) |
| `NewLoraRegistry` | How a request names its model and adapter, the load and unload endpoints' bodies, and when an adapter stops being available | [lora.go](../pkg/engine/vllm/lora.go) |

A `Configuration` group can be engine-owned even though the struct lives in `pkg/common`: `lora` and
`kvcache` are declared there so the core can read them, but their defaults, flag names, YAML keys, and
validation rules all come from the engine, and `pkg/common` holds no defaults for them. `fake-metrics`
goes further: `Configuration.FakeMetrics` is an interface, so the engine supplies the concrete type as
well, since the set of fakeable fields mirrors its own real metrics.

A single `Configuration` field can be engine-named the same way, for a setting the engines disagree on the
name of. `MaxModelLen`, `MaxNumSeqs` and `MaxWaitingQueueLength` are tagged `yaml:"-"` and have no core
flag: the core keeps the field, its default and its validation, and each engine's `BindFlags` registers
its own flag for the field and claims its own config-file key (`common.ClaimYAMLInt`). Every engine
declares all three, so neither engine's names are the ones the other deviates from, and a name one engine
does not declare reaches no field: the flag is unknown, and the config key is unclaimed and reported like
any other. `common.DeclareConfigIntField` does the three jobs one such field needs — claim the key, register
the flag, record the name — and the recorded name is what `/admin/config` and the startup log report the
field as, so no reader is shown a name the running engine does not take.

## What stays engine-neutral

An engine inherits these rather than reimplementing them. They are the parts most likely to be mistaken
for engine-specific:

- The OpenAI-compatible endpoints and every HTTP handler body. `BindHTTP` chooses which routes exist, the
  handlers themselves are shared and live in `pkg/communication`.
- The worker queue, latency model, dataset-backed response generation, and the tokenizer.
- The KV-event topic (`kv@<ip>:<port>@<model>`), batch envelope, ZMQ framing, and the publisher. The seam
  sits at the individual event, which is why `EventEncoder` encodes one event rather than a whole batch.
  See [KV cache](kv-cache.md).
- Block-cache accounting: eviction order, reference counting, and per-model block keys.
- The batch slots LoRA adapters occupy (`max-loras` under vLLM, `max-loras-per-batch` under SGLang), and
  the scheduling that runs a waiting request on an adapter already in the batch ahead of one on an adapter
  that is not.


## Startup order

The order matters because each step's output is the next step's input. `main` resolves the engine, then:

1. `common.ParseCommandParamsAndLoadConfig`:
   1. `NewConfig` sets the common defaults and leaves the engine-owned groups zero-valued.
   2. `ApplyDefaults` fills those groups in.
   3. A `--config` file is loaded, overwriting defaults and yielding the raw YAML tree.
   4. Common flags are registered, then `BindFlags` registers the engine's own and claims the keys of the
      fields it names, the raw tree in hand. Each flag defaults to the value the config already holds, so
      an unset flag preserves the YAML value and a set flag wins.
   5. Any top-level YAML key still unclaimed is reported as unrecognized. The engine claims its own
      groups by deleting them from the raw tree as it reads them, so an engine that does not implement
      a feature gets the rejection for free and never names another engine's keys. Another engine's name
      for an engine-named field is unclaimed too, so a file written for a different engine is reported
      the same way.
   6. `f.Parse` reads the command line.
   7. Common environment variables are applied, then `ApplyEnv`. It receives `f.Changed` so an environment
      variable can act as a fallback for an unset flag rather than an override of a set one.
   8. `Configuration.validate` checks the common fields, then `ValidateConfig` checks the engine's.
2. `simulator.Start` builds each rank's `SimContext`, which calls `NewLoraRegistry` to hold the rank's
   LoRA adapters, `NewMetricsAdapter` to wire the metrics bus and, when the KV cache is enabled and multi-modal encoder-only mode is not, `NewKVEventEncoder` to
   wire the block cache.
3. `Communication.Start` calls `NewRequestValidator` when strict validation is enabled. It consults
   `BindGRPC`, unless multi-modal encoder-only mode is enabled, and opens a gRPC listener only if it
   returns true. It then calls `BindHTTP` while building the HTTP router.


## How an error is framed

An error body's shape depends on the route as much as on the engine: both engines answer `/v1/messages` in
the Anthropic envelope and `/v1/responses` nested under an `error` key, because a client of either route is
an SDK whose typed error classes parse nothing else. So the route selects the family (`api.ErrorRoute`,
derived from the request path) and the engine fills it in (`ErrorBody`, `StreamErrorBody`).

| route family | vLLM | SGLang |
| --- | --- | --- |
| the OpenAI-shaped routes | `{"error": {message, type, param, code}}` | the same fields at the top level, next to `"object": "error"` |
| `/v1/responses` | as above | nested, with every error type spelled `invalid_request_error` whatever the status |
| `/v1/messages` | `{"type": "error", "error": {type, message}}`, keeping the error type it uses elsewhere | the same envelope, with the type mapped to Anthropic's own vocabulary (`invalid_request_error`, `rate_limit_error`, ...) and a 5xx message replaced by `Internal server error` |

A streaming frame is framed separately, since an engine need not frame one the way it frames a whole body:
SGLang wraps the OpenAI-shaped routes' frames under an `error` key, and keeps the error's own type on
`/v1/responses` where a whole body would not. The SSE frame around it belongs to the route: the Messages API
names the event (`event: error`) and ends the stream with no terminator, where the other routes send a bare
`data:` frame followed by `[DONE]`.

What an error body *says* is not engine-specific, except for a LoRA error, which the engine's
`NewLoraRegistry` words. Otherwise the message wording and the `param` field are the simulator's own, so
SGLang's `The model 'x' does not exist` with `"param": "model"` reads as ``The model `x` does not exist.``
with a null param.


## Adding an engine

1. Create `pkg/engine/<name>/` implementing `engine.Engine`. The closest reference for each hook is the
   vLLM file in the table above.
2. Add one entry to the `registry` map in [pkg/engine/registry.go](../pkg/engine/registry.go). The registry
   holds a constructor, but `vllm.Engine` is an empty struct: per-run state belongs in the objects the
   factory hooks return, not on the engine itself.
3. Document the engine's own flags, environment variables, and metric names in
   [configuration.md](configuration.md) and [metrics.md](metrics.md), and extend the `engine` flag's
   description.
4. Add a test suite in `pkg/engine/<name>/` for the parts with an external contract: the KV-event encoder
   should be asserted against the matching `engineadapter` parser from `llm-d-router`, which is the
   consumer these events exist to feed. In `pkg/tests`, a spec that exercises only the engine-neutral
   surface belongs in a container wrapped in `forEachEngine`, which runs it once per registered engine, so
   a new engine inherits that coverage. Every other spec names its own engine. Suites resolve the engine
   the same way `main` does, so setting `SIM_ENGINE` selects it for the specs that do not.
