# Engines

The simulator separates behavior that every inference engine shares from behavior an engine puts its own
name on. If a client could tell one engine from another by it, it is engine-specific, everything else is
the simulator core. That boundary is the `engine.Engine` interface in
[pkg/engine/engine.go](../pkg/engine/engine.go). `vllm`, in [pkg/engine/vllm/](../pkg/engine/vllm/), is the
only registered engine.

## Selecting an engine

The `engine` setting selects which one runs: see [Configuration](configuration.md#general) for the flag and
[Configuration precedence](configuration.md#configuration-precedence) for how the flag, `SIM_ENGINE`, and
a YAML value rank against each other.

`common.ResolveEngineName` resolves the name before configuration parsing begins, because the chosen
engine registers its own flags and validates its own fields as part of that parse. `engine.Select` then
maps the name to an implementation, rejecting one that is not registered.

## What an engine owns

| Hook | Owns | vLLM implementation |
| --- | --- | --- |
| `Name` | The engine's identifier, also reported as `owned_by` in `/v1/models` | [vllm.go](../pkg/engine/vllm/vllm.go) |
| `ApplyDefaults` | Default values of the configuration groups the engine owns | [defaults.go](../pkg/engine/vllm/defaults.go) |
| `BindFlags` | The engine's CLI flags, and unmarshaling its own YAML blocks out of the raw config tree | [flags.go](../pkg/engine/vllm/flags.go) |
| `ApplyEnv` | The engine's environment variables | [env.go](../pkg/engine/vllm/env.go) |
| `ValidateConfig` | Validation rules for the engine's own fields | [validate.go](../pkg/engine/vllm/validate.go) |
| `BindHTTP` | HTTP routes beyond the OpenAI-compatible set | [transport.go](../pkg/engine/vllm/transport.go) |
| `BindGRPC` | The gRPC service, and whether the engine has one at all | [transport.go](../pkg/engine/vllm/transport.go) |
| `NewMetricsAdapter` | Prometheus metric names, labels, and the fake-metrics schema | [metrics.go](../pkg/engine/vllm/metrics.go), [fakemetrics.go](../pkg/engine/vllm/fakemetrics.go) |
| `NewKVEventEncoder` | The wire format of a single KV-cache event | [kvevents.go](../pkg/engine/vllm/kvevents.go) |

A `Configuration` group can be engine-owned even though the struct lives in `pkg/common`: `lora` and
`kvcache` are declared there so the core can read them, but their defaults, flag names, YAML keys, and
validation rules all come from the engine, and `pkg/common` holds no defaults for them. `fake-metrics`
goes further: `Configuration.FakeMetrics` is an interface, so the engine supplies the concrete type as
well, since the set of fakeable fields mirrors its own real metrics.

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


## Startup order

The order matters because each step's output is the next step's input. `main` resolves the engine, then:

1. `common.ParseCommandParamsAndLoadConfig`:
   1. `NewConfig` sets the common defaults and leaves the engine-owned groups zero-valued.
   2. `ApplyDefaults` fills those groups in.
   3. A `--config` file is loaded, overwriting defaults and yielding the raw YAML tree.
   4. Common flags are registered, then `BindFlags` registers the engine's. Each flag defaults to the
      value the config already holds, so an unset flag preserves the YAML value and a set flag wins.
   5. `f.Parse` reads the command line.
   6. Common environment variables are applied, then `ApplyEnv`. It receives `f.Changed` so an environment
      variable can act as a fallback for an unset flag rather than an override of a set one.
   7. `Configuration.validate` checks the common fields, then `ValidateConfig` checks the engine's.
2. `simulator.Start` builds each rank's `SimContext`, which calls `NewMetricsAdapter` to wire the metrics
   bus and, when the KV cache is enabled and multi-modal encoder-only mode is not, `NewKVEventEncoder` to
   wire the block cache.
3. `Communication.Start` consults `BindGRPC` first, unless multi-modal encoder-only mode is enabled, and
   opens a gRPC listener only if it returns true. It then calls `BindHTTP` while building the HTTP router.

## Adding an engine

1. Create `pkg/engine/<name>/` implementing `engine.Engine`. The closest reference for each hook is the
   vLLM file in the table above.
2. Add one entry to the `registry` map in [pkg/engine/engine.go](../pkg/engine/engine.go). The registry
   holds a constructor, but `vllm.Engine` is an empty struct: per-run state belongs in the objects the
   factory hooks return, not on the engine itself.
3. Document the engine's own flags, environment variables, and metric names in
   [configuration.md](configuration.md) and [metrics.md](metrics.md), and extend the `engine` flag's
   description.
4. Add a test suite in `pkg/engine/<name>/` for the parts with an external contract: the KV-event encoder
   should be asserted against the matching `engineadapter` parser from `llm-d-router`, which is the
   consumer these events exist to feed. Integration suites under `pkg/tests` resolve the engine the same
   way `main` does, so setting `SIM_ENGINE` runs them against a different engine.
