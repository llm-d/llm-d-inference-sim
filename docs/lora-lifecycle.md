# LoRA handling in SGLang, vLLM and the simulator

This page describes how each engine registers, serves, evicts and unloads LoRA adapters, and how the
simulator models that. The user-facing contract of the simulator's SGLang engine is in
[Engines](engine-backends.md#lora-adapters); this page is the reference behind it.

Everything here was read from source, not run. Upstream behavior is quoted from SGLang commit `6c3db0f`
(`python/sglang/srt/...`) and vLLM commit `1b2a666` (`vllm/...`). Simulator behavior is read from the code
in this repository; its tests were not consulted. Behavior that was not read is listed under
[Not checked](#not-checked).

## Terms

Both engines keep adapters at up to three levels. The names below are used throughout.

| Level | Meaning |
|---|---|
| Registered | The engine knows the adapter by name and can serve requests naming it |
| Resident | The adapter's weights are held in CPU memory by the engine's worker |
| Active | The adapter occupies a GPU slot, which a running batch needs |

## SGLang

### Components

Two processes hold state.

The tokenizer manager process (`managers/tokenizer_manager.py`, `managers/tokenizer_control_mixin.py`,
`lora/lora_registry.py`):

- `LoRARegistry`: an ordered dict from adapter name to `LoRARef` (name, path, pinned flag, `lora_id`), kept in
  least-recently-used order, plus one `ConcurrentCounter` of in-flight requests per `lora_id`. `acquire` and
  `get_unregistered_loras` move a name to the most-recently-used end.
- `lora_ref_cache`: a dict from name to the latest `LoRARef` of every adapter ever loaded, including the
  `--lora-paths` entries. Nothing deletes from it.
- `lora_update_lock`: one asyncio lock held for the whole of every load and unload.
- `pending_lora_unloads`: names whose unload started but did not complete.

The scheduler process (`lora/lora_manager.py`, `lora/mem_pool.py`, `lora/eviction_policy.py`):

- `LoRAManager`: `lora_refs`, `configs` and `loras` (the weights, "cached in CPU memory", per the comment in
  `init_lora_adapters`), and `num_pinned_loras`.
- `LoRAMemoryPool`: a GPU buffer allocated at startup with `max_loras_per_batch` slots, each sized for
  `--max-lora-rank` and `--lora-target-modules`, plus an eviction policy (`lru` or `fifo`).

So the levels map as: registered = in `LoRARegistry`; resident = in `LoRAManager.loras`; active = in a
`LoRAMemoryPool` slot.

### Startup

`--lora-paths` entries reach `LoRAManager.init_state`, which loads them one by one with the same
`_load_lora_adapter` that serves `/load_lora_adapter`. The first failure aborts startup with
`Failed to load LoRA adapter {name}: {error}`. The tokenizer manager registers the same entries in
`LoRARegistry` and seeds `lora_ref_cache`.

The pool shape is fixed from `--max-lora-rank` and `--lora-target-modules`, or inferred from the startup
adapters. With no `--lora-paths`, both must be given (`init_state` asserts it).

Flag checks (`arg_groups/lora_hook.py`): `max_loaded_loras >= max_loras_per_batch`, and
`len(lora_paths) <= max_loaded_loras`. With overlap loading enabled, `max_loaded_loras` must be at most
`2 * max_loras_per_batch`.

### Loading an adapter (`POST /load_lora_adapter`)

Body: `lora_name`, `lora_path`, optional `pinned` (default false). Response: `LoRAUpdateOutput` with
`success`, `error_message`, `loaded_adapters` (name to path), status 200 on success and 400 on failure.

1. If LoRA is disabled, fail with ``LoRA is not enabled. Please set `--enable-lora` to enable LoRA.``
2. Take `lora_update_lock`. Other loads and unloads wait here.
3. If the name is in `pending_lora_unloads`, fail with `LoRA adapter '{name}' has an incomplete unload.
   Retry the unload before loading it again.`
4. Build a `LoRARef` and send the load to the scheduler. `LoRAManager._load_lora_adapter` then:
   - reads the adapter config;
   - `validate_new_adapter` rejects adapters that add vocabulary tokens, DoRA adapters, a name that is already
     loaded (`Failed to load LoRA adapter {name} because it is already loaded`), an adapter whose rank or
     target modules do not fit the pool (`... is incompatible with the current LoRA memory pool
     configuration ...`), and a pinned adapter when `num_pinned_loras >= max_loras_per_batch - 1`;
   - reads the weights into CPU memory (`load_lora_weights`);
   - records the `LoRARef` in `lora_refs` and counts it in `num_pinned_loras` if pinned.
5. On success the tokenizer manager registers the adapter as most recently used and stores it in
   `lora_ref_cache`.
6. If `max_loaded_loras` is set, evict while the registry holds more than that (see
   [Eviction at max-loaded-loras](#eviction-at-max-loaded-loras)). This step runs whether or not the load
   succeeded.

A load does not put the adapter in a GPU slot. The load path never touches the pool; a pinned adapter is
counted in `num_pinned_loras` but gets no slot until a batch needs it.

### A request naming an adapter

Naming (`entrypoints/openai/serving_base.py`, used by `serving_completions.py` and `serving_chat.py`): the
`model` field is split on its first colon into `base:adapter`, both parts stripped; an empty adapter counts
as none. An adapter from `model` takes precedence over the request's `lora_path`, which is a string or a list
with one entry per prompt (`None` for the base model). Other endpoints were not read.

`_validate_and_resolve_lora` (tokenizer manager) then runs before the request is tokenized or sent to the
scheduler. It does nothing when `lora_path` is empty, so a base-model request skips everything below.

1. LoRA disabled: fail with `LoRA adapter '{first}' was requested, but LoRA is not enabled. Please launch
   the server with --enable-lora flag and preload adapters using --lora-paths or /load_lora_adapter
   endpoint.`
2. If `max_loaded_loras` is set and the request names more unique adapters than that (a `None` entry in a
   `lora_path` list counts as one), fail with `Received request with {n} unique loras requested but max
   loaded loras is {m}`.
3. For each name missing from the registry (`get_unregistered_loras`; names present are touched):
   - not in `lora_ref_cache`: fail with `Got LoRA adapter that has never been loaded: {name}` and the list of
     known names;
   - otherwise log `Reloading evicted adapter` and call `load_lora_adapter` with the cached name, path and
     pinned flag. That is a full load, with its lock wait and its possible eviction. A failure other than
     `already loaded` fails the request with `Failed to implicitly load LoRA adapter {path}: {error}`.
4. `lora_registry.acquire` looks every name up again, moves it to the most-recently-used end and increments
   its counter. A name that disappeared in between fails with `The following requested LoRA adapters are not
   loaded: {name}`.
5. The request is sent to the scheduler after all of the above succeeds.
6. When the request finishes or aborts, `_release_lora_once` decrements each counter, once per request.

The request therefore holds each adapter from step 4 until completion, queue time included.

### Scheduling and GPU slots

For each waiting request the scheduler calls `can_schedule_lora_req` (`managers/scheduler.py`), with
`running_loras` built from the running batch plus requests already admitted in this pass. The set includes
`None` for base-model requests.

- The adapter is already in `running_loras`: admit.
- Otherwise, with overlap loading off, admit if `LoRAManager.validate_lora_batch({adapter} | running_loras)`
  holds. It requires `len(set) <= max_loras_per_batch` and, when pinned adapters exist,
  `len(set) - pinned_in_batch <= max_loras_per_batch - num_pinned_loras`.
- A request that fails the check is skipped and later requests are still considered.

Base-model requests count in the set. Pinned adapters reduce the slots available to others whether or not
they are running.

Once a batch is formed, `fetch_new_loras` calls `LoRAMemoryPool.prepare_lora_batch`. For each adapter in the
batch without a slot it takes an empty slot or evicts one, then copies the weights from CPU memory into the
slot. Eviction candidates exclude adapters in the current batch and pinned adapters, and prefer real adapters
over the base model. The policy picks the victim: `lru` evicts the least recently marked-used, `fifo` the
first inserted. With no candidate it raises `No available buffer slots found ...`. The evicted adapter stays
registered and resident; only its slot is lost, and a later batch copies it in again.

### Unloading an adapter (`POST /unload_lora_adapter`)

Body: `lora_name`. Same response shape and statuses as load. Also the path taken by eviction.

1. LoRA disabled: fail with the not-enabled message above.
2. Take `lora_update_lock`.
3. If the name is not in `pending_lora_unloads`, remove it from `LoRARegistry` (`unregister`) and record it in
   `pending_lora_unloads`. An unknown name fails with `LoRA with name {name} does not exist. Loaded LoRAs:
   {keys}`.
4. A new request naming it does not find it in the registry. It takes the reload path above, which waits on
   `lora_update_lock` behind this unload.
5. `wait_for_unload` blocks until the adapter's counter is zero. There is no timeout. Requests already
   holding the adapter finish normally.
6. The unload is sent to the scheduler. `_unload_lora_adapter` clears the adapter's GPU slot if it has one,
   and removes it from `configs`, `loras` and `lora_refs`, decrementing `num_pinned_loras` if pinned.
7. On success the name leaves `pending_lora_unloads`. On failure it stays, and a retry skips step 3 and keeps
   the same `lora_id`.

`lora_ref_cache` is not touched, so the adapter remains reloadable by any later request.

Scenarios:

| Scenario | Result |
|---|---|
| Unload an idle adapter | Registry and scheduler drop it; response lists the remaining adapters |
| Unload with a request running on it | The request completes; the unload call returns only after it does, and holds `lora_update_lock` meanwhile |
| Load or unload while another unload waits | Waits for `lora_update_lock` |
| Request for the adapter while its unload waits | Waits for the lock, then reloads it, then runs |
| Request for the adapter after the unload | Reloads it from `lora_ref_cache`, then runs |
| Unload a name that is not registered | 400, `LoRA with name ... does not exist` |

### Eviction at max-loaded-loras

After a load, while the registry holds more adapters than `max_loaded_loras`:

1. `lru_lora_name(exclude_pinned=True)` returns the first unpinned name in registry order, that is, the least
   recently used. With none, the load call fails with `Didn't find any LoRA adapters when trying to evict LRU
   LoRA adapter.`
2. That adapter goes through the full unload above, including the wait for its in-flight requests.
3. A failure there fails the load call.

The new adapter is registered before eviction starts. Eviction removes the weights from the process, not just
from the GPU. The evicted adapter can come back through the implicit reload.

### Pinning

`pinned` is set at load time (`--lora-paths` JSON form or `/load_lora_adapter`). Effects found in code:

- Load fails when `num_pinned_loras >= max_loras_per_batch - 1`.
- Excluded from `max_loaded_loras` eviction (`lru_lora_name(exclude_pinned=True)`).
- Skipped as a GPU-slot eviction candidate in `prepare_lora_batch`.
- Counted in `validate_lora_batch`'s vacancy computation (above).
- Preserved by implicit reload, through the cached `LoRARef`.

### Pool shape

`--max-lora-rank` and `--lora-target-modules` size every slot. `LoRAMemoryPool.can_support` accepts an adapter
only if its rank is at most the maximum and its target modules are a subset of the pool's (or `all`).

## vLLM

### Components

The API server process (`entrypoints/openai/models/serving.py`, class `OpenAIServingModels`):

- `lora_requests`: a dict from adapter name to `LoRARequest` (name, int id, path). It is the set of names the
  server accepts.
- `lora_id_counter`: allocates int ids.
- One lock per adapter name, taken by load, unload and resolve.

The worker (`lora/worker_manager.py` `LRUCacheWorkerLoRAManager`, `lora/model_manager.py`
`LRUCacheLoRAModelManager`), holding two LRU caches keyed by int id:

- `_registered_adapters`: capacity `max_cpu_loras`; the resident adapters.
- `_active_adapters`: capacity `max_loras`; the GPU slots (`lora_index_to_id`).

So the levels map as: registered = in `lora_requests`; resident = in `_registered_adapters`; active = in
`_active_adapters`. The first is kept by the server and the other two by the worker, and nothing keeps them
in step.

`LoRAConfig` (`config/lora.py`): `max_loras` defaults to 1; `max_cpu_loras` defaults to `max_loras` and must
not be smaller.

### Startup

`--lora-modules` entries go through `init_static_loras`, which runs the same `load_lora_adapter` as the HTTP
endpoint. Any failure raises and aborts startup.

### Loading an adapter (`POST /v1/load_lora_adapter`)

The load and unload routes exist only when `VLLM_ALLOW_RUNTIME_LORA_UPDATING` is set
(`entrypoints/serve/lora/api_router.py`). Body: `lora_name`, `lora_path`, optional `load_inplace`,
`is_3d_lora_weight`.

Under the per-name lock:

1. Reject, with status 400: a missing name or path (`Both 'lora_name' and 'lora_path' must be provided.`); a
   name equal to a served base model (`... conflicts with a served base model.`); a name already in
   `lora_requests` unless `load_inplace` (`The lora adapter '{name}' has already been loaded. If you want to
   load the adapter in place, set 'load_inplace' to True.`).
2. Choose the int id: the existing one if the name is already in `lora_requests` (in-place reload), otherwise
   a new one from `lora_id_counter`.
3. Call `engine_client.add_lora`, which the code comment says "will also preload it for incoming requests".
   In the worker, `add_adapter`:
   - loads the adapter from disk into CPU memory (`_load_adapter`; an unreadable adapter fails here);
   - removes an existing entry with the same id if any (in-place case);
   - if the resident cache would exceed `max_cpu_loras`, evicts the oldest unpinned entry;
   - adds the new one and calls `activate_adapter`, which puts it in a GPU slot, evicting the oldest active
     adapter if all `max_loras` slots are used.
4. Only then does the server record the name in `lora_requests`.
5. Success: status 200 with the text `Success: LoRA adapter '{name}' added successfully.`

A load error from the engine becomes an error response; any exception other than the adapter-not-found one
becomes a 500 `InternalServerError`.

So a vLLM load makes the adapter resident and active immediately, which differs from SGLang.

### A request naming an adapter

The `model` field is the adapter name; vLLM has no `lora_path` request field (`LoRARequest` carries the
path). In `_check_model` (`entrypoints/serve/engine/serving.py`):

1. A served base model name, or an empty model: valid, no adapter.
2. A name in `lora_requests`: valid.
3. Otherwise, only when `VLLM_ALLOW_RUNTIME_LORA_UPDATING` is set, `resolve_lora` is tried (below).
4. Otherwise, 404 `NotFoundError`: ``The model `{model}` does not exist. Valid aliases: {served names}.``

`_maybe_get_adapters` then returns the `LoRARequest` from `lora_requests` and the request is handed to the
engine with it. If LoRA is not enabled in the engine, the input processor rejects it
(`Got lora_request ... but LoRA is not enabled!`).

### Scheduling and GPU slots

In `schedule()` (`v1/core/sched/scheduler.py`), `scheduled_loras` is the set of int ids of scheduled requests
that have a `lora_request`. A waiting request with an adapter is skipped, and later requests are still
considered, when `len(scheduled_loras) == max_loras` and its id is not in the set. Base-model requests have no
`lora_request` and take no slot.

In the worker, `_apply_adapters` runs for each step's adapter set. It raises `RuntimeError` if the set exceeds
the slot count, then calls `add_adapter` for each adapter:

- not resident (or `load_inplace`): load from `lora_request.lora_path`, evict the oldest resident entry if
  over capacity, add;
- resident: touch it, moving it to the most-recently-used end;
- then `activate_adapter`: if not active, evict the oldest active adapter when all slots are full, copy the
  weights into a free slot, and touch.

So an adapter evicted from the resident cache is loaded from disk again by the next step that needs it, and
the request is not rejected. This was read in the code; no run confirmed it.

### Unloading an adapter (`POST /v1/unload_lora_adapter`)

Body: `lora_name`. Under the per-name lock:

1. Missing name: 400 `'lora_name' needs to be provided to unload a LoRA adapter.`
2. Name not in `lora_requests`: 404 `NotFoundError`, ``The lora adapter '{name}' cannot be found.``
3. `del lora_requests[name]`; success: status 200 with the text `Success: LoRA adapter '{name}' removed
   successfully.`

The handler never calls `engine_client.remove_lora`. A search of `vllm/entrypoints` found no caller of
`remove_lora`, and none for `pin_lora` either, so there is no HTTP route for pinning. Consequences read from
the code:

- The unload does not wait for running requests; they hold their own `LoRARequest` and continue.
- The worker keeps the weights in its resident cache, and in a GPU slot if active, until LRU eviction
  displaces them.
- Loading the same name again allocates a new int id (the name is absent from `lora_requests`), so the old
  id's cache entry is not reused and ages out.
- New requests naming it get the 404 above, unless a resolver finds it.

### Resolver plugins

Only when `VLLM_ALLOW_RUNTIME_LORA_UPDATING` is set. For a model name not in `lora_requests`,
`resolve_lora` takes the per-name lock, rechecks `lora_requests`, then asks each registered `LoRAResolver`
in turn. A resolver that returns a `LoRARequest` is given a new int id and loaded through `add_lora`; on a
load failure the next resolver is tried. Outcomes of `resolve_lora`: success returns the adapter and records
it in `lora_requests`; no resolver found it, a 404 `LoRA adapter {name} does not exist`; found but none
loaded, a 400 `LoRA adapter '{name}' was found but could not be loaded.` `_check_model` passes only the 400
through. For the 404 it answers with its own ``The model `{model}` does not exist. Valid aliases: ...``.

## Side by side

| | SGLang | vLLM |
|---|---|---|
| Name in OpenAI requests | `base:adapter` in `model`, or `lora_path` | adapter name as `model`; no `lora_path` |
| Dynamic load/unload routes | always present | only with `VLLM_ALLOW_RUNTIME_LORA_UPDATING` |
| Load response | JSON `LoRAUpdateOutput`, 200 or 400 | text `Success: ...`, 200; errors 400, 404 or 500 |
| Load puts adapter in | CPU memory | CPU memory and a GPU slot |
| Duplicate name on load | 400 | 400 unless `load_inplace` |
| Unload frees weights | yes, after in-flight requests finish | no engine call; weights age out of the worker cache |
| Unload vs running request | unload call waits for it | unload returns at once; request continues |
| Request for an unloaded name | reloaded from `lora_ref_cache` | 404, unless a resolver finds it |
| Cap on registered adapters | `max-loaded-loras`; LRU unpinned adapter is fully unloaded on a load | none on names; `max-cpu-loras` bounds the worker's resident cache |
| GPU slot limit | `max-loras-per-batch` | `max-loras` |
| Slot victim policy | `lru` or `fifo`, skipping batch members and pinned | oldest active adapter (LRU) |
| Base model against slot limit | counts | does not count |
| Pinning | at load; reserves slots in admission | `pin_lora` in the engine API; no HTTP route found |
| Admission when over the slot limit | request skipped, others tried | request skipped, others tried |

## The simulator

### Structure

The LoRA code is split in two (`pkg/lora/`, `pkg/simulator/lora.go`, `pkg/engine/*/lora.go`):

- The engine's `lora.Registry` (`NewLoraRegistry`) decides what names exist and what the load and unload
  endpoints do: `ResolveModel`, `Acquire`, `Release`, `IsAdapter`, `ModelCards`, `Load`, `Unload`.
- The core (`SimContext`, `loras` state) decides which adapters occupy the `max-loras` batch slots. It is
  shared by both engines and never consults the registry for slot decisions.

The simulator holds no weights, reads no adapter paths, and adds no delay for loads, unloads or reloads.

### Request flow in the core

`Simulator.HandleRequest`:

1. Failure injection, if configured.
2. `loraRegistry.ResolveModel(route, model field)` validates the model and sets the displayed model.
3. The request is split into sub-requests. `loraRegistry.Acquire(route, model, loraPaths)` returns one adapter
   per sub-request ("" for the base model), reloading evicted or unloaded adapters first. An error is returned
   to the client. Each sub-request records its adapter (`SetLoraAdapter`).
4. Each sub-request's done callback calls `loraRegistry.Release(adapter)`, so a request holds its adapter
   from admission to completion, queue time included.
5. The sub-requests enter the core's worker queue.

Slot accounting (`pkg/simulator/lora.go`, `simulator.go`): `loras.loadedLoras` maps an adapter name to the
number of running requests on it. An entry with count 0 stays until displaced.

- `loadLora(adapter)`: "" is always allowed. Otherwise allowed when the adapter is already in the map or the
  map has fewer than `maxLoras` entries. If neither, it drops the first entry with count 0 it finds (map
  iteration order), tells the KV cache helper the victim is unloaded, and allows the request. If no entry has
  count 0, it refuses. On success it increments the count, assigns a 1-based LoRA id from a `maxLoras`-sized
  table, and tells the KV cache helper the adapter is loaded.
- A new request takes a free worker and calls `loadLora`. With no free worker, or when `loadLora` refuses, it
  joins the waiting queue.
- When a worker finishes, `decrementLora` lowers the count; at zero it signals `loraRemovable`, and the loop
  re-scans the queue.
- `dequeue` scans the queue for the first request whose adapter is already in the map, and otherwise for the
  first whose `loadLora` succeeds.

A request that is not admitted is skipped and later ones are tried, as in both engines. The map is keyed by
name and independent of the registry, so an adapter unloaded from the registry stays in the map until
displaced.

### vLLM engine registry (`pkg/engine/vllm/lora.go`)

- State: a map from name to path, seeded from `--lora-modules`.
- `ResolveModel`: an adapter name is accepted and displayed as is; a served model name is displayed as the
  first served name; anything else is 404 ``The model `{x}` does not exist.``
- `Acquire`: the adapter is the `model` field if it names one; `lora_path` is ignored. `Release` does
  nothing.
- `Load`: decodes `lora_name` and `lora_path` and stores them. A body that is not valid JSON is 400. Status
  200 with an empty body otherwise.
- `Unload`: decodes the name and deletes it. Status 200, whether or not it existed.
- `max-cpu-loras` is validated (defaults to `max-loras`, must not be smaller) and not used afterwards.
  `max-loras` sets the core's slot count and is the `max_lora` label of the LoRA metric.
- The load and unload routes are always registered (`BindHTTP`).

### SGLang engine registry (`pkg/engine/sglang/lora.go`, `lora_request.go`, `validate.go`)

State: `registered` (loaded adapters, least recently used first), `known` (every adapter ever loaded, in the
order first loaded, a repeated name replacing its entry in place), `inFlight` counts, `numPinned`, a `sync.Cond`
signalled on release, and `updateMu`, which serializes loads and unloads as `lora_update_lock` does.

Settings: `max-loras-per-batch` is the core's slot count and the pin limit; `max-loaded-loras` is stored in
`Lora.MaxCPULoras` (0 means no limit) and caps `registered`.

Startup (`validate.go`, mirroring SGLang's checks and messages): `max-loras-per-batch` positive; target modules
from the allowed set; `all` alone; `max-lora-rank` and `lora-target-modules` required when LoRA is enabled
without `lora-paths`; `max-loaded-loras >= max-loras-per-batch` and at least the number of `lora-paths`;
duplicate names and over-limit pins rejected with SGLang's `failed to load LoRA adapter ...` wording. LoRA is
enabled when `lora-paths` is given unless disabled explicitly. The startup adapters are registered and known.

`Load` (body validated as FastAPI and pydantic would, a failure being a 400 `Bad Request`):

1. LoRA disabled: failed update, status 400.
2. Take `updateMu`.
3. `register`: a name already registered fails; a pinned load with `numPinned >= max-loras-per-batch - 1`
   fails; otherwise it appends to `registered` as most recently used, records it in `known`, counts a pin.
4. If `max-loaded-loras` is set: while `registered` exceeds it, take the first unpinned entry and unload it
   (waiting as `Unload` does); with none unpinned, fail with SGLang's `Didn't find any LoRA adapters ...`
   message. This runs after a failed register too.
5. Response: `success`, `error_message` and `loaded_adapters` (name to path); 200 on success, 400 otherwise.

`Unload`: disabled fails. Take `updateMu`. An unknown name fails with `LoRA with name {name} does not exist.
Loaded LoRAs: ...`. Otherwise remove it from `registered`, release its pin count, then wait on the condition
until `inFlight[name]` is zero. There is no timeout. `known` is untouched.

`Acquire` (non-Responses routes; the Responses route runs every request on the base model):

1. A `lora_path` that is neither a string nor a list of strings or nulls: 400.
2. The adapter per prompt is the part after the first colon of `model` when non-empty, else the prompt's
   `lora_path` entry, else "".
3. If any adapter is named and LoRA is disabled: 400 with SGLang's not-enabled message.
4. If `max-loaded-loras` is set and the unique adapters (plus one when a prompt is on the base model among
   adapters) exceed it: 400 `Received request with {n} unique loras requested but max loaded loras is {m}`.
5. Each named adapter not registered is reloaded from `known` through `load`; a name not in `known` gives 400
   `Got LoRA adapter that has never been loaded: ...`. A reload failure other than "already loaded" gives 400
   `Failed to implicitly load LoRA adapter ...`.
6. Under one lock, every named adapter must still be registered (else 400 `The following requested LoRA
   adapters are not loaded: ...`); then each gets `inFlight++` and moves to the most-recently-used end.

`ResolveModel`: nil model returns "". Only the Responses route checks the model, against the served names and
registered adapters, by whole name or the part before the colon; a miss is 404. Elsewhere the model is
accepted and reported as received.

`ModelCards`: none while LoRA is disabled; otherwise the registered adapters, least recently used first, rooted
at their paths, without `max_model_len`.

### Differences from the engines

Read from the code on both sides.

SGLang engine:

| Area | SGLang | Simulator |
|---|---|---|
| Base model against `max-loras-per-batch` | counts | does not count (see [Engines](engine-backends.md#lora-adapters)) |
| Pinned adapters in admission | reserve slots | no effect beyond the load-time pin limit and eviction exemption |
| GPU-slot victim | `lru` or `fifo`, skipping batch members and pinned | first idle entry in map iteration order; `lora-eviction-policy` is accepted and ignored |
| Adapter shape checks | rank and target modules against the pool; DoRA and added-token adapters rejected | none; `max-lora-rank` and `lora-target-modules` are validated at startup and otherwise unused |
| Adapter path | read at load; failure fails the load | never read |
| Weight copy and reload cost | disk read on load and implicit reload; host-to-device copy into a slot | no delay |
| Overlap loading, drain threshold, backends | change scheduling | accepted and ignored |
| Failed unload | `pending_lora_unloads` blocks reloading the name until the unload is retried | no equivalent |

vLLM engine:

| Area | vLLM | Simulator |
|---|---|---|
| Load and unload routes | present only with `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | always registered |
| Load of an existing name | 400 unless `load_inplace` | accepted, path overwritten; `load_inplace` ignored |
| Load of a base model's name | 400 | accepted |
| Load with a missing name or path | 400 | accepted |
| Unload of an unknown name | 404 | 200 |
| Success bodies | text `Success: ...` | empty |
| Unknown model message | ends with `Valid aliases: {names}.` | ends after `does not exist.` |
| Resident cache and `max-cpu-loras` | LRU cache of that size in the worker | not modeled; validated only |
| Resolver plugins | supported | not modeled |
| Disk read and slot activation cost | on load, and on a request whose adapter is not resident | no delay |

## Not checked

- Upstream behavior was not run. Statements about what vLLM does after a resident-cache eviction come from
  reading `add_adapter` and `_apply_adapters`, not from tracing a request.
- SGLang model-name handling was read for the completions and chat endpoints only. Whether other endpoints
  parse `base:adapter`, and whether SGLang validates the base part of `model`, was not read.
- The HTTP status and body SGLang returns when `_validate_and_resolve_lora` raises (the simulator answers 400),
  and what SGLang's `/v1/models` lists for adapters.
- That SGLang's `/v1/responses` runs every request on the base model; the simulator implements it.
- Multi-rank behavior beyond what the code comments state: SGLang merges per-rank load results, failing if any
  rank fails, and dynamic loading asserts a data-parallel size of 1.
- SGLang's overlap loader and drainer internals, the weight-copy code in `load_lora_weight_to_buffer`, the
  `csgmv` and `triton` backends, and the embedding and MoE adapter paths.
- vLLM's `default_mm_loras`, `lora_dtype`, the resolver plugin implementations, and the v0 engine.
- Defaults of SGLang's flags, which were not read from code.
