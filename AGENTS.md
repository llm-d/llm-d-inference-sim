# AGENTS.md

llm-d-inference-sim. Go service that simulates vLLM's HTTP and gRPC surface — request handling, streaming responses, latency, LoRA lifecycle, and KV-cache events over ZMQ — without loading real models. Used for load testing, routing/EPP integration testing, and end-to-end scenario reproduction. Multi-vendor open source under the llm-d project. Review bandwidth is a shared community resource, so scope work tightly and discuss substantive changes in the open before code lands.

`make help` lists targets. `make presubmit` is the pre-merge gate. Host Go and `make` are required.

## Agent operating rules

**Allowed.** Edit code, run `make` targets, read the codebase and GitHub state.

**Ask first.** Pushing commits to any branch (including feature branches), rewriting pushed history, edits under `.github/` or to `OWNERS`, dependency upgrades. When asking, propose the specific change and the reason in one message; do not start the work in the same turn.

**Never, without explicit per-turn authorization.** Public actions under the user's identity: GitHub comments, reviews, reactions, PR state changes, label or reviewer assignment, posts to Slack or any external surface. Draft such replies as quoted text for the user to send. Authorization is per-action and does not carry between actions or to sub-agents.

## Working in the codebase

- State your interpretation before coding. When the task has multiple valid reads, ask; don't pick one silently. For clear failure signals (logs, failing tests, reproducer), act; the ask rule is about unclear requirements, not unclear bugs.
- Define success as a checkable outcome: "add validation" becomes "write failing tests for invalid inputs, then make them pass". Where the issue is reproducible, the failing test IS the success criterion; write it first and let it gate the implementation.
- Before changing or extending a component, read an analogous one in the repository. The closest existing implementation is the canonical pattern; follow its structure, naming, and tests rather than introducing new conventions.
- Orientation: [docs/http-endpoints.md](docs/http-endpoints.md) describes the API surface; [docs/latency-simulation.md](docs/latency-simulation.md), [docs/response_generation.md](docs/response_generation.md), [docs/kv-cache.md](docs/kv-cache.md), and [docs/metrics.md](docs/metrics.md) cover the four core subsystems. Code lives under `pkg/api/` (wire types), `pkg/communication/` (HTTP + gRPC handlers), `pkg/llm-d-inference-sim/` (simulator core: worker queue, latency, generation, LoRA), `pkg/kv-cache/` (block cache + ZMQ publisher), `pkg/dataset/` (response corpora), `pkg/tokenizer/` (HuggingFace tokenizer + render client), and `pkg/common/` (config, logging). Cross-package integration tests live under `pkg/tests/`.
- Tests in the same package describe the contract. Read them before changing behavior.
- Verify behavior against the code, not from filenames or familiarity. Run the build or read the test when uncertain.
- Do not claim work is complete without running `make presubmit` (or the targeted test) and confirming the relevant output. "Tests pass" is a claim, not a fact, until the command output exists.
- If execution goes sideways (unexpected state, cascading failures, a fix that breaks adjacent code), stop and replan. Restate what you know, identify where the plan broke, propose a revised path before continuing.

## Pull requests

- Minimalism: smallest correct change inside the smallest scope.
- Non-trivial work must be tracked in an issue. If there isn't one, ask the user to file or link it.
- The PR addresses that issue and nothing else: no renames, reformatting, refactors, new abstractions, or pattern changes beyond what the issue requires.
- Unrelated improvements belong in their own issue and PR, not folded into this PR. If you spot dead code or unrelated bugs in passing, mention them; don't fix them.
- Self-check on the way out: if the change grew larger than expected or the fix feels hacky, rewrite the clean version before opening the PR.
- Verify the code passes `make presubmit` locally before submitting a PR.
- Always use the project's `.github/PULL_REQUEST_TEMPLATE.md`.
  - Fill in **What**, **Why**, and **How was this tested?**; tick only checkboxes that reflect what actually ran.
  - Link the driving issue under **Related Issues** (`Fixes #N`).
  - List only new tests under the testing section - indicate functionality verified, not test names.

## Code style

- Standard Go. `make format` and `make lint` are authoritative.
- Comments are terse and only present when the WHY is non-obvious. Never paraphrase the code.
- Docs and comments describe the current state on its own terms. No "previously", "now", "recently", "renamed from", "added to fix", or other temporal or conversational framing. A reader with no context for the change must still understand the text.
- State each fact once, in its canonical location. Do not duplicate across struct docs, prose, tables, inline comments, and examples.
- Do not use Unicode symbols or special characters in general, unless explicitly requested.

### Logging

The codebase uses `go-logr` directly (`github.com/go-logr/logr`). Verbosity constants are defined in [pkg/common/logging/levels.go](pkg/common/logging/levels.go) (`WARN=1`, `INFO=2`, `DEBUG=4`, `TRACE=5`).

**Level conventions:**

- `logger.V(logging.INFO).Info(...)` for service-lifecycle milestones that fire a bounded number of times (startup, dataset load, tokenizer init, request-processing start/done).
- `logger.V(logging.DEBUG).Info(...)` for once-per-request operational signals.
- `logger.V(logging.TRACE).Info(...)` for detailed state transitions (KV-cache block operations, worker queue events, token-by-token generation).
- `logger.Error(err, "msg", ...)` for recoverable errors that carry an underlying `error` value.

**Use named constants, not bare integers:**

```go
// wrong
logger.V(4).Info("running protocol", ...)

// correct
logger.V(logging.DEBUG).Info("running protocol", ...)
```

**Guard expensive log construction:**

```go
if v := logger.V(logging.DEBUG); v.Enabled() {
    v.Info("payload details", "data", expensiveSerialization())
}
```

## Git workflow

- DCO sign-off is required. Use `git commit -s`.
- Commit subject: imperative, ~72 characters. Body short and focused on the WHY; long narrative belongs in the PR description.
- Do not add machine-generated co-author trailers. Sign-off is the only required trailer.
- Do not bypass hooks (`--no-verify`) or signing checks.
