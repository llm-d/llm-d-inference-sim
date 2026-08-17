## Contributing Guidelines

Thank you for your interest in contributing to llm-d-inference-sim. Community involvement is highly valued and crucial for the project's growth and success. The llm-d-inference-sim project accepts contributions via GitHub pull requests. This outlines the process to help get your contribution accepted.

To ensure a clear direction and cohesive vision for the project, the project leads have the final decision on all contributions. However, these guidelines outline how you can contribute effectively to llm-d-inference-sim.

## How You Can Contribute

There are several ways you can contribute to llm-d:

* **Reporting Issues:** Help us identify and fix bugs by reporting them clearly and concisely.
* **Suggesting Features:** Share your ideas for new features or improvements.
* **Improving Documentation:** Help make the project more accessible by enhancing the documentation.
* **Submitting Code Contributions (with consideration):** While the project leads maintain final say, code contributions that align with the project's vision are always welcome.

## Community and Communication

* **Developer Slack:** [Join our developer Slack workspace](https://llm-d.slack.com/archives/C097SUE2HSL) to connect with the core maintainers and other contributors, ask questions, and participate in discussions.
* **Code**: Hosted in the [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) GitHub repo
* **Issues**: Project-scoped bugs or issues should be reported in [llm-d-inference/issues](https://github.com/llm-d/llm-d-inference-sim/issues)

## Contributing Process

We follow a **lazy consensus** approach: changes proposed by people with responsibility for a problem, without disagreement from others, within a bounded time window of review by their peers, should be accepted.

### Types of Contributions

#### 1. Features with Public APIs or New Components

All features involving public APIs, behavior between core components, or new core repositories/subsystems must be accompanied by an **approved project proposal**.

**Process:**

1. Open a GitHub issue in [llm-d-inference-sim/issues](https://github.com/llm-d/llm-d-inference-sim/issues) describing the proposal. The issue should include:
   * **Summary**: A sentence or two suitable for any contributor or user to understand the change proposed and the outcome.
   * **Motivation**: The problem to be solved, including Goals/Non-Goals, and any necessary background.
   * **Proposed Solution**: What is the desired outcome and how do we measure success? Can include User Stories ("As a User I want to X"). Should have enough detail that reviewers can understand exactly what you're proposing.
   * **Design Details** (optional at the issue stage, expected before implementation): Enough information that the specifics of the change are understandable. May include API specs or code snippets. If there's any ambiguity about HOW the proposal will be implemented, discuss it here.
   * **Alternatives**: Alternative implementations/proposals and a short summary of why they were rejected.
   * **Release Notes**: Any impact on user-facing aspects, such as documentation, release notes, deprecation, and replacement of existing functionality.
2. Discuss the issue with impacted component maintainers.
3. Get approval from project maintainers on the issue before opening an implementation PR.

The proposal must be reviewed by the impacted component maintainers and approved by project maintainers. Proposal review should enforce overall principles and ensure consistency and coherence of the project. Approval of a proposal should reflect lazy consensus that the proposal is the right path, and the proposal should have high priority for review.

#### 2. Fixes, Issues, and Bugs

For changes that fix broken code or add small changes within a component:

* All bugs and commits must have a clear description of the bug, how to reproduce, and how the change is made
* Any other changes can be proposed in a pull-request to a component or an issue in [llm-d-inference-sim/issues](https://github.com/llm-d/llm-d-inference-sim/issues), a maintainer must approve the change (within the spirit of the component design and scope of change)
  * A good way to bring attention for moderate size changes is to create an RFC issue in GitHub, then engage in Slack
  * Within components, use project proposals when scope of change is large or impact to users is high

## Before You Write Code

AI tools have lowered the cost of opening a PR without lowering the cost of reviewing or maintaining one. The guidance below keeps contributions proportional to the project's capacity to support them.

### Contributions that Benefit from Discussion First

These are not requirements, but contributions in the following categories tend to land more smoothly when the approach is agreed on first. Opening an issue or proposing in the appropriate Slack channel is usually faster than writing a PR and iterating.

* **New features.** Even features that do not rise to a project proposal (no new public API or component) benefit from a brief issue describing the problem and the proposed approach before implementation.
* **New testing methodologies.** Fuzzing, property-based testing, chaos testing, load testing, or other testing approaches that introduce a new class of ongoing maintenance (new CI jobs, curated inputs, triage, release-gating policy). See the worked example below.
* **New external dependencies.** Require maintainer sign-off.
  **Note**: Please check the licensing of all new dependencies. Dependencies should comply with the [CNCF Allowed License Policy](https://github.com/cncf/foundation/blob/main/policies-guidance/allowed-third-party-license-policy.md)
* **Renames or other API-affecting changes.** See [API Changes and Deprecation](#api-changes-and-deprecation).

Every contribution creates ongoing cost: review time, CI time, flake triage, and future maintenance. A good problem statement captures that cost alongside the benefit, which is what an issue or proposal makes visible before code is written.

#### Worked Example: A New Testing Methodology

Adding a new testing framework (fuzzing, property-based, load, chaos) improves robustness on paper. In practice, the PR itself consumes significant review cycles to align on scope and ownership, and once merged it introduces a new CI job, inputs to curate, a new category of flake to triage, and an implicit policy call about whether findings block releases. An issue first absorbs the alignment work at a fraction of the cost.

Before opening a PR, please raise an issue or proposal covering:

* What robustness gap is this closing? (A reported bug, a history of failures in this area, a security concern.)
* What components are in scope?
* Who owns the inputs, CI job, and triage?
* What is the gating policy for findings from this methodology?

Once those are settled, the implementation PR is usually straightforward.

### Contributions We May Decline

Examples of patterns maintainers may close or redirect:

* **Speculative hardening.** Guards or error handling for conditions that cannot occur given current code invariants. If the condition can actually occur, please open an issue with a reproducer instead.
* **Defensive abstractions without a caller.** New interfaces, factory indirection, or generic wrappers introduced for anticipated future use. Maintainers may ask you to defer these until there is a concrete caller.

### AI-Assisted Contributions

AI-assisted contributions are welcome under the same standards as any other contribution. The human submitter is the author of record and must be able to defend the change on substance.

## Code Review Requirements

* **All code changes** must be submitted as pull requests (no direct pushes)
* **All changes** must be reviewed and approved by a maintainer other than the author
* **The repository** must gate merges on compilation and passing tests
* **All experimental features** must be off by default and require explicit opt-in

## Commit and Pull Request Style

* **Pull requests** should describe the problem succinctly
* **Descriptions** should accurately reflect what the diff does
* **PR ownership**: the submitting contributor is the author of record and should be able to explain the code, justify design choices, and respond to review on substance
* **Scope discipline**: keep changes sized to the stated problem; large, wide-ranging diffs may be asked to split or trim
* **Rebase and squash** before merging
* **Use minimal commits** and break large changes into distinct commits
* **Commit messages** should have:
  * Short, descriptive titles
  * Description of why the change was needed
  * Enough detail for someone reviewing git history to understand the scope
* **DCO Sign-off**: All commits must include a valid DCO sign-off line (`Signed-off-by: Name <email@domain.com>`)
  * Add automatically with `git commit -s`
  * Required for all contributions per [Developer Certificate of Origin](https://developercertificate.org/)

## API Changes and Deprecation

* **Breaking changes**: Once an API/protocol is in a GA release (non-experimental), breaking changes are strongly discouraged and only permitted in exceptional circumstances (for example, to address a security issue, correct a specification-level defect, or resolve behavior that blocks the project's long-term direction). Any such change must be proposed and discussed in a GitHub issue, explicitly approved by the project maintainers, and clearly documented in the release notes — including the motivation, the user-visible impact, and any available migration path.
* **Includes**: All protocols, API endpoints, internal APIs, command line flags/arguments
* **Exception**: Bug fixes that don't impact significant number of consumers (As the project matures, we will be stricter about such changes - Hyrum's Law is real)
* **Versioning**: All protocols and APIs should be versionable with clear forward and backward compatibility requirements. A new version may change behavior and fields.
* **Documentation**: All APIs must have documented specs describing expected behavior

## Testing Requirements

All code changes are expected to include tests, and every pull request must keep the existing test suite green. The project uses Go's standard testing tooling together with [Ginkgo](https://onsi.github.io/ginkgo/) suites under `pkg/`; new code should follow the same conventions and live alongside the code it exercises. 
Tests should cover the behavior being added or changed — including relevant error paths and edge cases — rather than mirror the implementation. 
Appropriate test coverage is an important part of code review, and reviewers may request additional cases before approving a change. 

Contributors are also encouraged to run the full suite locally (`make test`) before opening a PR.

## Security

Maintain appropriate security mindset for production serving. The project will establish a project email address for responsible disclosure of security issues that will be reviewed by the project maintainers. Prior to the first GA release we will formalize a security component and process.

## Documentation

Project documentation lives under [`docs/`](docs/) as plain Markdown. 
Any contribution that adds a new feature, changes existing behavior, adds or modifies a configuration option or command-line flag, or otherwise affects how users interact with the simulator must include a corresponding documentation update in the same pull request. 
When adding a new topic, create a new Markdown file under `docs/` and link it from the [README](README.md) so it is discoverable. 
Reviewers will treat missing or out-of-date documentation the same as missing tests.
