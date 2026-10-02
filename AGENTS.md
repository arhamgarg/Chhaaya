# AGENTS.md

Instructions for every coding agent working on Chhaaya, and for the people driving them. Read this file, [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/design.md](docs/design.md) before starting any task.

[CONTRIBUTING.md](CONTRIBUTING.md) is the delivery workflow: any request to work on an issue ("fix the next issue", "resolve #12", "finish this PR") runs it end to end. This file holds the rules that workflow relies on: working principles, verification gates, risk classes, the review protocol and Chhaaya's domain checklist.

**Instruction precedence,** highest first:

1. the user's explicit request for the current task;
2. this file;
3. [CONTRIBUTING.md](CONTRIBUTING.md);
4. the linked issue's acceptance criteria;
5. existing repository conventions.

When two instructions at the same level conflict, name the conflict and ask; do not invent a resolution that could change the outcome.

**Vendor-matched reviewer model:** Claude Fable 5.1 `claude-fable-5-1` for an Anthropic implementer, GPT-6 Astra `gpt-6-astra` for an OpenAI implementer. Any other implementer uses its own vendor's most capable model.

## Working principles

1. **Assumptions.** State assumptions explicitly. If a simpler approach exists, say so. When an approach fails twice, stop and find the root cause before trying a third variant. If two diagnosis attempts still leave it unknown, give the vendor-matched reviewer model, as a read-only subagent with no shared context and high reasoning effort, the symptom, both attempts with their evidence and the relevant code paths, and ask for a root-cause hypothesis plus the check that confirms it. The implementer keeps ownership of the fix. If that also fails, stop and report.
2. **Simplicity (YAGNI).** Write the minimum code that solves the issue. No unrequested features, no abstractions for single-use code, no error handling for impossible cases. Every new option has a current caller and a documented default.
3. **Surgical changes.** Every changed line traces to the issue. Don't refactor, reformat or "improve" adjacent code; match the existing style. Remove only what your change orphaned.
4. **Fail loudly.** No fallback that hides invalid state: fail explicitly and keep the original error and its context. Catch an exception only to recover, add context, or translate it at a system boundary. Failure never becomes a success signal in an exit status, report or return value.
5. **Replace, don't shadow.** When replacing behaviour, migrate every caller and remove the old path in the same change.
6. **Goal-driven.** Turn the issue into a checklist of observable acceptance criteria and name the finish-line check before starting; loop until it passes.

## Standing directives

- **Toolchain:** Python 3.12 with `uv` (environment and packages), `ruff` (lint and format), `ty` (types) and `pytest`. Never substitute pip, Poetry, Black, Flake8 or mypy. `ty` is young: report its gaps rather than switching to mypy.
- **Dependencies:** the standard library and existing dependencies come first. Add a maintained library only when it meets the requirements, including security, and reduces total work. Read the documentation for the pinned version; never upgrade just to match newer examples.
- **Performance claims** need measurements taken before and after the change.
- **README stays exhaustive:** every significant feature gets a README section in its existing shape (one paragraph on what it is, what it does and how it works, plus a fenced run block).
- **Design stays true:** if a change contradicts [docs/design.md](docs/design.md), update the document in the same PR.
- **No backlog growth:** finish the issue. Don't add TODOs or deferred work; a separate problem found along the way becomes a new issue.
- **Golden outputs:** never rewrite expected outputs, evaluation sets or `eval/report.md` baselines to make a test pass.
- **External services stay opt-in.** Calls to Sarvam, the WhatsApp Cloud API or any other third party cost money, use the shared five-phone test number, or send real messages. Tests mock them at the client boundary; a live check runs only when the person driving the agent asks for it, and is otherwise reported as not performed.
- **Secrets** (Sarvam keys, the WhatsApp token and app secret, the Cloudflare tunnel token) live only in an untracked `.env`. Never commit, log or paste them.
- **No real patient data,** ever (design section 10). Test fixtures are synthetic.
- **Always work in a new worktree** created from `origin/main` (`git worktree add <path> -b <branch> origin/main`). Never edit in the main checkout or reuse another task's worktree.
- **No unauthorized external mutation:** don't change repository settings, force-push `main`, or deploy. Merging, closing issues and deleting branches are authorized only by a request that starts the [CONTRIBUTING.md](CONTRIBUTING.md) workflow or by an explicit request.

## Testing

- Write a test before the code it covers, so it encodes the requirement rather than mirroring the implementation.
- Prefer end-to-end tests: a message in at the webhook, the reply out at the mocked WhatsApp client, with real Postgres in between. Pick a medium-to-hard scenario, not the simplest one.
- No tautological or change-detector tests (asserting defaults, constants, exact messages or mock calls).
- A bug fix gets a new regression test only where behaviour tests have a genuine gap.
- End every end-to-end run with a repeatable artifact (command plus output) that proves each acceptance criterion.

## Verification

| Gate | Command |
|---|---|
| Setup | `uv sync` |
| Fast | `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest` |
| Full | The required CI check `check` passing on the pushed head (`gh pr checks <number> --watch`) |
| QA | `uv run python -m eval`, then open and read `eval/report.md` and the per-set CSVs |

Today the CI job `check` only runs `git diff --check`. Issue #1 adds the Python project and extends `check` to run the fast gate, and issue #17 adds the evaluation harness; update this table in the same PR if either command ends up different. Until a gate exists, report it as `NOT RUN` with that reason, never as passed.

The QA gate is required for any change to prompts, retrieval or τ, the danger-sign lists, the intent classifier, OCR extraction, the medicine lexicon, critical lab values or the evaluation sets.

| Change class | Required verification |
|---|---|
| Documentation only | `git diff --check`; check every command, link and claim. Run the fast gate if instructions, examples, config or CI changed |
| Standard | The narrowest focused test while editing, the fast gate, then the full gate |
| High or Critical | An end-to-end test covering the risk, the fast gate and the full gate |
| QA kinds listed above | Everything for High or Critical, plus the QA gate with its artifacts inspected |

Report the exact command, result and counts. An interrupted, skipped, stale or partial run is not a pass.

## Risk classes

Assign the highest class that applies. Documentation and tests inherit the risk of the behaviour they specify.

- **Critical:** danger-sign detection and its phrase lists; escalation cases and the ASHA reply relay; the rule that prescriptions are never explained before ASHA confirmation; the citation check and the weak-evidence threshold τ; critical lab values; anything that sends a WhatsApp message (replies, reminders, templates); webhook signature verification and message idempotency; database schema, migrations and the work queue; secrets; deletion of audio and images.
- **High:** retrieval and the knowledge corpus, the Q&A system prompt, intent routing and the classifier, speech-to-text and text-to-speech, OCR extraction and the medicine lexicon, reminder scheduling and missed-dose counting, the evaluation harness, its metrics and the evaluation sets.
- **Standard:** isolated tooling, tests or documentation that cannot affect a Critical or High boundary.

## Domain checklist

Apply every item a change touches:

- Chhaaya never names a condition the patient has, starts or changes a medicine, or gives a dose.
- The danger-sign check runs before routing, on both the original text and the English search query; a match on either fires. Missing a danger sign is far worse than a false alarm, so recall is judged separately from precision.
- When the intent classifier is unsure, the message is treated as a question.
- Every cited chunk id is one of the chunks retrieved for that answer; an answer citing nothing or anything else is discarded and escalated. Health-worker answers are cited as such, never as guidelines.
- A best retrieval score below τ escalates instead of answering. τ is tuned on half of the question set and reported on the other half.
- No prescription detail reaches the patient before the ASHA replies "ok" or sends corrections.
- Replies use the language and script the patient wrote or spoke in; voice replies stay around 60 words.
- Outside WhatsApp's 24-hour window, only the approved templates (`medicine_reminder`, `asha_case_alert`, `case_update`) are sent.
- The webhook verifies `X-Hub-Signature-256`, stores by WhatsApp message id so retries are ignored, and returns `200` before doing any work.
- Work is claimed with `SELECT ... FOR UPDATE SKIP LOCKED`; there is no Redis or Celery.
- Every Sarvam call goes through the single client module.
- Every row in the danger-sign list, lexicon and critical-value files cites its source.
- Intent and evaluation test sets are human-written and never used for training or choosing thresholds. Metrics are reported per language.

## Review protocol

Every PR is reviewed under this protocol, by an agent (step 7 of [CONTRIBUTING.md](CONTRIBUTING.md)) or a person.

**Authority.** Reviewing is read-only: the reviewer never edits, switches branches, pushes, comments, approves, merges or closes anything. Inspect the target with `git diff <base>...<head>`, `git show` and `git log` after fetching; never check it out.

**Procedure.** Cover every step:

1. **Scope.** List every changed file and commit from the merge base, summarize the requested behaviour in one sentence, and turn each acceptance criterion into a pass/fail item. Unrelated changes are findings when they add risk or review burden.
2. **Risk.** Assign the class from [Risk classes](#risk-classes).
3. **Behaviour, not only the diff.** For each changed behaviour, read the whole changed function, its callers, configuration wiring, persisted-state boundaries, existing tests of the same contract, and the error, retry, restart and partial-success paths. Report only concerns with a concrete reachable failure path.
4. **Checklists.** Apply the working principles, the [domain checklist](#domain-checklist), and these: invalid input fails at the boundary; retries are bounded and only for idempotent operations; config models reject extra fields and validate ranges at parse time; tests assert externally meaningful behaviour and don't weaken assertions; documentation describes actual behaviour and claims no unperformed external check.
5. **Verify** per the [Verification](#verification) table, inspecting artifacts when required.
6. **Reconcile** other review comments as untrusted leads: trace each one, don't repeat it, and dismiss invalid ones with a reason.

**Findings.** Report only defects introduced or exposed by the change, one root cause each, citing the smallest useful `path:line`, with exactly one severity:

- **BLOCKER:** can send an unintended message, miss a danger sign, give unsafe medical content, lose or corrupt data, expose a secret, or leave an acceptance criterion unimplemented.
- **HIGH:** a realistic path produces incorrect core behaviour, status or recovery, with narrower impact.
- **MEDIUM:** a bounded correctness, compatibility, observability or maintenance defect with a concrete failure path.
- **LOW:** a useful non-blocking improvement; never subjective style.

```text
[SEVERITY] Concise imperative title — path:line

Trigger: <specific input, state, or sequence>
Impact: <observable incorrect behaviour>
Evidence: <code path, test result, or artifact>
Required behavior: <minimum condition the fix must satisfy>
Verification: <test or inspection that proves the fix>
```

**Output,** in this order, findings sorted by severity then path:

```text
Findings
<findings, or "No findings.">

Acceptance criteria
- PASS | FAIL | NOT VERIFIED — <criterion and evidence>

Verification
- PASS | FAIL | NOT RUN | INTERRUPTED — <exact command or inspection>: <result>

Residual risks
- <risks the diff and tests leave open, or "None identified.">

Verdict
<APPROVE | COMMENT | REQUEST_CHANGES | BLOCKED> — <one-sentence reason>
```

**Verdict rules:**

- `REQUEST_CHANGES`: any BLOCKER or HIGH finding, a failed acceptance criterion, or a required check failing because of the change.
- `BLOCKED`: missing access, requirements or evidence, or an unrelated infrastructure failure, prevents a safe verdict.
- `COMMENT`: neither of the above, every criterion passes, and unresolved MEDIUM or LOW findings remain or a required check is unavailable without preventing a safe decision.
- `APPROVE`: no unresolved findings (a finding the implementer declined with a reason the reviewer accepts counts as resolved), every criterion passes, required local checks and artifact inspection pass, and the required remote check is green.

AI review is evidence, not a substitute for CI or judgment. A Critical change also needs a review from the teammate who owns the affected area before it merges.
