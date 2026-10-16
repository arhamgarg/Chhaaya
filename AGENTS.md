# AGENTS.md

Instructions for every coding agent working on Chhaaya and the people driving them. Read this file, [CONTRIBUTING.md](CONTRIBUTING.md) (the delivery workflow that any request to work on an issue runs end to end) and [docs/design.md](docs/design.md) before starting a task.

**Instruction precedence,** highest first: the user's explicit request; this file; [CONTRIBUTING.md](CONTRIBUTING.md); the linked issue's acceptance criteria; repository conventions. When equals conflict, name the conflict and ask.

## Working principles

1. **Assumptions.** State assumptions. If a simpler approach exists, say so. When an approach fails twice, find the root cause before a third variant; if two diagnosis attempts leave it unknown, stop and report the symptom, both attempts and their evidence.
2. **Simplicity (YAGNI).** The minimum code that solves the issue: no unrequested features, no abstractions for single-use code, no handling for impossible cases. Every new option has a current caller and a documented default.
3. **Surgical changes.** Every changed line traces to the issue; match the existing style and leave adjacent code alone. Remove only what your change orphaned.
4. **Fail loudly.** No fallback that hides invalid state; keep the original error and its context. Catch only to recover, add context, or translate at a system boundary. Failure never becomes a success signal in an exit status, report or return value.
5. **Replace, don't shadow.** Migrate every caller and remove the old path in the same change.
6. **Goal-driven.** Turn the issue into observable acceptance criteria, name the finish-line check before starting, and loop until it passes.

## Standing directives

- **Toolchain:** Python 3.12 with `uv`, `ruff`, `ty` and `pytest`. Never pip, Poetry, Black, Flake8 or mypy; report `ty` gaps instead of switching to mypy.
- **Dependencies:** the standard library and existing dependencies first. Add a maintained library only when it meets the requirements, including security, and reduces total work. Read the docs for the pinned version; never upgrade to match newer examples.
- **Performance claims** need measurements before and after the change.
- **README stays exhaustive:** every significant feature gets a README section in the existing shape (one paragraph on what it is, what it does and how it works, plus a fenced run block).
- **Design stays true:** a change that contradicts [docs/design.md](docs/design.md) updates it in the same PR.
- **No backlog growth:** finish the issue; no TODOs or deferred work. A separate problem found along the way becomes a new issue.
- **Golden outputs:** never rewrite expected outputs, evaluation sets or `eval/report.md` baselines to make a test pass.
- **External services stay opt-in.** Sarvam, the WhatsApp Cloud API and any other third party cost money, use the shared five-phone test number, or send real messages. Tests mock them at the client boundary; a live check runs only when the person driving the agent asks, and is otherwise reported as not performed.
- **Secrets** (Sarvam keys, the WhatsApp token and app secret, the Cloudflare tunnel token) live only in an untracked `.env`. Never commit, log or paste them.
- **No real patient data,** ever (design section 10). Fixtures are synthetic.
- **Always work in a new worktree** from `origin/main` (`git worktree add <path> -b <branch> origin/main`). Never edit in the main checkout or reuse another task's worktree.
- **No unauthorized external mutation:** no repository-settings changes, force-pushes to `main`, or deploys. Merging, closing issues and deleting branches are authorized only by a request that starts the [CONTRIBUTING.md](CONTRIBUTING.md) workflow or by an explicit request.

## Testing

- Write the test before the code it covers, so it encodes the requirement rather than the implementation.
- Prefer end-to-end tests: a message in at the webhook, the reply out at the mocked WhatsApp client, real Postgres in between, on a medium-to-hard scenario.
- No tautological or change-detector tests (asserting defaults, constants, exact messages or mock calls). A bug fix gets a new test only where behaviour tests have a genuine gap.
- End every end-to-end run with a repeatable artifact (command plus output) that proves each acceptance criterion.

## Verification

| Gate | Command |
|---|---|
| Setup | `uv sync` |
| Fast | `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest` |
| Full | The required CI check `check` passing on the pushed head (`gh pr checks <number> --watch`) |
| QA | `uv run python -m eval`, then open and read `eval/report.md` and the per-set CSVs |

Issue #17 adds the evaluation harness; until it exists, report the QA gate as `NOT RUN` with that reason, never as passed, and update this row in the same PR if its command differs. The QA gate is required for any change to prompts, retrieval or τ, the danger-sign lists, the intent classifier, OCR extraction, the medicine lexicon, critical lab values or the evaluation sets.

| Change class | Required verification |
|---|---|
| Documentation only | `git diff --check`; check every command, link and claim. The fast gate if instructions, examples, config or CI changed |
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
- The danger-sign check runs before routing, on both the original text and the English search query; a match on either fires. Recall is judged separately from precision, since missing a danger sign is far worse than a false alarm.
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
