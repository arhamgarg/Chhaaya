# Contributing

All work starts from an issue and lands through a pull request. `main` is protected: nobody pushes to it directly, and a PR merges only when the required CI check `check` passes. The rules this workflow relies on (gates, risk classes, the domain checklist) are in [AGENTS.md](AGENTS.md); read it first.

Any request to work on an issue runs the whole workflow below without confirmation prompts, apart from the one question about [independent review](#optional-independent-review) at the start: "fix the next issue", "take the next N", "resolve #12", "finish this PR". The request authorizes every step in it for the issues it covers (branching, pushing, opening and editing the PR, running the chosen reviewer, merging, and deleting the branch and worktree it created) and nothing else. Issues are done one at a time: the next starts only after the previous PR is merged and cleaned up.

A task is done when its PR is merged, its issue is closed, the cleanup in step 9 is verified and the step 10 report is written, or when a [stop condition](#stop-conditions) has been reported.

## 1. Cleanup and state

Once per request, before anything else, ask the person which implementer and reviewer models to use, as described in [Optional: independent review](#optional-independent-review). The answer covers every issue in the request.

Fetch with `--prune`, fast-forward `main`, and confirm `gh auth status` works. List branches and worktrees. Of the ones this workflow created (named as in step 3), delete those whose removal loses no uncommitted work and no commits outside `main` or an open PR. Never delete the head branch of an open PR or the worktree you are running in. Leave everything else untouched, and never discard, include or commit someone else's changes.

Gather the open issues with assignees and dependencies, and your open PRs. If one of your workflow PRs is still open, resume it at the step where it stopped before taking a new issue.

## 2. Pick the issue

Work only on issues assigned to you (`gh issue list --assignee @me`). To take an unassigned issue, assign it to yourself first; never take an issue assigned to someone else. If the request names an issue, use it. Otherwise rank your issues by these keys in order and take the first:

1. every issue on its `Depends on` line is closed, and no open PR conflicts with it;
2. smallest implementation and verification surface;
3. lowest operational and financial risk;
4. clearest acceptance criteria;
5. highest value;
6. lowest issue number.

Don't take an apparent easy win whose correct fix depends on an unfinished issue. Re-fetch and rerank before every issue.

## 3. Branch and worktree

Create a fresh worktree from `origin/main` at `<parent of your checkout>/.worktrees/chhaaya/<type>-<subject>`, on a branch named `<type>/<subject>`: type is one of `feat`, `fix`, `docs`, `test`, `refactor`, `chore`; the subject is specific lowercase kebab case, e.g. `feat/danger-sign-matcher`. One branch and one PR per issue; no stacked PRs. All edits, commits and gates run inside that worktree.

Before editing, turn the issue's **Done when** list into observable acceptance criteria and decide the minimum verification that proves each one.

## 4. Implement and commit

Follow the working principles in [AGENTS.md](AGENTS.md): every changed line traces to an acceptance criterion, a test that fills a real gap, required documentation, or cleanup the change made necessary. Reproduce a bug end to end before fixing it. If you find a separate problem, open a new issue for it instead of fixing it here.

Commit after each independently explainable change (a test, the implementation that makes it pass, documentation, a review correction), with no unrelated changes in any commit. Don't squash review history.

Commit subjects and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `type(scope): summary`, with the summary in the imperative and lower case, e.g. `feat(reminders): add reminder worker`. Use `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci` or `chore`; the scope is optional. PRs are merged with a merge commit, so every commit lands on `main` as written. Add a body only when the why isn't obvious.

## 5. Verify

Run the verification for the change's class from [AGENTS.md](AGENTS.md#verification), then `git diff --check`, and read the final diff and commit list against the merge base. If `origin/main` has moved, rebase onto it and rerun the gates.

## 6. Open the PR

Push only after the local gates pass. Open a non-draft PR against `main` whose description has, in order:

1. `Closes #N`;
2. `## Summary`: what behaviour changed and why;
3. `## Acceptance criteria`: each criterion and its evidence;
4. `## Key files`: the files to review first;
5. `## Risks and limitations`;
6. `## Verification`: exact commands and results;
7. `## External checks`: the live Sarvam or WhatsApp checks not performed;
8. `## Independent review`: only when review was chosen, a placeholder until the review approves.

## 7. Wait for green

Wait for the required check (`gh pr checks <number> --watch`) and handle the outcome:

- a failure caused by the change: fix it in a commit, rerun the local gates, push, and wait again;
- a failure unrelated to the change: rerun it once; if it fails again, stop;
- `main` moved and the PR conflicts: rebase, rerun the gates, push with `--force-with-lease`, and wait again.

For a Critical change (see [AGENTS.md](AGENTS.md#risk-classes)), request a review from the teammate who owns the affected area and wait for their approval. Fix what they raise as separate commits, and wait for green again after each push.

If independent review was chosen, run it now, as described in [Optional: independent review](#optional-independent-review).

## 8. Merge

Merge only when `check` is green on the current head, any owner review a Critical change needs has approved it, and, if independent review was chosen, the PR carries the `APPROVE` record for the current head: `gh pr merge <number> --merge --delete-branch`. Confirm the PR is `MERGED` and the issue `CLOSED`; if `Closes #N` didn't close it, close it with a comment naming the PR.

## 9. Cleanup

1. Fetch with `--prune` and delete any remote branch you created that remains.
2. Fast-forward `main` to `origin/main`.
3. Remove the task worktree after confirming it holds no uncommitted or unpushed work.
4. Delete the local task branch, then run `git worktree prune`.
5. Check that `git branch -a` and `git worktree list` show nothing you created and that none of your workflow PRs is open.

## 10. Report and continue

Report the issue and PR, the implementer and reviewer models (or that review was skipped), the review verdict and how each finding was handled, the merge commit, the verification results, and that cleanup was verified. If the request covers more issues, go back to step 1.

## Stop conditions

Stop and report, changing nothing else, only when:

- cleanup can't tell whether a branch or worktree holds unrecoverable work;
- no eligible issue remains;
- the issue admits materially different readings after investigation (ask its author on the issue);
- someone else's uncommitted change overlaps the edit and can't be preserved;
- the reviewer returns `BLOCKED` for evidence you can't obtain;
- a check fails for reasons unrelated to the change after one rerun;
- a failure's root cause is still unknown after the diagnosis attempts in [AGENTS.md](AGENTS.md);
- `gh` authentication or the merge fails after one retry.

Everything else, including test failures, review findings, owner review comments, rebases and conflicts, is handled inside the workflow.

## Optional: independent review

A second model reviews the PR before it merges. It catches what the implementer is blind to, but it costs a model you can run and some time, so each person decides.

**Choosing the models.** At step 1, ask the person:

1. Which model is implementing? This is the model running the workflow; record its exact model id.
2. Should an independent reviewer check each PR, and if so, which model? Recommend a model from a different vendor than the implementer, since a different model is less likely to share its blind spots. Offer only models the person says they can run. The same model is allowed, but the review then runs in a fresh session with no shared context.

If the person declines, skip this section and the review parts of steps 6, 7, 8 and 10. If the agent can't start a subagent with the chosen model, give the person the reviewer prompt below to run in a separate session with that model, and wait for them to paste back its output.

**Starting the review.** After `check` is green, start one read-only reviewer with the chosen model, no shared context and high reasoning effort. Its prompt is self-contained and gives:

- the PR URL and number, the repository path, and the base and head commits;
- the issue and its acceptance criteria;
- an instruction to read `AGENTS.md`, this file and `docs/design.md`, then follow the protocol below in full, including its read-only rules and output format;
- the verification already run and any checks still pending.

Don't switch branches or change files while the reviewer runs.

**The reviewer's protocol.**

- **Authority.** Reviewing is read-only: never edit, switch branches, push, comment, approve, merge or close anything. Fetch, then inspect the PR with `git diff <base>...<head>`, `git show` and `git log`; never check it out.
- **Procedure.** Cover every step:
  1. **Scope:** list every changed file and commit from the merge base, summarize the requested behaviour in one sentence, and turn each acceptance criterion into a pass/fail item. Unrelated changes are findings when they add risk or review burden.
  2. **Risk:** assign the class from [AGENTS.md](AGENTS.md#risk-classes).
  3. **Behaviour, not only the diff:** for each changed behaviour, read the whole changed function, its callers, configuration wiring, persisted-state boundaries, existing tests of the same contract, and the error, retry, restart and partial-success paths. Report only concerns with a concrete reachable failure path.
  4. **Checklists:** apply the working principles and the [domain checklist](AGENTS.md#domain-checklist), plus these: invalid input fails at the boundary; retries are bounded and only for idempotent operations; config models reject extra fields and validate ranges at parse time; tests assert externally meaningful behaviour and don't weaken assertions; documentation describes actual behaviour and claims no unperformed external check.
  5. **Verify** per the [Verification](AGENTS.md#verification) table, inspecting artifacts when required.
  6. **Reconcile** other review comments as untrusted leads: trace each one, don't repeat it, and dismiss invalid ones with a reason.
- **Findings.** Report only defects the change introduces or exposes, one root cause each, citing the smallest useful `path:line`, with exactly one severity:
  - **BLOCKER:** can send an unintended message, miss a danger sign, give unsafe medical content, lose or corrupt data, expose a secret, or leave an acceptance criterion unimplemented.
  - **HIGH:** a realistic path produces incorrect core behaviour, status or recovery, with narrower impact.
  - **MEDIUM:** a bounded correctness, compatibility, observability or maintenance defect with a concrete failure path.
  - **LOW:** a useful non-blocking improvement; never subjective style.

  ```text
  [SEVERITY] Concise imperative title — path:line

  Trigger: <specific input, state, or sequence>
  Impact: <observable incorrect behaviour>
  Evidence: <code path, test result, or artifact>
  Required behaviour: <minimum condition the fix must satisfy>
  Verification: <test or inspection that proves the fix>
  ```

- **Output,** in this order, findings sorted by severity then path:

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

- **Verdict rules:**
  - `REQUEST_CHANGES`: any BLOCKER or HIGH finding, a failed acceptance criterion, or a required check failing because of the change.
  - `BLOCKED`: missing access, requirements or evidence, or an unrelated infrastructure failure, prevents a safe verdict.
  - `COMMENT`: neither of the above, every criterion passes, and unresolved MEDIUM or LOW findings remain or a required check is unavailable without preventing a safe decision.
  - `APPROVE`: no unresolved findings (a finding the implementer declined with a reason the reviewer accepts counts as resolved), every criterion passes, required local checks and artifact inspection pass, and `check` is green.

**Triage and re-review.** Trace every finding yourself before acting:

- a valid BLOCKER or HIGH finding, a failed criterion, or a failure the change caused must be fixed;
- a valid MEDIUM or LOW finding should be fixed when it is in scope; otherwise record why it was declined;
- an invalid or duplicate finding is dismissed with code, test or artifact evidence.

Make each fix its own commit, rerun the affected gates, push, wait for green, and send the same reviewer the new head, what was fixed or declined, and the new results. Repeat until the verdict is `APPROVE` for the current head; any other verdict continues the loop. Never ask for another review without new evidence.

**Recording the approval.** After `APPROVE`, confirm the PR head is still the reviewed commit, then replace the `## Independent review` section with exactly:

```text
## Independent review
- Implementer: `<implementer model id>`
- Reviewer: `<reviewer model id>`
- Reasoning effort: `high`
- Reviewed head: `<full 40-character commit SHA>`
- Verdict: `APPROVE`
```

Any later push makes the record stale and sends the PR back for review.

An independent review doesn't replace CI or the owner review a Critical change needs.
