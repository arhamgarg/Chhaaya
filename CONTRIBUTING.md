# Contributing

All work starts from an issue and lands through a pull request. `main` is protected: nobody pushes to it directly, and a PR merges only when the required CI check `check` passes. The rules this workflow relies on (gates, risk classes, the domain checklist) are in [AGENTS.md](AGENTS.md); read it first.

Any request to work on an issue runs the whole workflow below without confirmation prompts: "fix the next issue", "take the next N", "resolve #12", "finish this PR". The request authorizes every step in it for the issues it covers (branching, pushing, opening and editing the PR, spawning the reviewer, merging, and deleting the branch and worktree it created) and nothing else. Issues are done one at a time: the next starts only after the previous PR is merged and cleaned up.

A task is done when its PR is merged, its issue is closed, the cleanup in step 11 is verified and the step 12 report is written, or when a [stop condition](#stop-conditions) has been reported.

## 1. Cleanup and state

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
8. `## Independent review`: a placeholder until step 9.

## 7. Independent review

Wait for the required check (`gh pr checks <number> --watch`), handling failures as in step 9. Then spawn one read-only review subagent using the vendor-matched reviewer model from [AGENTS.md](AGENTS.md), with no shared context and high reasoning effort. Its prompt is self-contained and gives:

- the PR URL and number, the repository path, and the base and head commits;
- the issue and its acceptance criteria;
- an instruction to read `AGENTS.md` and this file, review the change against them and the issue without editing anything, and return each finding with a severity (BLOCKER, HIGH, MEDIUM or LOW) and `path:line`, then a verdict: `APPROVE`, `COMMENT`, `REQUEST_CHANGES` or `BLOCKED`;
- the verification already run and any checks still pending.

The reviewer works from the PR's base and head, not your checkout. Don't switch branches or change files while it runs.

## 8. Triage and re-review

Trace every finding yourself before acting:

- a valid BLOCKER or HIGH finding, a failed criterion, or a failure the change caused must be fixed;
- a valid MEDIUM or LOW finding should be fixed when it is in scope; otherwise record why it was declined;
- an invalid or duplicate finding is dismissed with code, test or artifact evidence.

Make each fix its own commit, rerun the affected gates, push, wait for the check, and send the same reviewer the new head, what was fixed or declined, and the new results. Repeat until the verdict is `APPROVE` for the current head; any other verdict continues the loop. Never ask for another review without new evidence.

For a Critical change, also request a review from the teammate who owns the affected area, and wait for their approval before merging.

## 9. Record the approval and wait for green

After `APPROVE`, confirm the PR head is still the reviewed commit, then replace the `## Independent review` section with exactly:

```text
## Independent review
- Model: `<reviewer model id>`
- Reasoning effort: `high`
- Reviewed head: `<full 40-character commit SHA>`
- Verdict: `APPROVE`
```

Any later push makes this stale and returns to step 7. Wait for the required check again and handle the outcome:

- a test failure caused by the change: fix it in a commit, push, and return to step 7;
- a failure unrelated to the change: rerun it once; if it fails again, stop;
- `main` moved and the PR conflicts: rebase, rerun the gates, push with `--force-with-lease`, and return to step 7.

## 10. Merge

Merge only when the current head carries the `APPROVE` record and `check` is green: `gh pr merge <number> --merge --delete-branch`. Confirm the PR is `MERGED` and the issue `CLOSED`; if `Closes #N` didn't close it, close it with a comment naming the PR.

## 11. Cleanup

1. Fetch with `--prune` and delete any remote branch you created that remains.
2. Fast-forward `main` to `origin/main`.
3. Remove the task worktree after confirming it holds no uncommitted or unpushed work.
4. Delete the local task branch, then run `git worktree prune`.
5. Check that `git branch -a` and `git worktree list` show nothing you created and that none of your workflow PRs is open.

## 12. Report and continue

Report the issue and PR, the reviewer model, its verdict and how each finding was handled, the merge commit, the verification results, and that cleanup was verified. If the request covers more issues, go back to step 1.

## Stop conditions

Stop and report, changing nothing else, only when:

- cleanup can't tell whether a branch or worktree holds unrecoverable work;
- no eligible issue remains;
- the issue admits materially different readings after investigation (ask its author on the issue);
- someone else's uncommitted change overlaps the edit and can't be preserved;
- the reviewer returns `BLOCKED` for evidence you can't obtain;
- a check fails for reasons unrelated to the change after one rerun;
- the escalated diagnosis in [AGENTS.md](AGENTS.md) can't find a root cause either;
- `gh` authentication or the merge fails after one retry.

Everything else, including review findings, test failures, stale approvals, rebases and conflicts, is handled inside the workflow.
