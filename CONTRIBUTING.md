# Contributing

All work starts from an issue and lands through a pull request. `main` is protected: nobody pushes to it directly, and a PR merges only when CI passes.

1. Assign the issue to yourself if it isn't already assigned.
2. Branch from an up-to-date `main`. Name the branch after the issue, e.g. `12-danger-sign-matcher`.
3. Keep a PR to one issue. If you find a separate problem along the way, open a new issue for it instead of fixing it in the same PR.
4. Write `Closes #12` in the PR description, followed by a few lines on what changed and how you checked it.
5. Ask the owner of the area you touched to review. Reviews aren't enforced, but don't merge your own non-trivial PR without one.
6. Squash-merge, then delete the branch.

Commits and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `type(scope): summary`, with the summary in the imperative and lower case, e.g. `feat(reminders): add reminder worker`. Use `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci` or `chore`; the scope is optional. The PR title becomes the squashed commit on `main`, so it matters most. Add a body only when the why isn't obvious.

If a change contradicts [docs/design.md](docs/design.md), update the document in the same PR.
