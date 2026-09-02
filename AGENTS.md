# GitHub Safety Policy

## Absolute Merge Prohibition

Codex must NEVER merge a pull request in this repository.

This includes:

- Do not run `gh pr merge`.
- Do not use GitHub API tools to merge a pull request.
- Do not enable auto-merge.
- Do not merge a branch into `main`.
- Do not push directly to `main`.
- Do not force-push to `main`.

Codex may:

- modify code
- run tests
- create commits
- push a feature branch
- create a pull request
- update an existing pull request

After creating or updating a pull request, STOP.

Creating a pull request does NOT imply permission to merge it.

Even if:
- tests pass
- CI succeeds
- the implementation is complete
- the PR has no conflicts
- the user asked to "push"
- the user asked to "create a PR"

Codex must leave the pull request open.

Only the human repository owner may merge pull requests.
