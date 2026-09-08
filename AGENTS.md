# GitHub Development Policy

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

- inspect and modify code
- run validation or tests when appropriate
- create commits on a feature branch
- push a feature branch
- create a pull request
- update an existing pull request
- continue modifying an existing pull request when explicitly requested by the user

Creating or updating a pull request does NOT imply permission to merge it.

Even if:

- tests pass
- CI succeeds
- the implementation is complete
- the PR has no conflicts
- the user asked to "push"
- the user asked to "create a PR"

Codex must leave the pull request open.

Only the human repository owner may merge pull requests.


## Commit Discipline

Keep commit history minimal, clean, and logically grouped.

### Default Rule

For a single user-requested task, prefer ONE commit.

Do not create multiple commits merely because multiple files are modified.

Changes that belong to the same logical task should be committed together.

Examples that should normally be ONE commit:

- applying the same configuration change to several config files
- implementing one feature across model, training, inference, and config files
- fixing one bug across preprocessing and downstream consumers
- adding several closely related experiment configurations
- correcting small issues discovered while implementing the same requested task

### Avoid Fragmented Commits

Do NOT:

- create one commit per file for the same logical change
- create repeated commits with the same or nearly identical commit message
- create a new commit for every small correction made during the same implementation
- create cleanup commits immediately after a commit when the cleanup could have been included in that commit
- split one coherent implementation into many tiny commits without a clear technical reason

Before committing, review all changes belonging to the current task and group them into the smallest reasonable number of logical commits.

### When Multiple Commits Are Acceptable

Multiple commits are acceptable only when the changes are genuinely independent and separating them materially improves reviewability.

Examples:

- an independent bug fix plus an unrelated new feature
- a repository-wide refactor followed by a logically separate experiment
- changes that the user explicitly requests to keep separate

When uncertain, prefer fewer commits.

### Fixups Before Merge

If several recent commits on the same feature branch represent one logical change, prefer consolidating them before merge when practical.

For example, these should normally become one commit:

```text
Set ranking temperature explicitly
Set ranking temperature explicitly
Set ranking temperature explicitly
