# General Instructions

- Do not add a lot of comments for each part of the code with history of why that change is made. Keep them succinct and to the point, only for what's essential.

- Ensure you always keep the README.md updated with the latest state of the code/library.

- For any markdown file, report generation or any writeups in the tracked repo or .localdev, follow ASD-STE100 language specifications at 80% strictness.

# Local Testing, Explorations and Planning

- For any local testing, research and explorations that requires generating any artifacts/reports, use the .localdev folder in the root (it's gitignored)

# Tasks Spanning Multiple Tickets

- Read every ticket first (`gh issue view <N>`), then order them by dependency: which files each touches and whether one needs another's behaviour (e.g. a check on preprocessed features goes after the preprocessing changes it must see).
- One subagent per ticket, each in its own worktree on a new branch `glin-<TICKET NUMBER>` (e.g. `glin-17`). Independent tickets run in parallel from `main`; a dependent ticket branches from its prerequisite's branch (`git switch -c glin-15 glin-17`).
- If two independent branches conflict on merge (check with `git merge-tree --write-tree --name-only <a> <b>`), rebase the later one onto the stack once the others are done and resolve the conflicts there.
- Subagents implement, update README, test and commit locally only (never push). The commit message follows the `asd-commitmsg-draft` skill with scope `#<N>` and no trailers.
- Subagents must use their own venv under `.localdev/`, not the main checkout's `.venv` (its editable install points at the main source). Verify `glin.__file__` resolves to the worktree.
- The supervisor re-runs each branch's full suite, reviews the diff, then pushes the branches, opens one PR per ticket against `main` (stacked ones state their prerequisite), and comments the PR link on each ticket.
