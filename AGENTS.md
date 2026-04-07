## Commit Rules
- Always escalate a permission error in a command to the user.
- Never touch the git index.lock. If escalation does not help, just retry.
- Never commit the checkout blindly with `-A`. Always add files explicitly, potentially multiple at a time.
- Commit in sensible chunks. Don't mix multiple topics into one commit.
- Use a commit title convention: "area/subarea: short what has been done". E.g. "model/nanoschnack: do x".
- Ensure to run `source .venv/bin/activate` when committing so the pre-commit hook works.

## Design Invariants
- Treat `model/config.py` as the single source of truth for hyperparameters; all code must read and write values there, not shadow them in local variables.
- Maintain backward compatibility when changing checkpoint/resume formats; provide migrations or fallbacks so older checkpoints still resume.

## Coding Conventions
- Comment style: add a one-line comment above small blocks of logically connected lines; no blank line before the first block comment in a scope, and a blank line before each subsequent block comment; add a 3–5 line class docstring to explain purpose and constraints.
- Avoid duplicate code; prefer shared helpers or a single source of truth.
- Keep a blank line above comments unless the comment starts a scope.
- Preserve existing formatting/line breaks unless changing semantics or improving clarity; avoid re-wrapping long expressions purely for style.
- If a comment is not the first line in a scope, add a blank line before it.
- Keep print statements on a single line when they fit within 120 columns.

## Testing
- Non-trivial infrastructure code must include unit tests in `tests/`.
- Before committing, run the tests.
- Use `source .venv/bin/activate` and then `python -m unittest discover -s tests` for the test run.

## Notebook Sync
- `model/training.py` is paired with `model/training.ipynb` via jupytext.
- The pre-commit hook syncs both directions when `.ipynb` files are staged.
- When editing only the `.py` file, also stage the `.ipynb` to trigger the hook, or run `jupytext --sync model/training.py` manually.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Use short, dashed, speaking `bd` issue IDs when creating or renaming issues, for example `ns-inference-spec`
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
