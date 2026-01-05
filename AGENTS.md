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
- `model/training.py` is exported from `model/training.ipynb`. When editing `training.py`, sync changes back to the notebook before committing.
- To sync: edit the corresponding cell in the `.ipynb` file with the same changes made to the `.py` file.
