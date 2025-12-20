# nanoschnack

## Goals

- Create a nanochat inspired chat bot (https://nanochat.ai).
- Train on German internet data only.
- Success criteria: you can chat in German on http://nanoschnack.de.

## Development setup

Install dev tooling (including pre-commit and jupytext):

```sh
uv sync --extra dev
```

Activate the uv-managed virtual environment:

```sh
source .venv/bin/activate
```

Install the pre-commit hook:

```sh
pre-commit install
```
