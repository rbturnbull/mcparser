# Development

```bash
uv sync --all-groups
uv run pytest                                          # tests, 100% line and branch coverage required
uv run sphinx-build -W docs docs/_build/html           # these docs
```

- Tests never call a real model: `pydantic_ai.models.ALLOW_MODEL_REQUESTS` is
  off. The agents are tested with scripted `FunctionModel`s (see
  `tests/scripted.py`), and the MCP server with the SDK's in-process client.
- Coverage below 100% (lines and branches) fails the test run.
- Implementation notes for contributors, including design decisions and
  environment gotchas, are in `AGENTS.md` at the repository root (read
  automatically by coding agents such as Codex and Claude Code).

## Layout

```text
mcparser/
  corpus/      CoNLL-U model, confidence, segmentation, validation, search, store
  mcp/         the MCP server and source specs
  agents/      roles, prompts, permission wrapper, agent team
  pipeline.py  score → refine rounds
  baseline.py  frequency baseline
  textinput.py raw text / YAML → CoNLL-U
  evaluate.py  CoNLL 2018-style scores
  settings.py  models and run settings
  cli.py       typer command line
```
