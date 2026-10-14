# llmparser — implementation notes

Notes for coding agents (Codex, Claude Code) and humans picking up this repo.
Keep this file current when you change the design or finish a milestone.

## Goal

Dependency parsing with LLM agents. Trusted CoNLL-U files (e.g. UD training
data) serve as examples; a prediction CoNLL-U file is created or supplied, then
scored and refined by agents over several rounds. All corpus access goes
through an MCP server so that the same server also works from Claude Desktop
and from other projects (e.g. llmaligner, which aligns parallel corpora).

## Status (2026-10-14)

| Milestone | State |
|---|---|
| 1. Corpus store (`llmparser/corpus/`) | done, 100% covered |
| 2. MCP server + CLI `mcp serve` / `mcp config` / `validate` | done, 100% covered |
| 3. Agents (pydantic-ai), per-task model config | not started |
| 4. Pipeline: `init` (raw text / YAML), score, refine rounds, `evaluate` | not started |
| 5. Sphinx docs | not started |

The previous implementation lives in `llmparserold/` (untracked; its tests are
in `llmparserold/tests/`). Reuse ideas from it, but it is not imported.

## Layout

```
llmparser/
  corpus/            # no agent or MCP dependencies; usable directly from Python
    conllu.py        # lossless CoNLL-U model (cells kept as raw strings) + parser
    confidence.py    # Conf=col:score,... in MISC
    segmentation.py  # split/merge/insert/delete words, add/remove multiword tokens
    validation.py    # structural, tagset and tree checks -> Issue
    index.py         # per-source postings, tag inventories, BM25
    models.py        # pydantic return types (also the MCP output schemas)
    source.py        # one file: ro/rw, sidecars, atomic save, conflict check
    store.py         # CorpusStore: the public API used by MCP and Python callers
  mcp/
    server.py        # build_server(store, allow_edits=True) -> MCPServer
    config.py        # SourceSpec "[NAME=]PATH[:ro|rw[:CONF]]", Claude Desktop JSON
  cli.py             # typer app (entry point `llmparser`)
tests/               # pytest; data in tests/data/{train,pred}.conllu
```

## Design decisions (and why)

- **Multiple sources in one store.** Each source is `ro` (trusted examples,
  default confidence 1.0) or `rw` (predictions, default 0.0). Read-only is
  enforced in `CorpusSource.require_writable`, not only in the MCP layer.
- **Confidence lives in MISC** as `Conf=form:0.90,lemma:0.85,upos:0.95,...`
  (user's choice). `form` = segmentation confidence. Columns without a score use
  the source default. Changing a value without giving a confidence *clears* that
  column's score (an old score must not vouch for a new value). Editing MISC
  directly can never change `Conf`.
- **Flags** (requests for another agent to review a column) and the **edit
  history** are sidecars next to a writable file: `<file>.review.json` and
  `<file>.history.jsonl`. Free-text reasons don't fit in MISC.
- **Every write is saved immediately** (`CorpusStore(autosave=True)`) with an
  atomic rename. `save()` raises `ConflictError` if the file's mtime changed
  since load (e.g. Claude Desktop and the CLI editing the same file); `reload`
  recovers.
- **Segmentation edits renumber via an object graph** (`segmentation._Graph`):
  HEAD/DEPS point at token objects during the edit, then ids are reassigned.
  Removed words redirect to their replacement (split → head piece, merge →
  merged word, delete → the deleted word's head). Results include `id_map`
  (old → new id). Open flags on that sentence are remapped. Word-changing edits
  automatically raise a sentence-level `head` flag so the syntax agent reviews.
- **Agents may flag any column.** Any agent can see the whole sentence/file and
  call `flag_issue(column=...)`; the agent responsible for that column picks
  flags up with `list_flags(column=...)` and closes them with `resolve_flag`.
  Column aliases: segmentation/tokenization→form, pos→upos, features→feats,
  arc→head, relation→deprel.
- **Search**: exact or `loose` matching (NFD, strip combining marks, casefold)
  so unvocalised Arabic / unaccented Greek match annotated forms. `root` searches
  the MISC `Root=` attribute (configurable with `--root-key`). Hits are sorted by
  confidence; `min_confidence` and `exclude_sentence="source:sent_id"` keep weak
  predictions and the sentence under edit out of the evidence. BM25 is a small
  in-house implementation (old chroma/retriv backends were dropped: heavy, slow).
- **Indexes are rebuilt lazily** after an edit to a source (`CorpusSource.changed`).
  Fine for prediction files of a few thousand sentences; make incremental if needed.
- **Tagsets** (`allowed_values`, `feature_inventory`, `known_values`) count
  read-only sources by default, so predictions don't define what is allowed.
- **Thread safety**: MCP runs sync tools in worker threads, so every public
  `CorpusStore` method takes an `RLock`.

## MCP tools

Read: `list_sources list_sentences get_sentence get_context get_token
search_sentences find_tokens analyse_form find_multiword find_arcs
allowed_values feature_inventory validate low_confidence list_flags get_history`.
Edit (omitted with `--read-only`): `set_cell update_cells split_word merge_words
delete_word insert_word add_multiword remove_multiword flag_issue resolve_flag
save reload`. Resources: `conllu://sources`, `conllu://{source}/{sent_id}`.
Write tools take `agent` (recorded in history; also used for flag ownership).

Claude Desktop: `uv run llmparser mcp config -s train.conllu -s pred=pred.conllu:rw`
prints the `mcpServers` JSON (uses the venv's python with `-m llmparser`).

## Environment gotchas

- Use **uv**. The installed uv is 0.3.2, which predates PEP 735
  `[dependency-groups]`, so dev deps are in `[tool.uv] dev-dependencies`.
- Python is pinned to **3.13** (`.python-version`). The uv-managed 3.12 on this
  machine is x86_64 and loads an arm64 `cryptography` wheel → ImportError. If
  the venv breaks: `rm -rf .venv && uv sync --python /opt/homebrew/opt/python@3.13/bin/python3.13`.
- **mcp is 2.x**: `FastMCP` is now `mcp.server.MCPServer`; raise
  `mcp.server.mcpserver.exceptions.ToolError` (any other exception reaches the
  client only as a generic message); `mcp.Client(server)` connects in-process
  for tests, `Client(StdioServerParameters(...))` over stdio.
- pydantic-ai's MCP support uses `fastmcp-slim` 4.x, which requires mcp 2.x — compatible.

## Conventions

- `uv run pytest` must stay at **100% line+branch coverage** (enforced by
  `--cov-fail-under=100` in `pyproject.toml`) and fast (~2 s now). No network
  in tests; use pydantic-ai `TestModel`/`FunctionModel` for agents.
- Raise subclasses of `CorpusError` for expected failures; the MCP layer turns
  them into tool errors.
- Return pydantic models from store methods; they become MCP output schemas.
- Token ids are strings everywhere (`"3"`, `"3-4"`, `"3.1"`).

## Next steps (agreed plan)

1. **Agents** (`llmparser/agents/`, pydantic-ai, MCP toolset): one per element —
   segmentation (form + multiword), lemma, upos/xpos, feats, head, deprel (arc
   labelling), deps optional, plus a scorer. Each sees the whole sentence and
   file, may only *write* its own columns (filter the MCP toolset), and may
   flag any column. Typed outputs (edits with confidence + reason). Prompts from
   `llmparserold`'s `agents/*/SKILL.md` as a starting point.
2. **Model config**: default `anthropic:claude-opus-5-5`; per-agent override in a
   config file and CLI (`--model`, `--model-for lemma=...`).
3. **Pipeline**: round 0 = given predictions or `llmparser init` from raw text
   (one sentence per line, or YAML like `ArabGr1-Ephesians-Parsed.yaml` with
   `text` + `verse`/ids; segmentation agent → frequency baseline from training →
   predict agents). Then score → refine cells below threshold, processing
   sentences from most to least confident, column order seg → upos/xpos → lemma
   → feats → head → deprel; then handle open flags; repeat for N rounds or until
   stable. `store.round` stamps edits with the round.
4. `llmparser evaluate PRED GOLD` (UPOS/XPOS/lemma/feats accuracy, UAS/LAS,
   segmentation F1).
5. Sphinx docs (myst-parser, furo, autodoc, sphinx-click for the typer CLI).
