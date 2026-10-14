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
| 3. Agents (pydantic-ai), per-role model config | done, 100% covered; verified with Opus 5.5 |
| 4. Pipeline: `init` (text / YAML) + baseline, score/refine rounds, `evaluate` | done, 100% covered |
| 5. Sphinx docs (`docs/`) | done; `sphinx-build -W` clean |

The previous implementation lives in `llmparserold/` (its tests in
`llmparserold/tests/`, its old Markdown/Quarto docs in `llmparserold/docs/`).
Reuse ideas from it, but it is not imported.

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
  agents/
    roles.py         # Role per column + scorer; tool sets; REFINE_ORDER
    prompts.py       # shared + per-role instructions, task prompts
    toolset.py       # RoleToolset: hides tools/`agent` arg, enforces columns & target
    outputs.py       # compact answers (Annotation/Scores), parsing, Applier (validate, then apply)
    evidence.py      # pre-fetched evidence per batch (analyses, context, endings, relations)
    compact.py       # compact text renderings of tool results
    team.py          # AgentTeam: one pydantic-ai Agent per role over one MCPToolset
  pipeline.py        # Pipeline: score -> refine rounds, stop rules, RunReport
  baseline.py        # frequency baseline (no LLM) used by `init`
  rules.py           # code that settles near-certain cells (no LLM)
  pricing.py         # list prices, cost estimates
  textinput.py       # raw text / YAML -> unannotated sentences (tokeniser)
  evaluate.py        # CoNLL 2018-style P/R/F1 with surface-span alignment
  settings.py        # Settings (YAML): model per role, threshold, rounds, cache
  cli.py             # typer app: init run score evaluate validate settings mcp
docs/                # Sphinx (myst-parser + furo); CLI page generated in conf.py
tests/               # pytest; data in tests/data/{train,pred}.conllu;
                     # tests/scripted.py = scripted FunctionModels for agent tests
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
- **One agent per column** (segmentation=form, upos, xpos, lemma, feats, head,
  deprel, deps [off by default]) + **scorer** (confidences only). Each role's
  column is also its flag column. Agents edit through MCP tools directly; the
  structured output is only a `TaskReport(summary)`.
- **Permissions live in `RoleToolset`** (client side): only the role's tools
  are listed, the `agent` argument is hidden from the schema and injected,
  writes must target the prediction source and the role's columns, the scorer
  may not send values. Refusals are `ModelRetry`s that tell the model to use
  `flag_issue` instead. It also accepts `updates` sent as a JSON string (Opus
  did this in a real run).
- **Pipeline**: per round, score sentences that changed since their last
  scorer pass (from the history: last scorer `confidence` seq vs last edit seq),
  then refine sentences most→least confident, roles in `REFINE_ORDER`, a role
  runs only if its column has cells < threshold or open flags. Stop when no
  tasks, a round made no edits, or the round limit; then a final scoring pass.
  A failing task is recorded and the run continues; `UserError` (e.g. missing
  API key) aborts. Reported confidence = mean over the enabled roles' columns.
- **Baseline** fills only cells that are `_` *and* unscored (`_` also means "no
  features"), so it is idempotent; splits use the most frequent multiword
  analysis when the token is split more often than not; `flag_head=False`.
- **Evaluation** aligns surface tokens by character span, then words inside a
  multiword token by form (SequenceMatcher), like the CoNLL 2018 scorer.
- **Prompt caching** on by default (`ModelSettings(cache=True)`): ~74% of
  input tokens were cache reads in a real run.
- **Rules before models** (`rules.py`): code settles cells whose evidence is
  overwhelming (≥ `min_count` examples, ≥ `min_share` agreeing): morphology per
  column by form (or by form after the previous word), XPOS from (UPOS, FEATS),
  DEPREL from `relation_stats` for the actual arc, `root` for HEAD 0, multiword
  splits / confirmations, and (optional) confident predictions as evidence for
  forms unseen in the treebank (only cells ≥ 0.96 count; rules set 0.95 so they
  never feed themselves). Rules only fill `_` or confirm an equal value; rules
  that build on other cells require those cells ≥ 0.9 (a guessed UPOS once
  produced a wrong XPOS). On Latin PROIEL: 265/265 morphology cells and 21/21
  relations correct. Rules run after `init`, at the start of `run`, and after
  each role on the sentences it touched.
- **Cross-check flags** (`Rules.cross_check`): after syntax, a UPOS whose share
  for (lemma, deprel) in the treebank is ≤ 1−min_share is flagged for the upos
  agent (deprel confidence ≥ 0.5 is enough, since it only flags). Needs ≥ 2
  rounds to take effect.
- Each run appends its report to `<pred>.runs.jsonl`; `evaluate
  --annotated-only` scores only fully parsed sentences.
- **Compact answers** (`agents/outputs.py`): every role but segmentation
  returns `Annotation`/`Scores` with `SENT_ID TOKEN_ID VALUE CONFIDENCE` lines;
  an output validator dry-runs the edits (`update_cells(dry_run=True)`), sends
  errors (and broken trees for HEAD) back as `ModelRetry`, then applies. These
  roles only see read tools. Segmentation still edits through tools.
- **Batching**: the pipeline is role-major (each role over all sentences, most
  confident first), with tasks of up to `batch_sentences`/`batch_cells`.
- **Three tiers**: strong Opus (segmentation, lemma, head, deprel, deps),
  efficient Sonnet (upos, feats, scorer), light Haiku (xpos).
- **Scorer** sees only cells last changed by a model/person (not by `rules` or
  `baseline`), not empty required cells, not XPOS/FEATS of untagged words.
- **Evidence** (`agents/evidence.py`, deduplicated per batch): form analyses;
  for ambiguous forms, analyses after the previous word; for rare/unseen forms,
  word-ending analogies with form→lemma examples (`find_tokens(suffix=...)`,
  indexed endings of 3–5 letters); for DEPREL and the scorer, `relation_stats`
  for the word's actual head; for HEAD, how similar words attach.
- Runs print tokens and an estimated price per model (`pricing.py`; update the
  table when prices change).
## Alignment (for llmaligner)

`corpus/alignment.py`: `AlignmentSet` links words of same-`sent_id` sentences in
two sources (Ugarit `1-2S 3-5P`, per-link confidence, JSONL file, ro/rw,
history sidecar). Store: `add_alignment`, `get_parallel`, `set_links`
(validated, `replace`, `dry_run`), `remove_links`, `translation_candidates`
(lemma → linked lemmas; lemma falls back to form), `low_confidence_links`. MCP
tools of the same names; CLI `--alignment NAME=A,B,PATH[:ro|rw[:CONF]]`. The
llmaligner repo (sibling directory) depends on this package; its handoff
instructions are in `../llmaligner/INSTRUCTIONS.md`.

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

- Use **uv** (0.13+). Dev deps are in `[dependency-groups] dev`.
- Python is pinned to **3.13** (`.python-version`). The uv-managed 3.12 on this
  machine is x86_64 and loads an arm64 `cryptography` wheel → ImportError. If
  the venv breaks: `rm -rf .venv && uv sync --python /opt/homebrew/opt/python@3.13/bin/python3.13`.
- **mcp is 2.x**: `FastMCP` is now `mcp.server.MCPServer`; raise
  `mcp.server.mcpserver.exceptions.ToolError` (any other exception reaches the
  client only as a generic message); `mcp.Client(server)` connects in-process
  for tests, `Client(StdioServerParameters(...))` over stdio.
- pydantic-ai's MCP support uses `fastmcp-slim` 4.x, which requires mcp 2.x — compatible.
  `MCPToolset(build_server(store))` connects in-process.
- pydantic-ai 2.x: `result.usage` is a property (not a method); build agents
  with `defer_model_check=True` so a missing API key only fails at run time.
- `MCPServer(log_level=...)` configures the *root* logger; `build_server`
  defaults to WARNING or every Anthropic HTTP request is logged.
- typer vendors its own click (`typer._click`), so sphinx-click can't render it;
  `docs/conf.py` writes `--help` output to `docs/_generated/cli.md` instead.
- `ANTHROPIC_API_KEY` is set on the dev machine: tests set
  `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False` in `conftest.py`.
- `PYDANTIC_AI_NO_BANNER=1` is set by the CLI and docs to silence the banner.

## Conventions

- `uv run pytest` must stay at **100% line+branch coverage** (enforced by
  `--cov-fail-under=100` in `pyproject.toml`) and fast (~2.5 s without
  coverage, ~6 s with). No network in tests; use `tests/scripted.py`.
- Docs: `./mkdocs.sh` (= `uv run --group docs sphinx-build -E -W ...`). Not
  `uv tool run`: the docs import llmparser.
- CI (`.github/workflows/`): `testing.yml` runs pytest on 3.11–3.14 with
  `uv sync --locked` (keep `uv.lock` current). `docs.yml` builds docs + coverage
  HTML on every push and deploys to GitHub Pages only from `main` (Pages source
  must be "GitHub Actions"); the coverage badge step needs a `GIST_SECRET`
  secret and writes `llmparser-coverage-badge.json`. `publish.yml` publishes
  on `v*.*.*` tags with `PYPI_TOKEN`, but the PyPI name `llmparser` is taken
  by another project, so the distribution name must change before publishing.
- Coverage differs by Python version: 3.11 maps branches inside `async with`
  differently, so avoid trailing `if` statements there (use a conditional
  expression).
- Raise subclasses of `CorpusError` for expected failures; the MCP layer turns
  them into tool errors.
- Return pydantic models from store methods; they become MCP output schemas.
- Token ids are strings everywhere (`"3"`, `"3-4"`, `"3.1"`).

## Real-model observations (Opus 5.5, Hittite UD demo)

- Sentence 14.10-2 (6 words), one round, all roles: UPOS/UAS/LAS 100%,
  lemma 5/6, feats 4/6. The scorer's lowest scores were exactly the two wrong
  cells (SÍSKUR lemma 0.50, feats 0.45). 28 requests, 377k input tokens before
  caching was enabled; ~2 minutes.
- Scorer + upos on a 7-word sentence with caching: 13 requests, 146k input
  (108k cached). The flag workflow worked (scorer flagged, upos resolved).
- Cost work, same 3 sentences (14.10-2, 14.15, 14.16-1; 20 words), 1 round:
  before (all Opus, JSON tool results, no prompt evidence) 114 requests, 1.72M
  input / 36.8k output tokens, ≈ $2.34–2.68; after (tiers + evidence + compact
  results + per-cell scoring + XPOS rule) 74 requests, 0.71M / 22.5k,
  ≈ $1.02 (Opus $0.87, Sonnet $0.15). Accuracy within noise: UPOS 95/95,
  lemma 90/90, FEATS 70→75, UAS 90/90, LAS 90→85 (one Opus deprel, obj→obl).

- Latin PROIEL dev, first 10 sentences (user run, old design): $2.43, 223
  requests; UPOS/XPOS/lemma 100%, FEATS 94%, UAS 99%, LAS 96%. Remaining errors
  were mostly gold inconsistencies (participle gender Neut vs treebank-majority
  Masc; `corde` nsubj) and genuinely ambiguous `-erint` forms (after `cum`: 9 Sub
  vs 7 Ind in train). New design on sentences 11–13: $0.25 for 3 sentences
  (13 requests), UPOS/lemma 100%, FEATS 96%, UAS 96%, LAS 91%.

- User run on Latin dev (sentences 1–30 parsed, 1 round): on those 30
  sentences UPOS 98.7, XPOS 98.4, lemma 99.3, UFeats 95.7, UAS 95.4, LAS 93.4
  (gold sentences + tokenisation). CoNLL 2018 best on la_proiel test (raw
  text, sentence segmentation ~39% F1): UPOS 96.86, UFeats 91.46, lemma 97.15,
  UAS 77.62, LAS 73.61 — not directly comparable.

- Ancient Greek PROIEL dev, first 10 sentences, 2 rounds, $1.36: UPOS 98.5,
  XPOS 94.1, lemma 98.5, UFeats 92.6, UAS 94.1, LAS 87.4 (CoNLL 2018 best, raw
  text: 97.75 / 97.91 / 97.27 / 91.98 / 82.76 / 79.25). Findings → fixes:
  baseline guesses (≤0.9 conf) were never reviewed → baseline capped at 0.7;
  Haiku wrote a UPOS tag into XPOS → closed-tagset values (XPOS, DEPREL, FEATS
  pairs) not in the treebank are rejected with a retry; after a UPOS change the
  old XPOS stayed → changing UPOS (HEAD) marks XPOS/FEATS (DEPREL) of that word
  for review (confidence 0); agents did not close flags they acted on → flags
  on a changed cell are auto-resolved; PROIEL's combined values
  (Gender=Masc,Neut) → FEATS guidance. A lemma-similarity check was tried and
  dropped (8 flags on Greek+Latin, 0 useful).

## Ideas / next steps

- Calibration report in `evaluate` (accuracy by confidence bucket).
- Incremental index updates if prediction files get large.
- Smarter `init` tokenisation per language (currently whitespace + punctuation;
  clitics are left to the baseline/segmentation agent).
- Anthropic Message Batches (50% off) now that most tasks are single requests.
- Coordination conventions (where shared dependents attach) as a computed hint.
