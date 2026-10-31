# MCParser — implementation notes

Notes for coding agents (Codex, Claude Code) and humans picking up this repo.
Keep this file current when you change the design or finish a milestone.

## Goal

Dependency parsing with LLM agents. Trusted CoNLL-U files (e.g. UD training
data) serve as examples; a prediction CoNLL-U file is created or supplied, then
scored and refined by agents over several rounds. All corpus access goes
through an MCP server so that the same server also works from Claude Desktop
and from other projects (e.g. llmaligner, which aligns parallel corpora).

## Status (2026-10-31)

| Milestone | State |
|---|---|
| 1. Corpus store (`mcparser/corpus/`) | done, 100% covered |
| 2. MCP server + CLI `mcp serve` / `mcp config` / `validate` | done, 100% covered |
| 3. Agents (pydantic-ai), per-role model config | done, 100% covered; verified with Opus 5.5 |
| 4. Pipeline: `init` (text / YAML) + baseline, score/refine rounds, `evaluate` | done, 100% covered |
| 5. Sphinx docs (`docs/`) | done; `sphinx-build -W` clean |

The previous implementation lives in `llmparserold/` (its tests in
`llmparserold/tests/`, its old Markdown/Quarto docs in `llmparserold/docs/`).
Reuse ideas from it, but it is not imported.

## Layout

```
mcparser/
  corpus/            # no agent or MCP dependencies; usable directly from Python
    conllu.py        # lossless CoNLL-U model (cells kept as raw strings) + parser
    confidence.py    # Conf=col:score,... in MISC
    segmentation.py  # split/merge/insert/delete words, add/remove multiword tokens
    validation.py    # structural, tagset and tree checks -> Issue
    index.py         # per-source postings, tag inventories, BM25
    textcorpus.py    # unannotated text: counts, neighbours, clitic variants, examples (no CoNLL-U)
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
  columns are also its flag columns. **Joint syntax** (default, setting
  `joint_syntax`, CLI `--joint-syntax/--separate-syntax`): role `syntax` with
  columns (head, deprel) replaces `head` + `deprel`; answers
  `S ID HEAD DEPREL CONF [DEPREL_CONF]` (`parse_syntax`); a head change does not
  mark its own DEPREL for review; review rounds get `arc_labels` for words with
  heads; `Rules.relation_check` flags model labels that contradict a dominant
  treebank label for the exact arc (on old runs: 6 flags, 2 real errors; cheap
  because a flag only costs a review). Batches count words, not cells. Agents edit through MCP tools directly; the
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
  tasks, a round made no edits, or the round limit; then, only with
  `final_scoring` / `--final-score` (off by default), a final scoring pass.
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
  overwhelming (≥ `min_count` = 5 examples, ≥ `min_share` = 0.98 agreeing): morphology per
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
  returns `Annotation`/`Scores` with `SENTENCE TOKEN_ID VALUE CONFIDENCE` lines,
  where SENTENCE is the task label `s1`, `s2`, ... (`Batch.resolve`; full
  sent_ids also accepted). Scorer wildcards `s1 3 *` / `s1 * *` expand to the
  cells listed in the task only (`Batch.cells`), so rule-set cells aren't re-scored;
  an output validator dry-runs the edits (`update_cells(dry_run=True)`), sends
  errors (and broken trees for HEAD) back as `ModelRetry`, then applies. These
  roles only see read tools. Segmentation still edits through tools.
- **Batching**: the pipeline is role-major (each role over all sentences, most
  confident first), with tasks of up to `batch_sentences`/`batch_cells`.
- **Three tiers**: strong Opus (segmentation, lemma, syntax/head/deprel, deps),
  efficient Sonnet (upos, feats, scorer), light Haiku (xpos).
- **Scorer** sees only cells last changed by a model/person (not by `rules` or
  `baseline`), not empty required cells, not XPOS/FEATS of untagged words.
- **Evidence** (`agents/evidence.py`, deduplicated per batch): form analyses;
  for ambiguous forms, analyses after the previous word, before the next word
  and before the next word's UPOS (`find_tokens(next_upos=...)`); for rare/unseen forms,
  word-ending analogies with form→lemma examples (`find_tokens(suffix=...)`,
  indexed endings of 3–5 letters); for DEPREL and the scorer, `relation_stats`
  for the word's actual head; for HEAD, how similar words attach.
- Runs print tokens and an estimated price per model (`pricing.py`; update the
  table when prices change).
## Unannotated text corpora (`--corpus`)

`corpus/textcorpus.py`: `TextCorpus` over plain-text files (one sentence or
paragraph per line), registered with `store.add_text_corpus(name, paths)` / CLI
`--corpus [NAME=]PATH[,PATH...]` (`run`, `score`, `mcp serve`, `mcp config`).
Index built lazily and once (30M words: ~10 s, ~200 MB): word counts plus at most 50 line
numbers per word and byte offsets, so sentences are read from the files on demand.
`find` returns count, neighbours (over those ≤ 50 lines), longer words that
end/begin with the form (≤ 4 extra letters: proclitic/enclitic variants) and
examples. MCP tools `list_corpora` / `corpus_find` exist only when a corpus is given;
`corpus_find` is in `READ_TOOLS` for agents. `EvidenceBuilder.corpus` adds a
`text ×N; after ..; before ..; prefixed ..` note for forms with < `RARE_COUNT`
treebank examples (morphology roles and scorer); `EvidenceBuilder.stems` gives
segmentation the counts of an unseen token and of what remains after cutting 1–2 letters.
Corpus words are never settled by `rules.py` and the prompt says treebank evidence outranks it.

Setting `corpus_tool` (CLI `--corpus-tool/--no-corpus-tool`, default on): whether agents may call
`corpus_find` themselves (`CORPUS_TOOLS` in `roles.py`, added per team in `role_toolset(extra=...)`) or only see
the evidence in the prompt. Evidence and `stems` match loosely (the ancientcorpora texts are lowercased and
unaccented), `corpus_find` defaults to exact matching.

Measured, 2026-11-01. (1) Arabic PADT with the classical-Arabic corpus, 20 sentences / 507 words chosen
for ≥ 3 unseen forms, morphology + segmentation roles, 2 rounds: no corpus UPOS 95.3 / 96.5 (two
identical runs), lemma 94.9 / 95.1, UFeats 92.9 / 91.7; corpus UPOS 96.3, lemma 96.1, UFeats 92.9; cost
$2.19 / $2.40 vs $2.83. The identical runs differ on 13 UPOS and 24 FEATS words, as much as the corpus does.
(2) Hebrew PTNK, Greek PROIEL, Latin PROIEL, Coptic Scriptorium (`experiments/cl2/`, 23–30 sentences each,
≥ 3 (Coptic 2) forms unseen in train, one found in the corpus), arms: no corpus ×2, corpus + tool, corpus
evidence only; mean over the 4 languages UPOS/XPOS/lemma/UFeats/AllTags: 97.95/97.47/97.53/94.70/93.34 and
97.83/97.30/97.71/94.90/93.38 (the two no-corpus runs), 97.98/97.46/97.81/94.99/93.63 (corpus + tool),
97.79/97.36/97.70/94.24/92.78 (evidence only); cost $9.29 / $9.15 vs $10.64 / $10.30. Corpus effect
0.1–0.3 points, below the spread of identical runs (up to 0.9 per language). **No demonstrated gain**; keep
it optional. **Leakage**: the Greek and Latin corpora contain the treebanks' own text (80% / 83% of dev
sentences have ≥ 50% of their 5-grams in the corpus; Coptic 32%, Hebrew 6%), so the tests use only
sentences outside the corpus (`experiments/select_clean.py`); an unfiltered test would flatter the corpus.
The Arabic corpus is classical text, PADT is newswire: only ~30% of the unseen dev words occur ≥ 5 times.
Probably useful only with a corpus of the treebank's genre and a language the model knows poorly.

Offline negative results from the same runs: the lemma's treebank gender would fix only ~17 of 115 gender
errors on nouns/proper nouns (mostly Hebrew); the systematic FEATS errors are `Gender=Masc,Neut`
conventions (PROIEL) and lexical gender; for punctuation, the gold head is a non-crossing candidate 98–99.9%
of the time, but choosing among the candidates (highest / lowest / nearest) is right only 1–57%, and agents
make crossing punctuation arcs only 7 times in 1,439 (16% of punctuation heads are wrong), so a
projectivity check would change nothing. Spend: about $52 of the second $80 budget.

## Alignment (for llmaligner)

`corpus/alignment.py`: `AlignmentSet` links words of same-`sent_id` sentences in
two sources (Ugarit `1-2S 3-5P`, per-link confidence, JSONL file, ro/rw,
history sidecar). Store: `add_alignment`, `get_parallel`, `set_links`
(validated, `replace`, `dry_run` — also on the MCP tool), `remove_links`, `translation_candidates`
(lemma → linked lemmas; lemma falls back to form), `low_confidence_links`. Each link
records `agent` (who last set or changed it; an unchanged link sent again keeps
its agent; `set_links` copies the links it is given, so callers' and stored
`Link` objects are never mutated); `get_parallel` returns `agents`. MCP
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

Claude Desktop: `uv run mcparser mcp config -s train.conllu -s pred=pred.conllu:rw`
prints the `mcpServers` JSON (uses the venv's python with `-m mcparser`).

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
  `uv tool run`: the docs import mcparser.
- CI (`.github/workflows/`): `testing.yml` runs pytest on 3.11–3.14 with
  `uv sync --locked` (keep `uv.lock` current). `docs.yml` builds docs + coverage
  HTML on every push and deploys to GitHub Pages only from `main` (Pages source
  must be "GitHub Actions"); the coverage badge step needs a `GIST_SECRET`
  secret and writes `mcparser-coverage-badge.json`. `publish.yml` publishes
  on `v*.*.*` tags with `PYPI_TOKEN`. The project was renamed from llmparser
  (taken on PyPI) to MCParser / `mcparser` (free on PyPI as of 2026-10-15).
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

- Greek batch 2 (sentences 11–30, $3.17) and Arabic PADT (10 sentences,
  $3.10) error study → changes: rule threshold 0.95 → 0.98 (replaying the
  morphology rule on blanked dev sentences of 4 treebanks: 53 → 16 wrong of
  ~5k/4.5k settled; e.g. ل ADP vs CCONJ, αὐτόν PRON vs DET were context
  readings); role order now segmentation, upos, lemma, head, deprel, feats,
  xpos (Case errors came from FEATS running before syntax at 0.9 confidence);
  FEATS → XPOS dependency, and a doubtful XPOS (< 0.5) is re-derived from
  UPOS+FEATS; HEAD/DEPREL evidence now includes the most similar treebank
  sentence (BM25) per sentence and the treebank's fixed expressions
  (≥3 times, ≥50% of occurrences). Several "errors" were gold inconsistencies
  with train (ἢν δὲ μή is never fixed in grc train; Gender=Masc,Neut usage).

- Coptic Scriptorium dev, 10 sentences (575 words), 2 rounds, $8.15 (Opus
  $4.54, Sonnet $3.57, Haiku $0.04; 45 tasks, 91 requests, 3.68M input of which
  2.28M cache reads and 1.40M cache writes, 293k output): UPOS 94.4, XPOS 93.0,
  lemma 95.7, UFeats 93.6, UAS 93.6, LAS 87.1. Costs: output ≈ 45% (long
  sent_ids repeated on every answer line; scorer listed every cell), cache
  writes ≈ 45% (long Coptic sentences and examples), 14 scorer tasks incl. the
  final pass. → sentence labels `s1`, scorer `s1 * *` wildcard, final scoring
  off by default, per-task and per-role usage in the report, examples ≤ 40
  words. Accuracy: tag errors were context readings of short function words
  (`ⲩ` DET before a noun vs PRON before a verb; `ⲉ` SCONJ/ADP/PART before
  `ⲩ`; `ϭⲉ` PART vs DET) → evidence by next form and next UPOS. Several rule
  "errors" follow train against the dev gold (`ⲉϩⲣⲁⲓ ⲉ` is fixed 1/26 in train,
  `ⲉⲃⲟⲗ ϩⲓⲧⲟⲟⲧ` 14/15, `ⲉⲧⲃⲉ ϫⲉ` 5/6). Remaining syntax errors: converters
  `mark` vs `aux`, punctuation attachment.

- Coptic batch 2 (dev sentences 11–20, 309 words, 3 rounds, $4.28 vs $8.15):
  UPOS 98.4, XPOS 97.7, lemma 97.4, UFeats 96.4, UAS 93.9, LAS 90.6. Rules
  set ~55% of morphology cells with 100% accuracy. Calibration: cells at
  0.85–0.9 were 92% right, ≥ 0.95 99%; the scorer piles cells at 0.85.
  Cache writes were 68% of the cost (single-request tasks write their whole
  prompt). → cell lists as id ranges (`1..36 at 0.00`), trailing `_` columns
  dropped, each form's analyses once per task with role-specific columns
  (`ANALYSIS_COLUMNS`), context readings per occurrence only when informative
  (`INFORMATIVE_GAIN`), sentence labels in evidence: prompts 30–50% smaller
  (head −40%, deprel −51%, upos −40%, lemma −47%, scorer −30%). Errors →
  `ⲛ` ADP vs plural article (after ADP: DET 230 / ADP 2) → context by previous
  UPOS; nouns `nsubj` instead of `dislocated` → `single_relations()` (deprels a
  word almost never has twice, from trusted data) and a once-only retry
  (`Applier._doubled`, `Batch.warned`); on old runs 13 such doublings, 11
  containing a gold error. `TaskResult.retries` records retry reasons.
  Measured but not used: punctuation heads (gold is the left or right
  adjacent unit's head 88% in Coptic, but neither side alone > 56%), and
  treebank support of an arc as a doubt signal (lowest-support arcs only ~35%
  wrong). Remaining lemma errors are unseen lemmas or train/dev
  inconsistencies (`ⲛⲛⲁϩⲣⲛ` → `ⲛⲁϩⲣⲛ` in train, `ⲛⲛⲁϩⲣⲛ` in dev).

- Experiment sweep 2026-10-31 (Gothic, OCS, Ancient Hebrew, Classical Armenian,
  Georgian, Coptic Bohairic, plus Greek/Latin/Arabic/Coptic Scriptorium; 12-sentence
  dev batches, 2 rounds; ~$1–3 per batch; scripts and logs in `experiments/`,
  report `experiments/report.html`). Pooled over 12 batches, v1 → v2: UAS
  92.0 → 92.8, LAS 89.0 → 89.8 at the same cost. Changes: `Applier._chained`
  (a relation attached to a word with the same relation, e.g. conj chain, is
  questioned once; `store.unchained_relations`, index `chained`), `Rules.voice_check`
  (nsubj/csubj/aux subtype by head Voice; flag only; first version also fired on
  obl subtypes and was right 3/24 times, so it is limited to those relations and
  to labels rare for that voice: replayed on 51 saved predictions 2 flags, both
  right; `experiments/replay_voice.py`; `store.deprels_for_voice`, index `by_voice`), UD reminders in the syntax prompt (first conjunct, cop,
  final punctuation, passive subjects), `evaluate --calibration`
  (`evaluate.calibration`). Measured, not adopted: `--strong-thinking high`
  (+0–2 LAS, +20–110% cost), `--threshold 0.9` (no gain, +15–35% cost), a
  per-form punctuation head rule (final marks attach to the root 93–97%, but commas/maqaf
  vary: the best of {previous, next, head of either, root} per form gave only 40–75%
  of punctuation heads right). Run-to-run noise on 12 sentences is about ±1.5 UAS/LAS,
  so differences smaller than that are not evidence. Calibration over ~70k cells: ≥ 0.95
  → 99.4% right, 0.8–0.9 → 91.8%, 0.5–0.7 → 57%; HEAD cells rarely reach 0.95
  (most sit at 0.8–0.95 and are 91–95% right).

## Ideas / next steps

- Incremental index updates if prediction files get large.
- Smarter `init` tokenisation per language (currently whitespace + punctuation;
  clitics are left to the baseline/segmentation agent).
- Anthropic Message Batches (50% off) now that most tasks are single requests.
- Coordination conventions (where shared dependents attach) as a computed hint.
- Compare joint vs separate syntax on gold batches (Coptic 11–20, Arabic) and
  record UAS/LAS/cost here. Merging UPOS+FEATS was rejected: FEATS must come
  after syntax (Case errors in Greek/Arabic).
- llmaligner still imports `llmparser`; it must switch to `mcparser`.
- A punctuation-attachment rule/hint (UD: attach to the head of the adjacent clause).
