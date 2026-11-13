# The MCP server

The MCP server exposes a {class}`~mcparser.corpus.store.CorpusStore` to any
MCP client. It is the only way the agents touch the data, and it can be used
on its own from Claude Desktop or other projects.

## Running it

```bash
# stdio (what Claude Desktop launches)
uv run mcparser mcp serve -s train=ud-train.conllu -s pred=pred.conllu:rw

# streamable HTTP
uv run mcparser mcp serve -s ud-train.conllu --transport streamable-http --port 8000

# search tools only, no editing
uv run mcparser mcp serve -s ud-train.conllu --read-only
```

`--root-key` names the MISC attribute that holds a word's root (default
`Root`), which `find_tokens(root=...)` searches.

## Claude Desktop

```bash
uv run mcparser mcp config -s train=ud-train.conllu -s pred=pred.conllu:rw
```

This prints the entry to merge into `mcpServers` in
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS).
It uses absolute paths and the project's Python, so Claude Desktop doesn't
need uv. Restart Claude Desktop afterwards. If the tools don't appear, check
`~/Library/Logs/Claude/mcp-server-conllu.log`.

Prompts to try:

- *"List the sources, then annotate the first sentence of `pred`. Use
  `analyse_form` and `find_arcs` against `train`, and set each value with a
  confidence."*
- *"Which cells in `pred` have confidence below 0.6? Are there open flags?"*
- *"How is the Arabic token بشيه split in the training data?"*

## Tools

All tools return structured JSON. Errors (an unknown sentence, an edit to a
read-only source, an invalid tag) come back as tool errors with a message the
model can act on.

**Browsing**

| Tool | Purpose |
|---|---|
| `list_sources` | sources with mode, size, default confidence, open flags |
| `list_sentences` | paged sentences; `order` by file or confidence; `max_confidence` |
| `get_sentence` | CoNLL-U, mean confidence, open flags and validation issues |
| `get_context` | neighbouring sentences |
| `get_token` | per-column confidence, head, dependents, enclosing multiword token |

**Search** (all accept `sources`, `min_confidence`, `exclude_sentence="source:sent_id"`)

| Tool | Purpose |
|---|---|
| `analyse_form` | counts of (lemma, upos, xpos, feats) for a form, with examples |
| `find_tokens` | words by form, lemma, UPOS, XPOS, features, relation, root, word ending (`suffix`) or neighbouring words (`previous_form`, `next_form`), grouped by any columns |
| `find_multiword` | how a surface token is split, and how often it is left unsplit |
| `find_arcs` | arcs by dependent, head (`head_upos="ROOT"` for roots), relation and `direction`, with relation and direction counts |
| `relation_stats` | relation counts for arcs like a given one (lemma pair, or lemma + head UPOS + direction, or UPOS pair) |
| `search_sentences` | BM25 search over sentence text |

`loose=true` ignores case and diacritics (Unicode NFD with combining marks
removed), so unvocalised Arabic or unaccented Greek matches annotated forms.
A relation without a subtype (`nmod`) also matches its subtypes (`nmod:gen`).

**Tagsets and quality**

| Tool | Purpose |
|---|---|
| `allowed_values` | values of UPOS, XPOS, DEPREL or FEATS in trusted data, optionally for one UPOS |
| `feature_inventory` | features and their values, optionally for one UPOS |
| `validate` | ids, ranges, tags, features, single root, no cycles, unseen values |
| `low_confidence` | cells below a threshold, least confident first |
| `list_flags`, `get_history` | review flags and edit history |

**Editing** (writable sources only)

| Tool | Purpose |
|---|---|
| `set_cell`, `update_cells` | set values and/or confidences (batch is all or nothing) |
| `score_cells` | confidences only, compactly: `{"3:upos": 0.9, "4:*": 0.95}` |
| `split_word`, `merge_words`, `insert_word`, `delete_word` | segmentation, with renumbering |
| `add_multiword`, `remove_multiword` | multiword surface tokens |
| `flag_issue`, `resolve_flag` | review flags |
| `save`, `reload` | normally not needed; edits save automatically |

**Alignment** (see [Word alignment](alignment.md)): `list_alignments`,
`get_parallel`, `translation_candidates`, `low_confidence_links`, and (writable
sets) `set_links`, `remove_links`.

**Text corpora** (only registered when the server is started with
`--corpus [NAME=]PATH[,PATH...]`; see
[Unannotated text](quickstart.md#unannotated-text)): `list_corpora`, and
`corpus_find`, which returns for a word its frequency, the words before and
after it, longer words that contain it (a stem with a clitic or suffix) and
example lines in context.

**Resources:** `conllu://sources` and `conllu://{source}/{sent_id}`.

## Using the corpus from another project

The store doesn't depend on the agents. A project such as a parallel-corpus
aligner can use it directly:

```python
from mcparser.corpus import CorpusStore

store = CorpusStore()
store.add_source("grc", "greek.conllu")
store.add_source("ar", "arabic.conllu", mode="rw")
store.analyse_form("λόγος", loose=True)
store.find_tokens(root="k t b", sources=["ar"])
store.set_cell("ar", "B10K1V1", "3", "misc", "AlignedTo=grc:5", agent="aligner")
```

Or it can talk to the server over MCP, e.g. with pydantic-ai:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from mcparser.mcp import build_server

agent = Agent("anthropic:claude-opus-5-5", toolsets=[MCPToolset(build_server(store))])
```
