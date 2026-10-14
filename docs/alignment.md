# Word alignment

The corpus store and MCP server can hold **word alignments** between two
sources, for example a translation and its original. This is what
[llmaligner](https://github.com/rbturnbull/llmaligner) builds on.

## Alignment sets

An alignment set pairs the sentences with the same `sent_id` in two loaded
sources (A and B) and stores word links between them, Ugarit style:

```text
1-2S 2-3S 3-5P      word in A - word in B; S = sure, P = probable
```

Many-to-many alignments are several links (`1-2S 1-3S`). Each link can carry a
confidence. Sets are JSON Lines files, one sentence pair per line:

```json
{"sent_id": "B10K1V1", "links": [{"source": "1", "target": "2", "sure": true, "confidence": 0.95}]}
```

Like sources, a set is read-only (trusted, e.g. gold alignments, default
confidence 1.0) or writable (default 0.0); a writable file is created when
first saved, edits are saved immediately, and every change is appended to
`<file>.history.jsonl`.

## On the command line

```bash
uv run llmparser mcp serve \
    -s arabic=arabic.conllu -s greek=greek.conllu \
    -a verses=arabic,greek,alignments/verses.jsonl \
    -a gold=arabic,greek,alignments/gold.jsonl:ro
```

Here `verses` is writable (the default) and `gold` is read-only.

`--alignment NAME=SOURCE_A,SOURCE_B,PATH[:ro|rw[:CONFIDENCE]]` is accepted by
`mcp serve` and `mcp config` (for Claude Desktop).

## Tools

| Tool | Purpose |
|---|---|
| `list_alignments` | alignment sets with their sources, mode and size |
| `get_parallel` | a sentence pair: both CoNLL-U sentences, the links with confidences, unlinked words on each side |
| `translation_candidates` | what a lemma (of A, or of B with `side="b"`) is linked to, with counts, unaligned occurrences and examples |
| `low_confidence_links` | links below a threshold, least confident first |
| `set_links` | set (or with `replace=false` add) the links of a pair; word ids are checked against both sentences |
| `remove_links` | remove links from a pair |

## From Python

```python
from llmparser.corpus import CorpusStore

store = CorpusStore()
store.add_source("arabic", "arabic.conllu")
store.add_source("greek", "greek.conllu")
store.add_alignment("verses", "arabic", "greek", "verses.jsonl")
store.set_links("verses", "B10K1V1", "1-1S 2-2S 3-3P", confidence=0.9, agent="aligner")
store.translation_candidates("رَسُول", side="a").candidates     # → [ValueCount(value='ἀπόστολος', count=...)]
```
