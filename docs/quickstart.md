# Quickstart

## Install

MCParser uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```bash
git clone <repository> && cd mcparser
uv sync
export ANTHROPIC_API_KEY=...   # or the key for whichever provider your models use
```

## A held-out test on a UD treebank

The quickest way to see what the agents do is to hide the annotation of a few
gold sentences, let the agents re-annotate them, and compare with the gold.

```bash
mkdir -p examples/demo
uv run python - <<'EOF'
from pathlib import Path
from mcparser.corpus import parse_conllu, format_conllu

out = Path("examples/demo")
sents = parse_conllu(Path("hit_hittb-ud-test.conllu").read_text())
train, test = sents[:-10], sents[-10:]
(out / "train.conllu").write_text(format_conllu(train))
(out / "gold.conllu").write_text(format_conllu(test))
for s in test:                      # keep the forms, blank everything else
    for t in s.tokens:
        if t.is_word:
            for col in ("lemma", "upos", "xpos", "feats", "head", "deprel", "deps"):
                t.set(col, "_")
(out / "pred.conllu").write_text(format_conllu(test))
EOF
```

Run the agents on one sentence first, to see what they do and what it costs:

```bash
uv run mcparser run examples/demo/pred.conllu -s examples/demo/train.conllu \
    --sentence 14.10-2 --rounds 1
```

Each agent reports what it did as it finishes, for example:

```text
[round 1] 14.10-2 head: I set all six heads, treating it as a copular clause: the predicate
adjective pár-ku-iš is the root, and nu, BE-EL, nam-ma and e-eš-du attach to it. ...
Round 1: scored 1 sentences, 6 agent tasks, 28 edits, 0 open flags, mean confidence 0.910
```

Then score the result:

```bash
uv run mcparser evaluate examples/demo/pred.conllu examples/demo/gold.conllu
```

Every edit is recorded in `examples/demo/pred.conllu.history.jsonl`, and review
flags in `examples/demo/pred.conllu.review.json`. To start again, recreate the
files and delete these two.

## Starting from raw text

`init` turns raw text into a prediction file. The input is either one sentence
per line or YAML mapping ids to entries with a `text` field:

```yaml
B10K1V1:
  text: بولس رسول يسوع المسيح بشيه الله: الي القديسين ...
  verse: Ephesians 1:1
```

```bash
uv run mcparser init ephesians.yaml -o ephesians.conllu -s arabic-train.conllu
uv run mcparser run ephesians.conllu -s arabic-train.conllu
```

`init` tokenises the text (whitespace, with punctuation split off), then
applies a [frequency baseline](pipeline.md#round-0-the-baseline) and the
rules learned from the trusted sources. `run` then scores and refines the
result. `init` also accepts a gold CoNLL-U file (annotation removed,
segmentation kept) and `--max-sentences N`, which is the easiest way to test
on a held-out set:

```bash
uv run mcparser init la_proiel-ud-dev.conllu -o pred.conllu -s la_proiel-ud-train.conllu --max-sentences 100
uv run mcparser run pred.conllu -s la_proiel-ud-train.conllu --max-sentences 20
uv run mcparser evaluate pred.conllu la_proiel-ud-dev.conllu --predicted-only
```

## Exploring a treebank from Claude Desktop

```bash
uv run mcparser mcp config -s train=examples/demo/train.conllu -s pred=examples/demo/pred.conllu:rw
```

Add the printed `mcpServers` entry to Claude Desktop's
`claude_desktop_config.json` and restart Claude Desktop. See
[The MCP server](mcp.md) for details.
