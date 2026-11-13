# Quickstart

## Install

MCParser needs Python 3.11 or later.

```bash
pip install --pre mcparser           # alpha release; or: uv tool install --prerelease allow mcparser
export ANTHROPIC_API_KEY=...         # the key for whichever provider your models use
mcparser --version
```

To work from a clone instead, run `uv sync` in the repository and prefix the
commands in these docs with `uv run`. Other providers are installed with extras
(`pip install "mcparser[openai]"`); see [Models and providers](providers.md).

Before spending anything, `mcparser validate train.conllu` checks that your
trusted file is well formed, and `--sentence ID` or `--max-sentences N` limit a run
to a few sentences.

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

## Unannotated text

A plain-text file (one sentence or paragraph per line) can serve as extra
evidence for forms the treebank has rarely or never seen. It has no analyses,
so it cannot say what a word is, but it shows how the word is used: how often
it occurs, which words come before and after, and which longer words contain it
(a stem next to the same stem with a clitic). Give it with `--corpus` (repeat the
option for several corpora; several files for one corpus are joined with commas):

```bash
uv run mcparser run pred.conllu -s ar-train.conllu \
    --corpus arabic=train.txt,valid.txt
```

The corpus is indexed once per run (about 10 seconds and 200 MB for 30 million
words). Agents then see, for each form with fewer than three treebank examples,
a `text ×N; after ...; before ...; prefixed ...` note in the evidence (the
segmentation agent sees how often an unseen token and its remainder after
cutting one or two letters occur) and may call `corpus_find` for example lines.
`--no-corpus-tool` keeps the notes in the prompt but stops agents calling `corpus_find` themselves (a little cheaper).
The same option works for `mcp serve` and `mcp config`, so Claude Desktop can
query the corpus too. Treebank evidence outranks the text, which may be of
another period or genre. In tests on Arabic, Hebrew, Greek, Latin and Coptic (corpora from
`ancientcorpora-output`, sentences that are not in the corpus) the gain was within run-to-run noise
at 11–25% more cost, so try it on a small batch first; it should help most when the text matches
the treebank's genre. Check that your corpus does not contain the very text you are annotating if
you want to measure its effect.

## Exploring a treebank from Claude Desktop

```bash
uv run mcparser mcp config -s train=examples/demo/train.conllu -s pred=examples/demo/pred.conllu:rw
```

Add the printed `mcpServers` entry to Claude Desktop's
`claude_desktop_config.json` and restart Claude Desktop. See
[The MCP server](mcp.md) for details.
