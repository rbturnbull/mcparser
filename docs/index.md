# MCParser

MCParser produces [Universal Dependencies](https://universaldependencies.org/)
annotation (CoNLL-U) with a team of LLM agents. Trusted CoNLL-U files, such as
a treebank's training data, provide the examples. The agents annotate a
prediction file, score every cell, and refine the least certain cells over
several rounds.

```text
raw text ──init──▶ prediction.conllu ──run──▶ scored, refined prediction.conllu ──evaluate──▶ scores
              (tokenise + baseline)      (scorer + one agent per column)
```

All access to the data goes through an [MCP](https://modelcontextprotocol.io)
server over CoNLL-U files. The agents use it, and so can Claude Desktop or other
projects (for example a parallel-corpus aligner).

Main features:

- **One agent per CoNLL-U element**: segmentation (FORM and multiword tokens),
  UPOS, XPOS, LEMMA, FEATS, syntax (HEAD and DEPREL decided together, or
  separately) and, optionally, DEPS, plus a **scorer** that judges every cell.
- **A confidence for every cell**, stored in the MISC column
  (`Conf=upos:0.95,head:0.40,...`), which decides what gets refined next.
- **Cross-agent review**: any agent can flag a probable mistake in a column
  it doesn't own, and the responsible agent picks the flag up.
- **Read-only trusted sources and writable predictions**, enforced by the
  server, with a full edit history.
- **A configurable model for each role** (default `anthropic:claude-opus-5-5`)
  via [pydantic-ai](https://ai.pydantic.dev).

```{toctree}
:maxdepth: 2
:caption: Guide

quickstart
concepts
pipeline
agents
mcp
alignment
evaluation
```

```{toctree}
:maxdepth: 1
:caption: Reference

_generated/cli
api
development
```
