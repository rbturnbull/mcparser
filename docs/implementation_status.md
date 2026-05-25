# Implementation Status

## Implemented

- Structured CoNLL-U models: `ConlluSentence`, `ConlluToken`
- Correction result schemas: `CorrectionPatch`, `TokenCorrectionResult`, `SentenceCorrectionResult`
- Segmentation patch schemas: `TokenSplitPatch`, `TokenMergePatch`, `TokenInsertPatch`, `TokenDeletePatch`
- Token evidence and segmentation evidence helpers
- Rule-based checkers for lemma, UPOS/XPOS, FEATS, and conservative HEAD/DEPREL repair
- Correction YAML config for rule checkers
- Rule-to-agent chaining abstractions
- Runtime-neutral agent checker adapters
- MCP-style correction session and tool registry
- CLI correction workflow for rule-based correction
- Column-by-column prediction runner with pluggable predictors

## Partly Implemented

- Prediction module: patch/audit runner exists, but no real LLM predictor is wired.
- Agent-backed checkers: config models and adapters exist, but no real LLM runtime is wired into the CLI.
- MCP support: correction tool logic and registry exist, but no MCP SDK transport is wired.
- Segmentation correction: row-level patches and evidence exist, but no linguistic segmenter is implemented.

## Not Implemented

- Real MCP server process.
- Real LLM/agent client integration.
- Automatic Arabic segmentation agent.

## Local Smoke Test

Build a local sparse database:

```bash
llmparser db examples/training.conllu example-db --sparse
```

Run correction:

```bash
llmparser correct examples/predictions.conllu \
  --db example-db \
  --output examples/corrected.yml \
  --config examples/correction.yml
```

Run tests:

```bash
poetry run pytest
```
