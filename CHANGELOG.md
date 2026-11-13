# Changelog

All notable changes to MCParser. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0a1]

First alpha release on PyPI. Install with `pip install --pre mcparser`.

### Added
- `mcparser --version`.
- Evaluation now follows the official CoNLL 2018 scorer: UFeats counts universal features only, LAS ignores
  subtypes, and CLAS, MLAS and BLEX are reported. `--strict` adds the full-feature and full-relation
  variants and `--official PATH` runs the official script on the same files.
- `--max-cost` / `max_cost`: stop starting agent tasks at an estimated spend; run again to continue.
- A `conventions` hint that tells the FEATS agent where the treebank writes combined feature values.
- Cost estimates for non-Anthropic models (through `genai-prices`), and extras for other providers
  (`openai`, `google`, `bedrock`, `groq`, `mistral`, `xai`, `openrouter`, `all-providers`).
- Documentation: models and providers, results, troubleshooting. A project logo.
- `LICENSE`, `CONTRIBUTING.md`, package metadata and a `py.typed` marker.

### Fixed
- When an agent task needed a retry, answers for sentences that had already passed were dropped, which left
  sentences without any HEAD. They are now kept.
- Tests no longer depend on terminal colour (they failed under GitHub Actions).

## [0.2.0]

- Renamed from llmparser to MCParser. Corpus store, MCP server, agent team, pipeline, rules, baseline,
  evaluation and documentation.
