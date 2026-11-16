# Changelog

All notable changes to MCParser. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- `--batch` / `batch`: send the model requests through Anthropic's Message Batches API for half the price, for long runs
  that are not urgent. The requests are batched, not the run, so agents, tools and retries are unchanged: a
  `BatchingClient` (`mcparser.agents.batching`, reusable) queues each request and a dispatcher submits a batch whenever every
  task in flight is waiting, polls it, and hands the answers back. Settings `batch_poll_seconds` and
  `batch_max_wait_seconds`, `batch_cache` (5m, 1h or off; default 1h), CLI `--batch-poll-seconds` and `--batch-cache`. Prices are halved and marked in the
  report, each batch is logged and listed in the run report, and `--batch` with a non-Anthropic model is an error.
- `cheap_first` / `--cheap-first`: round 1 runs one model tier down (Opus → Sonnet, Sonnet → Haiku) and later rounds use each
  role's own tier, so only cells still below the threshold reach the stronger model.
- `mcparser align import` and `align export`: convert alignments to and from plain text (Pharaoh/Ugarit lines,
  0- or 1-based) and YAML (`sent_id: links`). Functions in `mcparser.corpus.alignment_io`.

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
