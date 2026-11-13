|logo| |text|

.. |logo| image:: https://raw.githubusercontent.com/rbturnbull/mcparser/main/docs/_static/mcparser.svg
   :alt: MCParser logo
   :height: 220px
   :align: middle

.. |text| image:: https://raw.githubusercontent.com/rbturnbull/mcparser/main/docs/_static/mcparser-text.svg
   :alt: MCParser
   :height: 220px
   :align: middle

=========================
MCParser
=========================

.. image:: https://github.com/rbturnbull/mcparser/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/rbturnbull/mcparser/actions/workflows/testing.yml
   :alt: Tests

.. image:: https://img.shields.io/badge/license-Apache%202.0-blue.svg
   :target: https://github.com/rbturnbull/mcparser/blob/main/LICENSE
   :alt: Apache 2.0 licence

**Dependency parsing with a team of LLM agents.** MCParser annotates text in the
`Universal Dependencies <https://universaldependencies.org/>`_ CoNLL-U format
(segmentation, lemma, UPOS, XPOS, FEATS, HEAD, DEPREL and optionally enhanced
DEPS), using a treebank you trust, such as a UD training file, as the examples.

One agent looks after each CoNLL-U column, and a scorer gives every cell a
confidence. Cells the treebank makes near-certain are settled by plain code
without calling a model. The remaining low-confidence cells are refined over several
rounds. All access to the data goes through an `MCP <https://modelcontextprotocol.io>`_
server, so the same corpus tools also work from Claude Desktop and other
projects.

It is aimed at low-resource and historical languages, where a small treebank and
a capable LLM together beat either alone, and where you want to see *why* a cell
has the value it has. Every edit is logged with the agent that made it.

Installation
============

MCParser needs Python 3.11 or later.

.. code-block:: bash

   pip install --pre mcparser      # alpha release; or: uv tool install --prerelease allow mcparser
   export ANTHROPIC_API_KEY=...    # the default models are Anthropic's

From a clone of the repository: ``uv sync``, then prefix the commands below with ``uv run``.
Other model providers (OpenAI, Google, Bedrock, Groq, Mistral, xAI, OpenRouter) are
installed with extras, for example ``pip install "mcparser[openai]"``. See
`Models and providers <https://rbturnbull.github.io/mcparser/providers.html>`_.

Quick start
===========

.. code-block:: bash

   # 1. Raw text (one sentence per line, or YAML with 'text' fields) -> CoNLL-U with a frequency baseline
   mcparser init input.txt -o pred.conllu -s ud-train.conllu

   # 2. Try one sentence first to see what it costs
   mcparser run pred.conllu -s ud-train.conllu --sentence SENT_ID --rounds 1

   # 3. Annotate everything, stopping at a spending limit (run again to continue)
   mcparser run pred.conllu -s ud-train.conllu --rounds 3 --max-cost 20

   # 4. Compare with gold annotation using the CoNLL 2018 metrics
   mcparser evaluate pred.conllu gold.conllu

``-s`` names the trusted CoNLL-U files (read-only examples). The prediction file is
edited in place and every run can be resumed. ``mcparser settings`` prints all settings
(models for each role, confidence threshold, rounds, batch sizes, cost limit).

How well does it work?
======================

On 100 held-out Ancient Greek (PROIEL) dev sentences, annotated from raw tokens with
the default models (Opus 5.5 for segmentation, lemma and syntax, Sonnet 5.5 for UPOS,
FEATS and scoring, Haiku 5.5 for XPOS) in two rounds, scored by the official
CoNLL 2018 script's metrics:

=========  ======  ======  ======  =======  ======  ======  ======
UPOS       XPOS    UFeats  Lemmas  UAS      LAS     CLAS    MLAS
=========  ======  ======  ======  =======  ======  ======  ======
97.95      97.87   94.18   99.02   92.04    89.09   87.18   77.56
=========  ======  ======  ======  =======  ======  ======  ======

The run cost about $7.80 ($0.078 per sentence, $0.006 per word). Cost per sentence
depends on language and length; it is roughly $0.03 to $0.12 with Sonnet for the strong
roles and one round. Differences of about 1.5 points on a hundred sentences are noise.
Results on four complete dev files, the cost breakdown and the caveats are in
`Results <https://rbturnbull.github.io/mcparser/results.html>`_.

Using it from Claude Desktop
============================

.. code-block:: bash

   mcparser mcp config -s train=ud-train.conllu -s pred=pred.conllu:rw

prints the ``mcpServers`` entry to add to ``claude_desktop_config.json``. Sources
are ``[NAME=]PATH[:ro|rw[:CONFIDENCE]]``: read-only sources hold trusted examples and
writable sources hold predictions. ``mcparser mcp serve`` runs the server directly
(stdio or streamable HTTP; ``--read-only`` removes the editing tools).

From Python
===========

.. code-block:: python

   import anyio
   from mcparser.agents.team import AgentTeam
   from mcparser.corpus import CorpusStore
   from mcparser.pipeline import Pipeline
   from mcparser.settings import Settings

   store = CorpusStore()
   store.add_source("train", "train.conllu")
   store.add_source("pred", "pred.conllu", mode="rw")
   settings = Settings(rounds=2, max_cost=20)
   team = AgentTeam(store, "pred", settings)
   report = anyio.run(Pipeline(store, "pred", team, settings).run)

The corpus store (search, validation, editing, confidence) is also usable without any
agents or model. See the `API reference <https://rbturnbull.github.io/mcparser/api.html>`_.

Limitations
===========

- LLM output varies from run to run, and the agents inherit the treebank's own
  inconsistencies. Check a sample of the output before relying on it, and use the
  confidence scores to decide what to review by hand.
- Cost grows with the number of words, and long sentences are expensive. Use
  ``--max-sentences``, ``--sentence`` and ``--max-cost`` while you tune a setup.
- Only the Anthropic models have been tested in real runs. Other providers are
  supported through pydantic-ai but are untested here.
- The tokeniser in ``init`` is simple (whitespace and punctuation). Clitics and
  multiword tokens are left to the segmentation agent.

Documentation
=============

Full documentation: https://rbturnbull.github.io/mcparser (build it locally with
``uv run --group docs sphinx-build -W docs docs/_build/html``). Design notes for contributors are in
``AGENTS.md``, and changes in ``CHANGELOG.md``.

Contributing
============

.. code-block:: bash

   uv sync --all-groups
   uv run pytest        # the tests never call a model; 100% line and branch coverage is required

See ``CONTRIBUTING.md``. Please report bugs and ideas on the issue tracker.

Licence
=======

Apache 2.0, see ``LICENSE``. The vendored CoNLL 2018 evaluation script in ``tests/official/``
is under the Mozilla Public Licence 2.0.
