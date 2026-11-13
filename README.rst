|logo| |text|

.. |logo| image:: https://raw.githubusercontent.com/rbturnbull/mcparser/main/docs/_static/mcparser.svg
   :alt: MCParser logo
   :height: 140px
   :align: middle

.. |text| image:: https://raw.githubusercontent.com/rbturnbull/mcparser/main/docs/_static/mcparser-text.svg
   :alt: MCParser
   :height: 140px
   :align: middle

=========================
MCParser
=========================

Uses LLM agents to perform dependency parsing, with CoNLL-U files as examples.

A team of pydantic-ai agents (one per CoNLL-U column, plus a scorer) annotates
a prediction file, gives every cell a confidence (stored in MISC as
``Conf=...``), and refines the least certain cells over several rounds. All
data access goes through an MCP server over CoNLL-U files, which can also be
used from Claude Desktop or other projects.

Full documentation: ``uv run sphinx-build docs docs/_build/html`` and open
``docs/_build/html/index.html``. Design notes for contributors: ``AGENTS.md``.

Installation
============

.. code-block:: bash

   uv sync

Annotating
==========

.. code-block:: bash

   export ANTHROPIC_API_KEY=...
   # raw text (one sentence per line, or YAML with 'text' fields) -> CoNLL-U + baseline
   uv run mcparser init input.txt -o pred.conllu -s ud-train.conllu
   # score and refine with the agents (default model anthropic:claude-opus-5-5)
   uv run mcparser run pred.conllu -s ud-train.conllu --rounds 3
   # compare with gold
   uv run mcparser evaluate pred.conllu gold.conllu

Try ``--sentence ID`` first to see what one sentence costs. ``mcparser settings``
prints the configurable settings (per-role models, threshold, rounds, ...).

The CoNLL-U MCP server
======================

Each source is ``[NAME=]PATH[:ro|rw[:CONFIDENCE]]``: read-only (``ro``) sources
hold trusted examples, writable (``rw``) sources hold predictions.

.. code-block:: bash

   uv run mcparser mcp serve -s train=ud-train.conllu -s pred=pred.conllu:rw
   # JSON to add to Claude Desktop's claude_desktop_config.json:
   uv run mcparser mcp config -s train=ud-train.conllu -s pred=pred.conllu:rw

Development
===========

.. code-block:: bash

   uv sync --all-groups
   uv run pytest                                  # 100% line and branch coverage required
   uv run sphinx-build -W docs docs/_build/html
