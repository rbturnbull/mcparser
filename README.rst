=========================
LLM Parser
=========================

Uses LLM agents to perform dependency parsing, with CoNLL-U files as examples.

All corpus access goes through an MCP server over CoNLL-U files, which can also
be used from Claude Desktop or other projects. Agents and the refinement
pipeline are in progress; see ``AGENTS.md`` for the design and status.

Installation
============

.. code-block:: bash

   uv sync

The CoNLL-U MCP server
======================

Each source is a CoNLL-U file given as ``[NAME=]PATH[:ro|rw[:CONFIDENCE]]``.
Read-only (``ro``) sources hold trusted examples; writable (``rw``) sources hold
predictions that can be edited. Confidence per cell is stored in MISC as
``Conf=upos:0.95,lemma:0.80,...``.

.. code-block:: bash

   uv run llmparser mcp serve -s train=ud-train.conllu -s pred=pred.conllu:rw

To use it from Claude Desktop, print the configuration and add it to
``claude_desktop_config.json``:

.. code-block:: bash

   uv run llmparser mcp config -s train=ud-train.conllu -s pred=pred.conllu:rw

Add ``--read-only`` to expose only the search tools.

The same API is available in Python without MCP:

.. code-block:: python

   from llmparser.corpus import CorpusStore

   store = CorpusStore()
   store.add_source("train", "ud-train.conllu")
   store.add_source("pred", "pred.conllu", mode="rw")
   store.analyse_form("al", loose=True)

Validate a file (unseen tags are checked against the trusted sources):

.. code-block:: bash

   uv run llmparser validate pred.conllu -s ud-train.conllu

Development
===========

.. code-block:: bash

   uv run pytest

Tests must keep 100% line and branch coverage.
