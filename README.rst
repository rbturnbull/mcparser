=========================
LLM Parser
=========================

Uses an LLM to perform dependency parsing.


Correction workflow
===================

The correction workflow takes CoNLL-U predictions from another model and checks
them against examples from a training CoNLL-U database.

Build a local sparse training-example database:

.. code-block:: bash

   llmparser db examples/training.conllu example-db --sparse

Run rule-based correction:

.. code-block:: bash

   llmparser correct examples/predictions.conllu \
     --db example-db \
     --output examples/corrected.yml \
     --config examples/correction.yml

The example correction config enables the implemented rule-based checkers:

.. code-block:: yaml

   lemma:
     rule: true

   upos_xpos:
     rule: true

   feats:
     rule: true

   head_deprel:
     rule: true

The correction output includes the corrected CoNLL-U, detailed audit entries,
and an ``audit_summary`` for quick inspection.

The ``--sparse`` database option uses the local retriv backend. Omitting it
uses the OpenAI embedding/Chroma backend and requires working OpenAI embedding
credentials.

Current limitations
-------------------

The command-line correction path is currently rule-based. Agent-backed checker
configuration exists in the Python API, but the CLI does not yet provide a real
agent runtime. MCP correction tool logic and a registry exist, but the MCP SDK
transport is not wired yet. Segmentation correction has row-level patch
infrastructure and evidence helpers, but no automatic linguistic segmenter.

Agent-backed checkers
---------------------

Agent configuration is supported by the Python config model, but the command
line does not yet provide a real agent runtime. If a config contains
``agent:`` entries, the CLI will fail clearly instead of silently pretending
to run agents. Rule-based correction is the supported end-to-end CLI path for
now.


Developer notes
===============

Run the test suite with:

.. code-block:: bash

   poetry run pytest

Render the pipeline documentation with Quarto:

.. code-block:: bash

   quarto render docs/pipeline.qmd --to html
