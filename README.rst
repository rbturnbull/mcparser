=========================
LLM Parser
=========================

Uses an LLM to perform dependency parsing.


Correction workflow
===================

The correction workflow takes CoNLL-U predictions from another model and checks
them against examples from a training CoNLL-U database.

Build a training-example database:

.. code-block:: bash

   llmparser db training.conllu training-db

Run rule-based correction:

.. code-block:: bash

   llmparser correct examples/predictions.conllu \
     --db training-db \
     --output corrected.yml \
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

Agent-backed checkers
---------------------

Agent configuration is supported by the Python config model, but the command
line does not yet provide a real agent runtime. If a config contains
``agent:`` entries, the CLI will fail clearly instead of silently pretending
to run agents. Rule-based correction is the supported end-to-end CLI path for
now.
