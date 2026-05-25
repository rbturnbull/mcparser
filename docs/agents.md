# Agents

This file describes each agent in the pipeline: what it receives, what it is allowed to change, and what it must return.

Agents are runtime-neutral. The adapter classes (`AgentTokenChecker`, `AgentSentenceChecker`, `AgentSegmentationChecker`) hold an `AgentClient`, which is typed as:

```python
AgentClient = Callable[[dict[str, Any]], dict[str, Any]]
```

The adapter serialises the sentence, token, evidence, and skill text into a payload dict, calls `client(payload)`, and validates the response against a Pydantic model. Nothing in the adapter assumes what is on the other side of that call — it could be an LLM API, a local model, a hand-written function, or a test stub. Swapping the runtime means supplying a different callable; no adapter code changes.

At the moment, agent-backed checkers are available through the Python API by passing `agent_clients` to `build_checkers`. The CLI correction path is rule-based only.

---

## `predict_column`

**Skill file:** `agents/predict_column/SKILL.md`  
**Adapter:** `llmparser/prediction.py` — `ColumnPredictor` callable  
**Result model:** `TokenCorrectionResult`

### Purpose

Predicts the value of one CoNLL-U column for one token at a time, before any correction is run. This is the initial prediction step, not a correction step. It operates on a token that may have many columns still blank (`_`).

### Inputs

- The full sentence as it currently stands (partially or fully blank columns are normal here).
- The current token row.
- The target column name.
- A `TokenEvidence` bundle with retrieved training examples (may be empty for the first column in a chain).

### What it may change

Only the single target column for the current token. It must not touch any other token or column.

### What it must return

A `TokenCorrectionResult` bearing zero or more `CorrectionPatch` objects. Each patch:
- sets `token_id` to the current token's ID
- sets `column` to the target column
- sets `value` to the predicted value
- may include an optional `reason` and `evidence_sent_ids`

Use `status: "uncertain"` when evidence is insufficient to make a confident prediction.

### When it runs

`predict_sentence_columns` in `llmparser/prediction.py` iterates over configured columns in order and calls the column predictor for every token. Columns are processed left-to-right, so later columns can assume that earlier columns have already been filled.

---

## `correct_segmentation`

**Skill file:** `agents/correct_segmentation/SKILL.md`  
**Adapter:** `llmparser/agent_checker.py` — `AgentSegmentationChecker`  
**Result model:** `SegmentationCorrectionResult`

### Purpose

Fixes token boundary errors before any column-level correction begins. Segmentation problems include conjunctions, prepositions, clitics, or pronominal suffixes that have been incorrectly joined to or split from a neighbouring token.

### Inputs

- The original sentence text.
- The predicted CoNLL-U sentence.
- An optional `SegmentationEvidence` bundle, which contains:
  - `surface_examples` — training sentences containing the same surface form
  - `subform_examples` — training sentences containing each expected sub-piece
  - `sequence_examples` — training sentences containing the expected segmented sequence
  - `similar_sentence_examples` — training sentences retrieved by sentence similarity

### What it may change

Row structure: it can split one token into several, merge adjacent tokens into one, insert a new token, or delete a token. This is the only agent that operates at the row level rather than the cell level.

### What it must return

A `SegmentationCorrectionResult` bearing zero or more row-level patches:
- `TokenSplitPatch` — replaces one token with two or more replacement tokens
- `TokenMergePatch` — replaces two or more adjacent tokens with one replacement token
- `TokenInsertPatch` — inserts one or more tokens before or after an anchor token
- `TokenDeletePatch` — removes a token

Any result with `status: "patched"` must also set `needs_syntax_correction: true`, because row changes invalidate any existing HEAD/DEPREL values. Downstream syntax correction is expected to run after segmentation.

Replacement tokens should leave `head` and `deprel` blank rather than guessing syntax.

### When it runs

Before all column-level correction. `correct_predictions` runs segmentation first, then token and sentence checkers on the (possibly restructured) sentence.

---

## `correct_lemma`

**Skill file:** `agents/correct_lemma/SKILL.md`  
**Adapter:** `llmparser/agent_checker.py` — `AgentTokenChecker`  
**Result model:** `TokenCorrectionResult`

### Purpose

Reviews and, if necessary, corrects the `LEMMA` column for one token. The rule-based equivalent is `RuleBasedLemmaChecker`, which compares the predicted lemma against a majority vote from training examples filtered to the same form and compatible UPOS. The agent is expected to apply the same logic with more linguistic flexibility when the rule is uncertain.

### Inputs

- The full predicted sentence.
- The current token row.
- A `TokenEvidence` bundle with:
  - `form_examples` — sentences containing the same word form
  - `lemma_examples` — sentences containing the same predicted lemma
  - `form_upos_examples` — sentences containing the same form tagged with the same UPOS
  - `similar_sentence_examples` — sentences retrieved by similarity

### What it may change

Only `LEMMA` for the current token. Must not touch ID, FORM, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, or MISC.

### What it must return

A `TokenCorrectionResult`. If a correction is needed:
- one `CorrectionPatch` with `column: lemma` and `value` set to the corrected lemma
- `reason` explaining why the training evidence supports the correction
- `evidence_sent_ids` listing the relevant training sentence IDs

Use `status: "uncertain"` when evidence is absent, weak, or contradictory.

### Chaining

When configured with both `rule: true` and an `agent` block, a `ChainedTokenChecker` wraps the rule checker and the agent. The agent is only called when the rule returns `uncertain` (controlled by `only_when_uncertain` in the config).

---

## `correct_upos_xpos`

**Skill file:** `agents/correct_upos_xpos/SKILL.md`  
**Adapter:** `llmparser/agent_checker.py` — `AgentTokenChecker`  
**Result model:** `TokenCorrectionResult`

### Purpose

Reviews and, if necessary, corrects the `UPOS` and `XPOS` columns for one token. The rule-based equivalent (`RuleBasedPosChecker`) votes across training examples filtered to the same form and compatible lemma. A single checker instance is reused for both the `upos` and `xpos` column passes.

### Inputs

- The full predicted sentence.
- The current token row.
- A `TokenEvidence` bundle (same structure as for the lemma agent).
- The target column (`upos` or `xpos`).

### What it may change

Only `UPOS` and/or `XPOS` for the current token. It may patch both columns in one result by including two `CorrectionPatch` objects. Must not modify FORM, LEMMA, FEATS, HEAD, DEPREL, DEPS, or MISC.

### What it must return

A `TokenCorrectionResult` with up to two patches — one for each of `upos` and `xpos` if both need correction. Patches should always be aligned: if evidence supports a particular UPOS/XPOS pair, both values should be updated together.

### Chaining

Same chaining mechanism as the lemma agent. The same agent instance is used for both the `upos` and `xpos` correction passes.

---

## `correct_feats`

**Skill file:** `agents/correct_feats/SKILL.md`  
**Adapter:** `llmparser/agent_checker.py` — `AgentTokenChecker`  
**Result model:** `TokenCorrectionResult`

### Purpose

Reviews and, if necessary, corrects the `FEATS` column for one token. The rule-based equivalent (`RuleBasedFeatsChecker`) votes across training examples filtered to the same form, lemma, UPOS, and XPOS. Because FEATS correction runs *after* UPOS/XPOS correction, it can assume POS tags are already as good as the pipeline can make them.

### Inputs

- The full predicted sentence (UPOS and XPOS are already corrected at this point in the column order).
- The current token row.
- A `TokenEvidence` bundle.

### What it may change

Only `FEATS` for the current token. The feature bundle is stored as a dict when structured (e.g. `{"Case": "Acc", "Number": "Sing"}`) or as the string `"_"` when empty. Must not modify any other column.

### What it must return

A `TokenCorrectionResult` with at most one `CorrectionPatch` for `feats`. The patch `value` should be either a dict of feature key/value pairs or the string `"_"`.

### Chaining

Same chaining mechanism as the lemma and POS agents.

---

## `correct_head_deprel`

**Skill file:** `agents/correct_head_deprel/SKILL.md`  
**Adapter:** `llmparser/agent_checker.py` — `AgentSentenceChecker`  
**Result model:** `SentenceCorrectionResult`

### Purpose

Reviews and, if necessary, repairs the dependency tree. Unlike the token-level agents, this one inspects the *whole sentence* at once. The rule-based equivalent (`RuleBasedSyntaxChecker`) only fixes simple inconsistencies (e.g. a token labelled `root` but pointing to a non-zero head, or a token pointing at a non-existent head). The agent is expected to handle structurally valid but linguistically wrong attachments that the rule cannot detect.

The same checker instance is registered for both the `head` and `deprel` column passes.

### Inputs

- The full predicted sentence (all earlier columns corrected).
- A list of similar training sentences retrieved by sentence similarity.

### What it may change

Only `HEAD` and `DEPREL` values. Must not modify ID, FORM, LEMMA, UPOS, XPOS, FEATS, DEPS, or MISC.

### What it must return

A `SentenceCorrectionResult`. When corrections are needed, the result includes one `CorrectionPatch` per affected token, each patching `head` or `deprel` with an evidence-based `reason`.

Return `status: "uncertain"` rather than guessing when:
- there are missing HEAD references that cannot safely be repaired
- there are multiple plausible roots
- attachments are genuinely ambiguous

The rule checker already handles the two common trivially fixable cases:
- `DEPREL=root` but `HEAD≠0` → patch HEAD to 0
- `HEAD=0` but `DEPREL≠root` → patch DEPREL to `root`

### Chaining

When configured with both `rule: true` and an `agent` block, a `ChainedSentenceChecker` wraps the rule checker and the agent. The agent is only invoked when the rule result is `uncertain`.
