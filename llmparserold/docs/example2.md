# Worked Pipeline Example 2

This example uses a six-token English sentence with five deliberate errors spread across four columns. Every rule-based correction stage fires at least once.

See [example1.md](example1.md) for a single-token introduction and full agent contract illustrations.

---

## Sentence

> The boys quickly walk home .

---

## Input prediction (with errors)

```conllu
# sent_id = example-2
# text = The boys quickly walk home .
1	The	the	NOUN	DT	Definite=Ind|PronType=Art	4	det	_	_
2	boys	boy	NOUN	NNS	Number=Sing	4	nsubj	_	_
3	quickly	quickly	ADV	RB	_	4	advmod	_	_
4	walk	walkd	VERB	VBP	Mood=Ind	0	nsubj	_	_
5	home	home	ADV	RB	_	4	advmod	_	_
6	.	.	PUNCT	.	_	4	punct	_	_
```

### Errors at a glance

| Token | Column | Predicted | Correct | Notes |
|---|---|---|---|---|
| 1 `The` | `upos` | `NOUN` | `DET` | wrong part of speech |
| 1 `The` | `feats` | `Definite=Ind\|PronType=Art` | `Definite=Def\|PronType=Art` | wrong definiteness value |
| 2 `boys` | `feats` | `Number=Sing` | `Number=Plur` | wrong number |
| 4 `walk` | `lemma` | `walkd` | `walk` | misspelled/wrong lemma |
| 4 `walk` | `deprel` | `nsubj` | `root` | HEAD is 0, so DEPREL must be `root` |

---

## Training evidence used by this example

The rule-based checkers vote across training examples retrieved from the database. The relevant examples for each token are shown below.

**Form `The` (retrieved for lemma and POS passes):**
```conllu
# sent_id = train-the-1
# text = The cat sat .
1	The	the	DET	DT	Definite=Def|PronType=Art	2	det	_	_

# sent_id = train-the-2
# text = The dog runs .
1	The	the	DET	DT	Definite=Def|PronType=Art	2	det	_	_
```

**Form `boys` (retrieved for feats pass):**
```conllu
# sent_id = train-boys-1
# text = The boys laughed .
2	boys	boy	NOUN	NNS	Number=Plur	3	nsubj	_	_

# sent_id = train-boys-2
# text = Two boys ran .
1	boys	boy	NOUN	NNS	Number=Plur	3	nsubj	_	_
```

**Form `walk` (retrieved for lemma pass):**
```conllu
# sent_id = train-walk-1
# text = They walk home .
2	walk	walk	VERB	VBP	Mood=Ind	0	root	_	_

# sent_id = train-walk-2
# text = Birds walk and fly .
1	walk	walk	VERB	VBP	Mood=Ind	0	root	_	_
```

---

## Step-by-step pipeline trace

### Step 1 — Initial deterministic validation

The validator scans every token for structural errors before any correction runs.

**Error found on token 4:**

```
code:    head-zero-deprel
message: Token 4 has HEAD 0 but DEPREL 'nsubj'; expected root.
token_id: 4
column:  deprel
```

This is flagged in `audit["validation"]`. Correction has not happened yet — validation only reports, it does not fix.

**Sentence at this point: unchanged.**

---

### Step 2 — Segmentation

Not configured for this example (no `segmentation` entry in the correction config), so this step is skipped.

---

### Step 3 — Lemma correction (`RuleBasedLemmaChecker`)

The checker runs once per token, in sentence order.

**Token 1 `The` (predicted `UPOS=NOUN`, `LEMMA=the`)**

The checker retrieves training examples for form `The` then filters them:

```python
if token.upos not in (None, "_", "") and training_token.upos != token.upos:
    continue
```

The predicted UPOS is `NOUN`. Both training examples have `UPOS=DET`. They are filtered out. No candidates remain.

Result: **`uncertain`** — `"No training evidence found for form 'The'."` No patch applied.

Note: the lemma itself (`the`) is correct. The `uncertain` status is a side-effect of the wrong UPOS preventing evidence from matching. Because lemma runs first in the column order, it cannot benefit from the UPOS fix that comes later.

**Token 2 `boys` (predicted `UPOS=NOUN`, `LEMMA=boy`)**

Training examples for form `boys` have `UPOS=NOUN` → they pass the filter. Both report `LEMMA=boy`, which matches the prediction.

Result: **`checked`** — lemma is already correct.

**Token 3 `quickly`** — no training error to correct. **`checked`.**

**Token 4 `walk` (predicted `UPOS=VERB`, `LEMMA=walkd`)**

Training examples for form `walk` have `UPOS=VERB` → they pass the filter. Both report `LEMMA=walk`. The predicted `LEMMA=walkd` does not match.

Majority: `walk` appears 2/2 times (ratio=1.0). Threshold met.

Result: **`patched`**.

Patch applied immediately (in-place on the sentence):

```yaml
token_id: 4
column: lemma
value: walk
reason: "Training examples use lemma 'walk' for form 'walk' in 2/2 comparable examples."
evidence_sent_ids: [train-walk-1, train-walk-2]
```

**Token 5 `home`** — `checked`. **Token 6 `.`** — `checked`.

---

**Sentence after lemma correction:**

```conllu
# sent_id = example-2
# text = The boys quickly walk home .
1	The	the	NOUN	DT	Definite=Ind|PronType=Art	4	det	_	_
2	boys	boy	NOUN	NNS	Number=Sing	4	nsubj	_	_
3	quickly	quickly	ADV	RB	_	4	advmod	_	_
4	walk	walk	VERB	VBP	Mood=Ind	0	nsubj	_	_
5	home	home	ADV	RB	_	4	advmod	_	_
6	.	.	PUNCT	.	_	4	punct	_	_
```

Changed: token 4 lemma `walkd` → `walk`.

---

### Step 4 — UPOS/XPOS correction (`RuleBasedPosChecker`)

The checker runs once per token for `upos`, then again for `xpos` (same checker instance for both passes).

**Token 1 `The` (predicted `UPOS=NOUN`, `XPOS=DT`, `LEMMA=the`)**

Training examples for form `The` are filtered by lemma:

```python
if token.lemma not in (None, "_", "") and training_token.lemma != token.lemma:
    continue
```

Predicted `LEMMA=the`. Training examples also have `LEMMA=the` → they pass.

Votes for `(UPOS, XPOS)` pair: `(DET, DT)` appears 2/2 times (ratio=1.0). Threshold met.

Predicted pair is `(NOUN, DT)` — `UPOS` differs.

Result: **`patched`**.

Patches applied:

```yaml
- token_id: 1
  column: upos
  value: DET
  reason: "Training examples use UPOS 'DET' for form 'The' in 2/2 comparable examples."
  evidence_sent_ids: [train-the-1, train-the-2]
- token_id: 1
  column: xpos
  value: DT
  reason: "Training examples use XPOS 'DT' for form 'The' in 2/2 comparable examples."
  evidence_sent_ids: [train-the-1, train-the-2]
```

(XPOS is already `DT` so that patch is a no-op, but it is still emitted by the checker.)

**Tokens 2–6:** all pass as `checked` — no POS errors in training evidence for these tokens.

---

**Sentence after UPOS/XPOS correction:**

```conllu
# sent_id = example-2
# text = The boys quickly walk home .
1	The	the	DET	DT	Definite=Ind|PronType=Art	4	det	_	_
2	boys	boy	NOUN	NNS	Number=Sing	4	nsubj	_	_
3	quickly	quickly	ADV	RB	_	4	advmod	_	_
4	walk	walk	VERB	VBP	Mood=Ind	0	nsubj	_	_
5	home	home	ADV	RB	_	4	advmod	_	_
6	.	.	PUNCT	.	_	4	punct	_	_
```

Changed: token 1 `UPOS` `NOUN` → `DET`.

---

### Step 5 — FEATS correction (`RuleBasedFeatsChecker`)

The feats checker runs after UPOS and XPOS have been corrected. It filters training examples by form, lemma, UPOS, and XPOS together, so it works with the updated values.

**Token 1 `The` (`UPOS=DET`, `XPOS=DT`, `LEMMA=the`)**

Training examples for form `The` are now filtered by `UPOS=DET` and `XPOS=DT` — both match. They vote for the feature bundle `{"Definite": "Def", "PronType": "Art"}` (2/2 times).

Predicted feats: `{"Definite": "Ind", "PronType": "Art"}` — `Definite` differs.

Result: **`patched`**.

Patch applied:

```yaml
token_id: 1
column: feats
value: {Definite: Def, PronType: Art}
reason: "Training examples use FEATS {'Definite': 'Def', 'PronType': 'Art'} for form 'The' in 2/2 comparable examples."
evidence_sent_ids: [train-the-1, train-the-2]
```

**Token 2 `boys` (`UPOS=NOUN`, `XPOS=NNS`, `LEMMA=boy`)**

Training examples for form `boys` filtered by lemma `boy`, UPOS `NOUN`, XPOS `NNS` → both match. They vote for `{"Number": "Plur"}` (2/2 times).

Predicted feats: `{"Number": "Sing"}` — differs.

Result: **`patched`**.

Patch applied:

```yaml
token_id: 2
column: feats
value: {Number: Plur}
reason: "Training examples use FEATS {'Number': 'Plur'} for form 'boys' in 2/2 comparable examples."
evidence_sent_ids: [train-boys-1, train-boys-2]
```

> **Why FEATS runs correctly here but lemma did not for token 1:**
> Lemma ran first, when UPOS was still `NOUN`. The lemma checker filters by UPOS and found no matching evidence. FEATS runs later, after UPOS was corrected to `DET`, so training evidence now matches. The two checkers see different snapshots of the sentence because patches are applied in-place as the pipeline moves through columns.

**Tokens 3–6:** `checked`.

---

**Sentence after FEATS correction:**

```conllu
# sent_id = example-2
# text = The boys quickly walk home .
1	The	the	DET	DT	Definite=Def|PronType=Art	4	det	_	_
2	boys	boy	NOUN	NNS	Number=Plur	4	nsubj	_	_
3	quickly	quickly	ADV	RB	_	4	advmod	_	_
4	walk	walk	VERB	VBP	Mood=Ind	0	nsubj	_	_
5	home	home	ADV	RB	_	4	advmod	_	_
6	.	.	PUNCT	.	_	4	punct	_	_
```

Changed: token 1 `Definite=Ind` → `Definite=Def`. Token 2 `Number=Sing` → `Number=Plur`.

---

### Step 6 — HEAD/DEPREL correction (`RuleBasedSyntaxChecker`)

The syntax checker inspects the whole sentence in a single call (not token-by-token).

It looks for two trivially fixable inconsistencies:

1. `DEPREL=root` but `HEAD≠0` → patch HEAD to 0
2. `HEAD=0` but `DEPREL≠root` → patch DEPREL to `root`

**Token 4 `walk`** has `HEAD=0` and `DEPREL=nsubj`. This is case 2.

There is exactly one token with `HEAD=0`, so the root structure is not ambiguous. The fix is safe.

Result: **`patched`**.

Patch applied:

```yaml
token_id: 4
column: deprel
value: root
reason: "Token has HEAD 0, so DEPREL should be root."
```

**What the agent would add:** `RuleBasedSyntaxChecker` only handles these two mechanical cases. If configured with an agent via `ChainedSentenceChecker`, the agent would be called *only* when the rule returns `uncertain` — for example, a sentence where there are two tokens with `HEAD=0`, or a token pointing to a non-existent head. In those ambiguous cases the rule gives up and the agent uses training examples to find the correct attachment.

---

**Sentence after HEAD/DEPREL correction:**

```conllu
# sent_id = example-2
# text = The boys quickly walk home .
1	The	the	DET	DT	Definite=Def|PronType=Art	4	det	_	_
2	boys	boy	NOUN	NNS	Number=Plur	4	nsubj	_	_
3	quickly	quickly	ADV	RB	_	4	advmod	_	_
4	walk	walk	VERB	VBP	Mood=Ind	0	root	_	_
5	home	home	ADV	RB	_	4	advmod	_	_
6	.	.	PUNCT	.	_	4	punct	_	_
```

Changed: token 4 `DEPREL` `nsubj` → `root`.

---

### Step 7 — Post-validation

The same structural checks run again on the corrected sentence.

- All `UPOS` values are in the valid set.
- Exactly one token has `HEAD=0` and `DEPREL=root` (token 4).
- All HEAD references point to valid token IDs or 0.

Result: **no errors**. The initial validation error on token 4 is resolved.

---

## Final corrected sentence

```conllu
# sent_id = example-2
# text = The boys quickly walk home .
1	The	the	DET	DT	Definite=Def|PronType=Art	4	det	_	_
2	boys	boy	NOUN	NNS	Number=Plur	4	nsubj	_	_
3	quickly	quickly	ADV	RB	_	4	advmod	_	_
4	walk	walk	VERB	VBP	Mood=Ind	0	root	_	_
5	home	home	ADV	RB	_	4	advmod	_	_
6	.	.	PUNCT	.	_	4	punct	_	_
```

---

## Audit summary (YAML)

```yaml
example-2:
  audit_summary:
    patched_columns: [lemma, upos, feats, deprel]
    uncertain_columns: []
    needs_syntax_correction: false
    initial_validation_errors: 1
    post_validation_errors: 0
```

Notes:
- `xpos` is technically patched by the POS checker (the existing value `DT` matched the training evidence, so the emitted patch is a no-op), but because the value did not actually change it would not appear in `patched_columns` without a separate tracking mechanism. The column-level status is reported as `patched` only when at least one `CorrectionPatch` is in the audit for that column.
- `lemma` is `patched` because token 4 was fixed. Token 1's `uncertain` lemma result is recorded in the individual `token_results` inside the audit but does not change the column-level status to `uncertain`, because another token in the same column pass was successfully patched.

---

## What was not corrected and why

| Token | Column | Issue | Why not fixed |
|---|---|---|---|
| 1 `The` | `lemma` | `uncertain` during lemma pass | Predicted `UPOS=NOUN` at that point filtered out all training evidence (which is `DET`). Lemma was already correct, so no harm done. |
| 4 `walk` | `head` | `HEAD=0` left as-is | Correct value. The patch moved `DEPREL` to `root`; `HEAD=0` was always right. |

---

## Column ordering matters

The pipeline processes columns in a fixed order: `lemma → upos → xpos → feats → head → deprel → deps → misc`. Patches are applied in-place before the next column runs. This means:

- The lemma checker sees the *original* UPOS values and can miss evidence if UPOS is wrong.
- The feats checker sees the *corrected* UPOS and XPOS, so it finds the right training evidence.
- A future improvement could re-run lemma after POS correction when lemma came back `uncertain`. That is not implemented today.

---

## Traces with agent checkers enabled

The section above shows the rule-only path. Here every rule checker is paired with an agent via `ChainedTokenChecker` or `ChainedSentenceChecker`. With `only_when_uncertain: true`, an agent is only called when its paired rule returns `uncertain`. If the rule patches or checks successfully, the agent is skipped entirely.

### Configuration

```yaml
lemma:
  rule: true
  agent:
    skill: agents/correct_lemma/SKILL.md
    only_when_uncertain: true

upos_xpos:
  rule: true
  agent:
    skill: agents/correct_upos_xpos/SKILL.md
    only_when_uncertain: true

feats:
  rule: true
  agent:
    skill: agents/correct_feats/SKILL.md
    only_when_uncertain: true

head_deprel:
  rule: true
  agent:
    skill: agents/correct_head_deprel/SKILL.md
    only_when_uncertain: true
```

---

### `predict_column` agent — annotating from blank

The `predict_column` agent runs in a separate prediction mode, before correction. It is driven by `predict_sentence_columns` in `llmparser/prediction.py`. The input is a tokenised sentence with all annotation columns blank:

```conllu
# sent_id = example-2
# text = The boys quickly walk home .
1	The	_	_	_	_	_	_	_	_
2	boys	_	_	_	_	_	_	_	_
3	quickly	_	_	_	_	_	_	_	_
4	walk	_	_	_	_	_	_	_	_
5	home	_	_	_	_	_	_	_	_
6	.	_	_	_	_	_	_	_	_
```

The predictor is called once per token per configured column in a fixed left-to-right order (`lemma`, then `upos`, then `xpos`, etc.). Each call applies its patch in-place, so by the time the `feats` predictor runs for token 1, `lemma`, `upos`, and `xpos` are already filled.

Payload sent to the agent for **token 1, column `lemma`** (abbreviated):

```json
{
  "skill": "# Column Prediction Agent\n...",
  "sentence": {
    "sent_id": "example-2",
    "text": "The boys quickly walk home .",
    "tokens": [
      {"id": 1, "form": "The", "lemma": "_", "upos": "_", "xpos": "_", "feats": null, "head": null, "deprel": null, "deps": null, "misc": null},
      {"id": 2, "form": "boys", ...},
      ...
    ]
  },
  "token": {"id": 1, "form": "The", "lemma": "_", "upos": "_", ...},
  "column": "lemma",
  "evidence": {
    "form_examples": [
      {"sent_id": "train-the-1", "conllu": "1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t2\tdet\t_\t_"},
      {"sent_id": "train-the-2", "conllu": "1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t2\tdet\t_\t_"}
    ],
    "lemma_examples": [],
    "form_upos_examples": [],
    "similar_sentence_examples": [...]
  }
}
```

Expected agent response:

```json
{
  "token_id": 1,
  "column": "lemma",
  "status": "patched",
  "patches": [
    {
      "token_id": 1,
      "column": "lemma",
      "value": "the",
      "reason": "Training examples consistently use lemma 'the' for form 'The'.",
      "evidence_sent_ids": ["train-the-1", "train-the-2"]
    }
  ]
}
```

Patch applied in-place. By the time the `upos` predictor runs for token 1, `lemma` is already `the`.

---

### `correct_segmentation` agent — not invoked

Segmentation is not configured for this example and does not typically apply to English text. The `correct_segmentation` agent is not invoked. For a full trace of segmentation splitting and the agent contract, see [example1.md](example1.md), Example B.

---

### `correct_lemma` agent — resolving `uncertain` on token 1

During the correction pipeline, the rule returned `uncertain` for token 1 because the predicted `UPOS=NOUN` filtered out all training evidence. The chained checker now calls the agent.

Payload sent to the agent (abbreviated). Note that the sentence still shows `UPOS=NOUN` for token 1 — the agent receives the current state of the sentence, not the corrected state:

```json
{
  "skill": "# Lemma Correction Agent\n...",
  "sentence": {
    "sent_id": "example-2",
    "text": "The boys quickly walk home .",
    "tokens": [
      {"id": 1, "form": "The", "lemma": "the", "upos": "NOUN", "xpos": "DT",
       "feats": {"Definite": "Ind", "PronType": "Art"}, "head": 4, "deprel": "det", ...},
      ...
    ]
  },
  "token": {"id": 1, "form": "The", "lemma": "the", "upos": "NOUN", "xpos": "DT", ...},
  "column": "lemma",
  "evidence": {
    "form_examples": [
      {"sent_id": "train-the-1", "conllu": "1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t2\tdet\t_\t_"},
      {"sent_id": "train-the-2", "conllu": "1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t2\tdet\t_\t_"}
    ],
    "lemma_examples": [
      {"sent_id": "train-the-1", "conllu": "..."}
    ],
    "form_upos_examples": [],
    "similar_sentence_examples": [...]
  }
}
```

The agent inspects the evidence and notes that form `The` maps to lemma `the` in all training examples, regardless of the mismatched UPOS. The predicted lemma is already `the`, so no correction is needed.

Agent response:

```json
{
  "token_id": 1,
  "column": "lemma",
  "status": "checked",
  "patches": [],
  "evidence_sent_ids": ["train-the-1", "train-the-2"],
  "notes": "Predicted lemma 'the' matches training evidence for form 'The'. UPOS mismatch in prediction was ignored."
}
```

**Effect:** no patch applied. The per-token result changes from `uncertain` to `checked` in the audit. The lemma column for token 1 is now confirmed correct.

> The rule was blocked by UPOS mismatch; the agent resolved the uncertainty without being constrained to exact-match filtering. This is the main value of chaining: the rule handles clear cases cheaply, and the agent handles the ambiguous residue.

---

### `correct_upos_xpos` agent — skipped

The rule patched token 1 from `NOUN` to `DET` and returned `status: "patched"`. With `only_when_uncertain: true`, the chain returns the rule result immediately without invoking the agent. Tokens 2–6 were `checked` by the rule. Agent not called for any token.

---

### `correct_feats` agent — skipped

The rule patched tokens 1 and 2 with full confidence (`patched`). Tokens 3–6 were `checked`. Agent not called for any token.

---

### `correct_head_deprel` agent — handling a double-root ambiguity

The trivial HEAD/DEPREL error in the original example (token 4, `HEAD=0`, `DEPREL=nsubj`) is handled by the rule — the agent is never called. To show the agent working, consider a harder variant of the sentence where two tokens are both incorrectly marked as root:

```conllu
# sent_id = example-2-hard
# text = The boys walk quickly .
1	The	the	DET	DT	Definite=Def|PronType=Art	2	det	_	_
2	boys	boy	NOUN	NNS	Number=Plur	0	root	_	_
3	walk	walk	VERB	VBP	Mood=Ind	0	root	_	_
4	quickly	quickly	ADV	RB	_	3	advmod	_	_
5	.	.	PUNCT	.	_	3	punct	_	_
```

Tokens 2 and 3 both have `HEAD=0` and `DEPREL=root`. Initial validation reports `root-count: 2`. The rule checker evaluates:

- `root_tokens` (HEAD=0): 2 tokens
- `deprel_root_tokens` (DEPREL=root): 2 tokens

Both counts are non-one, so the rule returns:

```json
{"status": "uncertain", "patches": [], "notes": "Cannot safely repair root count 2; expected exactly 1 root."}
```

The chain now calls the agent. Payload (abbreviated):

```json
{
  "skill": "# HEAD/DEPREL Correction Agent\n...",
  "sentence": {
    "sent_id": "example-2-hard",
    "text": "The boys walk quickly .",
    "tokens": [
      {"id": 1, "form": "The",     "upos": "DET",  "head": 2, "deprel": "det",    ...},
      {"id": 2, "form": "boys",    "upos": "NOUN", "head": 0, "deprel": "root",   ...},
      {"id": 3, "form": "walk",    "upos": "VERB", "head": 0, "deprel": "root",   ...},
      {"id": 4, "form": "quickly", "upos": "ADV",  "head": 3, "deprel": "advmod", ...},
      {"id": 5, "form": ".",       "upos": "PUNCT","head": 3, "deprel": "punct",  ...}
    ]
  },
  "examples": [
    {"sent_id": "train-walk-1", "conllu": "2\twalk\twalk\tVERB\tVBP\tMood=Ind\t0\troot\t_\t_"},
    {"sent_id": "train-walk-2", "conllu": "1\twalk\twalk\tVERB\tVBP\tMood=Ind\t0\troot\t_\t_"}
  ]
}
```

The agent reasons:
- Token 3 (`walk`) is a `VERB`. Training evidence shows verbs are the clausal root.
- Tokens 4 and 5 already attach to token 3, which is consistent with token 3 being the head.
- Token 2 (`boys`) is a `NOUN` and the natural subject of `walk`; nouns are not clausual roots in these training examples.

Agent response:

```json
{
  "status": "patched",
  "patches": [
    {
      "token_id": 2,
      "column": "head",
      "value": 3,
      "reason": "boys is the nominal subject of walk; the verb walk is the clausal root.",
      "evidence_sent_ids": ["train-walk-1", "train-walk-2"]
    },
    {
      "token_id": 2,
      "column": "deprel",
      "value": "nsubj",
      "reason": "boys is a nominal subject, not a root.",
      "evidence_sent_ids": ["train-walk-1", "train-walk-2"]
    }
  ],
  "notes": "Resolved double-root: walk (VERB, token 3) is the clausal root; boys (NOUN, token 2) is nsubj."
}
```

Both patches are applied. The resulting sentence has exactly one root (token 3):

```conllu
# sent_id = example-2-hard
# text = The boys walk quickly .
1	The	the	DET	DT	Definite=Def|PronType=Art	2	det	_	_
2	boys	boy	NOUN	NNS	Number=Plur	3	nsubj	_	_
3	walk	walk	VERB	VBP	Mood=Ind	0	root	_	_
4	quickly	quickly	ADV	RB	_	3	advmod	_	_
5	.	.	PUNCT	.	_	3	punct	_	_
```

Post-validation: no errors.

---

### Agent invocation summary for example-2

| Stage | Rule result | Agent called? | Agent result |
|---|---|---|---|
| lemma, token 1 | `uncertain` | **yes** | `checked` (confirmed prediction correct) |
| lemma, tokens 2–6 | `checked` or `patched` | no | — |
| upos_xpos, token 1 | `patched` | no | — |
| upos_xpos, tokens 2–6 | `checked` | no | — |
| feats, tokens 1–2 | `patched` | no | — |
| feats, tokens 3–6 | `checked` | no | — |
| head_deprel (original example) | `patched` | no | — |
| head_deprel (double-root variant) | `uncertain` | **yes** | `patched` (resolved ambiguity) |
