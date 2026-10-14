# Worked Pipeline Example 1

This file walks through concrete pipeline behavior in this repository.

It shows:
- input
- what each pipeline step changes
- why the step behaves that way
- how each agent is expected to behave

Notes about runtime:
- The CLI correction path is currently rule-based.
- Agent-backed checkers are available through the Python API with configured agent clients.
- The `predict_column` agent contract exists and is implemented at module level (`llmparser/prediction.py`), but there is no production LLM predictor wiring in CLI yet.

## Example A: Column-by-Column Correction (`Hello`)

### A1) Input prediction

From `examples/predictions.conllu`:

```conllu
# sent_id = example-1
# text = Hello
1	Hello	hello	NOUN	NN	Case=Nom	0	root	_	_
```

### A2) Relevant training evidence

From `examples/training.conllu`:

```conllu
# sent_id = train-1
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_

# sent_id = train-2
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
```

### A3) Step-by-step pipeline trace

Pipeline order in `correct_predictions` is:
1. initial validation
2. optional segmentation
3. per-column correction in order (`lemma`, `upos`, `xpos`, `feats`, `head`, `deprel`, `deps`, `misc`)
4. post-validation

#### Step 1: Initial deterministic validation

Result: `checked` (no blocking structural errors here).

Why:
- UPOS `NOUN` is valid.
- HEAD/DEPREL consistency is valid (`0/root`).
- One root token.

#### Step 2: Segmentation correction

Not configured in `examples/correction.yml`, so this step is skipped.

#### Step 3: `lemma` correction (`RuleBasedLemmaChecker`)

Input token at this moment:
- `FORM=Hello`, `LEMMA=hello`, `UPOS=NOUN`

Observed result: `uncertain`, no patch.

Why:
- Lemma checker filters training examples by matching `FORM` and compatible `UPOS` (if UPOS is present in prediction).
- Prediction says `UPOS=NOUN`, but evidence is `UPOS=INTJ`.
- No comparable examples remain after filtering, so checker returns `uncertain`.

Sentence after step: unchanged.

#### Step 4: `upos` correction (`RuleBasedPosChecker`)

Input token at this moment:
- `FORM=Hello`, `LEMMA=hello`, `UPOS=NOUN`, `XPOS=NN`

Observed result: `patched`.

Patch:
- `upos: NOUN -> INTJ`
- `xpos: NN -> Uh`

Why:
- Training evidence has a strong majority for `(UPOS=INTJ, XPOS=Uh)` for `Hello` with lemma `hello`.
- Rule threshold is met (count and ratio).

Sentence after step:

```conllu
# sent_id = example-1
# text = Hello
1	Hello	hello	INTJ	Uh	Case=Nom	0	root	_	_
```

#### Step 5: `xpos` correction (same POS checker)

Observed result: usually `checked`.

Why:
- Previous step already patched `xpos` to `Uh`.
- Running again for column `xpos` finds prediction already aligned.

Sentence after step: unchanged.

#### Step 6: `feats` correction (`RuleBasedFeatsChecker`)

Input token at this moment:
- `FORM=Hello`, `LEMMA=hello`, `UPOS=INTJ`, `XPOS=Uh`, `FEATS=Case=Nom`

Observed result: `patched`.

Patch:
- `feats: Case=Nom -> _`

Why:
- Strong matching evidence says FEATS is empty (`_`) for this token profile.

Sentence after step:

```conllu
# sent_id = example-1
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
```

#### Step 7: `head` / `deprel` correction (`RuleBasedSyntaxChecker`)

Observed result: `checked`.

Why:
- `HEAD=0` and `DEPREL=root` are already consistent.
- Exactly one root.

#### Step 8: `deps` and `misc`

No configured checker, so these are `checked` with no patches.

#### Step 9: Post-validation and summary

Post-validation: no new errors expected.

Typical summary interpretation:
- `patched_columns`: includes `upos`, `feats` (and often `xpos` if patched in that pass)
- `uncertain_columns`: includes `lemma` in this specific ordering
- `needs_syntax_correction`: `false`

### A4) Final corrected sentence

```conllu
# sent_id = example-1
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
```

Important behavior to notice:
- Lemma can remain `uncertain` if POS is wrong early, because lemma runs before POS in the default order and is not revisited automatically.

## Example B: Segmentation-First Correction (`وكتابه جاء`)

This mirrors the tested behavior where segmentation splits one row into two.

### B1) Input prediction

```conllu
# sent_id = ar1
# text = وكتابه جاء
1	وكتابه	وكتابه	NOUN	_	_	2	nsubj	_	_
2	جاء	جاء	VERB	_	_	0	root	_	_
```

### B2) Segmentation checker output

A segmentation checker can return a `TokenSplitPatch`:
- split token `1` (`وكتابه`) into:
1. `و`
2. `كتابه`

After `apply_segmentation_patch`, token IDs are renumbered.

Resulting rows:

```conllu
# sent_id = ar1
# text = وكتابه جاء
1	و	و	CCONJ	_	_	_	_	_	_
2	كتابه	كتاب	NOUN	_	_	_	_	_	_
3	جاء	جاء	VERB	_	_	0	root	_	_
```

Why:
- Segmentation is row-level, not just one-column edits.
- Any split/merge/insert/delete sets `needs_syntax_correction=true`.
- Token and sentence checkers then run on this new tokenization.

## Agent Illustrations (All Agents)

These are minimal examples of expected contract behavior.

## 1) `predict_column` agent

Scope:
- one token
- one target column

Example patch output:

```json
{
  "token_id": 1,
  "column": "lemma",
  "status": "patched",
  "patches": [
    {
      "token_id": 1,
      "column": "lemma",
      "value": "hello",
      "reason": "Predicted lemma from form and context."
    }
  ]
}
```

Why:
- Predict only the target column and leave the rest unchanged.

## 2) `correct_segmentation` agent

Scope:
- sentence-level row operations (`split`, `merge`, `insert`, `delete`)

Example output:

```json
{
  "status": "patched",
  "needs_syntax_correction": true,
  "patches": [
    {
      "type": "split",
      "token_id": 1,
      "replacement_tokens": [
        {
          "id": 1,
          "form": "و",
          "lemma": "و",
          "upos": "CCONJ",
          "xpos": "_",
          "feats": null,
          "head": null,
          "deprel": null,
          "deps": null,
          "misc": null
        },
        {
          "id": 2,
          "form": "كتابه",
          "lemma": "كتاب",
          "upos": "NOUN",
          "xpos": "_",
          "feats": null,
          "head": null,
          "deprel": null,
          "deps": null,
          "misc": null
        }
      ],
      "reason": "Training examples split conjunction proclitic.",
      "evidence_sent_ids": ["train1"]
    }
  ]
}
```

Why:
- Segmentation must change rows, not just cell values.

## 3) `correct_lemma` agent

Scope:
- one token, `lemma` only

Example patch:

```json
{
  "token_id": 4,
  "column": "lemma",
  "status": "patched",
  "patches": [
    {
      "token_id": 4,
      "column": "lemma",
      "value": "δοκέω",
      "reason": "Majority of comparable training rows uses δοκέω.",
      "evidence_sent_ids": ["train12", "train91"]
    }
  ]
}
```

## 4) `correct_upos_xpos` agent

Scope:
- one token, `upos`/`xpos` only

Example patch set:

```json
{
  "token_id": 1,
  "column": "upos",
  "status": "patched",
  "patches": [
    {
      "token_id": 1,
      "column": "upos",
      "value": "INTJ",
      "reason": "Comparable rows tag greeting as INTJ.",
      "evidence_sent_ids": ["train1", "train2"]
    },
    {
      "token_id": 1,
      "column": "xpos",
      "value": "Uh",
      "reason": "Comparable rows tag greeting as Uh.",
      "evidence_sent_ids": ["train1", "train2"]
    }
  ]
}
```

## 5) `correct_feats` agent

Scope:
- one token, `feats` only

Example patch:

```json
{
  "token_id": 7,
  "column": "feats",
  "status": "patched",
  "patches": [
    {
      "token_id": 7,
      "column": "feats",
      "value": {
        "Case": "Acc",
        "Gender": "Masc",
        "Number": "Sing"
      },
      "reason": "Most comparable rows agree on this feature bundle.",
      "evidence_sent_ids": ["train4", "train8"]
    }
  ]
}
```

## 6) `correct_head_deprel` agent

Scope:
- sentence-level tree repair, patch only `head`/`deprel`

Example patch:

```json
{
  "status": "patched",
  "patches": [
    {
      "token_id": 1,
      "column": "head",
      "value": 0,
      "reason": "Token marked root must have HEAD 0."
    }
  ],
  "notes": "Applied root consistency fix."
}
```

If repair is ambiguous, expected output is:

```json
{
  "status": "uncertain",
  "patches": [],
  "notes": "Multiple plausible roots; cannot safely repair."
}
```

## Practical Mapping To Code

- Pipeline driver: `llmparser/correction.py`
- Checker configuration and chaining: `llmparser/correction_config.py`
- Agent adapters: `llmparser/agent_checker.py`
- Rule checkers:
  - `llmparser/lemma_checker.py`
  - `llmparser/pos_checker.py`
  - `llmparser/feats_checker.py`
  - `llmparser/syntax_checker.py`
- Segmentation patch application: `llmparser/segmentation.py`
- Evidence retrieval: `llmparser/evidence.py`
- Validation rules: `llmparser/validation.py`

