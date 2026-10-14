# The pipeline

```text
                         ┌────────────────────────────── round n ──────────────────────────────┐
prediction ──▶ rules ──▶ │ score: scorer rates cells a model changed since it last scored them  │──▶ stop? ──▶ final scoring
(+ baseline)             │ refine: each role in turn, batches of sentences (most confident      │      │
                         │ first), cells below the threshold + open flags; rules after each role│◀─────┘ next round
                         └──────────────────────────────────────────────────────────────────────┘
```

## Round 0: the baseline

`llmparser init` creates the prediction file. When trusted sources are given,
it also applies a **frequency baseline** that needs no LLM:

1. **Segmentation.** A surface token that the trusted data splits into a
   multiword token more often than it leaves it whole is split the same way,
   e.g. Arabic *بشيه* → *ب* + *شيه*.
2. **Morphology.** Each word gets the most frequent (LEMMA, UPOS, XPOS, FEATS)
   analysis of the same form. If there's no exact match, the lookup ignores
   case and diacritics.
3. **Confidence** reflects the evidence: the share of examples agreeing,
   discounted when there are fewer than three, and capped at 0.9.

The baseline only fills empty cells, so running it again changes nothing.
HEAD and DEPREL are left for the agents. You can also skip `init` and give
`run` any CoNLL-U file, for example the output of another parser.

## Each round

**Score.** The scorer agent rates every cell of each sentence that has never
been scored, or that changed after it was last scored. It sends only
confidences, never values, and flags cells it's fairly sure are wrong.

**Refine.** Sentences are visited **from most to least confident**, so the
sentences that settle first can serve as evidence (with `min_confidence`) for
the harder ones. In each sentence, the role agents run in this order:

```text
segmentation → upos → xpos → lemma → feats → head → deprel → (deps)
```

Later columns build on earlier ones. A role runs on a sentence only if it has
cells below the threshold (default 0.8) or open flags for its column.

**Stop.** The loop ends when no role had anything to do, when a round made no
edits, or when the round limit (default 3) is reached. A final scoring pass
then re-scores whatever the last round changed.

## Settings

`llmparser settings` prints the defaults as YAML. Save them, edit them, and
pass the file with `--config`:

```yaml
strong_model: anthropic:claude-opus-5-5       # segmentation, lemma, head, deprel, deps
efficient_model: anthropic:claude-sonnet-5-5  # efficient_roles
light_model: anthropic:claude-haiku-5-5       # light_roles
efficient_roles: [upos, feats, scorer]
light_roles: [xpos]
strong_thinking: low          # reasoning effort: false, true, minimal, low, medium, high, xhigh
threshold: 0.8                # refine cells below this confidence
rounds: 3                     # maximum rounds
batch_sentences: 8            # sentences per agent task
batch_cells: 120              # cells per agent task
concurrency: 1                # tasks run at the same time
request_limit: 60             # model requests per agent task
cache: true                   # prompt caching
rules:
  min_count: 5                # examples needed before code settles a cell
  min_share: 0.95             # share of them that must agree
  use_predictions: true       # confident predictions count as evidence for unseen forms
roles:
  head:
    thinking: medium                     # per-role reasoning effort
  deps:
    enabled: true                        # enhanced dependencies are off by default
```

Command-line overrides: `--strong-model`, `--efficient-model`, `--light-model`,
`--efficient-roles`, `--light-roles`, `--strong-thinking`,
`--efficient-thinking`, `--light-thinking`, `--model` (one model for every
role), `--model-for ROLE=MODEL`, `--roles`, `--rounds`, `--threshold`,
`--batch-sentences`, `--concurrency`, `--sentence` and `--max-sentences`.
`llmparser score` runs only the rules and the scorer.

## Cost

Each run ends with token counts and an estimated price per model (list prices
in {mod}`llmparser.pricing`). What keeps the cost down, roughly in order of
effect:

- **Rules first.** Code settles every cell the treebank makes near-certain (see
  {mod}`llmparser.rules`): on Latin PROIEL about three quarters of the LEMMA,
  UPOS, XPOS and FEATS cells, with no errors in testing, and a fifth of the
  relation labels once heads are known. Those cells never reach a model. With
  `use_predictions`, forms the treebank lacks are settled from confident
  predictions elsewhere in the file, so a long text gets cheaper as it goes.
- **Batches.** Each task covers up to `batch_sentences` sentences, and evidence
  for a repeated form or arc is shown once.
- **One request per task.** Agents (except segmentation) answer with compact
  lines (`SENT_ID TOKEN_ID VALUE CONFIDENCE`) that code checks and applies; an
  invalid answer (unknown tag, broken tree) is sent back to be fixed.
- **Evidence in the prompt**: the form's analyses, analyses after the previous
  word when the form is ambiguous, word-ending analogies for unseen forms, and
  relation counts for each word's actual head.
- **Three model tiers**: Haiku for XPOS, Sonnet for UPOS, FEATS and scoring,
  Opus for segmentation, lemmas and syntax.
- **The scorer only judges what a model or person wrote**, never cells set by
  code, unannotated cells, or XPOS/FEATS of untagged words.
- **Compact tool results**, **prompt caching**, and columns the treebank never
  uses settled without an agent.

On 10 Latin sentences the first version cost $2.43; the current one costs
about $0.08 per sentence of similar length.

Before running a whole file, try `--sentence` on a few sentences and look at
the estimate. `--report run.json` saves the numbers.

With `concurrency` above 1, several sentences are worked on at once. That is
faster, but a sentence can then no longer rely on the more confident sentences
having been finished first.
