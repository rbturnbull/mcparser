# The pipeline

```text
                 ┌──────────────────────────── round n ───────────────────────────┐
prediction ─────▶│ score: scorer rates every cell of sentences changed since last │──▶ stop? ──▶ final scoring
(+ baseline)     │ refine: most→least confident sentence; in each, run the roles  │      │
                 │ in order on cells below the threshold and on their open flags  │◀─────┘ next round
                 └────────────────────────────────────────────────────────────────┘
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
model: anthropic:claude-opus-5-5   # default for every role
threshold: 0.8                     # refine cells below this confidence
rounds: 3                          # maximum rounds
concurrency: 1                     # sentences worked on at the same time
request_limit: 60                  # model requests per agent task
cache: true                        # prompt caching
roles:
  lemma:
    model: anthropic:claude-sonnet-5-5   # a different model for one role
  deps:
    enabled: true                        # enhanced dependencies are off by default
  head:
    instructions: Attach enclitic particles to the verb.  # appended to the role's prompt
```

The most useful command-line overrides are `--model`,
`--model-for ROLE=MODEL`, `--roles upos,lemma`, `--rounds`, `--threshold`,
`--concurrency` and `--sentence`. Use `llmparser score` to run only the
scorer.

## Cost

Each agent task is a short tool-using conversation. In a test on a 6-word
Hittite sentence, one round with every role took 28 model requests. With
prompt caching (on by default), most input tokens are cache reads. Before
running a whole file, try `--sentence` on one or two sentences, and look at
the token counts printed at the end of the run (`--report run.json` writes
them to a file).

With `concurrency` above 1, several sentences are worked on at once. That is
faster, but a sentence can then no longer rely on the more confident sentences
having been finished first.
