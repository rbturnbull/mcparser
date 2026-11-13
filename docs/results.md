# Results

These numbers show what to expect, not a benchmark. They come from the
development of MCParser, with the caveats below, and the evaluator is a port of the
official CoNLL 2018 scorer (see [Evaluation](evaluation.md)).

## 100 Ancient Greek sentences, complete pipeline

Greek PROIEL dev sentences, 1,200 words, annotated from the gold tokenisation
with all roles on the default models, two rounds. The first column leaves the
FEATS convention hint off, the second has it on (the default).

| Metric | Hint off | Hint on |
|---|---|---|
| UPOS | 97.87 | 97.95 |
| XPOS | 97.70 | 97.87 |
| UFeats | 93.77 | 94.18 |
| AllTags | 92.21 | 92.62 |
| Lemmas | 99.10 | 99.02 |
| UAS | 92.95 | 92.04 |
| LAS | 89.91 | 89.09 |
| CLAS | 88.38 | 87.34 |
| MLAS | 77.87 | 77.70 |
| BLEX | 87.39 | 86.23 |
| Cost | $7.93 | $7.83 |

On a hundred sentences, run-to-run noise is about ±1.5 points for UAS and LAS and
the two columns differ by less than that.

## Four full dev files

Sonnet 5.5 in the strong roles, Haiku 5.5 for XPOS, one round, with a cost cap
per language. **None of these runs finished**: every cap was reached before XPOS
and part of FEATS ran, and a since-fixed bug left 13-27% of the sentences
without any head. The table scores only the sentences with a complete tree, so
the true full-file numbers are lower. The pipeline has not yet been rerun on whole files
after the fix.

| Treebank | Sentences scored / total | UPOS | UFeats | Lemmas | UAS | LAS | Cost |
|---|---|---|---|---|---|---|---|
| Arabic PADT | 790 / 909 | 96.86 | 75.85 | 96.06 | 83.95 | 78.71 | $65.5 |
| Ancient Greek PROIEL | 821 / 1,019 | 97.85 | 93.30 | 98.59 | 91.07 | 86.71 | $34.3 |
| Latin PROIEL | 905 / 1,233 | 97.45 | 93.35 | 98.28 | 89.28 | 84.41 | $34.2 |
| Coptic Scriptorium | 313 / 381 | 97.10 | 96.62 | 97.36 | 88.93 | 84.81 | $44.8 |

Arabic UFeats stopped after 10 of about 176 tasks, so most of that column is the baseline
and rules.

## What a sentence costs

| Setup | Cost per word | Cost per sentence |
|---|---|---|
| Opus 5.5 in the strong roles, two rounds (Greek, ~12 words per sentence) | $0.0064 | $0.078 |
| Sonnet 5.5 in the strong roles, one round, estimated for a finished run | $0.003 to $0.0045 | $0.03 (Latin, Greek) to $0.12 (Coptic, Arabic) |

Cost scales with sentence length and with how much the rules cannot settle
(unseen forms are expensive). Syntax is about 40% of the bill and FEATS about 20%.
Use `--max-cost` and `--max-sentences` to bound a run. See [Cost](pipeline.md#cost)
for what keeps it down.

## Caveats

- The treebank's own inconsistencies show up as "errors": several remaining
  differences were cases where the dev gold disagrees with the training data.
- Some Greek and Latin treebank text is also in common web corpora, so models may have
  seen it. Test on text you are sure the models cannot have seen when it matters.
- Unannotated-text evidence (`--corpus`) gave no measurable gain on five languages
  (differences were inside the spread of identical runs) and costs 11-25% more.
- More reasoning (`--strong-thinking high`) gave 0-2 points of LAS for 20-110% more cost.
