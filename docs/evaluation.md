# Evaluation

```bash
uv run mcparser evaluate pred.conllu gold.conllu          # table
uv run mcparser evaluate pred.conllu gold.conllu --json   # machine-readable
uv run mcparser evaluate pred.conllu gold.conllu --annotated-only  # only parsed sentences
```

`--calibration` adds a table of accuracy by the confidence recorded in MISC
(`Conf=`), pooled over UPOS, XPOS, lemma, FEATS, HEAD and DEPREL cells. On
about 70,000 cells from ten treebanks, cells at 0.95 or above were 99.4% right,
0.8–0.9 91.8%, and 0.5–0.7 57%, so the scorer ranks cells well; use the table to
choose `--threshold`.

`--predicted-only` ignores gold sentences missing from the predictions;
`--annotated-only` also leaves out predicted sentences that have not been
parsed yet (some HEAD still `_`), e.g. the rest of a file after
`run --max-sentences 30`. Without it, unparsed sentences count as wrong
attachments and pull UAS/LAS down.

The metrics are those of the CoNLL 2018 shared task, ported from the official
`conll18_ud_eval.py` and tested against it (`tests/official/`, 150 random
sentence sets with different tokens, multiword tokens and cells, and 50 whole
files): the numbers agree with the official script. Sentences are matched by
`sent_id` instead of by position, so a partly annotated file can be scored.

| Metric | A word counts as correct when … |
|---|---|
| Tokens | the surface token has the same characters as a gold token |
| Words | it is aligned with a gold word (segmentation) |
| UPOS / XPOS / Lemmas | the column matches |
| UFeats | the **universal** features match (language-specific ones such as `Clitic` are ignored; order is ignored) |
| AllTags | UPOS, XPOS and UFeats all match |
| UAS | its head is the aligned gold head |
| LAS | UAS, and the relation matches **without its subtype** (`obl:arg` = `obl`) |
| CLAS | like LAS, over content words only (precision and recall differ) |
| MLAS | CLAS, plus UPOS, UFeats and the same functional dependents |
| BLEX | CLAS, plus the lemma |

`--strict` adds `UFeatsAll`, `AllTagsAll` and `LASFull`, which count all features
and the full relation. They are lower where a treebank uses subtypes and
language-specific features (Coptic UFeats 95.8 → 95.0, Arabic LAS 81.2 → 80.7).
Numbers in older notes were strict.

`--official PATH` also runs the official script itself (download
<https://universaldependencies.org/conll18/conll18_ud_eval.py>) on the sentences
that were predicted. The official script refuses a tree with a missing HEAD,
several roots or a cycle, so such sentences are listed (`would refuse N sentences`)
and repaired for the official run only (missing heads become the root, extra
roots and cycles are attached to the first root as `dep`). Those repairs are
wrong heads, so the scores are not flattered. A difference to the official
script can only come from whole-file alignment when surface tokens differ at a
sentence boundary; with the tokens of the input it does not occur.

**Alignment.** Surface tokens are aligned by their character span in the
running text. Words inside a multiword token are then matched by form, in
order. A prediction that leaves *del* unsplit therefore loses the words *de*
and *el*, but everything after it still aligns.

Each metric reports **precision** (over predicted words), **recall** (over
gold words) and **F1**. With identical segmentation, all three equal accuracy.
Gold sentences missing from the predictions count against recall, and are
listed.

Because each cell carries a confidence, you can also check calibration. Cells
the scorer rates highly should be right more often; in practice, the lowest
scores pick out the cells most worth checking by hand.
