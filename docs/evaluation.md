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

Sentences are matched by `sent_id`. The metrics follow the CoNLL 2018 shared
task:

| Metric | A word counts as correct when … |
|---|---|
| Words | it is aligned with a gold word (segmentation) |
| UPOS / XPOS / Lemmas | the column matches |
| UFeats | the feature set matches (order is ignored) |
| AllTags | UPOS, XPOS and FEATS all match |
| UAS | its head is the aligned gold head |
| LAS | UAS, and the relation matches (including the subtype) |

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
