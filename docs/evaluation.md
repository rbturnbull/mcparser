# Evaluation

```bash
uv run llmparser evaluate pred.conllu gold.conllu          # table
uv run llmparser evaluate pred.conllu gold.conllu --json   # machine-readable
```

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
