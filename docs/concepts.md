# Concepts

## Sources

A **source** is one CoNLL-U file loaded into the corpus store under a name. On
the command line a source is written as `[NAME=]PATH[:MODE[:CONFIDENCE]]`:

| Spec | Meaning |
|---|---|
| `train.conllu` | read-only source named `train` |
| `ud=ud-train.conllu:ro` | read-only source named `ud` |
| `pred=out.conllu:rw` | writable source named `pred` |
| `silver=silver.conllu:ro:0.8` | read-only, but its cells only count as 0.8 confident |

- **Read-only (`ro`)** sources hold trusted examples. Their cells default to
  confidence 1.0. Any attempt to edit them is refused by the store itself, so
  no client (agent, Claude Desktop or script) can change them.
- **Writable (`rw`)** sources hold predictions. Their cells default to
  confidence 0.0 until something scores them.

Tagset tools (`allowed_values`, `feature_inventory`) and the "unseen value"
validation warnings count only the read-only sources, so predictions never
define what is allowed.

## Confidence

Every word cell in FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL and DEPS has a
confidence between 0 and 1. It is stored in the MISC column as one `Conf`
attribute:

```text
5	mercado	mercado	NOUN	NCMS000	Gender=Masc|Number=Sing	1	obl	_	SpaceAfter=No|Conf=lemma:0.95,upos:0.98,head:0.60
```

- `form` stands for **segmentation**: how sure we are that this word is
  correctly split off.
- A column without a score has the source's default confidence.
- Setting a value without a confidence **clears** that column's score. An old
  score must not vouch for a new value.
- Editing MISC directly never changes `Conf`; scores change only through the
  confidence API.

Search results are ordered by confidence and accept `min_confidence`. This is
how agents can use earlier, well-settled predictions as evidence alongside the
training data.

## Review flags

Any agent (or person) can raise a **flag** on a word or a whole sentence for a
column: "the lemma of word 3 looks wrong; suggest *dog*". The agent responsible
for that column sees the flag in its next task and closes it as `resolved`
(changed something) or `dismissed` (no change needed, with a note). Flags are
stored next to the prediction file in `<file>.review.json`.

Changing the segmentation of a sentence (splitting, merging, inserting or
deleting words) automatically raises a sentence-level flag for HEAD, because
the tree almost always needs another look.

## History

Every change to a writable source is appended to `<file>.history.jsonl`: the
old and new value, the confidence, the reason given, the agent and the round.
`get_history` reads it back, filtered by sentence or word.

## Segmentation edits

Splitting a word (`split_word`), merging words (`merge_words`), inserting or
deleting a word, and adding or removing multiword tokens all renumber the
sentence. HEAD and DEPS are rewritten to follow the words:

- the dependents of a split word attach to the piece that keeps its head;
- the dependents of merged words attach to the merged word;
- the dependents of a deleted word attach to its head.

Multiword ranges and empty nodes are kept consistent too. Each result includes
an `id_map` from old to new ids, and open flags on the sentence are remapped
automatically.

## Saving and conflicts

Writable sources are saved after every edit, using an atomic rename. If the
file changes on disk underneath the store (for example, Claude Desktop and a
`llmparser run` editing the same file), the next save is refused with a
conflict error rather than overwriting the other change. Call `reload` to pick
up the file's current state.
