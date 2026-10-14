"""Score predictions against gold CoNLL-U, in the style of the CoNLL 2018 shared task.

Sentences are matched by ``sent_id``. Surface tokens are aligned by their
character span in the running text; words inside multiword tokens are then
matched by form, so segmentation differences are counted rather than breaking
the comparison. Each metric reports precision (over
predicted words), recall (over gold words) and F1; with identical segmentation
all three equal accuracy.
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from pydantic import BaseModel

from .corpus.conllu import Sentence, Token

METRICS = ("Words", "UPOS", "XPOS", "Lemmas", "UFeats", "AllTags", "UAS", "LAS")


class MetricScore(BaseModel):
    metric: str
    correct: int
    gold: int
    predicted: int
    precision: float
    recall: float
    f1: float


class Evaluation(BaseModel):
    sentences: int
    missing: list[str]
    extra: list[str]
    scores: list[MetricScore]

    def score(self, metric: str) -> MetricScore:
        return next(score for score in self.scores if score.metric == metric)

    def table(self) -> str:
        lines = [f"{'Metric':<8} {'Precision':>9} {'Recall':>9} {'F1':>9}"]
        lines += [f"{s.metric:<8} {s.precision:>9.2%} {s.recall:>9.2%} {s.f1:>9.2%}" for s in self.scores]
        return "\n".join(lines)


Span = tuple[int, int]


def _units(sentence: Sentence) -> dict[Span, list[Token]]:
    """Surface tokens keyed by their character span, each with its words.

    A multiword token is one unit holding its words; any other word is a unit
    of its own. Spans are computed over the surface forms, so a split and an
    unsplit analysis of the same token cover the same span.
    """
    units: dict[Span, list[Token]] = {}
    position, covered = 0, 0
    for token in sentence.tokens:
        if token.is_empty_node or (token.is_word and int(token.id) <= covered):
            continue
        span = (position, position + len(token.form))
        position = span[1]
        if token.is_multiword:
            start, covered = token.span()
            units[span] = [word for word in sentence.words if start <= int(word.id) <= covered]
        else:
            units[span] = [token]
    return units


def _align(pred: Sentence, gold: Sentence) -> list[tuple[Token, Token]]:
    """Pairs of aligned words: same surface span, then same forms in order within it."""
    gold_units = _units(gold)
    pairs = []
    for span, pred_words in _units(pred).items():
        gold_words = gold_units.get(span)
        if gold_words is None:
            continue
        matcher = SequenceMatcher(None, [w.form.lower() for w in pred_words], [w.form.lower() for w in gold_words], autojunk=False)
        for block in matcher.get_matching_blocks():
            pairs += zip(pred_words[block.a : block.a + block.size], gold_words[block.b : block.b + block.size], strict=True)
    return pairs


def _compare(pred: Token, gold: Token, pred_to_gold: dict[str, str]) -> set[str]:
    correct = {"Words"}
    same = {
        "UPOS": pred.upos == gold.upos,
        "XPOS": pred.xpos == gold.xpos,
        "Lemmas": pred.lemma == gold.lemma,
        "UFeats": pred.feats_dict == gold.feats_dict,
    }
    correct |= {metric for metric, ok in same.items() if ok}
    if all(same[metric] for metric in ("UPOS", "XPOS", "UFeats")):
        correct.add("AllTags")
    head = "0" if pred.head == "0" else pred_to_gold.get(pred.head)
    if head == gold.head:
        correct.add("UAS")
        if pred.deprel == gold.deprel:
            correct.add("LAS")
    return correct


def evaluate(predicted: list[Sentence], gold: list[Sentence], predicted_only: bool = False) -> Evaluation:
    """Score ``predicted`` against ``gold`` for every metric in :data:`METRICS`.

    With ``predicted_only``, gold sentences missing from the predictions are
    ignored (useful when only part of a test set was annotated).
    """
    by_id = {sentence.sent_id: sentence for sentence in predicted}
    if predicted_only:
        gold = [sentence for sentence in gold if sentence.sent_id in by_id]
    gold_ids = {sentence.sent_id for sentence in gold}
    counts: Counter[str] = Counter()
    gold_words = predicted_words = matched = 0
    missing: list[str] = []
    for gold_sentence in gold:
        gold_words += len(gold_sentence.words)
        pred_sentence = by_id.get(gold_sentence.sent_id)
        if pred_sentence is None:
            missing.append(gold_sentence.sent_id)
            continue
        matched += 1
        predicted_words += len(pred_sentence.words)
        aligned = _align(pred_sentence, gold_sentence)
        pred_to_gold = {p.id: g.id for p, g in aligned}
        for pred_word, gold_word in aligned:
            counts.update(_compare(pred_word, gold_word, pred_to_gold))

    scores = []
    for metric in METRICS:
        correct = counts[metric]
        precision = correct / predicted_words if predicted_words else 0.0
        recall = correct / gold_words if gold_words else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(
            MetricScore(metric=metric, correct=correct, gold=gold_words, predicted=predicted_words, precision=precision, recall=recall, f1=f1)
        )
    extra = [sentence.sent_id for sentence in predicted if sentence.sent_id not in gold_ids]
    return Evaluation(sentences=matched, missing=missing, extra=extra, scores=scores)
