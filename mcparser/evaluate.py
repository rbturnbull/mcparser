"""Score predictions against gold CoNLL-U with the CoNLL 2018 shared task metrics.

The definitions and the word alignment are ported from the official
``conll18_ud_eval.py`` and tested against it (``tests/official/``), so the numbers
agree with it exactly. Sentences are matched by ``sent_id`` instead of by
position, so a partly annotated file can be scored. Surface tokens are aligned by
their character span in the running text; words inside multiword tokens are then
matched by form, so segmentation differences are counted rather than breaking
the comparison. Each metric reports precision (over predicted words), recall
(over gold words) and F1; with identical segmentation all three equal accuracy.
:func:`run_official` runs the official script itself.
"""

from __future__ import annotations

import importlib.util
import io
import unicodedata
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from .corpus.confidence import read_confidence
from .corpus.conllu import EMPTY, Sentence, Token, format_conllu
from .corpus.errors import CorpusError

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
    invalid: list[str] = []
    scores: list[MetricScore]

    def score(self, metric: str) -> MetricScore:
        return next(score for score in self.scores if score.metric == metric)

    def table(self) -> str:
        lines = [f"{'Metric':<10} {'Precision':>9} {'Recall':>9} {'F1':>9}"]
        lines += [f"{s.metric:<10} {s.precision:>9.2%} {s.recall:>9.2%} {s.f1:>9.2%}" for s in self.scores]
        return "\n".join(lines)


CONTENT_DEPRELS = frozenset(
    "nsubj obj iobj csubj ccomp xcomp obl vocative expl dislocated advcl advmod discourse nmod appos nummod acl amod conj fixed flat "
    "compound list parataxis orphan goeswith reparandum root dep".split()
)
FUNCTIONAL_DEPRELS = frozenset("aux cop mark det clf case cc".split())
UNIVERSAL_FEATURES = frozenset(
    "PronType NumType Poss Reflex Foreign Abbr Gender Animacy Number Case Definite Degree VerbForm Mood Tense Aspect Voice Evident "
    "Polarity Person Polite".split()
)
"""The relations and features the CoNLL 2018 scorer distinguishes (copied from ``conll18_ud_eval.py``)."""

OFFICIAL = ("Tokens", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX")
STRICT = ("UFeatsAll", "AllTagsAll", "LASFull")
"""Extra metrics with ``strict=True``: all features (not only the universal ones) and the full relation (with its subtype)."""
METRICS = OFFICIAL


class _Invalid:
    """The head of a word whose HEAD is missing or points outside its sentence: equal to no gold word."""


INVALID = _Invalid()


class _Word:
    """A word with the character span of its surface token (the whole multiword token for the words inside one)."""

    __slots__ = ("token", "start", "end", "multiword", "parent", "functional_children")

    def __init__(self, token: Token, start: int, end: int, multiword: bool):
        self.token, self.start, self.end, self.multiword = token, start, end, multiword
        self.parent: _Word | _Invalid | None = None
        self.functional_children: list[_Word] = []

    @property
    def deprel(self) -> str:
        return self.token.deprel.split(":")[0]

    @property
    def content(self) -> bool:
        return self.deprel in CONTENT_DEPRELS

    @property
    def feats(self) -> str:
        return "|".join(sorted(feat for feat in self.token.feats.split("|") if feat.split("=", 1)[0] in UNIVERSAL_FEATURES))


def _surface(form: str) -> int:
    """Length of a form without its spaces (any Unicode space separator), as the official scorer counts it."""
    return sum(unicodedata.category(char) != "Zs" for char in form)


def _words(sentence: Sentence) -> tuple[list[tuple[int, int]], list[_Word]]:
    """The surface token spans and the words of a sentence, with heads and functional children resolved."""
    spans: list[tuple[int, int]] = []
    words: list[_Word] = []
    position, covered = 0, 0
    for token in sentence.tokens:
        if token.is_empty_node or (token.is_word and int(token.id) <= covered):
            continue
        start, position = position, position + _surface(token.form)
        spans.append((start, position))
        if token.is_multiword:
            first, covered = token.span()
            words += [_Word(word, start, position, True) for word in sentence.words if first <= int(word.id) <= covered]
        else:
            words.append(_Word(token, start, position, False))
    by_id = {word.token.id: word for word in words}
    for word in words:
        head = word.token.head
        word.parent = None if head == "0" else by_id.get(head, INVALID)
        if isinstance(word.parent, _Word) and word.deprel in FUNCTIONAL_DEPRELS:
            word.parent.functional_children.append(word)
    return spans, words


def official_problems(sentence: Sentence) -> list[str]:
    """What the official scorer would reject in this sentence: a missing or impossible HEAD, a cycle, not exactly one root."""
    words = sentence.words
    ids = {word.id for word in words}
    problems = [f"word {word.id} has no valid HEAD" for word in words if word.head != "0" and word.head not in ids]
    if problems:
        return problems
    if sum(word.head == "0" for word in words) != 1:
        problems.append("not exactly one root")
    problems += [f"cycle through word {word.id}" for word in words[:1] if any(_cycle(words, other) for other in words)]
    return problems


def _lcs(gold: list[_Word], system: list[_Word]) -> list[list[int]]:
    table = [[0] * (len(system) + 1) for _ in range(len(gold) + 1)]
    for g in reversed(range(len(gold))):
        for s in reversed(range(len(system))):
            same = gold[g].token.form.lower() == system[s].token.form.lower()
            table[g][s] = max(table[g + 1][s], table[g][s + 1], 1 + table[g + 1][s + 1] if same else 0)
    return table


def _beyond(words: list[_Word], i: int, end: int) -> bool:
    if i >= len(words):
        return True
    return words[i].start >= end if words[i].multiword else words[i].end > end


def _multiword_span(gold: list[_Word], system: list[_Word], gi: int, si: int) -> tuple[int, int, int, int]:
    """The smallest stretch of words, starting here, whose character span is the same in both analyses."""
    if gold[gi].multiword:
        end = gold[gi].end
        if not system[si].multiword and system[si].start < gold[gi].start:
            si += 1
    else:
        end = system[si].end
        if gold[gi].start < system[si].start:
            gi += 1
    first_gold, first_system = gi, si
    while not _beyond(gold, gi, end) or not _beyond(system, si, end):
        if gi < len(gold) and (si >= len(system) or gold[gi].start <= system[si].start):
            word, gi = gold[gi], gi + 1
        else:
            word, si = system[si], si + 1
        if word.multiword and word.end > end:
            end = word.end
    return first_gold, first_system, gi, si


def _align_words(gold: list[_Word], system: list[_Word]) -> list[tuple[_Word, _Word]]:
    """Pairs of aligned (gold, system) words, as in the CoNLL 2018 scorer.

    Words outside multiword tokens match when they cover the same characters;
    inside a stretch with a multiword token, they match by form (longest common subsequence).
    """
    pairs: list[tuple[_Word, _Word]] = []
    gi = si = 0
    while gi < len(gold) and si < len(system):
        if gold[gi].multiword or system[si].multiword:
            gs, ss, gi, si = _multiword_span(gold, system, gi, si)
            if gi > gs and si > ss:
                table = _lcs(gold[gs:gi], system[ss:si])
                g = s = 0
                while g < gi - gs and s < si - ss:
                    if gold[gs + g].token.form.lower() == system[ss + s].token.form.lower():
                        pairs.append((gold[gs + g], system[ss + s]))
                        g += 1
                        s += 1
                    elif table[g][s] == table[g + 1][s]:
                        g += 1
                    else:
                        s += 1
        elif (gold[gi].start, gold[gi].end) == (system[si].start, system[si].end):
            pairs.append((gold[gi], system[si]))
            gi += 1
            si += 1
        elif gold[gi].start <= system[si].start:
            gi += 1
        else:
            si += 1
    return pairs


def _align(pred: Sentence, gold: Sentence) -> list[tuple[Token, Token]]:
    """Pairs of aligned (predicted, gold) tokens."""
    return [(p.token, g.token) for g, p in _align_words(_words(gold)[1], _words(pred)[1])]


def _keys(strict: bool):
    """Metric name -> (function giving the value to compare for a word, filter on words or None).

    The first argument of each function is the word, the second maps a word to
    the gold word it stands for (the gold word itself, or for a predicted word
    its aligned gold word, or ``"NotAligned"``).
    """

    def lemma(word, gold_of):
        return word.token.lemma if gold_of(word).token.lemma != EMPTY else EMPTY

    def parent(word, gold_of):
        return gold_of(word.parent)

    keys = {
        "UPOS": (lambda w, g: w.token.upos, None),
        "XPOS": (lambda w, g: w.token.xpos, None),
        "UFeats": (lambda w, g: w.feats, None),
        "AllTags": (lambda w, g: (w.token.upos, w.token.xpos, w.feats), None),
        "Lemmas": (lemma, None),
        "UAS": (parent, None),
        "LAS": (lambda w, g: (parent(w, g), w.deprel), None),
        "CLAS": (lambda w, g: (parent(w, g), w.deprel), lambda w: w.content),
        "MLAS": (
            lambda w, g: (
                parent(w, g), w.deprel, w.token.upos, w.feats,
                [(g(c), c.deprel, c.token.upos, c.feats) for c in w.functional_children],
            ),  # fmt: skip
            lambda w: w.content,
        ),
        "BLEX": (lambda w, g: (parent(w, g), w.deprel, lemma(w, g)), lambda w: w.content),
    }
    if strict:
        keys |= {
            "UFeatsAll": (lambda w, g: w.token.feats_dict, None),
            "AllTagsAll": (lambda w, g: (w.token.upos, w.token.xpos, w.token.feats_dict), None),
            "LASFull": (lambda w, g: (parent(w, g), w.token.deprel), None),
        }
    return keys


def is_annotated(sentence: Sentence) -> bool:
    """Whether every word has a HEAD, i.e. the sentence has been parsed."""
    return bool(sentence.words) and all(word.head != "_" for word in sentence.words)


def evaluate(
    predicted: list[Sentence],
    gold: list[Sentence],
    predicted_only: bool = False,
    annotated_only: bool = False,
    strict: bool = False,
) -> Evaluation:
    """Score ``predicted`` against ``gold`` with the metrics of the CoNLL 2018 shared task (:data:`OFFICIAL`).

    The definitions follow ``conll18_ud_eval.py``: UFeats and AllTags look at the
    universal features only, relation subtypes are ignored (``obl:arg`` is
    ``obl``), CLAS, MLAS and BLEX cover content words. With ``strict``, three more
    metrics count language-specific features and relation subtypes
    (:data:`STRICT`). Sentences are matched by ``sent_id`` rather than by position.

    With ``predicted_only``, gold sentences missing from the predictions are
    ignored (useful when only part of a test set was annotated). With
    ``annotated_only``, only predicted sentences that have been parsed (every
    word has a HEAD) are scored, so unparsed sentences do not hide the accuracy
    of the parsed ones. ``Evaluation.invalid`` lists scored sentences the official
    scorer would refuse (no HEAD, a cycle, several roots); their wrong heads count as errors.
    """
    if annotated_only:
        predicted = [sentence for sentence in predicted if is_annotated(sentence)]
        predicted_only = True
    by_id = {sentence.sent_id: sentence for sentence in predicted}
    if predicted_only:
        gold = [sentence for sentence in gold if sentence.sent_id in by_id]
    gold_ids = {sentence.sent_id for sentence in gold}
    keys = _keys(strict)
    names = [*OFFICIAL, *(STRICT if strict else ())]
    correct: Counter[str] = Counter()
    gold_total: Counter[str] = Counter()
    system_total: Counter[str] = Counter()
    missing: list[str] = []
    invalid: list[str] = []
    matched = 0
    for gold_sentence in gold:
        gold_spans, gold_words = _words(gold_sentence)
        gold_total["Tokens"] += len(gold_spans)
        gold_total["Words"] += len(gold_words)
        for name, (_, keep) in keys.items():
            gold_total[name] += sum(keep is None or keep(word) for word in gold_words)
        pred_sentence = by_id.get(gold_sentence.sent_id)
        if pred_sentence is None:
            missing.append(gold_sentence.sent_id)
            continue
        matched += 1
        if official_problems(pred_sentence):
            invalid.append(pred_sentence.sent_id)
        pred_spans, pred_words = _words(pred_sentence)
        system_total["Tokens"] += len(pred_spans)
        system_total["Words"] += len(pred_words)
        for name, (_, keep) in keys.items():
            system_total[name] += sum(keep is None or keep(word) for word in pred_words)
        correct["Tokens"] += len(set(gold_spans) & set(pred_spans))
        aligned = _align_words(gold_words, pred_words)
        gold_of = {pred: gold_word for gold_word, pred in aligned}
        correct["Words"] += len(aligned)

        def gold_aligned(word: _Word | _Invalid | None):
            if word is None:
                return None
            return gold_of.get(word, "NotAligned") if isinstance(word, _Word) else "NotAligned"

        for gold_word, pred_word in aligned:
            for name, (key, keep) in keys.items():
                if (keep is None or keep(gold_word)) and key(gold_word, lambda w: w) == key(pred_word, gold_aligned):
                    correct[name] += 1
    scores = []
    for name in names:
        right, expected, got = correct[name], gold_total[name], system_total[name]
        precision = right / got if got else 0.0
        recall = right / expected if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(MetricScore(metric=name, correct=right, gold=expected, predicted=got, precision=precision, recall=recall, f1=f1))
    extra = [sentence.sent_id for sentence in predicted if sentence.sent_id not in gold_ids]
    return Evaluation(sentences=matched, missing=missing, extra=extra, invalid=invalid, scores=scores)


def scorable(sentence: Sentence) -> Sentence | None:
    """A copy of ``sentence`` the official scorer accepts, or ``None`` if it needs no change.

    For scoring only, never for output: a missing or impossible HEAD becomes the root,
    extra roots and cycles are attached to the first root (relation ``dep``), and a
    sentence without a root gets one at its first word. Every repaired word is a wrong
    head and relation in all likelihood, so the scores are not flattered.
    """
    if not official_problems(sentence):
        return None
    fixed = sentence.model_copy(deep=True)
    words = fixed.words
    ids = {word.id for word in words}
    for word in words:
        if word.head != "0" and word.head not in ids:
            word.head = "0"
    roots = [word for word in words if word.head == "0"]
    if not roots:
        roots = [words[0]]
        words[0].head = "0"
    root = roots[0]
    for word in roots[1:]:
        word.head, word.deprel = root.id, "dep"
    while problems := [word for word in words if _cycle(words, word)]:
        problems[0].head, problems[0].deprel = root.id, "dep"
    return fixed


def _cycle(words: list[Token], word: Token) -> bool:
    heads = {w.id: w.head for w in words}
    seen, current = set(), word.id
    while current != "0" and current not in seen:
        seen.add(current)
        current = heads[current]
    return current != "0"


class OfficialResult(BaseModel):
    """The output of the official ``conll18_ud_eval.py`` for the sentences of a prediction file."""

    sentences: int
    repaired: list[str]
    skipped: list[str] = []
    """Gold sentences the official script cannot read (e.g. Arabic PADT ``afp.20000715.0008:p3u1``): left out of both files."""
    scores: dict[str, MetricScore]

    def table(self) -> str:
        lines = [f"{'Metric':<10} {'Precision':>9} {'Recall':>9} {'F1':>9}"]
        lines += [f"{s.metric:<10} {s.precision:>9.2%} {s.recall:>9.2%} {s.f1:>9.2%}" for s in self.scores.values()]
        return "\n".join(lines)


def run_official(script: Path | str, predicted: list[Sentence], gold: list[Sentence]) -> OfficialResult:
    """Run the official scorer (``conll18_ud_eval.py``, loaded from ``script``) on the sentences in ``predicted``.

    The official scorer needs the same text in both files, so only gold sentences
    that were predicted are used, in gold order. Sentences it would refuse are
    repaired with :func:`scorable` first and listed in ``repaired``.
    """
    spec = importlib.util.spec_from_file_location("conll18_ud_eval", script)
    if spec is None or spec.loader is None:
        raise CorpusError(f"Cannot load the official scorer from {script}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    by_id = {sentence.sent_id: sentence for sentence in predicted}
    gold = [sentence for sentence in gold if sentence.sent_id in by_id]
    skipped: list[str] = []
    readable = []
    for sentence in gold:
        try:
            module.load_conllu(io.StringIO(format_conllu([sentence])))
        except module.UDError:
            skipped.append(sentence.sent_id)
        else:
            readable.append(sentence)
    gold = readable
    repaired: list[str] = []
    system = []
    for sentence in (by_id[g.sent_id] for g in gold):
        fixed = scorable(sentence)
        if fixed is not None:
            repaired.append(sentence.sent_id)
        system.append(fixed or sentence)
    try:
        result = module.evaluate(
            module.load_conllu(io.StringIO(format_conllu(gold))), module.load_conllu(io.StringIO(format_conllu(system)))
        )
    except module.UDError as error:
        raise CorpusError(f"The official scorer refused the files: {error}") from error
    scores = {
        name: MetricScore(
            metric=name, correct=score.correct, gold=score.gold_total, predicted=score.system_total,
            precision=score.precision, recall=score.recall, f1=score.f1,
        )  # fmt: skip
        for name, score in result.items()
    }
    return OfficialResult(sentences=len(gold), repaired=repaired, skipped=skipped, scores=scores)


CALIBRATION_EDGES = (0.5, 0.7, 0.8, 0.9, 0.95)
"""Lower edges of the confidence buckets (the first bucket starts at 0)."""

CALIBRATION_COLUMNS = ("upos", "xpos", "lemma", "feats", "head", "deprel")


class CalibrationBucket(BaseModel):
    low: float
    high: float
    cells: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.cells if self.cells else 0.0

    @property
    def label(self) -> str:
        return f"{self.low:.2f}-{self.high:.2f}"


def calibration(predicted: list[Sentence], gold: list[Sentence], columns: tuple[str, ...] = CALIBRATION_COLUMNS) -> list[CalibrationBucket]:
    """How accurate cells are, by the confidence recorded in MISC (``Conf=``).

    Only cells of aligned words that carry a confidence count. A good scorer
    gives low scores to the cells that are wrong, so accuracy should rise with
    confidence; this shows where to set the refinement threshold.
    """
    bounds = [0.0, *CALIBRATION_EDGES, 1.0]
    buckets = [CalibrationBucket(low=low, high=high, cells=0, correct=0) for low, high in zip(bounds, bounds[1:])]
    by_id = {sentence.sent_id: sentence for sentence in predicted}
    for gold_sentence in gold:
        pred_sentence = by_id.get(gold_sentence.sent_id)
        if pred_sentence is None:
            continue
        aligned = _align(pred_sentence, gold_sentence)
        pred_to_gold = {p.id: g.id for p, g in aligned}
        for pred_word, gold_word in aligned:
            scores = read_confidence(pred_word)
            for column in columns:
                if column not in scores:
                    continue
                if column == "head":
                    ok = ("0" if pred_word.head == "0" else pred_to_gold.get(pred_word.head)) == gold_word.head
                elif column == "feats":
                    ok = pred_word.feats_dict == gold_word.feats_dict
                else:
                    ok = pred_word.get(column) == gold_word.get(column)
                bucket = next(b for b in buckets if scores[column] < b.high or b is buckets[-1])
                bucket.cells += 1
                bucket.correct += ok
    return buckets


def calibration_table(buckets: list[CalibrationBucket]) -> str:
    lines = [f"{'Confidence':<11} {'Cells':>6} {'Correct':>8} {'Accuracy':>9}"]
    lines += [f"{b.label:<11} {b.cells:>6} {b.correct:>8} {b.accuracy:>9.1%}" for b in buckets]
    return "\n".join(lines)
