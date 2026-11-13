"""Settle near-certain cells in code, without a model.

A rule fires when the evidence is overwhelming: at least ``min_count``
examples and a share of at least ``min_share`` for one value. Rules never
replace a different value: they fill empty cells, or confirm a value that is
already there by raising its confidence. Cells set by rules are not sent to the
role agents (their confidence is above the threshold) or to the scorer.

The rules are:

* **morphology** — a column (LEMMA, UPOS, XPOS, FEATS) whose value is
  overwhelming among the trusted analyses of the form, either overall or after
  the same previous word;
* **XPOS from tags** — the XPOS used with exactly this UPOS and FEATS;
* **DEPREL** — ``root`` for the word with HEAD 0, otherwise the relation used
  for arcs like this one (see :meth:`~mcparser.corpus.CorpusStore.relation_stats`);
* **segmentation** — surface tokens that the trusted data overwhelmingly
  splits (split them) or leaves whole (confirm them); if the trusted data has no
  multiword tokens at all, every word is confirmed;
* **cross-checks** — once a word's relation is confident, a UPOS that the
  treebank (almost) never uses for that lemma in that relation is flagged for
  the UPOS agent (e.g. PROIEL's *omnis* is DET as ``det`` but PRON as a
  subject or object). Cross-checks only flag; they never change a value.
* **relation checks** — a DEPREL set by a model that differs from the label
  the treebank (almost) always uses for arcs like it is flagged for review,
  with the treebank's label as suggestion.
* optionally, **the predictions themselves**: for a form the trusted data never
  has, analyses given with high confidence elsewhere in the prediction file
  (from agents or people, not from rules) count as evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, Field

from .corpus import CellUpdate, CorpusStore
from .corpus.confidence import read_confidence
from .corpus.conllu import EMPTY, Sentence, Token
from .corpus.index import direction_of

RULE_AGENT = "rules"
PROGRAMMATIC_AGENTS = frozenset({"rules", "baseline"})
"""Agents whose edits come from code; the scorer does not re-check them."""

TRUSTED_CONFIDENCE = 0.97
PREDICTION_CONFIDENCE = 0.95
PREDICTION_EVIDENCE = 0.96
"""Minimum confidence for prediction cells to count as evidence (above what rules set)."""

MORPHOLOGY = ("lemma", "upos", "xpos", "feats")

REDERIVE_BELOW = 0.5
"""An XPOS below this confidence (e.g. marked for review after its FEATS changed) may be re-derived."""

INPUT_CONFIDENCE = 0.9
FLAG_INPUT_CONFIDENCE = 0.5
"""Cross-checks only raise flags, so a merely plausible relation is enough to check against."""
"""Rules that build on other cells (XPOS from UPOS+FEATS, DEPREL from the arc) only
fire when those cells are at least this confident, so guesses do not propagate."""


VOICE_SHARE = 0.9
"""Share of dependents of passive (or active) heads that must have one label before a mismatch is flagged."""
VOICE_RARE = 0.1
"""The word's own label must be at most this common among dependents of such heads (obl:tmod next to a passive verb is fine)."""
VOICE_RELATIONS = frozenset({"nsubj", "csubj", "aux"})
"""Relations whose subtype follows the head's voice (nsubj:pass, csubj:pass, aux:pass)."""


class RuleSettings(BaseModel):
    min_count: int = Field(5, ge=1, description="Examples needed before a rule fires.")
    min_share: float = Field(
        0.98, gt=0.5, le=1,
        description="Share of the examples that must agree (0.98 had a third of the errors of 0.95 on four treebanks).",
    )  # fmt: skip
    use_predictions: bool = Field(True, description="Use confident predictions as evidence for forms unseen in the trusted data.")


class RulesReport(BaseModel):
    cells: dict[str, int] = Field(default_factory=dict, description="Cells settled per column.")
    splits: int = 0
    flags: int = 0

    @property
    def total(self) -> int:
        return sum(self.cells.values())


def dominant(counts: Counter[str], settings: RuleSettings) -> str | None:
    """The value with at least ``min_count`` examples and ``min_share`` of the total, if any."""
    if not counts:
        return None
    value, count = counts.most_common(1)[0]
    if count >= settings.min_count and count / sum(counts.values()) >= settings.min_share:
        return value
    return None


class Rules:
    """Applies the rules to one writable source. Trusted lookups are cached per instance."""

    def __init__(self, store: CorpusStore, target: str, settings: RuleSettings | None = None):
        self.store = store
        self.target = target
        self.settings = settings or RuleSettings()
        self.source = store.source(target)
        self.trusted = [info.name for info in store.list_sources() if info.mode == "ro"]
        self._cache: dict[tuple, dict[str, Counter[str]]] = {}

    # ------------------------------------------------------------ evidence

    def _analyses(self, form: str, previous: str | None, sources: list[str], min_confidence: float = 0.0, exclude: str | None = None) -> dict[str, Counter[str]]:
        result = self.store.analyse_form(
            form, previous_form=previous, sources=sources, min_confidence=min_confidence, exclude_sentence=exclude, limit=1
        )
        counts: dict[str, Counter[str]] = {column: Counter() for column in MORPHOLOGY}
        for group in result.groups:
            for column in MORPHOLOGY:
                counts[column][group.values[column]] += group.count
        return counts

    def trusted_analyses(self, form: str, previous: str | None = None) -> dict[str, Counter[str]]:
        key = (form, previous)
        if key not in self._cache:
            self._cache[key] = self._analyses(form, previous, self.trusted)
        return self._cache[key]

    def predicted_analyses(self, form: str, sent_id: str) -> dict[str, Counter[str]]:
        """Confident analyses of ``form`` elsewhere in the predictions, set by agents or people."""
        counts: dict[str, Counter[str]] = {column: Counter() for column in MORPHOLOGY}
        for sentence in self.source.sentences:
            if sentence.sent_id == sent_id:
                continue
            for word in sentence.words:
                if word.form != form:
                    continue
                scores = read_confidence(word)
                for column in MORPHOLOGY:
                    if scores.get(column, 0.0) >= PREDICTION_EVIDENCE:
                        counts[column][word.get(column)] += 1
        return counts

    # ------------------------------------------------------------ rules

    def _confident(self, word: Token, *columns: str) -> bool:
        return all(word.get(c) != EMPTY and self.source.confidence(word, c) >= INPUT_CONFIDENCE for c in columns)

    def _settle(self, word: Token, column: str, value: str | None, confidence: float, reason: str) -> CellUpdate | None:
        """An update if ``value`` may fill or confirm the cell."""
        if value is None or self.source.confidence(word, column) >= confidence:
            return None
        if word.get(column) not in (EMPTY, value):
            return None
        return CellUpdate(token_id=word.id, column=column, value=value, confidence=confidence, reason=reason)

    def morphology(self, sentence: Sentence) -> list[CellUpdate]:
        updates = []
        words = sentence.words
        for index, word in enumerate(words):
            previous = words[index - 1].form if index else None
            overall = self.trusted_analyses(word.form)
            context = self.trusted_analyses(word.form, previous) if previous else None
            predicted = None
            if self.settings.use_predictions and not sum(overall["upos"].values()):
                predicted = self.predicted_analyses(word.form, sentence.sent_id)
            for column in MORPHOLOGY:
                value = dominant(overall[column], self.settings)
                reason = f"{column.upper()} of '{word.form}' in the trusted data"
                if value is None and context is not None:
                    value = dominant(context[column], self.settings)
                    reason = f"{column.upper()} of '{word.form}' after '{previous}' in the trusted data"
                confidence = TRUSTED_CONFIDENCE
                if value is None and predicted is not None:
                    value = dominant(predicted[column], self.settings)
                    reason, confidence = f"{column.upper()} of '{word.form}' in confident predictions", PREDICTION_CONFIDENCE
                update = self._settle(word, column, value, confidence, reason)
                if update:
                    updates.append(update)
        return updates

    def xpos_from_tags(self, sentence: Sentence) -> list[CellUpdate]:
        updates = []
        for word in sentence.words:
            if not self._confident(word, "upos", "feats"):
                continue
            counts = Counter({v.value: v.count for v in self.store.xpos_for_tags(word.upos, word.feats)})
            value = dominant(counts, self.settings)
            reason = f"XPOS used with {word.upos} {word.feats}"
            if value and word.xpos != value and self.source.confidence(word, "xpos") < REDERIVE_BELOW:
                # XPOS follows from UPOS and FEATS: a doubtful XPOS left over from an earlier analysis is replaced
                update = CellUpdate(token_id=word.id, column="xpos", value=value, confidence=TRUSTED_CONFIDENCE, reason=reason)
            else:
                update = self._settle(word, "xpos", value, TRUSTED_CONFIDENCE, reason)
            if update:
                updates.append(update)
        return updates

    def deprel(self, sentence: Sentence) -> list[CellUpdate]:
        updates = []
        words = {word.id: word for word in sentence.words}
        for word in sentence.words:
            if word.head == "0" and self._confident(word, "head"):
                update = self._settle(word, "deprel", "root", 0.99, "HEAD 0 is the root")
            elif word.head in words and self._confident(word, "head", "upos") and self._confident(words[word.head], "upos"):
                head = words[word.head]
                stats = self.store.relation_stats(
                    word.lemma, word.upos, head.lemma, head.upos, direction_of(word.id, word.head),
                    min_total=self.settings.min_count,
                )  # fmt: skip
                counts = Counter({v.value: v.count for v in stats.deprels})
                update = self._settle(
                    word, "deprel", dominant(counts, self.settings), TRUSTED_CONFIDENCE, f"relation for arcs like this ({stats.basis})"
                )
            else:
                continue
            if update:
                updates.append(update)
        return updates

    def cross_check(self, sentence: Sentence) -> list[tuple[str, str, str]]:
        """``(token_id, current UPOS, expected UPOS)`` for words whose UPOS (almost) never has their relation.

        Flags only ask for a review, so a plausible relation (confidence at
        least 0.5) is enough to check against.
        """
        mismatches = []
        for word in sentence.words:
            if word.upos == EMPTY or word.lemma == EMPTY or word.deprel == EMPTY:
                continue
            if self.source.confidence(word, "deprel") < FLAG_INPUT_CONFIDENCE:
                continue
            counts = Counter({v.value: v.count for v in self.store.upos_for_function(word.lemma, word.deprel)})
            total = sum(counts.values())
            if total >= self.settings.min_count and counts[word.upos] / total <= 1 - self.settings.min_share:
                mismatches.append((word.id, word.upos, counts.most_common(1)[0][0]))
        return mismatches

    def relation_check(self, sentence: Sentence) -> list[tuple[str, str, str, str]]:
        """``(token_id, current DEPREL, expected DEPREL, basis)`` where a model's label contradicts the treebank.

        Uses the same evidence and thresholds as :meth:`deprel`, which can only
        fill empty cells; once a model has labelled an arc, a disagreement is
        flagged instead of silently accepted.
        """
        words = {word.id: word for word in sentence.words}
        mismatches = []
        for word in sentence.words:
            head = words.get(word.head)
            if head is None or word.deprel == EMPTY:
                continue
            if self.source.confidence(word, "head") < FLAG_INPUT_CONFIDENCE or not (
                self._confident(word, "upos") and self._confident(head, "upos")
            ):
                continue
            stats = self.store.relation_stats(
                word.lemma, word.upos, head.lemma, head.upos, direction_of(word.id, word.head), min_total=self.settings.min_count
            )
            expected = dominant(Counter({v.value: v.count for v in stats.deprels}), self.settings)
            if expected is not None and expected != word.deprel:
                count = next(v.count for v in stats.deprels if v.value == expected)
                mismatches.append((word.id, word.deprel, expected, f"{count} of {stats.total}, by {stats.basis}"))
        return mismatches

    def voice_check(self, sentence: Sentence) -> list[tuple[str, str, str, str]]:
        """``(token_id, current DEPREL, expected DEPREL, basis)`` where the label ignores the head's voice.

        Treebanks that mark passive subjects (``nsubj:pass``) use the subtype
        almost exactly when the head is passive. Flags only ask for a review, so
        the share needed is lower than for rules that write cells.
        """
        words = {word.id: word for word in sentence.words}
        mismatches = []
        for word in sentence.words:
            head = words.get(word.head)
            if head is None or word.deprel.split(":")[0] not in VOICE_RELATIONS or head.feats == EMPTY:
                continue
            if self.source.confidence(word, "deprel") < FLAG_INPUT_CONFIDENCE or self.source.confidence(head, "feats") < FLAG_INPUT_CONFIDENCE:
                continue
            passive = head.feats_dict.get("Voice") == "Pass"
            by_voice = {
                flag: Counter({v.value: v.count for v in self.store.deprels_for_voice(word.deprel, flag)}) for flag in (True, False)
            }
            if by_voice[True].most_common(1)[:1] == by_voice[False].most_common(1)[:1]:
                continue  # the head's voice makes no difference to this relation
            counts = by_voice[passive]
            total = sum(counts.values())
            expected, count = counts.most_common(1)[0] if counts else (word.deprel, 0)
            rare = counts[word.deprel] / total <= VOICE_RARE if total else False
            if total >= self.settings.min_count and count / total >= VOICE_SHARE and rare and expected != word.deprel:
                voice = "passive" if passive else "not passive"
                mismatches.append((word.id, word.deprel, expected, f"{count} of {total} relations like this have a {voice} head"))
        return mismatches

    def segmentation(self, sentence: Sentence) -> tuple[list[CellUpdate], list[tuple[str, list[str], int]]]:
        """Confirmations of unsplit words, and splits ``(token_id, pieces, head_index)`` to make."""
        trusted_has_multiwords = any(
            self.store.source(name).index.multiword_exact for name in self.trusted
        )
        covered = {
            str(i)
            for token in sentence.tokens
            if token.is_multiword
            for i in range(token.span()[0], token.span()[1] + 1)
        }
        confirm, splits = [], []
        for word in sentence.words:
            if word.id in covered:
                continue
            if not trusted_has_multiwords:
                update = self._settle(word, "form", word.form, TRUSTED_CONFIDENCE, "the trusted data has no multiword tokens")
            else:
                found = self.store.find_multiword(word.form, sources=self.trusted, limit=1)
                total = found.total + found.as_single_word
                split_counts = Counter({"split": found.total, "whole": found.as_single_word})
                decision = dominant(split_counts, self.settings)
                update = None
                if decision == "whole":
                    update = self._settle(word, "form", word.form, TRUSTED_CONFIDENCE, f"never split in {total} trusted examples")
                elif decision == "split" and found.analyses[0].count / total >= self.settings.min_share:
                    analysis = found.analyses[0]
                    head = next((i for i, tag in enumerate(analysis.upos) if tag not in ("ADP", "CCONJ", "SCONJ", "DET", "PART")), 0)
                    splits.append((word.id, analysis.pieces, head))
            if update:
                confirm.append(update)
        return confirm, splits

    # ------------------------------------------------------------ driver

    def apply(self, sent_ids: Iterable[str] | None = None, columns: Iterable[str] | None = None) -> RulesReport:
        """Apply the rules for ``columns`` (default all) to the given sentences (default all)."""
        wanted = set(columns or ("form", "lemma", "upos", "xpos", "feats", "deprel"))
        report = RulesReport()
        for sent_id in list(sent_ids) if sent_ids is not None else [s.sent_id for s in self.source.sentences]:
            if "form" in wanted:
                confirm, splits = self.segmentation(self.source.sentence(sent_id))
                for token_id, pieces, head in reversed(splits):  # right to left keeps earlier ids valid
                    self.store.split_word(
                        self.target, sent_id, token_id, pieces, multiword=True, head_index=head,
                        confidence=TRUSTED_CONFIDENCE, reason="split as in the trusted data", agent=RULE_AGENT, flag_head=False,
                    )  # fmt: skip
                    report.splits += 1
                if splits:
                    confirm, _ = self.segmentation(self.source.sentence(sent_id))
                self._write(sent_id, confirm, report)
            sentence = self.source.sentence(sent_id)
            updates = self.morphology(sentence) if wanted & set(MORPHOLOGY) else []
            updates = [u for u in updates if u.column in wanted]
            self._write(sent_id, updates, report)
            if "xpos" in wanted:
                self._write(sent_id, self.xpos_from_tags(self.source.sentence(sent_id)), report)
            if "deprel" in wanted:
                self._write(sent_id, self.deprel(self.source.sentence(sent_id)), report)
            self._flag(sent_id, report, wanted)
        return report

    def _raise(self, sent_id: str, column: str, token_id: str, reason: str, suggestion: str, report: RulesReport) -> None:
        """Raise a flag unless the rules raised one for this cell before (it may have been dismissed)."""
        previous = self.store.list_flags(self.target, column=column, sent_id=sent_id, status="all", limit=10_000).flags
        if any(flag.token_id == token_id and flag.raised_by == RULE_AGENT for flag in previous):
            return
        self.store.flag_issue(self.target, sent_id, column, reason, token_id=token_id, suggestion=suggestion, agent=RULE_AGENT)
        report.flags += 1

    def _flag(self, sent_id: str, report: RulesReport, columns: set[str]) -> None:
        sentence = self.source.sentence(sent_id)
        checks = self.cross_check(sentence) if "deprel" in columns else []
        for token_id, current, expected in checks:
            word = sentence.token(token_id)
            reason = f"In the treebank, '{word.lemma}' as {word.deprel} is {expected}, not {current}."
            self._raise(sent_id, "upos", token_id, reason, expected, report)
        for token_id, current, expected, basis in self.voice_check(sentence) if columns & {"deprel", "feats"} else []:
            reason = f"The treebank labels this {expected} ({basis}), not {current}; check the label or the head's voice."
            self._raise(sent_id, "deprel", token_id, reason, expected, report)
        for token_id, current, expected, basis in self.relation_check(sentence) if "deprel" in columns else []:
            reason = f"The treebank labels arcs like this {expected} ({basis}), not {current}; check the label or the head."
            self._raise(sent_id, "deprel", token_id, reason, expected, report)

    def _write(self, sent_id: str, updates: list[CellUpdate], report: RulesReport) -> None:
        if updates:
            self.store.update_cells(self.target, sent_id, updates, agent=RULE_AGENT)
            for update in updates:
                report.cells[update.column] = report.cells.get(update.column, 0) + 1
