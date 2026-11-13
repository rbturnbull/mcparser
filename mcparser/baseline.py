"""A frequency baseline that needs no LLM.

For each surface token it applies the most frequent multiword split seen in the
trusted sources, and for each word it copies the most frequent (LEMMA, UPOS,
XPOS, FEATS) analysis of the same form. Only empty cells (``_`` with no
confidence recorded) are filled, so running it twice changes nothing. The
confidence reflects how much evidence there was, so the scorer and the role
agents know where to look first. HEAD and DEPREL are left to the agents.
"""

from __future__ import annotations

from pydantic import BaseModel

from .corpus import CellUpdate, CorpusStore
from .corpus.confidence import read_confidence

CONTENT_UPOS = ("NOUN", "PROPN", "VERB", "ADJ", "NUM", "PRON", "ADV", "X")
MORPHOLOGY = ("lemma", "upos", "xpos", "feats")


class BaselineReport(BaseModel):
    sentences: int = 0
    splits: int = 0
    analysed: int = 0
    unknown: int = 0


BASELINE_MAX_CONFIDENCE = 0.7
"""Below the default refinement threshold: a most-frequent analysis is only a
candidate, so an agent always reviews it. Near-certain cells are settled by
:mod:`mcparser.rules` instead (at 0.97)."""


def evidence_confidence(count: int, total: int) -> float:
    """Agreement among the examples, discounted when there are few (at most 0.7)."""
    return round(min(BASELINE_MAX_CONFIDENCE, count / total * min(1.0, total / 3)), 2)


def run_baseline(store: CorpusStore, target: str, agent: str = "baseline") -> BaselineReport:
    trusted = [info.name for info in store.list_sources() if info.mode == "ro"]
    source = store.source(target)
    report = BaselineReport()
    for sent_id in [sentence.sent_id for sentence in source.sentences]:
        report.sentences += 1
        position = 1
        while position <= len(source.sentence(sent_id).words):
            word = source.sentence(sent_id).token(str(position))
            found = store.find_multiword(word.form, sources=trusted, limit=1)
            if found.total > found.as_single_word:
                analysis = found.analyses[0]
                head = next((i for i, tag in enumerate(analysis.upos) if tag in CONTENT_UPOS), 0)
                store.split_word(
                    target, sent_id, word.id, analysis.pieces, multiword=True, head_index=head,
                    confidence=evidence_confidence(found.total, found.total + found.as_single_word),
                    reason=f"split as in {analysis.count} of {found.total + found.as_single_word} trusted examples",
                    agent=agent, flag_head=False,
                )  # fmt: skip
                report.splits += 1
                position += len(analysis.pieces)
            else:
                position += 1

        updates: list[CellUpdate] = []
        for word in source.sentence(sent_id).words:
            scored = read_confidence(word)
            empty = [column for column in MORPHOLOGY if word.get(column) == "_" and column not in scored]
            if not empty:
                continue
            result = store.analyse_form(word.form, sources=trusted, limit=1)
            if result.total == 0:
                result = store.analyse_form(word.form, loose=True, sources=trusted, limit=1)
            if result.total == 0:
                report.unknown += 1
                continue
            best = result.groups[0]
            confidence = evidence_confidence(best.count, result.total)
            reason = f"most frequent analysis ({best.count} of {result.total} trusted examples)"
            updates += [
                CellUpdate(token_id=word.id, column=column, value=best.values[column], confidence=confidence, reason=reason)
                for column in empty
            ]
            report.analysed += 1
        if updates:
            store.update_cells(target, sent_id, updates, agent=agent)
    return report
