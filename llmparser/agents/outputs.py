"""Compact structured answers from the agents, checked and applied in code.

Instead of writing through tool calls and then sending a summary (two or more
requests), an agent returns one answer whose cell values are plain lines::

    12841 3 NOUN 0.95
    12841 4 VERB 0.9

Lines cost a fraction of the output tokens of JSON objects. The answer is
validated before anything is written: an unknown sentence or word, an invalid
value or (for HEAD) a broken tree is sent back to the model to fix, and only a
fully valid answer is applied, all at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from ..corpus import CellUpdate, CorpusError, CorpusStore
from ..corpus.store import normalise_column
from .roles import Role

TREE_ERRORS = frozenset({"cycle", "root-count", "self-loop", "invalid-head"})


class FlagRequest(BaseModel):
    sent_id: str
    token_id: str | None = Field(None, description="Omit for the whole sentence.")
    column: str = Field(description="The column that looks wrong (form, lemma, upos, xpos, feats, head, deprel).")
    reason: str
    suggestion: str | None = None


class FlagResolution(BaseModel):
    flag_id: int
    status: Literal["resolved", "dismissed"]
    note: str = ""


class Annotation(BaseModel):
    """Your answer: the values for your column."""

    values: str = Field(
        description=(
            'One line per cell you set or confirm: "SENT_ID TOKEN_ID VALUE CONFIDENCE", '
            'e.g. "12841 3 NOUN 0.95". Use _ for an empty value.'
        )
    )
    flags: list[FlagRequest] = Field(default_factory=list, description="Problems in other columns.")
    resolved: list[FlagResolution] = Field(default_factory=list, description="Flags addressed to you that you closed.")
    summary: str = Field("", description="At most 15 words.")


class Scores(BaseModel):
    """Your answer: confidences for the cells."""

    scores: str = Field(
        description=(
            'One line per cell: "SENT_ID TOKEN_ID COLUMN CONFIDENCE", e.g. "12841 3 upos 0.95"; '
            'COLUMN "*" scores every listed cell of the word.'
        )
    )
    flags: list[FlagRequest] = Field(default_factory=list, description="Cells you are fairly sure are wrong, with a fix.")
    summary: str = Field("", description="At most 15 words.")


@dataclass
class Batch:
    """The sentences of one task; passed to the agent run as its dependencies."""

    target: str
    sent_ids: list[str]


def _lines(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def parse_values(text: str, role: Role, batch: Batch) -> tuple[dict[str, list[CellUpdate]], list[str]]:
    """``SENT_ID TOKEN_ID VALUE CONFIDENCE`` lines as updates per sentence, plus error messages."""
    updates: dict[str, list[CellUpdate]] = {}
    errors = []
    for number, parts in enumerate(_lines(text), start=1):
        if len(parts) < 4:
            errors.append(f"line {number}: expected SENT_ID TOKEN_ID VALUE CONFIDENCE, got {' '.join(parts)!r}")
            continue
        sent_id, token_id, value, confidence = parts[0], parts[1], " ".join(parts[2:-1]), parts[-1]
        if sent_id not in batch.sent_ids:
            errors.append(f"line {number}: sentence {sent_id!r} is not in this task")
            continue
        try:
            score = float(confidence)
        except ValueError:
            errors.append(f"line {number}: confidence {confidence!r} is not a number")
            continue
        updates.setdefault(sent_id, []).append(
            CellUpdate(token_id=token_id, column=role.column, value=value, confidence=min(max(score, 0.0), 1.0))
        )
    return updates, errors


def parse_scores(text: str, batch: Batch) -> tuple[dict[str, dict[str, float]], list[str]]:
    """``SENT_ID TOKEN_ID COLUMN CONFIDENCE`` lines as ``score_cells`` dicts per sentence."""
    scores: dict[str, dict[str, float]] = {}
    errors = []
    for number, parts in enumerate(_lines(text), start=1):
        if len(parts) != 4:
            errors.append(f"line {number}: expected SENT_ID TOKEN_ID COLUMN CONFIDENCE, got {' '.join(parts)!r}")
            continue
        sent_id, token_id, column, confidence = parts
        if sent_id not in batch.sent_ids:
            errors.append(f"line {number}: sentence {sent_id!r} is not in this task")
            continue
        try:
            score = float(confidence)
        except ValueError:
            errors.append(f"line {number}: confidence {confidence!r} is not a number")
            continue
        column = "*" if column == "*" else normalise_column(column)
        scores.setdefault(sent_id, {})[f"{token_id}:{column}"] = min(max(score, 0.0), 1.0)
    return scores, errors


class Applier:
    """Checks an agent's answer against the store, then applies it."""

    def __init__(self, store: CorpusStore, role: Role):
        self.store = store
        self.role = role

    def _retry(self, errors: list[str]) -> None:
        if errors:
            raise ModelRetry("Fix these problems and answer again:\n" + "\n".join(f"- {e}" for e in errors[:20]))

    def _check_flags(self, flags: list[FlagRequest], batch: Batch, errors: list[str]) -> None:
        for flag in flags:
            if flag.sent_id not in batch.sent_ids:
                errors.append(f"flag on sentence {flag.sent_id!r}, which is not in this task")

    def _apply_flags(self, flags: list[FlagRequest], batch: Batch) -> None:
        for flag in flags:
            self.store.flag_issue(
                batch.target, flag.sent_id, flag.column, flag.reason, token_id=flag.token_id,
                suggestion=flag.suggestion, agent=self.role.name,
            )  # fmt: skip

    def annotation(self, answer: Annotation, batch: Batch) -> Annotation:
        updates, errors = parse_values(answer.values, self.role, batch)
        for sent_id, cells in updates.items():
            try:
                result = self.store.update_cells(batch.target, sent_id, cells, agent=self.role.name, dry_run=True)
            except CorpusError as error:
                errors.append(f"sentence {sent_id}: {error}")
                continue
            if self.role.column == "head":
                errors += [
                    f"sentence {sent_id}: {issue.message}" for issue in result.issues if issue.code in TREE_ERRORS
                ]
        self._check_flags(answer.flags, batch, errors)
        self._retry(errors)
        for sent_id, cells in updates.items():
            self.store.update_cells(batch.target, sent_id, cells, agent=self.role.name)
        self._apply_flags(answer.flags, batch)
        for resolution in answer.resolved:
            try:
                self.store.resolve_flag(batch.target, resolution.flag_id, resolution.status, resolution.note, agent=self.role.name)
            except CorpusError:
                pass  # an unknown or already closed flag is harmless
        return answer

    def scores(self, answer: Scores, batch: Batch) -> Scores:
        scores, errors = parse_scores(answer.scores, batch)
        for sent_id, cells in scores.items():
            try:
                self.store.score_cells(batch.target, sent_id, cells, agent=self.role.name, dry_run=True)
            except CorpusError as error:
                errors.append(f"sentence {sent_id}: {error}")
        self._check_flags(answer.flags, batch, errors)
        self._retry(errors)
        for sent_id, cells in scores.items():
            self.store.score_cells(batch.target, sent_id, cells, agent=self.role.name)
        self._apply_flags(answer.flags, batch)
        return answer

