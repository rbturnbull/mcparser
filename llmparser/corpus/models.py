"""Pydantic models returned by the corpus store (and hence by the MCP tools)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .conllu import Token
from .validation import Issue

FlagStatus = Literal["open", "resolved", "dismissed"]


class Flag(BaseModel):
    """A request for the agent responsible for ``column`` to review a cell or sentence."""

    id: int
    sent_id: str
    token_id: str | None = Field(None, description="None for a sentence-level flag.")
    column: str = Field(description="The element to review; 'form' means segmentation.")
    reason: str
    suggestion: str | None = None
    raised_by: str
    status: FlagStatus = "open"
    resolved_by: str | None = None
    note: str | None = None
    created: str
    round: int | None = None


class Edit(BaseModel):
    """One entry in a source's edit history."""

    seq: int
    time: str
    sent_id: str
    token_id: str | None = None
    action: str
    column: str | None = None
    old: str | None = None
    new: str | None = None
    confidence: float | None = None
    reason: str = ""
    agent: str
    round: int | None = None


class SourceInfo(BaseModel):
    name: str
    path: str
    mode: Literal["ro", "rw"]
    default_confidence: float
    sentences: int
    words: int
    open_flags: int


class SentenceSummary(BaseModel):
    source: str
    sent_id: str
    position: int
    text: str
    words: int
    confidence: float = Field(description="Mean confidence over the scored cells of the sentence's words.")
    open_flags: int


class SentencePage(BaseModel):
    total: int
    sentences: list[SentenceSummary]


class SentenceView(BaseModel):
    source: str
    sent_id: str
    position: int
    text: str
    conllu: str
    tokens: list[Token] | None = None
    confidence: float
    open_flags: list[Flag]
    issues: list[Issue]


class TokenRef(BaseModel):
    id: str
    form: str
    lemma: str
    upos: str
    deprel: str


class TokenView(BaseModel):
    source: str
    sent_id: str
    token: Token
    confidence: dict[str, float] = Field(description="Effective confidence for each annotation column.")
    head: TokenRef | None = Field(None, description="The head word (None for the root or an unknown head).")
    dependents: list[TokenRef]
    multiword: TokenRef | None = Field(None, description="The multiword token containing this word, if any.")


class TokenHit(BaseModel):
    source: str
    sent_id: str
    token_id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    misc: str
    head_form: str | None = Field(None, description="Form of the head word; None for the root.")
    confidence: float
    context: str = Field(description="Neighbouring words with the hit marked as [[form]].")


class GroupCount(BaseModel):
    values: dict[str, str]
    count: int


class TokenSearch(BaseModel):
    total: int
    groups: list[GroupCount] = Field(description="All matches grouped by the requested columns, most frequent first.")
    hits: list[TokenHit]


class SentenceHit(BaseModel):
    source: str
    sent_id: str
    text: str
    score: float
    confidence: float


class MultiwordAnalysis(BaseModel):
    pieces: list[str]
    upos: list[str]
    count: int
    examples: list[str] = Field(description="'source:sent_id:range_id' references.")


class MultiwordSearch(BaseModel):
    surface: str
    total: int
    analyses: list[MultiwordAnalysis]
    as_single_word: int = Field(description="How often the surface form occurs as an unsplit word.")


class ArcHit(BaseModel):
    source: str
    sent_id: str
    dependent: TokenRef
    head: TokenRef | None = Field(description="None when the dependent is the root.")
    deprel: str
    direction: Literal["left", "right", "root"] = Field(description="Side of the dependent on which its head lies.")
    distance: int
    confidence: float
    context: str


class ValueCount(BaseModel):
    value: str
    count: int


class ArcSearch(BaseModel):
    total: int
    deprels: list[ValueCount]
    directions: list[ValueCount]
    hits: list[ArcHit]


class CellRef(BaseModel):
    sent_id: str
    token_id: str
    form: str
    column: str
    value: str
    confidence: float


class CellPage(BaseModel):
    total: int
    cells: list[CellRef]


class FlagPage(BaseModel):
    total: int
    flags: list[Flag]


class SentenceIssue(BaseModel):
    sent_id: str
    issue: Issue


class ValidationReport(BaseModel):
    total: int
    errors: int
    warnings: int
    issues: list[SentenceIssue]


class CellUpdate(BaseModel):
    """A change to one cell: a new value, a new confidence, or both."""

    token_id: str
    column: str
    value: str | None = Field(None, description="New value; omit to change only the confidence.")
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    reason: str = ""


class EditResult(BaseModel):
    source: str
    sent_id: str
    edits: list[Edit]
    conllu: str = Field(description="The sentence after the edit.")
    issues: list[Issue]
    id_map: dict[str, str | None] | None = Field(None, description="For segmentation edits: old id -> new id.")
    new_ids: list[str] | None = None
