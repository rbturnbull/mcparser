"""MCP server exposing a :class:`~llmparser.corpus.CorpusStore`.

The same server is used by the llmparser agents (in-process or over stdio), by
Claude Desktop, and by other projects such as llmaligner.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from ..corpus import CorpusError, CorpusStore
from ..corpus.models import (
    ArcSearch,
    CellPage,
    CellUpdate,
    Edit,
    EditResult,
    Flag,
    FlagPage,
    MultiwordSearch,
    SentenceHit,
    SentencePage,
    SentenceSummary,
    SentenceView,
    SourceInfo,
    TokenSearch,
    TokenView,
    ValidationReport,
    ValueCount,
)

INSTRUCTIONS = """\
Search and edit CoNLL-U dependency treebanks.

Sources: each source is one CoNLL-U file. Read-only sources ("ro") hold trusted
examples such as treebank training data; writable sources ("rw") hold
predictions that you may edit. Call list_sources first.

Confidence: every word cell in FORM (segmentation), LEMMA, UPOS, XPOS, FEATS,
HEAD, DEPREL and DEPS has a confidence between 0 and 1, stored in MISC as
Conf=column:score,... A cell without a score has its source's default
confidence (usually 1.0 for read-only sources and 0.0 for predictions). Search
results are ordered by confidence and accept min_confidence to ignore weak
predictions. When you set a value, also give your confidence in it.

Evidence: use analyse_form for how a word is usually analysed, find_tokens for
words by lemma, tag, features or root (MISC Root=), find_multiword for how a
surface token is split, find_arcs for how words attach, allowed_values and
feature_inventory for the tagset, and search_sentences for similar sentences.
Pass exclude_sentence="source:sent_id" to avoid citing the sentence you are
editing.

Review: if you notice a likely mistake in a column you are not responsible for,
call flag_issue with that column so the responsible agent reviews it; use
list_flags to find work flagged for you and resolve_flag when done.

Token ids are strings: "3" for a word, "3-4" for a multiword token, "3.1" for
an empty node. Segmentation edits renumber words and keep heads consistent;
the result includes an id_map from old to new ids.
"""

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
EDIT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)

F = TypeVar("F", bound=Callable[..., Any])

Source = Annotated[str, Field(description="Name of a source (see list_sources).")]
SentId = Annotated[str, Field(description="The sentence's sent_id.")]
TokenId = Annotated[str, Field(description='Token id, e.g. "3", "3-4" or "3.1".')]
Sources = Annotated[list[str] | None, Field(description="Sources to search; default all.")]
Loose = Annotated[bool, Field(description="Ignore case and diacritics (accents, vowel points) when matching.")]
MinConfidence = Annotated[float, Field(ge=0, le=1, description="Ignore results below this confidence.")]
Exclude = Annotated[str | None, Field(description='Skip this sentence, as "source:sent_id".')]
Limit = Annotated[int, Field(ge=1, le=500, description="Maximum number of results.")]
Confidence = Annotated[float | None, Field(ge=0, le=1, description="Your confidence in the new value (0-1).")]
Reason = Annotated[str, Field(description="Short justification, recorded in the history.")]
Agent = Annotated[str, Field(description="Name of the calling agent, recorded in the history.")]


def _guard(function: F) -> F:
    """Report corpus errors to the client as tool errors."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except CorpusError as error:
            raise ToolError(str(error)) from error

    return wrapper  # type: ignore[return-value]


def build_server(
    store: CorpusStore,
    name: str = "conllu",
    allow_edits: bool = True,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
) -> MCPServer:
    """Create an MCP server over ``store``.

    With ``allow_edits=False`` only the read-only tools are registered, which is
    useful for exploring a corpus from Claude Desktop without risk. The MCP SDK
    configures the root logger at ``log_level``; the WARNING default keeps
    per-request logs out of the terminal.
    """
    server = MCPServer(name, instructions=INSTRUCTIONS, log_level=log_level)

    def tool(annotations: ToolAnnotations) -> Callable[[F], F]:
        def register(function: F) -> F:
            server.tool(annotations=annotations)(_guard(function))
            return function

        return register

    # ------------------------------------------------------------ browsing

    @tool(READ)
    def list_sources() -> list[SourceInfo]:
        """List the loaded sources with their mode (ro/rw), size, default confidence and open flags."""
        return store.list_sources()

    @tool(READ)
    def list_sentences(
        source: Source,
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Limit = 20,
        order: Literal["file", "confidence_asc", "confidence_desc"] = "file",
        max_confidence: Annotated[float | None, Field(ge=0, le=1, description="Only sentences at or below this confidence.")] = None,
    ) -> SentencePage:
        """Page through a source's sentences with their text, mean confidence and open flag count."""
        return store.list_sentences(source, offset=offset, limit=limit, order=order, max_confidence=max_confidence)

    @tool(READ)
    def get_sentence(
        source: Source,
        sent_id: SentId,
        include_tokens: Annotated[bool, Field(description="Also return the tokens as JSON objects.")] = False,
    ) -> SentenceView:
        """Get a sentence as CoNLL-U, with its confidence, open review flags and validation issues."""
        return store.get_sentence(source, sent_id, include_tokens=include_tokens)

    @tool(READ)
    def get_context(
        source: Source,
        sent_id: SentId,
        before: Annotated[int, Field(ge=0, le=50)] = 2,
        after: Annotated[int, Field(ge=0, le=50)] = 2,
    ) -> list[SentenceSummary]:
        """Get the neighbouring sentences in the file, for discourse context."""
        return store.get_context(source, sent_id, before=before, after=after)

    @tool(READ)
    def get_token(source: Source, sent_id: SentId, token_id: TokenId) -> TokenView:
        """Get one token with its per-column confidence, head word, dependents and multiword token."""
        return store.get_token(source, sent_id, token_id)

    # ------------------------------------------------------------ search

    @tool(READ)
    def search_sentences(
        query: Annotated[str, Field(description="Words to search for.")],
        sources: Sources = None,
        limit: Limit = 5,
        min_confidence: MinConfidence = 0.0,
        exclude_sentence: Exclude = None,
    ) -> list[SentenceHit]:
        """Find sentences similar to the query text (BM25, ignoring case and diacritics)."""
        return store.search_sentences(query, sources=sources, limit=limit, min_confidence=min_confidence, exclude_sentence=exclude_sentence)

    @tool(READ)
    def find_tokens(
        form: str | None = None,
        lemma: str | None = None,
        upos: str | None = None,
        xpos: str | None = None,
        feats: Annotated[str | None, Field(description='Required features, e.g. "Case=Gen|Number=Sing".')] = None,
        deprel: Annotated[str | None, Field(description='Relation; "nmod" also matches "nmod:gen".')] = None,
        root: Annotated[str | None, Field(description="Value of the MISC Root attribute (e.g. a Semitic root).")] = None,
        loose: Loose = False,
        sources: Sources = None,
        min_confidence: MinConfidence = 0.0,
        exclude_sentence: Exclude = None,
        group_by: Annotated[list[str] | None, Field(description='Columns to count matches by; default ["lemma", "upos"].')] = None,
        limit: Limit = 10,
    ) -> TokenSearch:
        """Find words matching all the given criteria. Returns counts over all matches plus example hits in context."""
        return store.find_tokens(
            form=form, lemma=lemma, upos=upos, xpos=xpos, feats=feats, deprel=deprel, root=root,
            loose=loose, sources=sources, min_confidence=min_confidence,
            exclude_sentence=exclude_sentence, group_by=group_by, limit=limit,
        )  # fmt: skip

    @tool(READ)
    def analyse_form(
        form: str,
        loose: Loose = False,
        sources: Sources = None,
        min_confidence: MinConfidence = 0.0,
        exclude_sentence: Exclude = None,
        limit: Limit = 5,
    ) -> TokenSearch:
        """How is this word form analysed elsewhere? Counts of (lemma, upos, xpos, feats) with examples."""
        return store.analyse_form(form, loose=loose, sources=sources, min_confidence=min_confidence, exclude_sentence=exclude_sentence, limit=limit)

    @tool(READ)
    def find_multiword(surface: str, loose: Loose = False, sources: Sources = None, limit: Limit = 10) -> MultiwordSearch:
        """How is this surface token split into syntactic words (e.g. clitics), and how often is it left unsplit?"""
        return store.find_multiword(surface, loose=loose, sources=sources, limit=limit)

    @tool(READ)
    def find_arcs(
        dependent_form: str | None = None,
        dependent_lemma: str | None = None,
        dependent_upos: str | None = None,
        head_form: str | None = None,
        head_lemma: str | None = None,
        head_upos: Annotated[str | None, Field(description='UPOS of the head; "ROOT" finds root words.')] = None,
        deprel: Annotated[str | None, Field(description='Relation; "nmod" also matches "nmod:gen".')] = None,
        loose: Loose = False,
        sources: Sources = None,
        min_confidence: MinConfidence = 0.0,
        exclude_sentence: Exclude = None,
        limit: Limit = 10,
    ) -> ArcSearch:
        """Find dependency arcs by dependent, head and relation, with relation and direction counts."""
        return store.find_arcs(
            dependent_form=dependent_form, dependent_lemma=dependent_lemma, dependent_upos=dependent_upos,
            head_form=head_form, head_lemma=head_lemma, head_upos=head_upos, deprel=deprel, loose=loose,
            sources=sources, min_confidence=min_confidence, exclude_sentence=exclude_sentence, limit=limit,
        )  # fmt: skip

    # ------------------------------------------------------------ tagsets

    @tool(READ)
    def allowed_values(
        column: Annotated[Literal["upos", "xpos", "deprel", "feats"], Field(description="Column to list values for.")],
        upos: Annotated[str | None, Field(description="Only count words with this UPOS.")] = None,
        sources: Annotated[list[str] | None, Field(description="Sources to count; default the read-only sources.")] = None,
        limit: Limit = 100,
    ) -> list[ValueCount]:
        """List the values a column takes in the trusted data, most frequent first."""
        return store.allowed_values(column, upos=upos, sources=sources, limit=limit)

    @tool(READ)
    def feature_inventory(
        upos: Annotated[str | None, Field(description="Only features of words with this UPOS.")] = None,
        sources: Annotated[list[str] | None, Field(description="Sources to count; default the read-only sources.")] = None,
    ) -> dict[str, list[ValueCount]]:
        """List each morphological feature and its values in the trusted data."""
        return store.feature_inventory(upos=upos, sources=sources)

    # ------------------------------------------------------------ quality

    @tool(READ)
    def validate(source: Source, sent_id: Annotated[str | None, Field(description="Omit to validate the whole source.")] = None, limit: Limit = 50) -> ValidationReport:
        """Check ids, multiword ranges, tags, features and tree structure (single root, no cycles)."""
        return store.validate(source, sent_id=sent_id, limit=limit)

    @tool(READ)
    def low_confidence(
        source: Source,
        threshold: Annotated[float, Field(ge=0, le=1)] = 0.5,
        columns: Annotated[list[str] | None, Field(description="Columns to check; default all scored columns.")] = None,
        sent_id: Annotated[str | None, Field(description="Restrict to one sentence.")] = None,
        limit: Limit = 50,
    ) -> CellPage:
        """List word cells below a confidence threshold, least confident first."""
        return store.low_confidence(source, threshold=threshold, columns=columns, sent_id=sent_id, limit=limit)

    @tool(READ)
    def list_flags(
        source: Source,
        status: Literal["open", "resolved", "dismissed", "all"] = "open",
        column: Annotated[str | None, Field(description="Only flags for this column (e.g. the one you are responsible for).")] = None,
        sent_id: str | None = None,
        limit: Limit = 50,
    ) -> FlagPage:
        """List review flags raised by agents."""
        return store.list_flags(source, status=status, column=column, sent_id=sent_id, limit=limit)

    @tool(READ)
    def get_history(source: Source, sent_id: str | None = None, token_id: str | None = None, limit: Limit = 20) -> list[Edit]:
        """List recent edits to a source (newest first), with agent, reason and confidence."""
        return store.get_history(source, sent_id=sent_id, token_id=token_id, limit=limit)

    _add_resources(server, store)
    if not allow_edits:
        return server

    # ------------------------------------------------------------ editing

    @tool(EDIT)
    def set_cell(
        source: Source,
        sent_id: SentId,
        token_id: TokenId,
        column: Annotated[str, Field(description="form, lemma, upos, xpos, feats, head, deprel, deps or misc.")],
        value: str,
        confidence: Confidence = None,
        reason: Reason = "",
        agent: Agent = "agent",
    ) -> EditResult:
        """Set one cell of a writable source. Returns the updated sentence and its validation issues."""
        return store.set_cell(source, sent_id, token_id, column, value, confidence=confidence, reason=reason, agent=agent)

    @tool(EDIT)
    def update_cells(source: Source, sent_id: SentId, updates: list[CellUpdate], agent: Agent = "agent") -> EditResult:
        """Apply several cell changes to one sentence at once (all or nothing).

        Each update has token_id, column, and a value and/or confidence. Give only
        a confidence to re-score a cell without changing it.
        """
        return store.update_cells(source, sent_id, updates, agent=agent)

    @tool(EDIT)
    def split_word(
        source: Source,
        sent_id: SentId,
        token_id: TokenId,
        forms: Annotated[list[str], Field(min_length=2, description="The forms of the new words, in order.")],
        multiword: Annotated[bool, Field(description="Keep the original as a multiword surface token (e.g. clitics).")] = False,
        head_index: Annotated[int, Field(ge=0, description="Which new word keeps the original head, relation and tags.")] = 0,
        confidence: Confidence = None,
        reason: Reason = "",
        agent: Agent = "agent",
    ) -> EditResult:
        """Split a word into several words. Other pieces attach to the head piece; heads are renumbered."""
        return store.split_word(source, sent_id, token_id, forms, multiword=multiword, head_index=head_index, confidence=confidence, reason=reason, agent=agent)

    @tool(EDIT)
    def merge_words(
        source: Source,
        sent_id: SentId,
        token_ids: Annotated[list[str], Field(min_length=2, description="Ids of adjacent words.")],
        form: Annotated[str | None, Field(description="Form of the merged word; default the surface form.")] = None,
        confidence: Confidence = None,
        reason: Reason = "",
        agent: Agent = "agent",
    ) -> EditResult:
        """Merge adjacent words into one word; heads are renumbered."""
        return store.merge_words(source, sent_id, token_ids, form=form, confidence=confidence, reason=reason, agent=agent)

    @tool(DESTRUCTIVE)
    def delete_word(source: Source, sent_id: SentId, token_id: TokenId, reason: Reason = "", agent: Agent = "agent") -> EditResult:
        """Delete a word; its dependents are attached to its head."""
        return store.delete_word(source, sent_id, token_id, reason=reason, agent=agent)

    @tool(EDIT)
    def insert_word(
        source: Source,
        sent_id: SentId,
        after: Annotated[str, Field(description='Insert after this word id; "0" inserts at the start.')],
        form: str,
        confidence: Confidence = None,
        reason: Reason = "",
        agent: Agent = "agent",
    ) -> EditResult:
        """Insert a new unannotated word."""
        return store.insert_word(source, sent_id, after, form, confidence=confidence, reason=reason, agent=agent)

    @tool(EDIT)
    def add_multiword(
        source: Source,
        sent_id: SentId,
        start: int,
        end: int,
        form: Annotated[str, Field(description="The surface form covering the words.")],
        reason: Reason = "",
        agent: Agent = "agent",
    ) -> EditResult:
        """Group words start..end under a multiword surface token."""
        return store.add_multiword(source, sent_id, start, end, form, reason=reason, agent=agent)

    @tool(EDIT)
    def remove_multiword(source: Source, sent_id: SentId, range_id: Annotated[str, Field(description='e.g. "3-4".')], reason: Reason = "", agent: Agent = "agent") -> EditResult:
        """Remove a multiword token line, keeping its words."""
        return store.remove_multiword(source, sent_id, range_id, reason=reason, agent=agent)

    @tool(EDIT)
    def flag_issue(
        source: Source,
        sent_id: SentId,
        column: Annotated[str, Field(description="Element to review: form (segmentation), lemma, upos, xpos, feats, head, deprel or deps.")],
        reason: Annotated[str, Field(description="What looks wrong and why.")],
        token_id: Annotated[str | None, Field(description="The word concerned; omit for the whole sentence.")] = None,
        suggestion: Annotated[str | None, Field(description="A suggested value, if you have one.")] = None,
        agent: Agent = "agent",
    ) -> Flag:
        """Flag a probable mistake for the agent responsible for that column."""
        return store.flag_issue(source, sent_id, column, reason, token_id=token_id, suggestion=suggestion, agent=agent)

    @tool(EDIT)
    def resolve_flag(
        source: Source,
        flag_id: int,
        status: Literal["resolved", "dismissed"] = "resolved",
        note: Annotated[str, Field(description="What you did, or why no change is needed.")] = "",
        agent: Agent = "agent",
    ) -> Flag:
        """Close a review flag."""
        return store.resolve_flag(source, flag_id, status=status, note=note, agent=agent)

    @tool(EDIT)
    def save(source: Annotated[str | None, Field(description="Omit to save all writable sources.")] = None) -> list[str]:
        """Write writable sources to disk (edits are normally saved automatically)."""
        return store.save(source)

    @tool(DESTRUCTIVE)
    def reload(source: Source) -> SourceInfo:
        """Reload a source from disk, discarding unsaved changes (use after a conflict)."""
        return store.reload(source)

    return server


def _add_resources(server: MCPServer, store: CorpusStore) -> None:
    @server.resource("conllu://sources", name="sources", mime_type="application/json")
    def sources_resource() -> str:
        """The loaded sources."""
        return json.dumps([info.model_dump() for info in store.list_sources()], ensure_ascii=False)

    @server.resource("conllu://{source}/{sent_id}", name="sentence", mime_type="text/plain")
    def sentence_resource(source: str, sent_id: str) -> str:
        """One sentence as CoNLL-U."""
        try:
            return store.get_sentence(source, sent_id).conllu
        except CorpusError as error:
            raise ResourceError(str(error)) from error
