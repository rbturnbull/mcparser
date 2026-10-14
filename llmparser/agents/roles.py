"""The agent roles: one per annotated CoNLL-U column, plus the scorer."""

from __future__ import annotations

from dataclasses import dataclass

READ_TOOLS = frozenset({
    "get_sentence", "get_context", "search_sentences", "find_tokens", "analyse_form", "find_multiword",
    "find_arcs", "relation_stats", "allowed_values", "feature_inventory", "validate", "list_flags",
})  # fmt: skip
"""MCP tools every agent may use.

Deliberately smaller than the server's full set: the task prompt already gives
the sentence, the cells to work on, open flags and the main evidence, so tools
such as list_sources, get_token or low_confidence would only add tokens to
every request.
"""

REVIEW_TOOLS = frozenset({"flag_issue", "resolve_flag"})
CELL_TOOLS = frozenset({"set_cell", "update_cells", "score_cells"})
SEGMENTATION_TOOLS = frozenset(
    {"split_word", "merge_words", "delete_word", "insert_word", "add_multiword", "remove_multiword"}
)
WRITE_TOOLS = CELL_TOOLS | SEGMENTATION_TOOLS | REVIEW_TOOLS


@dataclass(frozen=True)
class Role:
    name: str
    title: str
    columns: tuple[str, ...]
    """Columns this role may change (and whose flags it handles)."""

    @property
    def is_scorer(self) -> bool:
        return self.name == "scorer"

    @property
    def column(self) -> str:
        """The column whose low-confidence cells and flags this role handles."""
        return self.columns[0]

    @property
    def edits_with_tools(self) -> bool:
        """Segmentation edits through tools (it renumbers words); the other roles return compact answers."""
        return self.name == "segmentation"

    @property
    def tools(self) -> frozenset[str]:
        if self.edits_with_tools:
            return READ_TOOLS | REVIEW_TOOLS | SEGMENTATION_TOOLS | {"update_cells"}
        return READ_TOOLS


ROLES: dict[str, Role] = {
    role.name: role
    for role in (
        Role("segmentation", "tokenisation and multiword tokens (FORM)", ("form",)),
        Role("upos", "universal part-of-speech tags (UPOS)", ("upos",)),
        Role("xpos", "language-specific part-of-speech tags (XPOS)", ("xpos",)),
        Role("lemma", "lemmas (LEMMA)", ("lemma",)),
        Role("feats", "morphological features (FEATS)", ("feats",)),
        Role("head", "dependency heads (HEAD)", ("head",)),
        Role("deprel", "dependency relation labels (DEPREL)", ("deprel",)),
        Role("deps", "enhanced dependencies (DEPS)", ("deps",)),
        Role("scorer", "confidence scores for every cell", ("form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps")),
    )
}

REFINE_ORDER = ("segmentation", "upos", "xpos", "lemma", "feats", "head", "deprel", "deps")
"""Order in which roles work on a sentence: later columns build on earlier ones."""

DEFAULT_DISABLED = frozenset({"deps"})
"""Roles that only run when enabled in the settings (enhanced dependencies are optional in UD)."""

COLUMN_ROLES = {role.column: role.name for role in ROLES.values() if not role.is_scorer}
"""The role responsible for each column."""
