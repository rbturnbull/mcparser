"""The agent roles: one per annotated CoNLL-U column, plus the scorer."""

from __future__ import annotations

from dataclasses import dataclass

READ_TOOLS = frozenset({
    "list_sources", "list_sentences", "get_sentence", "get_context", "get_token", "search_sentences",
    "find_tokens", "analyse_form", "find_multiword", "find_arcs", "allowed_values", "feature_inventory",
    "validate", "low_confidence", "list_flags", "get_history",
})  # fmt: skip
"""MCP tools every agent may use."""

REVIEW_TOOLS = frozenset({"flag_issue", "resolve_flag"})
CELL_TOOLS = frozenset({"set_cell", "update_cells"})
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
    def tools(self) -> frozenset[str]:
        if self.is_scorer:
            return READ_TOOLS | REVIEW_TOOLS | {"update_cells"}
        tools = READ_TOOLS | REVIEW_TOOLS | CELL_TOOLS
        if self.column == "form":
            tools |= SEGMENTATION_TOOLS
        return tools


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
