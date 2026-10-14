"""CoNLL-U corpora with confidence scores, search and safe editing.

This subpackage has no dependency on the agents and can be used on its own,
either directly in Python or through the MCP server in :mod:`mcparser.mcp`.
"""

from .conllu import ANNOTATION_COLUMNS, COLUMNS, UPOS_TAGS, Sentence, Token, format_conllu, parse_conllu
from .errors import ConflictError, ConlluFormatError, CorpusError, EditError, NotFoundError, ReadOnlyError
from .models import CellUpdate, Edit, Flag
from .source import CorpusSource, SourceMode
from .store import CorpusStore
from .validation import Issue, validate_sentence

__all__ = [
    "ANNOTATION_COLUMNS",
    "COLUMNS",
    "UPOS_TAGS",
    "CellUpdate",
    "ConflictError",
    "ConlluFormatError",
    "CorpusError",
    "CorpusSource",
    "CorpusStore",
    "Edit",
    "EditError",
    "Flag",
    "Issue",
    "NotFoundError",
    "ReadOnlyError",
    "Sentence",
    "SourceMode",
    "Token",
    "format_conllu",
    "parse_conllu",
    "validate_sentence",
]
