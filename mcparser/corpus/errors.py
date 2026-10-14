"""Exceptions raised by the corpus layer.

Every exception derives from :class:`CorpusError` so that callers (e.g. the MCP
server) can turn any expected failure into a message for the model.
"""

from __future__ import annotations


class CorpusError(Exception):
    """Base class for all expected corpus failures."""


class ConlluFormatError(CorpusError):
    """The CoNLL-U text could not be parsed."""


class NotFoundError(CorpusError):
    """A source, sentence, token or flag does not exist."""


class ReadOnlyError(CorpusError):
    """An edit was attempted on a read-only source."""


class ConflictError(CorpusError):
    """The file on disk changed since it was loaded."""


class EditError(CorpusError):
    """An edit was rejected because its arguments are invalid."""
