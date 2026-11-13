"""Per-cell confidence scores stored in the MISC column.

Scores are kept in a single ``Conf`` attribute, one ``column:score`` pair per
annotated column, e.g. ``Conf=upos:0.95,lemma:0.80,head:0.40``. ``form`` stands
for segmentation. A column without a score falls back to the source's default
confidence (1.0 for read-only training data, 0.0 for predictions).
"""

from __future__ import annotations

from .conllu import ANNOTATION_COLUMNS, Token
from .errors import EditError

CONF_KEY = "Conf"


def check_confidence(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise EditError(f"Confidence must be between 0 and 1, got {value}.")
    return value


def check_confidence_column(column: str) -> str:
    if column not in ANNOTATION_COLUMNS:
        raise EditError(
            f"Column {column!r} has no confidence; expected one of {', '.join(ANNOTATION_COLUMNS)}."
        )
    return column


def read_confidence(token: Token) -> dict[str, float]:
    """The explicit scores on ``token``. Malformed entries are ignored."""
    raw = token.misc_value(CONF_KEY)
    scores: dict[str, float] = {}
    if not raw:
        return scores
    for item in raw.split(","):
        column, _, score = item.partition(":")
        try:
            value = float(score)
        except ValueError:
            continue
        if column in ANNOTATION_COLUMNS and 0.0 <= value <= 1.0:
            scores[column] = value
    return scores


def write_confidence(token: Token, scores: dict[str, float]) -> None:
    """Replace the scores on ``token``; an empty dict removes ``Conf``."""
    ordered = [(column, scores[column]) for column in ANNOTATION_COLUMNS if column in scores]
    value = ",".join(f"{column}:{score:.2f}" for column, score in ordered) or None
    token.set_misc_value(CONF_KEY, value)


def set_confidence(token: Token, column: str, value: float | None) -> None:
    """Set one column's score, or remove it with ``None``."""
    check_confidence_column(column)
    scores = read_confidence(token)
    if value is None:
        scores.pop(column, None)
    else:
        scores[column] = check_confidence(value)
    write_confidence(token, scores)


def strip_confidence(misc: str, original: str) -> str:
    """Return ``misc`` with its ``Conf`` replaced by the one in ``original``.

    Used when MISC is edited directly so that confidence scores can only be
    changed through the confidence API.
    """
    edited = Token(id="1", misc=misc)
    edited.set_misc_value(CONF_KEY, Token(id="1", misc=original).misc_value(CONF_KEY))
    return edited.misc
