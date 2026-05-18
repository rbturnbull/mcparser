from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .conllu_model import ConlluSentence


UPOS_VALUES = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "_",
}


class ValidationError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    token_id: Any | None = None
    column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def validate_sentence(sentence: ConlluSentence) -> list[ValidationError]:
    errors: list[ValidationError] = []
    token_ids: set[int] = set()
    duplicate_ids: set[int] = set()
    for token in sentence.tokens:
        if not isinstance(token.id, int):
            continue
        if token.id in token_ids:
            duplicate_ids.add(token.id)
        token_ids.add(token.id)

    for token_id in sorted(duplicate_ids):
        errors.append(
            ValidationError(
                code="duplicate-id",
                message=f"Token ID {token_id} appears more than once.",
                token_id=token_id,
                column="id",
            )
        )

    root_count = 0
    for token in sentence.tokens:
        if not isinstance(token.id, int):
            continue

        if token.upos not in UPOS_VALUES:
            errors.append(
                ValidationError(
                    code="invalid-upos",
                    message=f"Token {token.id} has invalid UPOS value {token.upos!r}.",
                    token_id=token.id,
                    column="upos",
                )
            )

        if token.head == 0:
            root_count += 1
        elif token.head not in token_ids:
            errors.append(
                ValidationError(
                    code="invalid-head",
                    message=f"Token {token.id} has HEAD {token.head}, which is not 0 or a token ID.",
                    token_id=token.id,
                    column="head",
                )
            )

        if token.deprel == "root" and token.head != 0:
            errors.append(
                ValidationError(
                    code="root-deprel-head",
                    message=f"Token {token.id} has DEPREL root but HEAD {token.head}; expected HEAD 0.",
                    token_id=token.id,
                    column="head",
                )
            )

        if token.head == 0 and token.deprel != "root":
            errors.append(
                ValidationError(
                    code="head-zero-deprel",
                    message=f"Token {token.id} has HEAD 0 but DEPREL {token.deprel!r}; expected root.",
                    token_id=token.id,
                    column="deprel",
                )
            )

        if token.deprel in (None, "_", ""):
            errors.append(
                ValidationError(
                    code="missing-deprel",
                    message=f"Token {token.id} has no DEPREL value.",
                    token_id=token.id,
                    column="deprel",
                )
            )

    if sentence.tokens and root_count != 1:
        errors.append(
            ValidationError(
                code="root-count",
                message=f"Sentence has {root_count} root tokens; expected exactly 1.",
                column="head",
            )
        )

    return errors
