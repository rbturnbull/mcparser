from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .conllu_model import ConlluSentence


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
    token_ids = {token.id for token in sentence.tokens if isinstance(token.id, int)}

    root_count = 0
    for token in sentence.tokens:
        if not isinstance(token.id, int):
            continue

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
