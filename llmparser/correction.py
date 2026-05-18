from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .conllu_model import CONLLU_COLUMNS
from .conllu_model import ConlluSentence
from .io import write_yaml
from .validation import validate_sentence


DEFAULT_CORRECTION_COLUMNS = (
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
)


class CorrectionPatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_id: Any
    column: str
    value: Any
    reason: str | None = None
    evidence_sent_ids: list[str] = Field(default_factory=list)

    @field_validator("column")
    @classmethod
    def normalise_column(cls, column: str) -> str:
        column = column.lower()
        if column not in CONLLU_COLUMNS:
            raise ValueError(f"Unsupported CoNLL-U column: {column}")
        return column

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ColumnCorrectionAudit(BaseModel):
    status: str
    patches: list[CorrectionPatch] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


CorrectionPatchInput = CorrectionPatch | dict[str, Any]
CorrectionChecker = Callable[[ConlluSentence, str, list[dict[str, Any]]], list[CorrectionPatchInput]]


def load_prediction_sentences(path: Path | str) -> list[ConlluSentence]:
    return ConlluSentence.iter_from_text(Path(path).read_text(encoding="utf-8"))


def _sentence_key(sentence: ConlluSentence, index: int) -> str:
    return sentence.sent_id or f"sentence_{index + 1}"


def _normalise_patch(patch: CorrectionPatchInput) -> CorrectionPatch:
    if isinstance(patch, CorrectionPatch):
        return patch
    return CorrectionPatch.model_validate(patch)


def correct_predictions(
    prediction_path: Path | str,
    database,
    checker: CorrectionChecker | None = None,
    n_examples: int = 10,
    output: Path | str | None = None,
    columns: tuple[str, ...] = DEFAULT_CORRECTION_COLUMNS,
) -> dict[str, dict[str, Any]]:
    sentences = load_prediction_sentences(prediction_path)
    corrected: dict[str, dict[str, Any]] = {}

    for index, sentence in enumerate(sentences):
        examples = database(sentence.text, k=n_examples)
        audit: dict[str, Any] = {
            "validation": [error.to_dict() for error in validate_sentence(sentence)],
        }

        for column in columns:
            patches: list[CorrectionPatch] = []
            if checker is not None:
                patches = [_normalise_patch(patch) for patch in checker(sentence, column, examples)]
                for patch in patches:
                    sentence.apply_patch(patch.to_dict())

            audit[column] = ColumnCorrectionAudit(
                patches=patches,
                status="patched" if patches else "checked",
            ).to_dict()

        post_validation = [error.to_dict() for error in validate_sentence(sentence)]
        if post_validation:
            audit["post_validation"] = post_validation

        item = sentence.to_dict()
        item["audit"] = audit
        corrected[_sentence_key(sentence, index)] = item

    if output is not None:
        write_yaml(corrected, output)

    return corrected
