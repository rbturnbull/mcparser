from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .conllu_model import CONLLU_COLUMNS
from .conllu_model import ConlluSentence, ConlluToken
from .evidence import TrainingEvidence
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
    token_results: list["TokenCorrectionResult"] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class TokenCorrectionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_id: Any
    column: str
    status: str = "checked"
    patches: list[CorrectionPatch] = Field(default_factory=list)
    evidence_sent_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("column")
    @classmethod
    def normalise_column(cls, column: str) -> str:
        column = column.lower()
        if column not in CONLLU_COLUMNS:
            raise ValueError(f"Unsupported CoNLL-U column: {column}")
        return column

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


CorrectionPatchInput = CorrectionPatch | dict[str, Any]
CorrectionChecker = Callable[[ConlluSentence, str, list[dict[str, Any]]], list[CorrectionPatchInput]]
TokenCorrectionOutput = TokenCorrectionResult | list[CorrectionPatchInput]
TokenCorrectionChecker = Callable[
    [ConlluSentence, ConlluToken, str, list[dict[str, Any]]],
    TokenCorrectionOutput,
]


def load_prediction_sentences(path: Path | str) -> list[ConlluSentence]:
    return ConlluSentence.iter_from_text(Path(path).read_text(encoding="utf-8"))


def _sentence_key(sentence: ConlluSentence, index: int) -> str:
    return sentence.sent_id or f"sentence_{index + 1}"


def _normalise_patch(patch: CorrectionPatchInput) -> CorrectionPatch:
    if isinstance(patch, CorrectionPatch):
        return patch
    return CorrectionPatch.model_validate(patch)


def _normalise_token_result(
    token: ConlluToken,
    column: str,
    result: TokenCorrectionOutput,
) -> TokenCorrectionResult:
    if isinstance(result, TokenCorrectionResult):
        return result

    patches = [_normalise_patch(patch) for patch in result]
    return TokenCorrectionResult(
        token_id=token.id,
        column=column,
        status="patched" if patches else "checked",
        patches=patches,
    )


def correct_token_column(
    sentence: ConlluSentence,
    column: str,
    evidence: TrainingEvidence,
    checker: TokenCorrectionChecker,
    n_examples: int = 10,
) -> ColumnCorrectionAudit:
    patches: list[CorrectionPatch] = []
    token_results: list[TokenCorrectionResult] = []

    for token in sentence.tokens:
        examples = evidence.find_examples_for_token_column(
            sentence=sentence,
            token=token,
            column=column,
            k=n_examples,
        )
        token_result = _normalise_token_result(
            token=token,
            column=column,
            result=checker(sentence, token, column, examples),
        )
        for patch in token_result.patches:
            sentence.apply_patch(patch.to_dict())
        patches.extend(token_result.patches)
        token_results.append(token_result)

    return ColumnCorrectionAudit(
        patches=patches,
        token_results=token_results,
        status="patched" if patches else "checked",
    )


def correct_predictions(
    prediction_path: Path | str,
    database,
    checker: CorrectionChecker | None = None,
    token_checkers: dict[str, TokenCorrectionChecker] | None = None,
    n_examples: int = 10,
    output: Path | str | None = None,
    columns: tuple[str, ...] = DEFAULT_CORRECTION_COLUMNS,
) -> dict[str, dict[str, Any]]:
    sentences = load_prediction_sentences(prediction_path)
    corrected: dict[str, dict[str, Any]] = {}
    evidence = TrainingEvidence(database)
    token_checkers = token_checkers or {}

    for index, sentence in enumerate(sentences):
        examples = evidence.find_similar_sentences(sentence.text, k=n_examples)
        audit: dict[str, Any] = {
            "validation": [error.to_dict() for error in validate_sentence(sentence)],
        }

        for column in columns:
            if column in token_checkers:
                audit[column] = correct_token_column(
                    sentence=sentence,
                    column=column,
                    evidence=evidence,
                    checker=token_checkers[column],
                    n_examples=n_examples,
                ).to_dict()
                continue

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
