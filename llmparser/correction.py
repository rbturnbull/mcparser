from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .conllu_model import CONLLU_COLUMNS
from .conllu_model import ConlluSentence, ConlluToken
from .evidence import TokenEvidence, TrainingEvidence
from .io import write_yaml
from .segmentation import SegmentationCorrectionResult, apply_segmentation_patch
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


class SentenceCorrectionResult(BaseModel):
    status: str = "checked"
    patches: list[CorrectionPatch] = Field(default_factory=list)
    evidence_sent_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


CorrectionPatchInput = CorrectionPatch | dict[str, Any]
CorrectionChecker = Callable[[ConlluSentence, str, list[dict[str, Any]]], list[CorrectionPatchInput]]
TokenCorrectionOutput = TokenCorrectionResult | list[CorrectionPatchInput]
TokenCorrectionChecker = Callable[
    [ConlluSentence, ConlluToken, str, TokenEvidence],
    TokenCorrectionOutput,
]
SentenceCorrectionOutput = SentenceCorrectionResult | list[CorrectionPatchInput]
SentenceCorrectionChecker = Callable[
    [ConlluSentence, list[dict[str, Any]]],
    SentenceCorrectionOutput,
]
SegmentationCorrectionOutput = SegmentationCorrectionResult | dict[str, Any]
SegmentationChecker = Callable[
    [ConlluSentence, list[dict[str, Any]]],
    SegmentationCorrectionOutput,
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


def _normalise_sentence_result(result: SentenceCorrectionOutput) -> SentenceCorrectionResult:
    if isinstance(result, SentenceCorrectionResult):
        return result

    patches = [_normalise_patch(patch) for patch in result]
    return SentenceCorrectionResult(
        status="patched" if patches else "checked",
        patches=patches,
    )


def _normalise_segmentation_result(
    result: SegmentationCorrectionOutput,
) -> SegmentationCorrectionResult:
    if isinstance(result, SegmentationCorrectionResult):
        return result
    return SegmentationCorrectionResult.model_validate(result)


def correct_segmentation(
    sentence: ConlluSentence,
    examples: list[dict[str, Any]],
    checker: SegmentationChecker,
) -> SegmentationCorrectionResult:
    result = _normalise_segmentation_result(checker(sentence, examples))
    if result.status == "patched":
        applied_patches = []
        for patch in result.patches:
            applied = apply_segmentation_patch(sentence, patch)
            applied_patches.extend(applied.patches)
        result.patches = applied_patches
        result.needs_syntax_correction = True
    return result


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
        token_evidence = evidence.find_token_evidence(
            sentence=sentence,
            token=token,
            column=column,
            k=n_examples,
        )
        token_result = _normalise_token_result(
            token=token,
            column=column,
            result=checker(sentence, token, column, token_evidence),
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


def correct_sentence(
    sentence: ConlluSentence,
    examples: list[dict[str, Any]],
    checker: SentenceCorrectionChecker,
) -> SentenceCorrectionResult:
    result = _normalise_sentence_result(checker(sentence, examples))
    for patch in result.patches:
        sentence.apply_patch(patch.to_dict())
    return result


def correct_predictions(
    prediction_path: Path | str,
    database,
    checker: CorrectionChecker | None = None,
    segmentation_checker: SegmentationChecker | None = None,
    token_checkers: dict[str, TokenCorrectionChecker] | None = None,
    sentence_checkers: dict[str, SentenceCorrectionChecker] | None = None,
    n_examples: int = 10,
    output: Path | str | None = None,
    columns: tuple[str, ...] = DEFAULT_CORRECTION_COLUMNS,
) -> dict[str, dict[str, Any]]:
    sentences = load_prediction_sentences(prediction_path)
    corrected: dict[str, dict[str, Any]] = {}
    evidence = TrainingEvidence(database)
    token_checkers = token_checkers or {}
    sentence_checkers = sentence_checkers or {}

    for index, sentence in enumerate(sentences):
        examples = evidence.find_similar_sentences(sentence.text, k=n_examples)
        audit: dict[str, Any] = {
            "validation": [error.to_dict() for error in validate_sentence(sentence)],
        }
        if segmentation_checker is not None:
            audit["segmentation"] = correct_segmentation(
                sentence=sentence,
                examples=examples,
                checker=segmentation_checker,
            ).to_dict()

        for column in columns:
            if column in sentence_checkers:
                audit[column] = correct_sentence(
                    sentence=sentence,
                    examples=examples,
                    checker=sentence_checkers[column],
                ).to_dict()
                continue

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
        item["audit_summary"] = build_audit_summary(audit)
        corrected[_sentence_key(sentence, index)] = item

    if output is not None:
        write_yaml(corrected, output)

    return corrected


def build_audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    patched_columns: list[str] = []
    uncertain_columns: list[str] = []
    needs_syntax_correction = False

    for key, value in audit.items():
        if not isinstance(value, dict):
            continue
        status = value.get("status")
        if status == "patched":
            patched_columns.append(key)
        elif status == "uncertain":
            uncertain_columns.append(key)
        if value.get("needs_syntax_correction"):
            needs_syntax_correction = True

    return {
        "patched_columns": patched_columns,
        "uncertain_columns": uncertain_columns,
        "needs_syntax_correction": needs_syntax_correction,
        "initial_validation_errors": len(audit.get("validation", [])),
        "post_validation_errors": len(audit.get("post_validation", [])),
    }
