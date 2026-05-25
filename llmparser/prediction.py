from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .conllu_model import ConlluSentence, ConlluToken
from .correction import (
    ColumnCorrectionAudit,
    CorrectionPatch,
    CorrectionPatchInput,
    TokenCorrectionResult,
    TokenCorrectionOutput,
)
from .evidence import TokenEvidence


ColumnPredictor = Callable[
    [ConlluSentence, ConlluToken, str, TokenEvidence],
    TokenCorrectionOutput,
]


def predict_sentence_columns(
    sentence: ConlluSentence,
    predictors: dict[str, ColumnPredictor],
    columns: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}

    for column in columns:
        predictor = predictors.get(column)
        if predictor is None:
            audit[column] = ColumnCorrectionAudit(status="checked").to_dict()
            continue

        patches: list[CorrectionPatch] = []
        token_results: list[TokenCorrectionResult] = []
        for token in sentence.tokens:
            result = _normalise_prediction_result(
                token=token,
                column=column,
                result=predictor(sentence, token, column, TokenEvidence()),
            )
            for patch in result.patches:
                sentence.apply_patch(patch.to_dict())
            patches.extend(result.patches)
            token_results.append(result)

        audit[column] = ColumnCorrectionAudit(
            status="patched" if patches else "checked",
            patches=patches,
            token_results=token_results,
        ).to_dict()

    return audit


def _normalise_prediction_result(
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


def _normalise_patch(patch: CorrectionPatchInput) -> CorrectionPatch:
    if isinstance(patch, CorrectionPatch):
        return patch
    return CorrectionPatch.model_validate(patch)
