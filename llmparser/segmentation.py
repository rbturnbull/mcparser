from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .conllu_model import ConlluSentence, ConlluToken


class TokenSplitPatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_id: Any
    replacement_tokens: list[ConlluToken]
    reason: str | None = None
    evidence_sent_ids: list[str] = Field(default_factory=list)


class TokenMergePatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_ids: list[Any]
    replacement_token: ConlluToken
    reason: str | None = None
    evidence_sent_ids: list[str] = Field(default_factory=list)


class TokenDeletePatch(BaseModel):
    token_id: Any
    reason: str | None = None
    evidence_sent_ids: list[str] = Field(default_factory=list)


class TokenInsertPatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    anchor_token_id: Any
    position: Literal["before", "after"] = "after"
    tokens: list[ConlluToken]
    reason: str | None = None
    evidence_sent_ids: list[str] = Field(default_factory=list)


SegmentationPatch = TokenSplitPatch | TokenMergePatch | TokenDeletePatch | TokenInsertPatch


class SegmentationCorrectionResult(BaseModel):
    status: str = "checked"
    patches: list[SegmentationPatch] = Field(default_factory=list)
    needs_syntax_correction: bool = False
    evidence_sent_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def apply_token_split(
    sentence: ConlluSentence,
    patch: TokenSplitPatch,
) -> SegmentationCorrectionResult:
    replacement_count = len(patch.replacement_tokens)
    if replacement_count == 0:
        return SegmentationCorrectionResult(
            status="uncertain",
            notes=f"TokenSplitPatch for token {patch.token_id!r} has no replacement tokens.",
        )

    tokens: list[ConlluToken] = []
    found = False
    next_id = 1
    id_shift = replacement_count - 1

    for token in sentence.tokens:
        if token.id == patch.token_id:
            found = True
            for replacement in patch.replacement_tokens:
                replacement.id = next_id
                tokens.append(replacement)
                next_id += 1
            continue

        if found and isinstance(token.id, int):
            token.id = token.id + id_shift
        if isinstance(token.id, int):
            token.id = next_id
            next_id += 1
        tokens.append(token)

    if not found:
        return SegmentationCorrectionResult(
            status="uncertain",
            notes=f"No token with id {patch.token_id!r} was found for segmentation split.",
        )

    sentence.tokens = tokens
    return SegmentationCorrectionResult(
        status="patched",
        patches=[patch],
        needs_syntax_correction=True,
        evidence_sent_ids=patch.evidence_sent_ids,
        notes="Applied token split; downstream syntax correction should be rerun.",
    )


def apply_token_merge(
    sentence: ConlluSentence,
    patch: TokenMergePatch,
) -> SegmentationCorrectionResult:
    token_id_set = set(patch.token_ids)
    if not token_id_set:
        return SegmentationCorrectionResult(
            status="uncertain",
            notes="TokenMergePatch has no token IDs.",
        )

    tokens: list[ConlluToken] = []
    inserted = False
    found_ids: set[Any] = set()
    for token in sentence.tokens:
        if token.id in token_id_set:
            found_ids.add(token.id)
            if not inserted:
                tokens.append(patch.replacement_token)
                inserted = True
            continue
        tokens.append(token)

    if found_ids != token_id_set:
        missing = sorted(token_id_set - found_ids)
        return SegmentationCorrectionResult(
            status="uncertain",
            notes=f"Cannot merge missing token IDs: {missing}.",
        )

    sentence.tokens = _renumber_integer_tokens(tokens)
    return _patched_result(patch, "Applied token merge; downstream syntax correction should be rerun.")


def apply_token_delete(
    sentence: ConlluSentence,
    patch: TokenDeletePatch,
) -> SegmentationCorrectionResult:
    tokens = [token for token in sentence.tokens if token.id != patch.token_id]
    if len(tokens) == len(sentence.tokens):
        return SegmentationCorrectionResult(
            status="uncertain",
            notes=f"No token with id {patch.token_id!r} was found for deletion.",
        )

    sentence.tokens = _renumber_integer_tokens(tokens)
    return _patched_result(patch, "Applied token delete; downstream syntax correction should be rerun.")


def apply_token_insert(
    sentence: ConlluSentence,
    patch: TokenInsertPatch,
) -> SegmentationCorrectionResult:
    tokens: list[ConlluToken] = []
    found = False
    for token in sentence.tokens:
        if patch.position == "before" and token.id == patch.anchor_token_id:
            tokens.extend(patch.tokens)
            found = True
        tokens.append(token)
        if patch.position == "after" and token.id == patch.anchor_token_id:
            tokens.extend(patch.tokens)
            found = True

    if not found:
        return SegmentationCorrectionResult(
            status="uncertain",
            notes=f"No anchor token with id {patch.anchor_token_id!r} was found for insertion.",
        )

    sentence.tokens = _renumber_integer_tokens(tokens)
    return _patched_result(patch, "Applied token insert; downstream syntax correction should be rerun.")


def apply_segmentation_patch(
    sentence: ConlluSentence,
    patch: SegmentationPatch | dict[str, Any],
) -> SegmentationCorrectionResult:
    if isinstance(patch, dict):
        patch_type = patch.get("type")
        patch_data = {key: value for key, value in patch.items() if key != "type"}
        if patch_type == "split":
            patch = TokenSplitPatch.model_validate(patch_data)
        elif patch_type == "merge":
            patch = TokenMergePatch.model_validate(patch_data)
        elif patch_type == "delete":
            patch = TokenDeletePatch.model_validate(patch_data)
        elif patch_type == "insert":
            patch = TokenInsertPatch.model_validate(patch_data)
        else:
            raise ValueError(f"Unsupported segmentation patch type: {patch_type!r}")

    if isinstance(patch, TokenSplitPatch):
        return apply_token_split(sentence, patch)
    if isinstance(patch, TokenMergePatch):
        return apply_token_merge(sentence, patch)
    if isinstance(patch, TokenDeletePatch):
        return apply_token_delete(sentence, patch)
    if isinstance(patch, TokenInsertPatch):
        return apply_token_insert(sentence, patch)
    raise TypeError(f"Unsupported segmentation patch: {type(patch)!r}")


def _renumber_integer_tokens(tokens: list[ConlluToken]) -> list[ConlluToken]:
    next_id = 1
    for token in tokens:
        if isinstance(token.id, int):
            token.id = next_id
            next_id += 1
    return tokens


def _patched_result(
    patch: SegmentationPatch,
    notes: str,
) -> SegmentationCorrectionResult:
    return SegmentationCorrectionResult(
        status="patched",
        patches=[patch],
        needs_syntax_correction=True,
        evidence_sent_ids=patch.evidence_sent_ids,
        notes=notes,
    )
