from __future__ import annotations

from typing import Any

from .conllu_model import ConlluSentence
from .correction import CorrectionPatch, SentenceCorrectionResult


class RuleBasedSyntaxChecker:
    def __call__(
        self,
        sentence: ConlluSentence,
        examples: list[dict[str, Any]],
    ) -> SentenceCorrectionResult:
        token_ids = {token.id for token in sentence.tokens if isinstance(token.id, int)}
        root_tokens = [
            token
            for token in sentence.tokens
            if isinstance(token.id, int) and token.head == 0
        ]
        deprel_root_tokens = [
            token
            for token in sentence.tokens
            if isinstance(token.id, int) and token.deprel == "root"
        ]

        missing_heads = [
            token
            for token in sentence.tokens
            if isinstance(token.id, int) and token.head != 0 and token.head not in token_ids
        ]
        if missing_heads:
            missing = ", ".join(f"{token.id}->{token.head}" for token in missing_heads)
            return SentenceCorrectionResult(
                status="uncertain",
                notes=f"Cannot safely repair missing HEAD references: {missing}.",
            )

        if len(root_tokens) != 1 and len(deprel_root_tokens) != 1:
            return SentenceCorrectionResult(
                status="uncertain",
                notes=f"Cannot safely repair root count {len(root_tokens)}; expected exactly 1 root.",
            )

        patches: list[CorrectionPatch] = []
        for token in sentence.tokens:
            if not isinstance(token.id, int):
                continue

            if token.deprel == "root" and token.head != 0:
                patches.append(
                    CorrectionPatch(
                        token_id=token.id,
                        column="head",
                        value=0,
                        reason="Token has DEPREL root, so HEAD should be 0.",
                    )
                )
            elif token.head == 0 and token.deprel != "root":
                patches.append(
                    CorrectionPatch(
                        token_id=token.id,
                        column="deprel",
                        value="root",
                        reason="Token has HEAD 0, so DEPREL should be root.",
                    )
                )

        if not patches:
            return SentenceCorrectionResult(
                status="checked",
                notes="No deterministic HEAD/DEPREL correction needed.",
            )

        return SentenceCorrectionResult(
            status="patched",
            patches=patches,
            notes="Applied deterministic HEAD/DEPREL consistency fixes.",
        )
