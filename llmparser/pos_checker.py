from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .conllu_model import ConlluSentence, ConlluToken
from .correction import CorrectionPatch, TokenCorrectionResult
from .evidence import TokenEvidence
from .lemma_checker import _example_metadata


class RuleBasedPosChecker:
    def __init__(self, min_count: int = 2, min_ratio: float = 0.75):
        self.min_count = min_count
        self.min_ratio = min_ratio

    def __call__(
        self,
        sentence: ConlluSentence,
        token: ConlluToken,
        column: str,
        examples: TokenEvidence | list[dict[str, Any]],
    ) -> TokenCorrectionResult:
        if column not in {"upos", "xpos"}:
            return TokenCorrectionResult(
                token_id=token.id,
                column=column,
                status="checked",
                notes="RuleBasedPosChecker only checks upos/xpos.",
            )

        candidate_examples = examples.form_examples if isinstance(examples, TokenEvidence) else examples
        counts: Counter[tuple[str | None, str | None]] = Counter()
        pair_evidence: dict[tuple[str | None, str | None], list[str]] = defaultdict(list)

        for example in candidate_examples:
            metadata = _example_metadata(example)
            conllu_text = metadata.get("conllu")
            if not conllu_text:
                continue
            sent_id = str(metadata.get("sent_id", ""))

            for training_sentence in ConlluSentence.iter_from_text(conllu_text):
                if not sent_id and training_sentence.sent_id:
                    sent_id = training_sentence.sent_id
                for training_token in training_sentence.tokens:
                    if training_token.form != token.form:
                        continue
                    if token.lemma not in (None, "_", "") and training_token.lemma != token.lemma:
                        continue
                    if training_token.upos in (None, ""):
                        continue

                    pair = (training_token.upos, training_token.xpos)
                    counts[pair] += 1
                    if sent_id:
                        pair_evidence[pair].append(sent_id)

        if not counts:
            return TokenCorrectionResult(
                token_id=token.id,
                column=column,
                status="uncertain",
                notes=f"No POS evidence found for form {token.form!r}.",
            )

        pair, count = counts.most_common(1)[0]
        total = sum(counts.values())
        ratio = count / total
        evidence_sent_ids = list(dict.fromkeys(pair_evidence[pair]))
        upos, xpos = pair

        if count < self.min_count or ratio < self.min_ratio:
            return TokenCorrectionResult(
                token_id=token.id,
                column=column,
                status="uncertain",
                evidence_sent_ids=evidence_sent_ids,
                notes=(
                    f"Best POS pair {(upos, xpos)!r} has {count}/{total} matching examples; "
                    "evidence is below the configured threshold."
                ),
            )

        patches: list[CorrectionPatch] = []
        if token.upos != upos:
            patches.append(
                CorrectionPatch(
                    token_id=token.id,
                    column="upos",
                    value=upos,
                    reason=(
                        f"Training examples use UPOS {upos!r} for form {token.form!r} "
                        f"in {count}/{total} comparable examples."
                    ),
                    evidence_sent_ids=evidence_sent_ids,
                )
            )
        if token.xpos != xpos:
            patches.append(
                CorrectionPatch(
                    token_id=token.id,
                    column="xpos",
                    value=xpos,
                    reason=(
                        f"Training examples use XPOS {xpos!r} for form {token.form!r} "
                        f"in {count}/{total} comparable examples."
                    ),
                    evidence_sent_ids=evidence_sent_ids,
                )
            )

        if not patches:
            return TokenCorrectionResult(
                token_id=token.id,
                column=column,
                status="checked",
                evidence_sent_ids=evidence_sent_ids,
                notes=f"Predicted POS matches strongest training evidence for form {token.form!r}.",
            )

        return TokenCorrectionResult(
            token_id=token.id,
            column=column,
            status="patched",
            patches=patches,
            evidence_sent_ids=evidence_sent_ids,
            notes=f"Predicted POS differs from strongest training evidence for form {token.form!r}.",
        )
