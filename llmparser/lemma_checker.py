from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .conllu_model import ConlluSentence, ConlluToken
from .correction import CorrectionPatch, TokenCorrectionResult


def _example_metadata(example: Any) -> dict[str, Any]:
    if isinstance(example, dict):
        return example

    metadata = getattr(example, "metadata", None)
    if isinstance(metadata, dict):
        return metadata

    return {}


class RuleBasedLemmaChecker:
    def __init__(self, min_count: int = 2, min_ratio: float = 0.75):
        self.min_count = min_count
        self.min_ratio = min_ratio

    def __call__(
        self,
        sentence: ConlluSentence,
        token: ConlluToken,
        column: str,
        examples: list[dict[str, Any]],
    ) -> TokenCorrectionResult:
        if column != "lemma":
            return TokenCorrectionResult(
                token_id=token.id,
                column=column,
                status="checked",
                notes="RuleBasedLemmaChecker only checks lemma.",
            )

        counts: Counter[str] = Counter()
        lemma_evidence: dict[str, list[str]] = defaultdict(list)

        for example in examples:
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
                    if token.upos not in (None, "_", "") and training_token.upos != token.upos:
                        continue
                    if training_token.lemma in (None, "_", ""):
                        continue

                    lemma = str(training_token.lemma)
                    counts[lemma] += 1
                    if sent_id:
                        lemma_evidence[lemma].append(sent_id)

        if not counts:
            return TokenCorrectionResult(
                token_id=token.id,
                column="lemma",
                status="uncertain",
                notes=f"No training evidence found for form {token.form!r}.",
            )

        lemma, count = counts.most_common(1)[0]
        total = sum(counts.values())
        ratio = count / total
        evidence_sent_ids = list(dict.fromkeys(lemma_evidence[lemma]))

        if count < self.min_count or ratio < self.min_ratio:
            return TokenCorrectionResult(
                token_id=token.id,
                column="lemma",
                status="uncertain",
                evidence_sent_ids=evidence_sent_ids,
                notes=(
                    f"Best lemma {lemma!r} has {count}/{total} matching examples; "
                    "evidence is below the configured threshold."
                ),
            )

        if token.lemma == lemma:
            return TokenCorrectionResult(
                token_id=token.id,
                column="lemma",
                status="checked",
                evidence_sent_ids=evidence_sent_ids,
                notes=f"Predicted lemma matches strongest training evidence for form {token.form!r}.",
            )

        return TokenCorrectionResult(
            token_id=token.id,
            column="lemma",
            status="patched",
            patches=[
                CorrectionPatch(
                    token_id=token.id,
                    column="lemma",
                    value=lemma,
                    reason=(
                        f"Training examples use lemma {lemma!r} for form {token.form!r} "
                        f"in {count}/{total} comparable examples."
                    ),
                    evidence_sent_ids=evidence_sent_ids,
                )
            ],
            evidence_sent_ids=evidence_sent_ids,
            notes=f"Predicted lemma {token.lemma!r} differs from strongest training evidence.",
        )
