from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .conllu_model import ConlluSentence, ConlluToken
from .correction import CorrectionPatch, TokenCorrectionResult
from .evidence import TokenEvidence
from .lemma_checker import _example_metadata


def _normalise_feats(feats: Any) -> tuple[tuple[str, Any], ...]:
    if feats in (None, "_", ""):
        return ()
    if isinstance(feats, dict):
        return tuple(sorted(feats.items()))
    return (("value", str(feats)),)


def _feats_value(feats_key: tuple[tuple[str, Any], ...]) -> dict[str, Any] | str:
    if not feats_key:
        return "_"
    if len(feats_key) == 1 and feats_key[0][0] == "value":
        return str(feats_key[0][1])
    return dict(feats_key)


class RuleBasedFeatsChecker:
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
        if column != "feats":
            return TokenCorrectionResult(
                token_id=token.id,
                column=column,
                status="checked",
                notes="RuleBasedFeatsChecker only checks feats.",
            )

        candidate_examples = examples.form_examples if isinstance(examples, TokenEvidence) else examples
        counts: Counter[tuple[tuple[str, Any], ...]] = Counter()
        feats_evidence: dict[tuple[tuple[str, Any], ...], list[str]] = defaultdict(list)

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
                    if token.upos not in (None, "_", "") and training_token.upos != token.upos:
                        continue
                    if token.xpos not in (None, "_", "") and training_token.xpos != token.xpos:
                        continue

                    feats_key = _normalise_feats(training_token.feats)
                    counts[feats_key] += 1
                    if sent_id:
                        feats_evidence[feats_key].append(sent_id)

        if not counts:
            return TokenCorrectionResult(
                token_id=token.id,
                column="feats",
                status="uncertain",
                notes=f"No FEATS evidence found for form {token.form!r}.",
            )

        feats_key, count = counts.most_common(1)[0]
        total = sum(counts.values())
        ratio = count / total
        evidence_sent_ids = list(dict.fromkeys(feats_evidence[feats_key]))
        expected_feats = _feats_value(feats_key)

        if count < self.min_count or ratio < self.min_ratio:
            return TokenCorrectionResult(
                token_id=token.id,
                column="feats",
                status="uncertain",
                evidence_sent_ids=evidence_sent_ids,
                notes=(
                    f"Best FEATS value {expected_feats!r} has {count}/{total} matching examples; "
                    "evidence is below the configured threshold."
                ),
            )

        if _normalise_feats(token.feats) == feats_key:
            return TokenCorrectionResult(
                token_id=token.id,
                column="feats",
                status="checked",
                evidence_sent_ids=evidence_sent_ids,
                notes=f"Predicted FEATS match strongest training evidence for form {token.form!r}.",
            )

        return TokenCorrectionResult(
            token_id=token.id,
            column="feats",
            status="patched",
            patches=[
                CorrectionPatch(
                    token_id=token.id,
                    column="feats",
                    value=expected_feats,
                    reason=(
                        f"Training examples use FEATS {expected_feats!r} for form {token.form!r} "
                        f"in {count}/{total} comparable examples."
                    ),
                    evidence_sent_ids=evidence_sent_ids,
                )
            ],
            evidence_sent_ids=evidence_sent_ids,
            notes=f"Predicted FEATS {token.feats!r} differ from strongest training evidence.",
        )
