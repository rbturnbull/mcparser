from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .conllu_model import ConlluSentence
from .io import write_yaml
from .validation import validate_sentence


DEFAULT_CHECK_COLUMNS = (
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
)

Checker = Callable[[ConlluSentence, str, list[dict[str, Any]]], list[dict[str, Any]]]


def load_prediction_sentences(path: Path | str) -> list[ConlluSentence]:
    return ConlluSentence.iter_from_text(Path(path).read_text(encoding="utf-8"))


def _sentence_key(sentence: ConlluSentence, index: int) -> str:
    return sentence.sent_id or f"sentence_{index + 1}"


def _normalise_patch(patch: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(patch)
    if "column" in normalised:
        normalised["column"] = str(normalised["column"]).lower()
    return normalised


def check_external_predictions(
    prediction_path: Path | str,
    database,
    checker: Checker | None = None,
    n_examples: int = 10,
    output: Path | str | None = None,
    columns: tuple[str, ...] = DEFAULT_CHECK_COLUMNS,
) -> dict[str, dict[str, Any]]:
    sentences = load_prediction_sentences(prediction_path)
    checked: dict[str, dict[str, Any]] = {}

    for index, sentence in enumerate(sentences):
        examples = database(sentence.text, k=n_examples)
        audit: dict[str, Any] = {
            "validation": [error.to_dict() for error in validate_sentence(sentence)],
        }

        for column in columns:
            patches: list[dict[str, Any]] = []
            if checker is not None:
                patches = [_normalise_patch(patch) for patch in checker(sentence, column, examples)]
                for patch in patches:
                    sentence.apply_patch(patch)

            audit[column] = {
                "patches": patches,
                "status": "patched" if patches else "checked",
            }

        post_validation = [error.to_dict() for error in validate_sentence(sentence)]
        if post_validation:
            audit["post_validation"] = post_validation

        item = sentence.to_dict()
        item["audit"] = audit
        checked[_sentence_key(sentence, index)] = item

    if output is not None:
        write_yaml(checked, output)

    return checked
