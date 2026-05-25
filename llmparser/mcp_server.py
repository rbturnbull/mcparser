from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .conllu_model import ConlluSentence
from .correction import CorrectionPatch
from .evidence import TrainingEvidence
from .segmentation import TokenSplitPatch, apply_segmentation_patch, apply_token_split
from .validation import validate_sentence


class CorrectionSession:
    def __init__(self, sentences: dict[str, ConlluSentence], database):
        self.sentences = sentences
        self.evidence = TrainingEvidence(database)

    @classmethod
    def from_conllu_text(cls, conllu_text: str, database) -> "CorrectionSession":
        sentences = ConlluSentence.iter_from_text(conllu_text)
        return cls(_index_sentences(sentences), database)

    @classmethod
    def from_conllu_file(cls, path: Path | str, database) -> "CorrectionSession":
        return cls.from_conllu_text(Path(path).read_text(encoding="utf-8"), database)

    def _sentence(self, item_id: str) -> ConlluSentence:
        try:
            return self.sentences[item_id]
        except KeyError as exc:
            raise KeyError(f"No prediction sentence with item_id {item_id!r}") from exc

    def get_sentence(self, item_id: str) -> dict[str, Any]:
        return self._sentence(item_id).to_dict()

    def get_token(self, item_id: str, token_id: Any) -> dict[str, Any]:
        return self._sentence(item_id).token_by_id(token_id).to_dict()

    def get_token_evidence(
        self,
        item_id: str,
        token_id: Any,
        column: str,
        k: int = 10,
    ) -> dict[str, Any]:
        sentence = self._sentence(item_id)
        token = sentence.token_by_id(token_id)
        return self.evidence.find_token_evidence(sentence, token, column, k=k).model_dump()

    def get_segmentation_evidence(
        self,
        item_id: str,
        surface: str,
        pieces: list[str],
        k: int = 10,
    ) -> dict[str, Any]:
        sentence = self._sentence(item_id)
        return self.evidence.find_segmentation_evidence(
            surface=surface,
            pieces=pieces,
            sentence_text=sentence.text,
            k=k,
        ).model_dump()

    def validate_prediction(self, item_id: str) -> list[dict[str, Any]]:
        return [error.to_dict() for error in validate_sentence(self._sentence(item_id))]

    def apply_patch(self, item_id: str, patch: CorrectionPatch | dict[str, Any]) -> dict[str, Any]:
        patch = patch if isinstance(patch, CorrectionPatch) else CorrectionPatch.model_validate(patch)
        sentence = self._sentence(item_id)
        sentence.apply_patch(patch.to_dict())
        return sentence.to_dict()

    def apply_token_split(self, item_id: str, patch: TokenSplitPatch | dict[str, Any]) -> dict[str, Any]:
        patch = patch if isinstance(patch, TokenSplitPatch) else TokenSplitPatch.model_validate(patch)
        return apply_token_split(self._sentence(item_id), patch).to_dict()

    def apply_segmentation_patch(self, item_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        return apply_segmentation_patch(self._sentence(item_id), patch).to_dict()

    def serialize_sentence(self, item_id: str) -> str:
        return self._sentence(item_id).to_conllu()


class CorrectionToolRegistry:
    def __init__(self, session: CorrectionSession):
        self.session = session
        self._tools: dict[str, Callable[..., Any]] = {
            "apply_patch": self.session.apply_patch,
            "apply_segmentation_patch": self.session.apply_segmentation_patch,
            "apply_token_split": self.session.apply_token_split,
            "get_sentence": self.session.get_sentence,
            "get_segmentation_evidence": self.session.get_segmentation_evidence,
            "get_token": self.session.get_token,
            "get_token_evidence": self.session.get_token_evidence,
            "serialize_sentence": self.session.serialize_sentence,
            "validate_prediction": self.session.validate_prediction,
        }

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def call_tool(self, name: str, **kwargs) -> Any:
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"No correction tool named {name!r}") from exc
        return tool(**kwargs)


def _index_sentences(sentences: list[ConlluSentence]) -> dict[str, ConlluSentence]:
    indexed: dict[str, ConlluSentence] = {}
    for index, sentence in enumerate(sentences):
        item_id = sentence.sent_id or f"sentence_{index + 1}"
        indexed[item_id] = sentence
    return indexed
