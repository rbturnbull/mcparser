from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .conllu_model import ConlluSentence, ConlluToken
from .correction import SentenceCorrectionResult, TokenCorrectionResult
from .evidence import SegmentationEvidence, TokenEvidence
from .segmentation import SegmentationCorrectionResult


AgentClient = Callable[[dict[str, Any]], dict[str, Any]]


class AgentTokenChecker:
    def __init__(self, skill_path: Path | str, client: AgentClient):
        self.skill_path = Path(skill_path)
        self.client = client
        self.skill = self.skill_path.read_text()

    def __call__(
        self,
        sentence: ConlluSentence,
        token: ConlluToken,
        column: str,
        evidence: TokenEvidence,
    ) -> TokenCorrectionResult:
        payload = {
            "skill": self.skill,
            "sentence": sentence.to_dict(),
            "token": token.to_dict(),
            "column": column,
            "evidence": evidence.model_dump(),
        }
        response = self.client(payload)
        return TokenCorrectionResult.model_validate(response)


class AgentSentenceChecker:
    def __init__(self, skill_path: Path | str, client: AgentClient):
        self.skill_path = Path(skill_path)
        self.client = client
        self.skill = self.skill_path.read_text()

    def __call__(
        self,
        sentence: ConlluSentence,
        examples: list[dict[str, Any]],
    ) -> SentenceCorrectionResult:
        payload = {
            "skill": self.skill,
            "sentence": sentence.to_dict(),
            "examples": examples,
        }
        response = self.client(payload)
        return SentenceCorrectionResult.model_validate(response)


class AgentSegmentationChecker:
    def __init__(self, skill_path: Path | str, client: AgentClient):
        self.skill_path = Path(skill_path)
        self.client = client
        self.skill = self.skill_path.read_text()

    def __call__(
        self,
        sentence: ConlluSentence,
        examples: list[dict[str, Any]],
        segmentation_evidence: SegmentationEvidence | None = None,
    ) -> SegmentationCorrectionResult:
        payload = {
            "skill": self.skill,
            "sentence": sentence.to_dict(),
            "examples": examples,
            "segmentation_evidence": (
                segmentation_evidence.model_dump()
                if segmentation_evidence is not None
                else None
            ),
        }
        response = self.client(payload)
        return SegmentationCorrectionResult.model_validate(response)
