from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .agent_checker import (
    AgentClient,
    AgentSegmentationChecker,
    AgentSentenceChecker,
    AgentTokenChecker,
)
from .correction import SentenceCorrectionResult, TokenCorrectionResult
from .feats_checker import RuleBasedFeatsChecker
from .lemma_checker import RuleBasedLemmaChecker
from .pos_checker import RuleBasedPosChecker
from .syntax_checker import RuleBasedSyntaxChecker


class AgentConfig(BaseModel):
    skill: Path
    only_when_uncertain: bool = True


class CheckerConfig(BaseModel):
    rule: bool = False
    agent: AgentConfig | None = None


class CorrectionConfig(BaseModel):
    segmentation: CheckerConfig = CheckerConfig()
    lemma: CheckerConfig = CheckerConfig()
    upos_xpos: CheckerConfig = CheckerConfig()
    feats: CheckerConfig = CheckerConfig()
    head_deprel: CheckerConfig = CheckerConfig()


class ChainedTokenChecker:
    def __init__(self, first, second, only_when_uncertain: bool = True):
        self.first = first
        self.second = second
        self.only_when_uncertain = only_when_uncertain

    def __call__(self, sentence, token, column, evidence) -> TokenCorrectionResult:
        first_result = TokenCorrectionResult.model_validate(
            self.first(sentence, token, column, evidence)
        )
        if self.only_when_uncertain and first_result.status != "uncertain":
            return first_result
        return TokenCorrectionResult.model_validate(self.second(sentence, token, column, evidence))


class ChainedSentenceChecker:
    def __init__(self, first, second, only_when_uncertain: bool = True):
        self.first = first
        self.second = second
        self.only_when_uncertain = only_when_uncertain

    def __call__(self, sentence, examples) -> SentenceCorrectionResult:
        first_result = SentenceCorrectionResult.model_validate(self.first(sentence, examples))
        if self.only_when_uncertain and first_result.status != "uncertain":
            return first_result
        return SentenceCorrectionResult.model_validate(self.second(sentence, examples))


def load_correction_config(path: Path | str) -> CorrectionConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return CorrectionConfig.model_validate(data)


def config_has_agent_checkers(config: CorrectionConfig) -> bool:
    return any(
        checker.agent is not None
        for checker in (
            config.segmentation,
            config.lemma,
            config.upos_xpos,
            config.feats,
            config.head_deprel,
        )
    )


def build_checkers(
    config: CorrectionConfig,
    agent_clients: dict[str, AgentClient] | None = None,
):
    agent_clients = agent_clients or {}
    token_checkers = {}
    sentence_checkers = {}
    segmentation_checker = _build_segmentation_checker(
        config.segmentation,
        agent_key="segmentation",
        agent_clients=agent_clients,
    )

    lemma_checker = _build_token_checker(
        config.lemma,
        rule_checker=RuleBasedLemmaChecker() if config.lemma.rule else None,
        agent_key="lemma",
        agent_clients=agent_clients,
    )
    if lemma_checker is not None:
        token_checkers["lemma"] = lemma_checker

    pos_checker = _build_token_checker(
        config.upos_xpos,
        rule_checker=RuleBasedPosChecker() if config.upos_xpos.rule else None,
        agent_key="upos_xpos",
        agent_clients=agent_clients,
    )
    if pos_checker is not None:
        checker = pos_checker
        token_checkers["upos"] = checker
        token_checkers["xpos"] = checker

    feats_checker = _build_token_checker(
        config.feats,
        rule_checker=RuleBasedFeatsChecker() if config.feats.rule else None,
        agent_key="feats",
        agent_clients=agent_clients,
    )
    if feats_checker is not None:
        token_checkers["feats"] = feats_checker

    syntax_checker = _build_sentence_checker(
        config.head_deprel,
        rule_checker=RuleBasedSyntaxChecker() if config.head_deprel.rule else None,
        agent_key="head_deprel",
        agent_clients=agent_clients,
    )
    if syntax_checker is not None:
        checker = syntax_checker
        sentence_checkers["head"] = checker
        sentence_checkers["deprel"] = checker

    return token_checkers, sentence_checkers, segmentation_checker


def _build_token_checker(
    config: CheckerConfig,
    rule_checker: Any | None,
    agent_key: str,
    agent_clients: dict[str, AgentClient],
):
    agent_checker = None
    if config.agent is not None:
        agent_checker = AgentTokenChecker(
            config.agent.skill,
            _agent_client(agent_key, agent_clients),
        )

    if rule_checker is not None and agent_checker is not None:
        return ChainedTokenChecker(
            rule_checker,
            agent_checker,
            only_when_uncertain=config.agent.only_when_uncertain,
        )
    return agent_checker or rule_checker


def _build_sentence_checker(
    config: CheckerConfig,
    rule_checker: Any | None,
    agent_key: str,
    agent_clients: dict[str, AgentClient],
):
    agent_checker = None
    if config.agent is not None:
        agent_checker = AgentSentenceChecker(
            config.agent.skill,
            _agent_client(agent_key, agent_clients),
        )

    if rule_checker is not None and agent_checker is not None:
        return ChainedSentenceChecker(
            rule_checker,
            agent_checker,
            only_when_uncertain=config.agent.only_when_uncertain,
        )
    return agent_checker or rule_checker


def _build_segmentation_checker(
    config: CheckerConfig,
    agent_key: str,
    agent_clients: dict[str, AgentClient],
):
    if config.rule:
        raise ValueError("Rule-based segmentation checker is not implemented.")
    if config.agent is None:
        return None
    return AgentSegmentationChecker(
        config.agent.skill,
        _agent_client(agent_key, agent_clients),
    )


def _agent_client(agent_key: str, agent_clients: dict[str, AgentClient]) -> AgentClient:
    try:
        return agent_clients[agent_key]
    except KeyError as exc:
        raise ValueError(f"No agent client configured for {agent_key!r}") from exc
