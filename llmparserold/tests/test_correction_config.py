from pathlib import Path

from llmparser.correction_config import (
    ChainedSentenceChecker,
    ChainedTokenChecker,
    CorrectionConfig,
    build_checkers,
    load_correction_config,
)
from llmparser.conllu_model import ConlluSentence
from llmparser.correction import SentenceCorrectionResult, TokenCorrectionResult
from llmparser.evidence import TokenEvidence
from llmparser.feats_checker import RuleBasedFeatsChecker
from llmparser.lemma_checker import RuleBasedLemmaChecker
from llmparser.pos_checker import RuleBasedPosChecker
from llmparser.syntax_checker import RuleBasedSyntaxChecker


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, payload):
        self.calls.append(payload)
        return self.response


class StubTokenChecker:
    def __init__(self, status):
        self.status = status
        self.calls = 0

    def __call__(self, sentence, token, column, evidence):
        self.calls += 1
        return TokenCorrectionResult(token_id=token.id, column=column, status=self.status)


class StubSentenceChecker:
    def __init__(self, status):
        self.status = status
        self.calls = 0

    def __call__(self, sentence, examples):
        self.calls += 1
        return SentenceCorrectionResult(status=self.status)


def test_load_correction_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "correction.yml"
    config_path.write_text(
        """
segmentation:
  agent:
    skill: agents/correct_segmentation/SKILL.md
    only_when_uncertain: false
lemma:
  rule: true
upos_xpos:
  rule: true
feats:
  rule: false
head_deprel:
  rule: true
"""
    )

    config = load_correction_config(config_path)

    assert config.segmentation.agent is not None
    assert config.segmentation.agent.only_when_uncertain is False
    assert config.lemma.rule is True
    assert config.upos_xpos.rule is True
    assert config.feats.rule is False
    assert config.head_deprel.rule is True


def test_load_correction_config_reads_agent_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "correction.yml"
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# Skill\n")
    config_path.write_text(
        f"""
lemma:
  rule: true
  agent:
    skill: {skill_path}
    only_when_uncertain: true
"""
    )

    config = load_correction_config(config_path)

    assert config.lemma.agent is not None
    assert config.lemma.agent.skill == skill_path
    assert config.lemma.agent.only_when_uncertain is True


def test_build_checkers_from_config() -> None:
    config = CorrectionConfig.model_validate(
        {
            "lemma": {"rule": True},
            "upos_xpos": {"rule": True},
            "feats": {"rule": True},
            "head_deprel": {"rule": True},
        }
    )

    token_checkers, sentence_checkers, segmentation_checker = build_checkers(config)

    assert isinstance(token_checkers["lemma"], RuleBasedLemmaChecker)
    assert isinstance(token_checkers["upos"], RuleBasedPosChecker)
    assert token_checkers["upos"] is token_checkers["xpos"]
    assert isinstance(token_checkers["feats"], RuleBasedFeatsChecker)
    assert isinstance(sentence_checkers["head"], RuleBasedSyntaxChecker)
    assert sentence_checkers["head"] is sentence_checkers["deprel"]
    assert segmentation_checker is None


def test_build_checkers_with_empty_config_returns_no_checkers() -> None:
    token_checkers, sentence_checkers, segmentation_checker = build_checkers(CorrectionConfig())

    assert token_checkers == {}
    assert sentence_checkers == {}
    assert segmentation_checker is None


def test_build_checkers_can_create_agent_token_checker(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# Lemma Skill\n")
    client = FakeClient({"token_id": 1, "column": "lemma", "status": "checked"})
    config = CorrectionConfig.model_validate(
        {
            "lemma": {
                "agent": {
                    "skill": skill_path,
                    "only_when_uncertain": False,
                }
            }
        }
    )

    token_checkers, sentence_checkers, segmentation_checker = build_checkers(
        config,
        agent_clients={"lemma": client},
    )

    assert "lemma" in token_checkers
    assert sentence_checkers == {}
    assert segmentation_checker is None


def test_build_checkers_can_create_segmentation_agent_checker(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# Segmentation Skill\n")
    client = FakeClient({"status": "checked"})
    config = CorrectionConfig.model_validate(
        {
            "segmentation": {
                "agent": {
                    "skill": skill_path,
                    "only_when_uncertain": False,
                }
            }
        }
    )

    token_checkers, sentence_checkers, segmentation_checker = build_checkers(
        config,
        agent_clients={"segmentation": client},
    )

    assert token_checkers == {}
    assert sentence_checkers == {}
    assert segmentation_checker is not None


def test_chained_token_checker_calls_agent_only_when_uncertain() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	hello	INTJ	_	_	0	root	_	_
"""
    )
    rule_checker = StubTokenChecker("uncertain")
    agent_checker = StubTokenChecker("patched")
    checker = ChainedTokenChecker(rule_checker, agent_checker, only_when_uncertain=True)

    result = checker(sentence, sentence.tokens[0], "lemma", TokenEvidence())

    assert result.status == "patched"
    assert rule_checker.calls == 1
    assert agent_checker.calls == 1


def test_chained_token_checker_skips_agent_when_rule_checked() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	hello	INTJ	_	_	0	root	_	_
"""
    )
    rule_checker = StubTokenChecker("checked")
    agent_checker = StubTokenChecker("patched")
    checker = ChainedTokenChecker(rule_checker, agent_checker, only_when_uncertain=True)

    result = checker(sentence, sentence.tokens[0], "lemma", TokenEvidence())

    assert result.status == "checked"
    assert rule_checker.calls == 1
    assert agent_checker.calls == 0


def test_chained_sentence_checker_calls_agent_only_when_uncertain() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	hello	INTJ	_	_	0	root	_	_
"""
    )
    rule_checker = StubSentenceChecker("uncertain")
    agent_checker = StubSentenceChecker("patched")
    checker = ChainedSentenceChecker(rule_checker, agent_checker, only_when_uncertain=True)

    result = checker(sentence, [])

    assert result.status == "patched"
    assert rule_checker.calls == 1
    assert agent_checker.calls == 1
