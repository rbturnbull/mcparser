from pathlib import Path

import pytest
from pydantic import ValidationError

from llmparser.agent_checker import AgentSegmentationChecker, AgentSentenceChecker, AgentTokenChecker
from llmparser.conllu_model import ConlluSentence
from llmparser.evidence import SegmentationEvidence, TokenEvidence


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return self.response


def test_agent_token_checker_loads_skill_and_validates_response(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# Lemma Skill\nReturn a TokenCorrectionResult.\n")
    client = FakeClient(
        {
            "token_id": 1,
            "column": "lemma",
            "status": "patched",
            "patches": [
                {
                    "token_id": 1,
                    "column": "lemma",
                    "value": "δοκέω",
                    "reason": "Agent found matching examples.",
                    "evidence_sent_ids": ["train1"],
                }
            ],
            "evidence_sent_ids": ["train1"],
            "notes": "Corrected from training evidence.",
        }
    )
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = ἐδόκεε
1	ἐδόκεε	ἐδόκεε	VERB	V-	_	0	root	_	_
"""
    )
    evidence = TokenEvidence(form_examples=[{"sent_id": "train1"}])

    result = AgentTokenChecker(skill_path, client)(sentence, sentence.tokens[0], "lemma", evidence)

    assert result.status == "patched"
    assert result.patches[0].value == "δοκέω"
    payload = client.payloads[0]
    assert payload["skill"] == "# Lemma Skill\nReturn a TokenCorrectionResult.\n"
    assert payload["column"] == "lemma"
    assert payload["token"]["form"] == "ἐδόκεε"
    assert payload["evidence"]["form_examples"] == [{"sent_id": "train1"}]


def test_agent_token_checker_rejects_malformed_response(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# Lemma Skill\n")
    client = FakeClient({"token_id": 1, "column": "not-a-column", "status": "patched"})
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	hello	INTJ	_	_	0	root	_	_
"""
    )

    with pytest.raises(ValidationError):
        AgentTokenChecker(skill_path, client)(sentence, sentence.tokens[0], "lemma", TokenEvidence())


def test_agent_sentence_checker_loads_skill_and_validates_response(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# Syntax Skill\nReturn a SentenceCorrectionResult.\n")
    client = FakeClient(
        {
            "status": "patched",
            "patches": [
                {
                    "token_id": 1,
                    "column": "head",
                    "value": 0,
                    "reason": "root must have HEAD 0",
                }
            ],
            "notes": "Applied root repair.",
        }
    )
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Broken
1	Broken	broken	ADJ	_	_	2	root	_	_
2	token	token	NOUN	_	_	1	nsubj	_	_
"""
    )

    result = AgentSentenceChecker(skill_path, client)(sentence, [{"sent_id": "train1"}])

    assert result.status == "patched"
    assert result.patches[0].column == "head"
    payload = client.payloads[0]
    assert payload["skill"] == "# Syntax Skill\nReturn a SentenceCorrectionResult.\n"
    assert payload["sentence"]["sent_id"] == "pred"
    assert payload["examples"] == [{"sent_id": "train1"}]


def test_agent_segmentation_checker_includes_segmentation_evidence(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# Segmentation Skill\nReturn a SegmentationCorrectionResult.\n")
    client = FakeClient({"status": "checked"})
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = وكتابه جاء
1	وكتابه	وكتابه	NOUN	_	_	2	nsubj	_	_
2	جاء	جاء	VERB	_	_	0	root	_	_
"""
    )
    evidence = SegmentationEvidence(surface_examples=[{"sent_id": "train1"}])

    result = AgentSegmentationChecker(skill_path, client)(sentence, [], evidence)

    assert result.status == "checked"
    payload = client.payloads[0]
    assert payload["skill"] == "# Segmentation Skill\nReturn a SegmentationCorrectionResult.\n"
    assert payload["sentence"]["sent_id"] == "pred"
    assert payload["segmentation_evidence"]["surface_examples"] == [{"sent_id": "train1"}]
